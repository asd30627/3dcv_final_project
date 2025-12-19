from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CV_NUM_THREADS"] = "1" # 考慮一下有沒有要加? 配合 Line 37: cv2.setNumThreads(0)


import argparse
import os.path as osp
import time
import random
import numpy as np
import logging
from datetime import timedelta
import copy
import math
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import cv2

from mmengine import Config
from model.student import OccStudent
from dataset import get_dataloader

# =========================================================
# 多線程控制：OpenCV + PyTorch
# =========================================================

# # 關掉 OpenCV 自動多線程（很重要，不然每個 worker 再開一堆 thread）
cv2.setNumThreads(0)

import torch
import torch.nn.functional as F

# -------------------------
# KD verification / reduction switches (預設不會大量印 log)
# - KD_KL_REDUCTION: "mean"(穩定, 預設) 或 "batchmean"(保留舊行為)
# - KD_VERIFY=1 時會印出 teacher 是否為 prob、sum_over_class、KL ratio 等驗證資訊
# -------------------------
KD_VERIFY = bool(int(os.environ.get("KD_VERIFY", "0")))
KD_VERIFY_FIRST_CALLS = int(os.environ.get("KD_VERIFY_FIRST_CALLS", "3"))
KD_VERIFY_EVERY_CALLS = int(os.environ.get("KD_VERIFY_EVERY_CALLS", "2000"))
KD_EPS = float(os.environ.get("KD_EPS", "1e-6"))
KD_KL_REDUCTION = os.environ.get("KD_KL_REDUCTION", "mean").lower().strip()  # "mean" or "batchmean"
_KD_CALL_COUNTER = 0


def _infer_class_dim(s_logits, K=18):
    # 支援 (B,K,...) 或 (B,...,K)
    if s_logits.dim() >= 2 and s_logits.shape[1] == K:
        return 1
    if s_logits.dim() >= 1 and s_logits.shape[-1] == K:
        return -1
    raise ValueError(f"Cannot infer class dim from s_logits shape={tuple(s_logits.shape)} with K={K}")

def _squeeze_teacher(t, K=18):
    """
    把 teacher tensor 壓成常見型態：
    - BEV:  (B,K,H,W) 或 (K,H,W)
    - 3D:   (B,K,H,W,Z) 或 (K,H,W,Z)

    允許輸入：
    (B,1,K,H,W) / (B,1,K,H,W,Z) 這種多一個 1 維的情況
    """
    if t is None:
        return None
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    # 常見：t = (B,1,K,H,W) or (B,1,K,H,W,Z)
    if t.dim() >= 5 and t.shape[1] == 1 and t.shape[2] == K:
        t = t.squeeze(1)  # -> (B,K,H,W) or (B,K,H,W,Z)
    # 次常見：t = (1,K,H,W) collate 後變 (B,1,K,H,W) 已處理；若還有多餘 1 維再保守 squeeze
    # 只 squeeze「明顯是多餘」且不會吃到 class 維的位置
    while t.dim() >= 5 and t.shape[0] == 1 and t.shape[1] == K and t.shape[2] == 1:
        # 例如極少數 (1,K,1,H,W) 這種怪格式
        t = t.squeeze(2)
    return t

def _align_teacher_to_student(t, s_logits, K=18):
    """
    讓 teacher 的 layout 對齊 student layout。
    teacher:
      BEV: (B,K,H,W) or (K,H,W)
      3D : (B,K,H,W,Z) or (K,H,W,Z)
    student:
      (B,K,...) or (B,...,K)
    回傳:
      (B, same_layout_as_student_without_batch)
    """
    cd = _infer_class_dim(s_logits, K=K)
    B = int(s_logits.shape[0]) if s_logits.dim() >= 1 else 1

    t = _squeeze_teacher(t, K=K)

    if t.dim() in (3, 4):  # (K,H,W) or (K,H,W,Z)  -> 加 batch
        t = t.unsqueeze(0)  # (1,K,...)
    elif t.dim() in (4, 5):
        # already has batch
        pass
    else:
        raise ValueError(f"Unexpected teacher dim={t.dim()} shape={tuple(t.shape)}")

    # 現在 t 是 (Bt,K,H,W) 或 (Bt,K,H,W,Z)
    Bt = int(t.shape[0])
    if Bt != B:
        # 99% 情況 B=1；若不一致，嘗試 broadcast
        if Bt == 1:
            t = t.repeat(B, *([1] * (t.dim() - 1)))
        else:
            raise ValueError(f"Teacher batch {Bt} != student batch {B} | t={tuple(t.shape)} s={tuple(s_logits.shape)}")

    # layout 對齊
    if cd == 1:
        return t  # (B,K,...)
    else:
        # (B,K,H,W)   -> (B,H,W,K)
        # (B,K,H,W,Z) -> (B,H,W,Z,K)
        if t.dim() == 4:
            return t.permute(0, 2, 3, 1).contiguous()
        elif t.dim() == 5:
            return t.permute(0, 2, 3, 4, 1).contiguous()
        else:
            raise ValueError(f"Unexpected after-batch teacher dim={t.dim()} shape={tuple(t.shape)}")

def _teacher_to_prob(t_aligned, class_dim, T=2.0, eps=1e-6):
    """
    t_aligned 可能是 prob 也可能是 logits。
    - 如果看起來像 prob（非負且 sum~1），就做 normalize 保險
    - 否則當 logits：softmax(t/T)
    """
    t = t_aligned.to(dtype=torch.float32)

    # 判斷是否像 prob
    # (注意：fp16 存 prob 有時 max 會略 >1，所以放寬)
    looks_nonneg = (t.min() >= -1e-4)
    # sum over class
    s = t.sum(dim=class_dim, keepdim=True)
    mean_sum = float(s.mean().detach().cpu())
    looks_like_prob = looks_nonneg and (abs(mean_sum - 1.0) < 0.05)

    if looks_like_prob:
        t = t.clamp_min(eps)
        t = t / t.sum(dim=class_dim, keepdim=True).clamp_min(eps)
        return t
    else:
        return F.softmax(t / T, dim=class_dim)

def kd_kl_mean(s_logits, t_tensor, K=18, T=2.0, eps=1e-6):
    """
    s_logits: student logits (B,K,...) or (B,...,K)
    t_tensor: teacher prob/logits (各種可能 shape 都吃)
    """
    cd = _infer_class_dim(s_logits, K=K)
    t_aligned = _align_teacher_to_student(t_tensor, s_logits, K=K)
    t_prob = _teacher_to_prob(t_aligned, class_dim=cd, T=T, eps=eps)

    s = s_logits.to(dtype=torch.float32)
    log_p_s = F.log_softmax(s / T, dim=cd)
    return F.kl_div(log_p_s, t_prob, reduction="batchmean") * (T * T)




# =========================================================
# KD: 保留原本 kd_kl_mean（batchmean），但提供「穩定(mean) + 可驗證」版本
# =========================================================
kd_kl_mean_orig = kd_kl_mean  # ✅ 完整保留舊版行為（batchmean）

def kd_kl_mean_v2(s_logits, t_tensor, K=18, T=2.0, eps=1e-6):
    """
    取代版（不砍原本 kd_kl_mean）：
    - teacher 可能是 prob 或 logits：沿用 _teacher_to_prob 判斷並轉成 prob
    - KL 預設用 reduction='none'.mean()（避免被 voxel 數量放大到 1e8）
    - KD_VERIFY=1 時：少量 step 印出 teacher 分佈檢查、prob sum、KL(batchmean)/KL(mean) ratio
    - KD_KL_REDUCTION 可切回 'batchmean' 以對照舊版
    """
    global _KD_CALL_COUNTER
    _KD_CALL_COUNTER += 1
    debug = KD_VERIFY and ((_KD_CALL_COUNTER <= KD_VERIFY_FIRST_CALLS) or (_KD_CALL_COUNTER % KD_VERIFY_EVERY_CALLS == 0))

    cd = _infer_class_dim(s_logits, K=K)
    t_aligned = _align_teacher_to_student(t_tensor, s_logits, K=K)

    # teacher -> prob（沿用你原本的判斷邏輯）
    t_prob = _teacher_to_prob(t_aligned, class_dim=cd, T=T, eps=eps)

    # student logits -> log prob
    s = s_logits.to(dtype=torch.float32)
    log_p_s = F.log_softmax(s / T, dim=cd)

    # 兩種 reduction 都算，方便 debug 對照
    kl_batchmean = F.kl_div(log_p_s, t_prob, reduction="batchmean") * (T * T)
    kl_mean = F.kl_div(log_p_s, t_prob, reduction="none").mean() * (T * T)

    if debug:
        t_raw = t_aligned.detach().to(dtype=torch.float32)
        sumC = t_raw.sum(dim=cd, keepdim=False)
        t_min = float(t_raw.min().cpu())
        t_max = float(t_raw.max().cpu())
        t_mean = float(t_raw.mean().cpu())
        sum_mean = float(sumC.mean().cpu())
        sum_min = float(sumC.min().cpu())
        sum_max = float(sumC.max().cpu())
        neg_ratio = float((t_raw < 0).float().mean().cpu())
        has_nan = bool(torch.isnan(t_raw).any().cpu())
        has_inf = bool(torch.isinf(t_raw).any().cpu())

        # prob sum 檢查（理想應 ~1）
        sumP = t_prob.detach().sum(dim=cd, keepdim=False)
        ratio = float((kl_batchmean / (kl_mean + 1e-12)).detach().cpu())

        print(f"[KDVERIFY] call={_KD_CALL_COUNTER} K={K} T={T} class_dim={cd} reduction={KD_KL_REDUCTION}")
        print(f"[KDVERIFY] student_logits shape={tuple(s_logits.shape)} dtype={s_logits.dtype} "
              f"min={float(s.min().cpu()):.4f} max={float(s.max().cpu()):.4f} mean={float(s.mean().cpu()):.4f}")
        print(f"[KDVERIFY] teacher_raw   shape={tuple(t_aligned.shape)} dtype={getattr(t_aligned, 'dtype', None)} "
              f"min={t_min:.4f} max={t_max:.4f} mean={t_mean:.4f} neg_ratio={neg_ratio:.6f} nan={has_nan} inf={has_inf}")
        print(f"[KDVERIFY] teacher sum_over_class: mean={sum_mean:.4f} min={sum_min:.4f} max={sum_max:.4f} (≈1 -> prob)")
        print(f"[KDVERIFY] prob sum_over_class   : mean={float(sumP.mean().cpu()):.4f} min={float(sumP.min().cpu()):.4f} max={float(sumP.max().cpu()):.4f}")
        print(f"[KDVERIFY] KL(batchmean)={float(kl_batchmean.cpu()):.6f} KL(mean)={float(kl_mean.cpu()):.6f} ratio={ratio:.2f}")
        print("[KDVERIFY] ratio 很大 => 你原本 batchmean 會被 voxel 數量放大，導致 kd2d/kd3d 1e8 爆炸。")
        print("")
    if KD_KL_REDUCTION == "batchmean":
        return kl_batchmean
    else:
        return kl_mean

# ✅ 只改「呼叫到的 kd_kl_mean 名稱指向」，原本 kd_kl_mean_orig 完整保留
if KD_VERIFY or (KD_KL_REDUCTION != "batchmean"):
    kd_kl_mean = kd_kl_mean_v2

def student_bev_from_logits(s_logits, K=18):
    """
    把 student 3D logits 轉 BEV logits，用 max over Z。
    支援 (B,K,H,W,Z) or (B,H,W,Z,K)
    """
    cd = _infer_class_dim(s_logits, K=K)
    if cd == 1:
        # (B,K,H,W,Z) -> max over Z (last dim)
        return s_logits.max(dim=-1).values
    else:
        # (B,H,W,Z,K) -> max over Z (dim=-2)
        return s_logits.max(dim=-2).values


def compute_total_loss(student, teacher, batch, amp_enabled, kd_cfg):
    # 1) student forward (要回傳 distill 用的 feature/logits)
    out_s = student(batch, return_feats=True)
    s_logits = out_s["logits"]
    s_bev = out_s.get("bev_feat", None)

    # 2) teacher forward (no_grad)
    with torch.no_grad():
        out_t = teacher(batch, return_feats=True)
        t_logits = out_t["logits"]
        t_bev = out_t.get("bev_feat", None)

    # 3) supervised loss
    loss_dict = student.head.loss(
        occ_pred=s_logits,
        voxel_semantics=batch["occ_label"],
        mask_camera=batch["occ_cam_mask"],
    )
    occ_loss = loss_dict["loss_occ"]

    depth_loss = getattr(student, "last_depth_loss", None)
    if depth_loss is None:
        depth_loss = occ_loss.new_tensor(0.0)
    elif not torch.is_tensor(depth_loss):
        depth_loss = occ_loss.new_tensor(float(depth_loss))

    # 4) kd losses
    lambda_feat = kd_cfg.get("lambda_feat", 0.0)
    lambda_logit = kd_cfg.get("lambda_logit", 0.0)
    T = kd_cfg.get("T", 2.0)

    loss_kd_feat = occ_loss.new_tensor(0.0)
    if (s_bev is not None) and (t_bev is not None) and lambda_feat > 0:
        loss_kd_feat = torch.mean((s_bev - t_bev) ** 2)

    loss_kd_logit = occ_loss.new_tensor(0.0)
    if lambda_logit > 0:
        s = torch.log_softmax(s_logits / T, dim=1)
        t = torch.softmax(t_logits / T, dim=1)
        loss_kd_logit = torch.nn.functional.kl_div(s, t, reduction="batchmean") * (T * T)

    total = occ_loss + depth_loss + lambda_feat * loss_kd_feat + lambda_logit * loss_kd_logit

    # 你也可以回傳一個 dict 方便 log
    return total, {
        "occ": occ_loss.detach(),
        "depth": depth_loss.detach(),
        "kd_feat": loss_kd_feat.detach(),
        "kd_logit": loss_kd_logit.detach(),
    }


# ---------------------------------------------------------
# 工具函數區塊
# ---------------------------------------------------------
def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True  # 固定輸入大小通常可以加速


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.ema_model.state_dict()


def format_seconds(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def get_logger(log_file):
    logger = logging.getLogger("FlashOCCTrain")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def adjust_learning_rate(optimizer, current_iter, warmup_iters, max_iters, base_lr):
    if current_iter < warmup_iters:
        alpha = float(current_iter) / max(warmup_iters, 1)
        lr = base_lr * (0.001 + (1 - 0.001) * alpha)
    else:
        progress = (current_iter - warmup_iters) / max((max_iters - warmup_iters), 1)
        lr = base_lr * 0.5 * (1. + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def to_device(batch, device):
    """把 batch 裡 tensor / numpy array 搬到 GPU。"""
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)

    if isinstance(batch, np.ndarray):
        # teacher dump 會進到這裡
        if batch.dtype == object:
            batch = np.stack([np.asarray(x, dtype=np.float32) for x in list(batch)], axis=0)
        return torch.from_numpy(batch).to(device, non_blocking=True)

    if isinstance(batch, dict):
        out = {}
        for k, v in batch.items():
            out[k] = to_device(v, device)
        return out

    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(x, device) for x in batch)

    return batch



# =========================================================================
# 🚀 主程式
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--py-config', default='config/nuscenes_student_nobase.py')
    parser.add_argument('--work-dir', default=None, help='path to save logs and ckpts')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', default=None, help='resume from checkpoint path')
    parser.add_argument('--max-epochs', type=int, default=24, help='force max epochs if not in config')
    parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision (AMP)')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size in config')

    # 🔧 新增：控制 PyTorch CPU 線程數
    # parser.add_argument(
    #     '--torch_threads',
    #     type=int,
    #     default=2,
    #     help='Max PyTorch CPU threads per process (torch.set_num_threads)'
    # )

    args = parser.parse_args()

    # # =====================================================
    # # 設定 PyTorch 線程數（這才是真的有用的地方）
    # # =====================================================
    # if args.torch_threads is not None and args.torch_threads > 0:
    #     torch.set_num_threads(args.torch_threads)
    #     torch.set_num_interop_threads(max(1, args.torch_threads // 2))
    #     print(f"[THREAD] torch_threads = {args.torch_threads}, "
    #           f"interop_threads = {max(1, args.torch_threads // 2)}")
    # else:
    #     print("[THREAD] torch_threads not set or <=0, using PyTorch default.")

    # 1. 讀取 Config
    cfg = Config.fromfile(args.py_config)

    
    # 覆寫 Batch Size
    if args.batch_size is not None and cfg.get('train_loader'):
        cfg.train_loader['batch_size'] = args.batch_size
        print(f"🚀 [Command Override] Batch size set to {args.batch_size}")

    set_random_seed(args.seed, deterministic=False)

    # Work Dir
    if args.work_dir:
        work_dir = args.work_dir
    else:
        work_dir = cfg.get('work_dir') or 'work_dirs/flash_occ_torch'

    os.makedirs(work_dir, exist_ok=True)

    logger = get_logger(osp.join(work_dir, 'train.log'))
    writer = SummaryWriter(work_dir)

    logger.info(f"🚀 Start training with config: {args.py_config}")
    logger.info(f"📂 Work dir: {work_dir}")

    # 顯示最終的 Batch Size
    current_bs = cfg.train_loader.get('batch_size', 'Unknown') if cfg.get('train_loader') else 'Unknown'
    logger.info(f"📦 Batch Size: {current_bs}")
    logger.info(f"⚡ AMP Enabled: {args.amp}")

    # 2. Dataloader
    logger.info("🛠 Building dataloaders...")
    train_loader, val_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False,
        iter_resume=False,
    )

    # 3. Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🤖 Building OccStudent Model on {device}...")

    model = OccStudent(
        bev_h=200,
        bev_w=200,
        depth_bins=16,
        num_classes=18,
        backbone_pretrained=True,
        backbone_frozen_stages=1,
        input_size=(480, 640),
        numC_Trans=128,
        pc_range=(-50.0, -50.0, -5.0, 50.0, 50.0, 3.0),
    ).to(device)

    # 4. Optimizer
    optimizer_cfg = cfg.get('optimizer', {})
    if isinstance(optimizer_cfg, dict):
        base_lr = optimizer_cfg.get('lr', 1e-4)
        weight_decay = optimizer_cfg.get('weight_decay', 1e-2)
    else:
        base_lr = 1e-4
        weight_decay = 1e-2

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    # 5. EMA
    ema = ModelEMA(model, decay=0.999)

    # 6. Resume
    start_epoch = 1
    global_iter = 0
    if args.resume and os.path.exists(args.resume):
        logger.info(f"🔄 Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'ema_state' in ckpt:
            ema.ema_model.load_state_dict(ckpt['ema_state'])
        start_epoch = ckpt['epoch'] + 1
        global_iter = ckpt['global_iter']

    # 7. Training Setup
    runner_cfg = cfg.get('runner', {})
    if isinstance(runner_cfg, dict):
        max_epochs = runner_cfg.get('max_epochs', args.max_epochs)
    else:
        max_epochs = args.max_epochs

    logger.info(f"📅 Total Epochs: {max_epochs}")

    scaler = GradScaler(enabled=args.amp)

    total_iters = max_epochs * len(train_loader)
    warmup_iters = 200

    start_time = time.time()

    # -----------------------------------------------------
    # 🧠 Training Loop
    # -----------------------------------------------------
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        epoch_start_time = time.time()

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{max_epochs}",
            leave=False
        )

        running_loss = 0.0

        for batch_idx, batch in pbar:
            global_iter += 1

            lr = adjust_learning_rate(optimizer, global_iter, warmup_iters, total_iters, base_lr)
            optimizer.zero_grad()

            # 整個 batch 先搬到 GPU
            batch = to_device(batch, device)

            kd_cfg = dict(
                lambda_kd_3d=1.0,   # 你可調
                lambda_kd_bev=0.5,  # 你可調
                T=2.0,
            )

            with autocast(enabled=args.amp):
                out_s = model(batch, return_feats=True)
                s_logits = out_s["logits"]

                depth_loss = getattr(model, "last_depth_loss", None)
                loss_dict = model.head.loss(
                    occ_pred=s_logits,
                    voxel_semantics=batch["occ_label"],
                    mask_camera=batch["occ_cam_mask"],
                )
                occ_loss = loss_dict["loss_occ"]

                if depth_loss is None:
                    depth_loss = occ_loss.new_tensor(0.0)
                elif not torch.is_tensor(depth_loss):
                    depth_loss = occ_loss.new_tensor(float(depth_loss))

            # ✅ KD 建議用 fp32 算（避免 AMP 讓 KL 不穩）
            loss_kd_3d = occ_loss.new_tensor(0.0)
            loss_kd_bev = occ_loss.new_tensor(0.0)

            t_occ = batch.get("teacher_occ_prob", None)
            t_bev = batch.get("teacher_bev_prob", None)

            with torch.cuda.amp.autocast(enabled=False):
                if (t_occ is not None) and kd_cfg["lambda_kd_3d"] > 0:
                    loss_kd_3d = kd_kl_mean(s_logits, t_occ, K=18, T=kd_cfg["T"])

                if (t_bev is not None) and kd_cfg["lambda_kd_bev"] > 0:
                    s_bev_logits = student_bev_from_logits(s_logits, K=18)
                    loss_kd_bev = kd_kl_mean(s_bev_logits, t_bev, K=18, T=kd_cfg["T"])


            loss = occ_loss + depth_loss \
                + kd_cfg["lambda_kd_3d"] * loss_kd_3d \
                + kd_cfg["lambda_kd_bev"] * loss_kd_bev




            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            ema.update(model)

            running_loss += loss.item()
            current_loss = loss.item()

            if global_iter % 50 == 0:
                writer.add_scalar('Train/Loss', current_loss, global_iter)
                writer.add_scalar('Train/LR', lr, global_iter)
                if global_iter % 200 == 0:
                    logger.info(f"Iter {global_iter} | Loss: {current_loss:.4f} | LR: {lr:.6f}")

            pbar.set_postfix({
            'loss': f"{current_loss:.4f}",
            'occ': f"{occ_loss.item():.3f}",
            'dep': f"{depth_loss.item():.3f}",
            'kd3d': f"{loss_kd_3d.item():.3f}",
            'kd2d': f"{loss_kd_bev.item():.3f}",
            'lr': f"{lr:.6f}",
            })


        avg_loss = running_loss / max(len(train_loader), 1)
        epoch_duration = time.time() - epoch_start_time
        logger.info(f"[TRAIN] Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}. Time: {format_seconds(epoch_duration)}")

        # -------------------------------------------------
        # 🔍 Validation
        # -------------------------------------------------
        logger.info(f"🔍 Validating Epoch {epoch}...")
        val_loss = run_validation(model, val_loader, device)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        logger.info(f"[VAL] Epoch {epoch} Val Loss: {val_loss:.4f}")

        # -------------------------------------------------
        # 💾 Save Checkpoint
        # -------------------------------------------------
        ckpt_path = osp.join(work_dir, f"epoch_{epoch}.pth")
        save_dict = {
            "epoch": epoch,
            "global_iter": global_iter,
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
        }
        torch.save(save_dict, ckpt_path)
        torch.save(save_dict, osp.join(work_dir, "latest.pth"))
        logger.info(f"💾 Saved checkpoint to {ckpt_path}")

    total_time = time.time() - start_time
    logger.info(f"✅ Training Finished. Total time: {format_seconds(total_time)}")
    writer.close()


@torch.no_grad()
def run_validation(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    count = 0

    for batch in tqdm(val_loader, desc="Validation", leave=False):
        batch = to_device(batch, device)

        occ_pred = model(batch)
        occ_label = batch["occ_label"]
        occ_cam_mask = batch["occ_cam_mask"]

        loss_dict = model.head.loss(
            occ_pred=occ_pred,
            voxel_semantics=occ_label,
            mask_camera=occ_cam_mask,
        )

        occ_loss = loss_dict["loss_occ"]
        depth_loss = getattr(model, "last_depth_loss", None)
        if depth_loss is None:
            depth_loss = occ_loss.new_tensor(0.0)
        elif not torch.is_tensor(depth_loss):
            depth_loss = occ_loss.new_tensor(float(depth_loss))

        loss = occ_loss + depth_loss



        total_loss += loss.item()
        count += 1


    return total_loss / max(count, 1)


if __name__ == "__main__":
    main()
