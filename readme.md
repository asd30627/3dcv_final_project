# 3dcv_final_project

本專案為 3D/BEV Occupancy 相關實作與實驗紀錄，包含：

- **OccStudent**：以多相機影像輸入的輕量 BEV-based occupancy student model  
- **Teacher dump / Distillation**：由 teacher 輸出  
  - 3D voxel logits/prob：`200×200×16×18`  
  - BEV logits/prob：`200×200×18`  
  並在 dataloader 讀取後進行蒸餾訓練

---

## Dataset（nuScenes）

1. 先至 nuScenes 官網下載完整 `trainval` 與 `test`：
   - https://www.nuscenes.org/

2. Dataset 整理方式可參考 FlashOCC 的資料準備流程，並依其方式下載對應的 GT：
   - https://github.com/Yzichen/FlashOCC/tree/master

---

## 環境需求（Teacher 端：GaussianFormer）

本專案的環境依 **GaussianFormer** 安裝方式建置。請先 clone 下列 repository，照其安裝步驟完成環境建置，並輸出蒸餾所需的 teacher dump 檔案：

- Teacher repo（GaussianFormer 版本）：
  - https://github.com/asd30627/3dcv_final_project_gaussianformer

---

## 如何產生蒸餾檔案（Teacher dump）

1. Clone teacher repo（請確認網址是否為你要的版本）：

   - https://github.com/asd30627/3dcv_final_project_gaussianformer

2. 進入 teacher repo 後執行以下指令產生 dump（請自行修改路徑與檔名）：

```bash
python eval.py \
  --py-config config/prob/nuscenes_gs25600.py \
  --work-dir /home/ivlab3/GaussianFormer/out/prob/dump_gf2 \
  --resume-from /home/ivlab3/GaussianFormer/out/prob/state_dict.pth \
  --splits train,val \
  --spatial-shape 200,200,16 \
  --num-classes 18 \
  --free-id 17 \
  --dump-occ3d-dir /mnt/xs1000/teacher_dump/occ3d_sem \
  --dump-bev-dir /mnt/xs1000/teacher_dump/bev_sem \
  --dump-bev-vis-dir /mnt/xs1000/teacher_dump/bev_vis \
  --dump-bev-prob-dir /mnt/xs1000/teacher_dump/bev_prob \
  --bev-prob-type logits \
  --bev-prob-key pred_occ \
  --dump-3d-dir /mnt/xs1000/teacher_dump/occ3d_logits \
  --dump-3d-type logits \
  --dump-3d-key pred_occ \
  --print-freq 50 \
  --skip-existing

## 其他：Clone 本專案後的訓練與驗證（最常用）

接下來 clone 本專案後，直接執行以下指令即可開始跑：
python train_student.py --py-config config/nuscenes_student_nobase.py --work-dir ./work_dirs/1219_KD

## 驗證
python eval_occ_metrics_from_pth.py   --py-config config/nuscenes_student_nobase.py   --ckpt work_dirs/occ_student_nobase/epoch_11.pth

