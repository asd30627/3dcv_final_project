#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from pathlib import Path

def short_stats(arr: np.ndarray, max_print=6):
    # 安全統計（避免超大陣列導致太慢）
    a = arr
    info = {}
    info["shape"] = tuple(a.shape)
    info["dtype"] = str(a.dtype)

    # 轉 float64 做統計比較安全，但避免 huge array 太慢
    # 這裡只做基本統計；如果是整數也可做 unique(抽樣)
    try:
        info["nan"] = bool(np.isnan(a).any()) if np.issubdtype(a.dtype, np.floating) else False
        info["inf"] = bool(np.isinf(a).any()) if np.issubdtype(a.dtype, np.floating) else False
    except Exception:
        info["nan"] = "?"
        info["inf"] = "?"

    # min/max/mean
    try:
        info["min"] = float(np.min(a))
        info["max"] = float(np.max(a))
        info["mean"] = float(np.mean(a))
    except Exception:
        info["min"] = info["max"] = info["mean"] = "?"

    # 額外：如果是小維度整數類，抽樣看 unique
    if np.issubdtype(a.dtype, np.integer):
        try:
            flat = a.reshape(-1)
            if flat.size <= 2_000_000:
                u = np.unique(flat)
                if u.size <= max_print:
                    info["unique"] = u.tolist()
                else:
                    info["unique"] = (u[:max_print].tolist(), f"... total_unique={u.size}")
            else:
                # 太大就抽樣
                idx = np.random.choice(flat.size, size=min(200000, flat.size), replace=False)
                u = np.unique(flat[idx])
                if u.size <= max_print:
                    info["unique_sample"] = u.tolist()
                else:
                    info["unique_sample"] = (u[:max_print].tolist(), f"... total_unique_sample={u.size}")
        except Exception:
            pass

    return info

def print_info(prefix, info: dict):
    # 統一漂亮輸出
    print(f"{prefix} shape={info.get('shape')} dtype={info.get('dtype')}"
          f" min={info.get('min')} max={info.get('max')} mean={info.get('mean')}"
          f" nan={info.get('nan')} inf={info.get('inf')}")
    if "unique" in info:
        print(f"    unique={info['unique']}")
    if "unique_sample" in info:
        print(f"    unique_sample={info['unique_sample']}")

def inspect_npy(path: Path):
    arr = np.load(path, allow_pickle=False)
    print(f"\n[NPY] {path}")
    info = short_stats(arr)
    print_info("  -", info)

def inspect_npz(path: Path):
    print(f"\n[NPZ] {path}")
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.keys())
        print(f"  keys={keys}")
        for k in keys:
            arr = data[k]
            info = short_stats(arr)
            print_info(f"  - key='{k}':", info)

def gather_files(root: Path, recursive: bool):
    if root.is_file():
        return [root]
    exts = {".npy", ".npz"}
    if recursive:
        files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    else:
        files = [p for p in root.glob("*") if p.suffix.lower() in exts]
    return sorted(files)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="單一檔案或資料夾（含 train/val 子資料夾也可）")
    ap.add_argument("--recursive", action="store_true", help="遞迴掃描子資料夾")
    ap.add_argument("--pick-first", type=int, default=20, help="最多檢查前 N 個檔案（避免掃太多）")
    args = ap.parse_args()

    root = Path(args.path)
    if not root.exists():
        raise FileNotFoundError(f"not found: {root}")

    files = gather_files(root, args.recursive)
    if not files:
        print(f"No .npy/.npz found under: {root}")
        return

    print(f"Found {len(files)} files. Showing first {min(args.pick_first, len(files))} files.")
    for p in files[: args.pick_first]:
        try:
            if p.suffix.lower() == ".npy":
                inspect_npy(p)
            else:
                inspect_npz(p)
        except Exception as e:
            print(f"\n[ERR] {p} -> {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
