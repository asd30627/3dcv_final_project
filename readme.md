# 3dcv_final_project（OccStudent / 3D & BEV Occupancy）

本專案為 3D/BEV Occupancy 相關實作與實驗紀錄，包含：
- **OccStudent**：以多相機影像輸入的輕量 BEV-based occupancy student model  
- **Teacher dump / Distillation（進階可選）**：由 teacher（GaussianFormer）輸出離線檔案，供 student dataloader 讀取後進行蒸餾訓練  
  - 3D voxel logits/prob：`200×200×16×18`  
  - BEV logits/prob：`200×200×18`

---

## Repo 版本（重要）

本 README 對應「純 student 版本（無蒸餾）」，但保留蒸餾流程說明以便後續擴充：

- **純 student 版本（無蒸餾）**
  - https://github.com/asd30627/3dcv_final_project_no_distallation

- **Teacher 端（GaussianFormer 版本，用於輸出 teacher dump）**
  - https://github.com/asd30627/3dcv_final_project_gaussianformer

---

## Dataset（nuScenes）

1. 至 nuScenes 官網下載完整 `trainval` 與 `test`：
- https://www.nuscenes.org/

2. Dataset 整理方式可參考 FlashOCC 的資料準備流程，並依其方式下載對應的 Ground Truth（gts 資料夾）：
- https://github.com/Yzichen/FlashOCC/tree/master

> 本專案的 dataloader 預期 nuScenes 與對應 gts 依 FlashOCC 流程整理完成。

---

## 安裝與環境

- 建議在 repo 根目錄下執行所有指令（可避免 import 問題）。
- 若遇到 `ModuleNotFoundError`（找不到 `dataset` 或 `model`），請改用：
  - `PYTHONPATH=. python xxx.py ...`

---

## 基本使用（純 Student）

### 1) 訓練

```bash
python train_student.py --py-config config/nuscenes_student_nobase.py

