# 3dcv_final_project

本專案為 3D/BEV Occupancy 相關實作與實驗紀錄，包含：
- **OccStudent**：以多相機影像輸入的輕量 BEV-based occupancy student model
- **Teacher dump / Distillation**：從 teacher 輸出 3D voxel logits/prob（200×200×16×18）與 BEV logits/prob（200×200×18），並在 dataloader 讀取進行蒸餾

##dataset
先在此網站下載完整的trainval、test
https://www.nuscenes.org/
可以參考flashocc的dataset整理，以及照他的gts下載
flashocc網址:
https://github.com/Yzichen/FlashOCC/tree/master

## 環境需求
我們的環境是照gaussianformer安裝的，首先先去我們這個repository照著安裝步驟安裝，並輸出蒸餾所需的檔案

repository網址: https://github.com/asd30627/3dcv_final_project_gaussianformer
##如何產生蒸餾檔案
clone 這個(https://github.com/asd30627/3dcv_final_project_gaussianform)後
執行
python eval.py   --py-config config/prob/nuscenes_gs25600.py   --work-dir /home/ivlab3/GaussianFormer/out/prob/dump_gf2   --resume-from /home/ivlab3/GaussianFormer/out/prob/state_dict.pth   --splits train,val   --spatial-shape 200,200,16   --num-classes 18   --free-id 17   --dump-occ3d-dir /mnt/xs1000/teacher_dump/occ3d_sem   --dump-bev-dir /mnt/xs1000/teacher_dump/bev_sem   --dump-bev-vis-dir /mnt/xs1000/teacher_dump/bev_vis   --dump-bev-prob-dir /mnt/xs1000/teacher_dump/bev_prob   --bev-prob-type logits   --bev-prob-key pred_occ   --dump-3d-dir /mnt/xs1000/teacher_dump/occ3d_logits   --dump-3d-type logits   --dump-3d-key pred_occ   --print-freq 50   --skip-existing
Namespace(py_config='config/prob/nuscenes_gs25600.py', work_dir='/home/ivlab3/GaussianFormer/out/prob/dump_gf2', resume_from='/home/ivlab3/GaussianFormer/out/prob/state_dict.pth', seed=42, splits='train,val', spatial_shape='200,200,16', num_classes=18, free_id=17, use_occ_mask=True, bev_scan_direction='low2high', bev_sem_mode='overwrite', bev_priority_ids='4,10,3,9,5,6,2,7,8,1', dump_occ3d_dir='/mnt/xs1000/teacher_dump/occ3d_sem', dump_bev_dir='/mnt/xs1000/teacher_dump/bev_sem', dump_bev_vis_dir='/mnt/xs1000/teacher_dump/bev_vis', vis_scale=4, dump_gt_bev_dir='', dump_gt_bev_vis_dir='', dump_gt_bev_mask_vis_dir='', dump_bev_prob_dir='/mnt/xs1000/teacher_dump/bev_prob', bev_prob_type='logits', bev_prob_key='pred_occ', bev_prob_occ_thresh=0.01, bev_prob_format='npz', bev_prob_compress=True, bev_rotate_k=0, bev_flip_x=False, bev_flip_y=False, dump_3d_dir='/mnt/xs1000/teacher_dump/occ3d_logits', dump_3d_type='logits', dump_3d_key='pred_occ', dump_3d_compress=True, max_samples=-1, skip_existing=True, no_metric=False, print_freq=50, amp=False, force_no_shuffle=True, debug_keys=False, gpus=1)

檔案位置以及專案名稱請更改成自己的

接下來clone本專案
執行
python train_student.py --py-config config/nuscenes_student_nobase.py --work-dir ./work_dirs/1219_KD即可開始跑
驗證miou等資訊跑
python eval_occ_metrics_from_pth.py   --py-config config/nuscenes_student_nobase.py   --ckpt work_dirs/occ_student_nobase/epoch_11.pth

