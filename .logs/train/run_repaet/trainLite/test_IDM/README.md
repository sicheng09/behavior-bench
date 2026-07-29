# trainLite/test_IDM · DriveLite mix Drive+LSTM vs IDM

训完后自动评测 **policy_0 = Drive + Recurrent**（Mixed A/B 的高等级槽）。  
协议：pufferinter + IDM；同卡复用训练 GPU。

| Run | Train tmux | W&B | GPU |
| --- | --- | --- | ---: |
| A shared | `drivelite-mixA_mb98k_shared_1500m-gpu3` | `uhnxy2lw` | 3 |
| A stratified | `drivelite-mixA_stratified_1500m-gpu4` | `r0jht7bf` | 4 |
| B shared | `drivelite-mixB_mb98k_shared_1500m-gpu5` | `d9fq08yq` | 5 |
| B stratified | `drivelite-mixB_stratified_1500m-gpu6` | `qgytyxt4` | 6 |

挂起：`bash .logs/train/run_repaet/trainLite/test_IDM/launch_wait_eval.sh`
