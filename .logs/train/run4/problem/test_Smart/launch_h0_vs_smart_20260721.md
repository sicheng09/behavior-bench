# H0 PPO ego vs SMART traffic

- date: 2026-07-21
- protocol: pufferinter / map-ids=all (same as test_IDM)
- ego: Drive + Recurrent (256,256) — three homogeneous 500M ckpts
- traffic: smart / `weights/SMART_epoch_030.pt`
- output root: `.logs/train/run4/problem/test_Smart/<run>/Drive_Recurrent/`

| Run | W&B | Ego weights | GPU | tmux |
| --- | --- | --- | ---: | --- |
| run1 | fdfw3v5e | `experiments/puffer_drive_fdfw3v5e/model_puffer_drive_000954.pt` | 0 | `h0-vs-smart-run1-gpu0` |
| run4_r1 | 1ytcyfuk | `experiments/puffer_drive_1ytcyfuk/model_puffer_drive_000954.pt` | 5 | `h0-vs-smart-r1-gpu5` |
| run4_r2 | u9ymfcqr | `experiments/puffer_drive_u9ymfcqr/model_puffer_drive_000954.pt` | 6 | `h0-vs-smart-r2-gpu6` |

Script: `run_one_policy_eval.sh`
