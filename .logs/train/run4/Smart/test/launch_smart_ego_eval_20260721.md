# SMART ego eval launch

- date: 2026-07-21T20:34:35+08:00
- ego: smart / weights/SMART_epoch_030.pt
- split: pufferinter / map-ids=all
- vs IDM: tmux smart-ego-eval-idm-gpu2 (GPU 2)
- vs PPO traffic: tmux smart-ego-eval-ppo-gpu4 (GPU 4), traffic=H0 r2 u9ymfcqr
- output: .logs/train/run4/Smart/test/smart_ego_epoch030/SMART/
- protocol aligned with try_IDM / try_Multi
