==== try_IDM cons_mix A/B stratified launch summary ====
date=2026-07-19T00:35:04+08:00
mode: mix_ppo_sampling=stratified policy_mb=32768,32768 update_steps=16,16
A: tmux=cons-mix-A-strat-1b-gpu0 wandb=e4gq9dom eval GPU 2=IDM 3=PPO
B: tmux=cons-mix-B-strat-1b-gpu1 wandb=u2qysy6b eval GPU 4=IDM 5=PPO
watch: wait-eval-tryIDM-A-strat / wait-eval-tryIDM-B-strat
train logs: /home/fanyuqi/wsc/behavior-bench/.logs/train/run4/try_IDM/train/cons_mix_*_stratified_1b_tryIDM_*.log
eval root:  /home/fanyuqi/wsc/behavior-bench/.logs/train/run4/try_IDM/test/cons_mix_*_stratified_1b_tryIDM/
doc:        /home/fanyuqi/wsc/behavior-bench/.logs/train/run4/try_IDM/train/stratified_mix_training.md
========================================
