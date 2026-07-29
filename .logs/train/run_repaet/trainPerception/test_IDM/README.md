# trainPerception/test_IDM · Perception mix 标准 Drive+LSTM vs IDM

训完后自动评测 **policy_2 = Drive + Recurrent**（高感知槽；p0/p1 是 PerceptionLow/Mid）。  
协议：pufferinter + IDM；同卡复用训练 GPU。

| Run | Train tmux | W&B | GPU | Policy slot |
| --- | --- | --- | ---: | ---: |
| shared | `perception-mb98k_shared_1500m-gpu0` | `yj5u9i7h` | 0 | 2 |
| stratified | `perception-stratified_1500m-gpu7` | `yov2516v` | 7 | 2 |

挂起：`bash .logs/train/run_repaet/trainPerception/test_IDM/launch_wait_eval.sh`
