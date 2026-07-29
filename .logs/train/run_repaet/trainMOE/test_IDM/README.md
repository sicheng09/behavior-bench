# train/test_IDM · MIXMOE Drive+LSTM vs IDM

训完后自动评测 **policy_0 = Drive + Recurrent**（pufferinter + IDM）。  
结果写在本目录；评测占用与训练相同的 GPU。

| Run | Train tmux | W&B | GPU | Results |
| --- | --- | --- | ---: | --- |
| shared | `mixmoe-mb98k-1500m-gpu1` | `ncn8qxko` | 1 | `mix_moe_*_mb98k_*/Drive_Recurrent/` |
| stratified | `mixmoe-strat-1500m-gpu2` | `o4trtoxc` | 2 | `mix_moe_*_stratified_*/Drive_Recurrent/` |

挂起：`bash .logs/train/run_repaet/trainMOE/test_IDM/launch_wait_eval.sh`
