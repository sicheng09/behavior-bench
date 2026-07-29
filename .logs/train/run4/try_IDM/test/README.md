# try_IDM · post-train eval（对齐 minibatch 后的 C-A / C-B）

训练结束后自动评测 ego（policy 0）在 **pufferinter** 上 vs **IDM** 与 vs **同权重 PPO**。  
协议与 `.logs/train/run4/try/test` / `.logs/val/validation.md` 一致；结果写在本目录。

## Layout

```text
.logs/train/run4/try_IDM/test/
  wait_and_eval_cons_mix.sh
  run_one_policy_eval.sh
  resolve_final_ego_checkpoint.sh
  launch_wait_eval_A_B.sh
  wait_and_eval_<train_dir>_*.log
  <train_dir>/Drive_Recurrent/
    eval_*_Drive_vs_idm_gpu*_*.log
    eval_*_Drive_vs_ppo_gpu*_*.log
    <timestamp>_<uuid>/   # summary.csv, per_map.csv, collision_snapshots.json
```

## 当前 run（mb65k，H0 优化器对齐）

| Run | Train tmux | W&B | Eval IDM | Eval PPO | Results dir |
| --- | --- | --- | ---: | ---: | --- |
| C-A | `cons-mix-A-mb65k-1b-gpu0` | `stq4hogz` | 2 | 3 | `cons_mix_A_steer0333_mb65k_1b_tryIDM/Drive_Recurrent/` |
| C-B | `cons-mix-B-mb65k-1b-gpu1` | `plvd8yph` | 4 | 5 | `cons_mix_B_shape_mb65k_1b_tryIDM/Drive_Recurrent/` |

Watcher：`wait-eval-tryIDM-A-mb65k`、`wait-eval-tryIDM-B-mb65k`  
一键挂起：`bash .logs/train/run4/try_IDM/test/launch_wait_eval_A_B.sh`

Checkpoint：`experiments/puffer_drive_conservative_mix_<wandb_id>/model_policy_0_*_000954.pt`（`MIN_EPOCH=953`）。

训练说明：`../train/README_optimizer_alignment.md`。

## 手动复评

```bash
WEIGHTS=experiments/puffer_drive_conservative_mix_stq4hogz/model_policy_0_puffer_drive_conservative_mix_000954.pt
MAP_IDS=all bash .logs/train/run4/try_IDM/test/run_one_policy_eval.sh \
  cons_mix_A_steer0333_mb65k_1b_tryIDM Drive_Recurrent "$WEIGHTS" \
  Drive Recurrent 256 256 2 idm
```
