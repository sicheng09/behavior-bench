# ConservativeMix run4 try — post-train eval

After `cons_mix_A` / `cons_mix_B` finish, auto-eval ego (policy 0) on **pufferinter** vs **IDM** and vs **same-weight PPO**.

## Layout

```text
.logs/train/run4/try/test/
  wait_and_eval_cons_mix.sh
  run_one_policy_eval.sh
  resolve_final_ego_checkpoint.sh
  wait_and_eval_<train_dir>_*.log          # watcher logs
  <train_dir>/Drive_Recurrent/
    eval_*_Drive_vs_idm_gpu*_*.log
    eval_*_Drive_vs_ppo_gpu*_*.log
    <timestamp>_<uuid>/                    # eval.py outputs
```

## Runs

### r1 (first seed; done)

| Run | Train tmux | W&B | Eval IDM GPU | Eval PPO GPU | Results dir |
| --- | --- | --- | ---: | ---: | --- |
| A | `cons-mix-A-batch2x-1b-gpu0` | `6qdti7k0` | 2 | 3 | `cons_mix_A_steer0333_batch2x_1b_run4/Drive_Recurrent/` |
| B | `cons-mix-B-batch2x-1b-gpu1` | `l8z87oaa` | 4 | 5 | `cons_mix_B_shape_batch2x_1b_run4/Drive_Recurrent/` |

### r2 (repeat; launched 2026-07-18)

| Run | Train tmux | W&B | Eval IDM GPU | Eval PPO GPU | Results dir |
| --- | --- | --- | ---: | ---: | --- |
| A | `cons-mix-A-batch2x-1b-r2-gpu0` | `t1uqsvzw` | 2 | 3 | `cons_mix_A_steer0333_batch2x_1b_run4_r2/Drive_Recurrent/` |
| B | `cons-mix-B-batch2x-1b-r2-gpu1` | `i1mvzmos` | 4 | 5 | `cons_mix_B_shape_batch2x_1b_run4_r2/Drive_Recurrent/` |

Watcher tmux: `wait-eval-cons-mix-A-r2`, `wait-eval-cons-mix-B-r2`.  
Launch: `.logs/train/run4/try/run/launch_cons_mix_A_B_r2.sh`

Checkpoint: `experiments/puffer_drive_conservative_mix_<wandb_id>/model_policy_0_*.pt` (min epoch 953).

## Manual re-run

```bash
WEIGHTS=experiments/puffer_drive_conservative_mix_6qdti7k0/model_policy_0_puffer_drive_conservative_mix_000954.pt
MAP_IDS=all bash .logs/train/run4/try/test/run_one_policy_eval.sh \
  cons_mix_A_steer0333_batch2x_1b_run4 Drive_Recurrent "$WEIGHTS" \
  Drive Recurrent 256 256 2 idm
```
