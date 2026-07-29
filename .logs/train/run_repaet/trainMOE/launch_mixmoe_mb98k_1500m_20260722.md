# Launch · MIXMOE mb98k 1500M (run_repaet)

- date: 2026-07-22T19:58:13+08:00
- tmux: `mixmoe-mb98k-1500m-gpu1`
- GPU: 1
- W&B: [`ncn8qxko`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/ncn8qxko)
- group: `run_repaet-mixmoe`
- log: `mix_moe_drive_moe_moe3_mb98k_1500m_run_repaet_20260722_195813.log`
- script: `start_mixmoe_mb98k_1500m.sh`

```text
mix_ppo_policy_mix   = drive:1,moe:1,moe3:1
mix_ppo_policy_names = Drive,DriveMoE,DriveMoE3
batch_size           = 1572864
total_timesteps      = 1500000000
minibatch_size       = 98304
→ ~500M env steps / policy, 16 opt steps/epoch, ~32768 samples/policy/step
```
