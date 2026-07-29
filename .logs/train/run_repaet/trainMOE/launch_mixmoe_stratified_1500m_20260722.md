# Launch · MIXMOE stratified 1500M (run_repaet)

- date: 2026-07-22T20:06:31+08:00
- tmux: `mixmoe-strat-1500m-gpu2`
- GPU: 2
- W&B: [`o4trtoxc`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/o4trtoxc)
- group: `run_repaet-mixmoe`
- log: `mix_moe_drive_moe_moe3_stratified_1500m_run_repaet_20260722_200631.log`
- script: `start_mixmoe_stratified_1500m.sh`
- companion shared: [`ncn8qxko`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/ncn8qxko) / `mixmoe-mb98k-1500m-gpu1`

```text
mix_ppo_sampling               = stratified
mix_ppo_policy_mix             = drive:1,moe:1,moe3:1
mix_ppo_policy_minibatch_sizes = 32768,32768,32768
mix_ppo_policy_update_steps    = 16,16,16
batch_size                     = 1572864
total_timesteps                = 1500000000
→ ~500M / policy; pool~524288; 16 opt steps; mb=32768 per pool
```

Note: first launch failed because `drive.ini` lacked stratified keys (CLI rejected).
Added default `mix_ppo_sampling=shared` (+ empty minibatch/update fields) to
`pufferlib/config/ocean/drive.ini`, then relaunched successfully.
