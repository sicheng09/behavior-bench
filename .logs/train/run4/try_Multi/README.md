# try_Multi · 冻结 C-B partner + ego 从零训练

日期：2026-07-21  
目录：`.logs/train/run4/try_Multi/`

## 设定

两路对照（配方相同，仅冻结对象不同）：

| 标签 | ego | 冻结 partner | 权重副本 | 启动脚本 |
| --- | --- | --- | --- | --- |
| **frozen B partner (p1)** | 从零 | `plvd8yph` **policy_1** | `weights/plvd8yph_policy_1_000954.pt` | `train/start_ego_scratch_frozen_B_partner_mb65k_1b.sh` |
| **frozen B ego0 (p0)** | 从零 | `plvd8yph` **policy_0**（vs-IDM Coll 2.55%） | `weights/plvd8yph_policy_0_000954.pt` | `train/start_ego_scratch_frozen_B_ego0_mb65k_1b.sh` |

公共项：`Drive`+`Recurrent`；`partner_mode=reward_shaping`；`mix_ppo_sampling=shared`，`ego:0.5,partner:0.5`；`trainable=True,False`。

来源评测日志（对应训练 run）：  
`.logs/train/run4/try_IDM/test/cons_mix_B_shape_mb65k_1b_tryIDM/.../eval_..._vs_idm_...211536.log`  
→ W&B [`plvd8yph`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/plvd8yph)（评测用的是 ego=`policy_0`；本实验冻结的是同次训练的 **`policy_1`**）。

## Ego 优化几何（对齐 H0 / try_IDM）

见 `.logs/train/run4/try_IDM/train/README_optimizer_alignment.md`：

| 量 | H0 同质 | 本实验（50:50 + mb×2） |
| --- | ---: | ---: |
| `batch_size` | 524288 | **1048576** |
| `total_timesteps` | 500M | **1B**（ego 环境步 ≈500M） |
| `minibatch_size` | 32768 | **65536** |
| 每 epoch opt.step | 16 | **16**（仅 ego；partner 无 optimizer） |
| 每步 ego 样本（期望） | 32768 | **~32768** |
| 外层 epoch | ≈953 | ≈953 |

冻结 partner：`--train.mix-ppo-policy-trainable True,False`；只产交互动作，不 `backward` / `opt.step`。

## 启动

```bash
cd "$HOME/wsc/behavior-bench"
# 默认 GPU 1；可覆盖：GPU_IDS=3 bash ...
bash .logs/train/run4/try_Multi/train/start_ego_scratch_frozen_B_partner_mb65k_1b.sh
```

| | |
| --- | --- |
| tmux | `try-multi-ego-scratch-frozenB-gpu1` |
| RUN_NAME | `ego_scratch_frozen_B_partner_mb65k_1b_tryMulti` |
| W&B group | `run4-try-Multi-frozen-partner` |
| 训练日志 | `train/ego_scratch_frozen_B_partner_mb65k_1b_tryMulti_*.log` |
| 评测用 ckpt | `experiments/puffer_drive_conservative_mix_<wandb>/model_policy_0_*_000954.pt` |
