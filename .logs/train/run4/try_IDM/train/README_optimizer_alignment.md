# try_IDM · 对齐 H0 优化器频率后复跑 ConservativeMix A/B

日期：2026-07-18  
目录：`.logs/train/run4/try_IDM/train/`  
对照旧实验：`.logs/train/run4/try/`（`minibatch_size=32768`）

---

## 1. 为什么要重跑？

`run4/try` 的 C-A / C-B 为了在 50:50 `mix_ppo` 下对齐 H0 的 **ego 环境样本量**，做了：

| 量 | H0（同质） | try（旧） |
| --- | ---: | ---: |
| `batch_size` | 524288 | **1048576**（×2） |
| `total_timesteps` | 500M | **1B**（×2） |
| 外层 epoch | ≈953 | ≈953 |
| 每 epoch ego 样本 | ≈524288 | ≈524288 |
| `minibatch_size` | 32768 | **32768**（未放大） |

这只对齐了 **rollout / 外层 epoch**，**没有**对齐 ego 的 **optimizer.step 结构**。

原因：mix_ppo **不是**两个策略各存一个经验池，而是：

1. 共享一个全局 rollout buffer；
2. 按全局 `batch_size / minibatch_size` 抽 minibatch；
3. 再按 `segment_policy_ids` 切开，分别对 ego / partner `backward` + `opt.step()`。

因此在 try 旧配置下：

| 量 | H0 | try（旧，有问题） |
| --- | ---: | ---: |
| 每 epoch `optimizer.step` 次数 | \(524288/32768=\)**16** | \(1048576/32768=\)**32**（×2） |
| 每步 ego 样本（期望，50%） | 32768 | \(32768×0.5≈\)**16384**（÷2） |
| ego 总环境步 | ≈500M | ≈500M（仍对齐） |

结论：旧 try 的 ego 是「**更新更碎、更频、每步样本更少**」地训满 500M，与 H0 的优化几何不一致。双卡 DDP 也消不掉该结构问题；必须按 mix 比例同步放大 `minibatch_size`（及 `max_minibatch_size`）。

实现依据：`pufferlib/pufferl.py` 中  
`total_minibatches = update_epochs * batch_size / minibatch_size`，以及 mix_ppo 分支「全局采样 → `policy_mask` → 各策略 `opt.step`」。

---

## 2. try_IDM 修正配方（与 try 唯一实质差异）

算法 / 伙伴设定与 try 相同：

- **C-A**：`partner_mode=action_constraint`，`partner_max_abs_steer=0.333`，`Drive` + `DriveSteerConstrained`
- **C-B**：`partner_mode=reward_shaping`，`partner_target_headway=1.8`，双 `Drive` + partner shaping  
- `mix_ppo`：`ego:0.5,partner:0.5`，均可训；训练 **不接** 评测 IDM（`mix_traffic=False`）

预算对齐修正：

| 量 | H0 | try（旧） | **try_IDM（本目录）** |
| --- | ---: | ---: | ---: |
| `batch_size` | 524288 | 1048576 | 1048576 |
| `total_timesteps` | 500M | 1B | 1B |
| `minibatch_size` | 32768 | 32768 | **65536** |
| `max_minibatch_size` | 32768 | 32768 | **65536** |
| 每 epoch opt steps | 16 | 32 | **16** |
| 每步 ego 样本（期望） | 32768 | ~16384 | **~32768** |
| 外层 epoch | ≈953 | ≈953 | ≈953 |

一般规则（ego 占比 \(f\)）：

```text
batch_size        = H0_batch / f
total_timesteps   = H0_steps / f
minibatch_size    = H0_minibatch / f
```

50:50 时 \(f=0.5\) → 三者均 ×2。

---

## 3. 如何启动

```bash
cd "$HOME/wsc/behavior-bench"
bash .logs/train/run4/try_IDM/train/launch_cons_mix_A_B_mb65k.sh
```

或分开：

```bash
bash .logs/train/run4/try_IDM/train/start_cons_mix_A_mb65k_1b.sh   # GPU0
bash .logs/train/run4/try_IDM/train/start_cons_mix_B_mb65k_1b.sh   # GPU1
```

| 标签 | tmux | RUN_NAME | W&B | GPU |
| --- | --- | --- | --- | ---: |
| C-A | `cons-mix-A-mb65k-1b-gpu0` | `cons_mix_A_steer0333_mb65k_1b_tryIDM` | [`stq4hogz`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/stq4hogz) | 0 |
| C-B | `cons-mix-B-mb65k-1b-gpu1` | `cons_mix_B_shape_mb65k_1b_tryIDM` | [`plvd8yph`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/plvd8yph) | 1 |

W&B group：`run4-try-IDM-cons-mix`（首次启动：2026-07-18 18:54）

日志：本目录下 `cons_mix_*_tryIDM_YYYYMMDD_HHMMSS.log`  
Checkpoint：`experiments/puffer_drive_conservative_mix_<wandb_id>/model_policy_0_*_000954.pt`（评测用 **policy 0**）

环境：`conda activate behavior-bench`；  
`DRIVE_BINARIES_DATA_ROOT=$REPO/resources/drive/binaries`。

---

## 4. 与旧 try 结果如何解读

- **旧 try 结果仍有效为「未对齐优化几何的 A/B」**，机制结论（B 更稳、A Front 偏高等）可作参考，但不宜再当作「与 H0 严格同预算同优化步结构」的主证据。
- **主比较应以本目录 try_IDM 跑完后的评测为准**（pufferinter vs IDM / vs PPO，协议同 `.logs/val/validation.md`）。
- 若 try_IDM 与旧 try 数字接近 → 该偏差对结论不敏感；若显著不同 → 旧 try 的 Coll/Fault 增益需按对齐后数字重写。

成功门槛（相对 H0 run4_r2，与 try 相同）：

| 指标 | H0 r2 | 目标 |
| --- | ---: | --- |
| vs IDM Coll | 5.77% | ≤ 4.90%（−15% rel） |
| vs IDM At-fault | 4.58% | ≤ 3.89%（−15% rel） |
| vs PPO Coll | 0.68% | ≤ 1.68%（+1 pp） |

---

## 5. 内存注意

`minibatch=65536` 约为旧 try 的 2×；共享 buffer 切片峰值上升，但切开后每策略约 32768，与 H0 单策略相当。若 OOM，可改为：

- 保持 `minibatch_size=65536`，降低 `max_minibatch_size`（例如 32768）并依赖 `accumulate_minibatches=2`（语义接近「两半再 step」，需单独记录）；或  
- 先确认单卡峰值后再决定是否双卡加速（**双卡 alone 不能替代 mb 缩放**）。

---

## 6. 训后评测

训完后自动评测挂在：`.logs/train/run4/try_IDM/test/`（协议同 `try/test`，结果写本树）。

```bash
bash .logs/train/run4/try_IDM/test/launch_wait_eval_A_B.sh
```

Watcher：`wait-eval-tryIDM-A-mb65k` / `wait-eval-tryIDM-B-mb65k` → 结果  
`try_IDM/test/cons_mix_*_tryIDM/Drive_Recurrent/`。

---

## 7. 一句话

**旧 try 只对齐了 ego 的 rollout 预算；try_IDM 把 `minibatch` 同步 ×2，使每 epoch 16 次更新、每步 ~32768 ego 样本，与 H0 优化器频率一致后再比较 C-A / C-B。**
