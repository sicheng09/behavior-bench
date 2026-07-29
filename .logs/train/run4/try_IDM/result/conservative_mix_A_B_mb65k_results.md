# Run4 try_IDM · ConservativeMix A / B（mb65k，优化器对齐）结果汇总

> 策略：ego `Drive` + `Recurrent(256,256)`；伙伴分别为 steer 约束（A）/ 奖励塑形（B）  
> 与旧 `try/` 的差异：全局 `minibatch_size=max_minibatch_size=65536`（仍为 `mix_ppo_sampling=shared`），使每 epoch ego 期望约 16 次 `opt.step`、每步约 32768 样本，对齐 H0 优化几何。  
> 更新时间：2026-07-18  
> 原始评测产物：`.logs/train/run4/try_IDM/test/`  
> 机器可读表：同目录 `metrics_summary.csv`  
> 对齐说明：`.logs/train/run4/try_IDM/train/README_optimizer_alignment.md`

---

## 1. 实验配置

两路训练共用 ConservativeMix 配方；在旧 try「batch×2 / steps×2」之上，**同步将 minibatch ×2**。

| 项目 | 取值 |
| --- | --- |
| 环境 / 入口 | `puffer_drive_conservative_mix` · `pufferlib/config/ocean/drive_conservative_mix.ini` |
| ego 策略 | `Drive` + `Recurrent`（`rnn_input=256`, `rnn_hidden=256`） |
| `mix_ppo` | `True` · `ego:0.5, partner:0.5` |
| `mix_ppo_sampling` | **`shared`**（本轮未用 stratified） |
| `batch_size` | **1,048,576**（2×524288；ego ≈524288/外层 update） |
| `total_timesteps` | **1,000,000,000**（2×500M；ego ≈500M） |
| 外层 updates | ≈953 → 最终 ego ckpt epoch **000954** |
| `minibatch_size` / `max_minibatch_size` | **65536 / 65536**（旧 try 为 32768） |
| 期望每 epoch opt steps | \(1048576/65536=\)**16**（与 H0 同） |
| 期望每步 ego 样本 | \(65536×0.5≈\)**32768**（与 H0 同） |
| `bptt_horizon` / `update_epochs` | 32 / 1 |
| `env.num_maps` | 10,000 |
| 交通混合 | `mix_traffic=False`（训练不接入评测 IDM） |
| 数据根（训练） | `resources/drive/binaries` |

### 1.1 两路差异

| 标签 | partner_mode | partner 策略 | 关键超参 | W&B | GPU |
| --- | --- | --- | --- | --- | ---: |
| **C-A mb65k** | `action_constraint` | `DriveSteerConstrained` | `partner_max_abs_steer=0.333` | [`stq4hogz`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/stq4hogz) | 0 |
| **C-B mb65k** | `reward_shaping` | `Drive`（同架构，伙伴专用 shaping） | `partner_target_headway=1.8`；`w_center/align/steer/gap` 默认 | [`plvd8yph`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/plvd8yph) | 1 |

训练日志：

| 标签 | 日志 |
| --- | --- |
| C-A | `.logs/train/run4/try_IDM/train/cons_mix_A_steer0333_mb65k_1b_tryIDM_20260718_185415.log` |
| C-B | `.logs/train/run4/try_IDM/train/cons_mix_B_shape_mb65k_1b_tryIDM_20260718_185415.log` |

Checkpoint（评测用 **policy 0 = ego**）：

| 标签 | 路径 |
| --- | --- |
| C-A | `experiments/puffer_drive_conservative_mix_stq4hogz/model_policy_0_puffer_drive_conservative_mix_000954.pt` |
| C-B | `experiments/puffer_drive_conservative_mix_plvd8yph/model_policy_0_puffer_drive_conservative_mix_000954.pt` |

W&B group：`run4-try-IDM-cons-mix`。

### 1.2 评测协议

与 H0 / 旧 try 相同（见 `.logs/val/validation.md`）：

| 项目 | vs IDM | vs PPO |
| --- | --- | --- |
| split | `pufferinter` | `pufferinter` |
| maps | `all`（有效 **589**） | 同左 |
| ego | 上表 policy-0 ckpt | 同左 |
| traffic | IDM（`v0=15`, `T=1.5`, `min_gap=1.0`, `a=1.0`, `b=3.0`） | **同一 ego 权重** PPO |
| 结果根 | `.logs/train/run4/try_IDM/test/<run>/Drive_Recurrent/` | 同左 |

| 标签 | vs IDM 输出目录 | vs PPO 输出目录 |
| --- | --- | --- |
| C-A | `…/20260718_210133_effda6/` | `…/20260718_210133_e72300/` |
| C-B | `…/20260718_211540_45275f/` | `…/20260718_211541_e713bc/` |

### 1.3 成功门槛（相对 H0 run4_r2）

与旧 try 相同：

| 指标 | H0 r2 | 目标 |
| --- | ---: | --- |
| vs IDM Collision | 5.77% | **≤ 4.90%**（相对 −15%） |
| vs IDM At-fault | 4.58% | **≤ 3.89%**（相对 −15%） |
| vs PPO Collision | 0.68% | **≤ 1.68%**（+1 pp absolute） |

通过条件：vs-IDM 的 Coll **或** At-fault 达到 −≥15% **且** vs-PPO Coll 不超上限。

---

## 2. 主结果表

基线 H0 = 同质 Drive+LSTM run4_r2（`u9ymfcqr`）。

### 2.1 评测 vs IDM（pufferinter）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward | 相对 H0 Coll | 相对 H0 Fault | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H0 run4_r2 | 81.66% | 5.77% | 4.58% | 0.85% | −0.675 | — | — | baseline |
| **C-A mb65k** | 82.17% | 4.92% | 4.41% | 1.02% | −0.703 | **−14.7%** | **−3.7%** | **FAIL** |
| **C-B mb65k** | **83.02%** | **2.55%** | **1.87%** | 1.02% | −0.587 | **−55.8%** | **−59.2%** | **PASS** |

### 2.2 评测 vs PPO（pufferinter，同权重）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward | Gate（≤1.68%） |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| H0 run4_r2 | 89.64% | 0.68% | 0.68% | 0.34% | −0.649 | baseline |
| **C-A mb65k** | **90.66%** | **0.00%** | 0.00% | 0.17% | −0.649 | **PASS** |
| **C-B mb65k** | 89.30% | **0.68%** | 0.34% | 0.51% | −0.617 | **PASS** |

### 2.3 判决

| 标签 | vs IDM 门槛 | vs PPO 门槛 | 总判 |
| --- | --- | --- | --- |
| C-A mb65k | FAIL（Coll 4.92% 略高于 4.90%；Fault 未到 −15%） | PASS | **未过关** |
| C-B mb65k | PASS（Coll 与 Fault 均远超 −15%） | PASS | **过关** |

---

## 3. 碰撞构成（vs IDM）

数据来自各 run 的 `collision_snapshots.json`（首碰 + `collision_classifier`）。

| Run | 碰撞 maps | Lateral | Front | Rear | Stopped | At-fault / 碰撞 | fault Lateral | fault Front |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 run4_r2 | 34 | **25** | 7 | 1 | 1 | 27/34（79%） | 20 | 7 |
| C-A mb65k | 29 | 17 | **10** | 1 | 1 | 26/29（90%） | 16 | 10 |
| C-B mb65k | **15** | **9** | **3** | 1 | 2 | 11/15（73%） | 7 | 3 |

要点：

1. **C-A**：相对 H0，Lateral 25→17、Front 7→10；仍偏「压侧向、抬纵向」，At-fault 降幅不足，Coll 卡在门槛边缘。  
2. **C-B**：撞图总数 34→15；Front 仅 3、Lateral 9 → Coll / Fault 大幅下降。  
3. Goal：A/B 在 IDM 上均 ≥ H0；Offroad 均为 1.02%（相对 H0 +0.17 pp，约 +1 map/589）。

---

## 4. 与旧 try（mb=32768，shared）对照

旧 try 数字见 `.logs/train/run4/try/result/conservative_mix_A_B_results.md`（此处用 **r1** 单跑对照同配方）。

| Run | IDM Coll | IDM Fault | IDM Goal | PPO Coll |
| --- | ---: | ---: | ---: | ---: |
| 旧 C-A r1 | 5.09% | 4.58% | 81.32% | 0.00% |
| **C-A mb65k** | **4.92%** | **4.41%** | **82.17%** | 0.00% |
| 旧 C-B r1 | 4.24% | 3.23% | 81.49% | 0.17% |
| **C-B mb65k** | **2.55%** | **1.87%** | **83.02%** | 0.68% |

解读：

- 对齐优化器几何后，**C-B 增益明显放大**（Coll 4.24%→2.55%，Fault 3.23%→1.87%），支持「旧 try 低估了 B 在正确更新频率下的效果」这一判断。  
- **C-A 仍未过关**，但略好于旧 r1；Front 偏高模式仍在（10 vs H0 的 7）。  
- 本轮仍是 **单次复跑**（非显式多 seed）；后续主线建议改用 `mix_ppo_sampling=stratified`（见 `train/stratified_mix_training.md`）再固定 seed 复验。

---

## 5. 解读

```text
旧 try（shared, mb=32768）
  → ego 每 epoch ~32 次更碎的 opt.step、每步 ~16384 样本

try_IDM（shared, mb=65536）
  → ego 每 epoch ~16 次 step、每步 ~32768 样本（对齐 H0）
  → C-B：Front/Lateral 同步大降，过关且幅度更大
  → C-A：结构问题（Front 补偿）仍在，门槛边缘 FAIL
```

**建议**：以 **C-B mb65k** 作为当前 shared 对齐下的主结果；下一阶段优先 **stratified + C-B**（或 B33），用日志 `policy_0/stratified_*` 验收后再评测，而不是继续加码 A 的硬 steer mask。

---

## 6. 文件索引

| 类型 | 路径 |
| --- | --- |
| 本汇总 | `.logs/train/run4/try_IDM/result/conservative_mix_A_B_mb65k_results.md` |
| 指标 CSV | `.logs/train/run4/try_IDM/result/metrics_summary.csv` |
| 优化器对齐说明 | `.logs/train/run4/try_IDM/train/README_optimizer_alignment.md` |
| Stratified 下一阶段 | `.logs/train/run4/try_IDM/train/stratified_mix_training.md` |
| 训练日志 | `.logs/train/run4/try_IDM/train/cons_mix_*_tryIDM_*.log` |
| 原始 eval | `.logs/train/run4/try_IDM/test/cons_mix_*_tryIDM/Drive_Recurrent/` |
| 旧 try 对照 | `.logs/train/run4/try/result/conservative_mix_A_B_results.md` |
| H0 基线汇总 | `.logs/train/run4/problem/homogeneous_drive_lstm_500m_results.md` |

---

## 7. 一句话

**在 shared 采样下把 minibatch 提到 65536 以对齐 H0 优化频率后：C-B 大幅过关（IDM Coll 2.55% / Fault 1.87%）；C-A 仍未过 IDM 门槛。优于旧 try 的 B，且支持继续用 stratified 做更干净的主实验。**
