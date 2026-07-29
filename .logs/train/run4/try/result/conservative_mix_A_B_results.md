# Run4 try · ConservativeMix A / B 结果汇总

> 策略：ego `Drive` + `Recurrent(256,256)`；伙伴分别为 steer 约束（A）/ 奖励塑形（B）  
> 更新时间：2026-07-18（§6 追加 r2 复跑）  
> 原始评测产物：`.logs/train/run4/try/test/`  
> 机器可读表：同目录 `metrics_summary.csv`

---

## 1. 实验配置

两路训练共用 ConservativeMix 配方，**ego 步数对齐同质 H0 的 500M**（50:50 mix → 总 batch / 总 steps 各 ×2）。

| 项目 | 取值 |
| --- | --- |
| 环境 / 入口 | `puffer_drive_conservative_mix` · `pufferlib/config/ocean/drive_conservative_mix.ini` |
| ego 策略 | `Drive` + `Recurrent`（`rnn_input=256`, `rnn_hidden=256`） |
| `mix_ppo` | `True` · `ego:0.5, partner:0.5` |
| `batch_size` | **1,048,576**（2×524288；ego ≈524288/update） |
| `total_timesteps` | **1,000,000,000**（2×500M；ego ≈500M） |
| updates | ≈953 → 最终 ego ckpt epoch **000954** |
| `minibatch_size` / `bptt_horizon` / `update_epochs` | 32768 / 32 / 1 |
| `env.num_maps` | 10,000 |
| 交通混合 | `mix_traffic=False`（训练不接入评测 IDM） |
| 数据根（训练） | `resources/drive/binaries` |

### 1.1 两路差异

| 标签 | partner_mode | partner 策略 | 关键超参 | W&B | GPU |
| --- | --- | --- | --- | --- | ---: |
| **C-A** | `action_constraint` | `DriveSteerConstrained` | `partner_max_abs_steer=0.333` | [`6qdti7k0`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/6qdti7k0) | 0 |
| **C-B** | `reward_shaping` | `Drive`（同架构，伙伴专用 shaping） | `partner_target_headway=1.8`；`w_center/align/steer/gap` 默认 | [`l8z87oaa`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/l8z87oaa) | 1 |

训练日志：

| 标签 | 日志 |
| --- | --- |
| C-A | `.logs/train/run4/try/run/cons_mix_A_steer0333_batch2x_1b_run4_20260718_040248.log` |
| C-B | `.logs/train/run4/try/run/cons_mix_B_shape_batch2x_1b_run4_20260718_040810.log` |

Checkpoint（评测用 **policy 0 = ego**）：

| 标签 | 路径 |
| --- | --- |
| C-A | `experiments/puffer_drive_conservative_mix_6qdti7k0/model_policy_0_puffer_drive_conservative_mix_000954.pt` |
| C-B | `experiments/puffer_drive_conservative_mix_l8z87oaa/model_policy_0_puffer_drive_conservative_mix_000954.pt` |

训练健康度（W&B，B）：`assignment/mixed_scene_rate ≈ 0.97`；`partner/gap_shaping_mean` 全程非零（最新 ≈0.30）。

### 1.2 评测协议

与 H0 相同（见 `.logs/val/validation.md` / H0 汇总 §1）：

| 项目 | vs IDM | vs PPO |
| --- | --- | --- |
| split | `pufferinter` | `pufferinter` |
| maps | `all`（manifest 1000；有效约 589） | 同左 |
| ego | 上表 policy-0 ckpt | 同左 |
| traffic | IDM（`v0=15`, `T=1.5`, `min_gap=1.0`, `a=1.0`, `b=3.0`） | **同一 ego 权重** PPO |
| 结果根 | `.logs/train/run4/try/test/<run>/Drive_Recurrent/` | 同左 |

| 标签 | vs IDM 输出目录 | vs PPO 输出目录 |
| --- | --- | --- |
| C-A | `…/20260718_061533_815c37/` | `…/20260718_061533_94bb7e/` |
| C-B | `…/20260718_063733_064710/` | `…/20260718_063734_73fee6/` |

### 1.3 成功门槛（相对 H0 run4_r2）

定义见 `.logs/train/run4/try/launch_commands.md`：

| 指标 | H0 r2 | 目标 |
| --- | ---: | --- |
| vs IDM Collision | 5.77% | **≤ 4.90%**（相对 −15%） |
| vs IDM At-fault | 4.58% | **≤ 3.89%**（相对 −15%） |
| vs PPO Collision | 0.68% | **≤ 1.68%**（+1 pp absolute） |

通过条件：vs-IDM 的 Coll **或** At-fault 达到 −≥15% **且** vs-PPO Coll 不超上限。

---

## 2. 主结果表

基线 H0 = 同质 Drive+LSTM run4_r2（`u9ymfcqr`），完整表见  
`.logs/train/run4/problem/homogeneous_drive_lstm_500m_results.md`。

### 2.1 评测 vs IDM（pufferinter）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward | 相对 H0 Coll | 相对 H0 Fault | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H0 run4_r2 | 81.66% | 5.77% | 4.58% | 0.85% | −0.675 | — | — | baseline |
| **C-A** | 81.32% | 5.09% | 4.58% | 1.70% | −0.640 | **−11.8%** | **0.0%** | **FAIL** |
| **C-B** | 81.49% | **4.24%** | **3.23%** | 1.36% | −0.711 | **−26.5%** | **−29.5%** | **PASS** |

### 2.2 评测 vs PPO（pufferinter，同权重）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward | Gate（≤1.68%） |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| H0 run4_r2 | 89.64% | 0.68% | 0.68% | 0.34% | −0.649 | baseline |
| **C-A** | 90.32% | **0.00%** | 0.00% | 0.34% | −0.643 | **PASS** |
| **C-B** | 89.98% | **0.17%** | 0.00% | 0.34% | −0.632 | **PASS** |

### 2.3 判决

| 标签 | vs IDM 门槛 | vs PPO 门槛 | 总判 |
| --- | --- | --- | --- |
| C-A | FAIL（Coll 只 −11.8%；Fault 持平） | PASS | **未过关** |
| C-B | PASS（Coll 与 Fault 均 ≤ −15%） | PASS | **过关** |

---

## 3. 碰撞构成（vs IDM）

数据来自各 run 的 `collision_snapshots.json`（首碰 + `collision_classifier`）。

| Run | 碰撞 maps | Lateral | Front | Rear | Stopped | At-fault / 碰撞 | fault Lateral | fault Front |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 run4_r2 | 34 | **25** | 7 | 1 | 1 | 27/34（79%） | 20 | 7 |
| C-A | 30 | 14 | **14** | 0 | 2 | 27/30（90%） | 12 | 14 |
| C-B | **25** | 18 | **5** | 1 | 1 | 19/25（76%） | 14 | 5 |

要点：

1. **C-A**：Lateral 从 25→14 明显下降，但 Front 从 7→14 翻倍（Front 一律 at-fault）→ 总 Coll 略降，**At-fault 率卡在 4.58%**。  
2. **C-B**：Lateral 与 Front 同时下降，Front 为三路最低（5）→ Coll / At-fault 双达标。  
3. Goal 三路均在 ~81%（IDM）/ ~90%（PPO），能力指标未牺牲。

---

## 4. 解读

```text
H0 同质自博弈：同伴会横向配合
   ↓ 分布偏移
评测 IDM：纵向保距 T=1.5、横向钝、不主动让道
   ↓
H0 失败：Lateral 为主 + Front 为辅

C-A（伙伴大转角 logits mask）
   → 训练中横向摩擦↓ → 评测 Lateral↓
   → ego 更偏纵向贴挤 → Front↑ → At-fault 不降

C-B（伙伴奖励塑形：居中/对齐/转角罚 + gap T_train=1.8）
   → 同伴更保守跟车/少乱转
   → ego 学到对「保守交通」的节奏 → Front↓ + Lateral↓ → 过关
```

**建议**：以 **C-B** 作为 ConservativeMix 主配方；后续可试 **C-AB**（约束+塑形）或 B 的 headway/权重消融，而不是单独加码 A。

---

## 5. 文件索引

| 类型 | 路径 |
| --- | --- |
| 本汇总 | `.logs/train/run4/try/result/conservative_mix_A_B_results.md` |
| 指标 CSV | `.logs/train/run4/try/result/metrics_summary.csv` |
| 设计文档 | `.logs/train/run4/try/2026-07-18-conservative-mixed-training-design.zh.md` |
| 启动说明 | `.logs/train/run4/try/launch_commands.md` |
| 原始 eval | `.logs/train/run4/try/test/cons_mix_{A,B}_*/Drive_Recurrent/` |
| H0 基线汇总 | `.logs/train/run4/problem/homogeneous_drive_lstm_500m_results.md` |

---

## 6. 复跑 r2（追加，2026-07-18）

> 目的：同配方再训 + 再测，估计运行间方差（ini `seed=42` 未改；`pufferl` 中 torch/numpy seed 未启用，故非 bit-exact 复现）。  
> 启动：`.logs/train/run4/try/run/launch_cons_mix_A_B_r2.sh`  
> 下文不改写 §1–§5；r1 结论仍以上文为准。

### 6.1 r2 运行标识

| 标签 | W&B | 训练日志 | ego ckpt |
| --- | --- | --- | --- |
| **C-A r2** | [`t1uqsvzw`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/t1uqsvzw) | `…/run/cons_mix_A_steer0333_batch2x_1b_run4_r2_20260718_112001.log` | `experiments/puffer_drive_conservative_mix_t1uqsvzw/model_policy_0_…_000954.pt` |
| **C-B r2** | [`i1mvzmos`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/i1mvzmos) | `…/run/cons_mix_B_shape_batch2x_1b_run4_r2_20260718_112001.log` | `experiments/puffer_drive_conservative_mix_i1mvzmos/model_policy_0_…_000954.pt` |

评测输出：

| 标签 | vs IDM | vs PPO |
| --- | --- | --- |
| C-A r2 | `test/…_run4_r2/…/20260718_134737_779c43/` | `…/20260718_134738_e60fea/` |
| C-B r2 | `test/…_run4_r2/…/20260718_141037_99d7c4/` | `…/20260718_141038_05841f/` |

### 6.2 r2 主表 · vs IDM

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward | 相对 H0 Coll | 相对 H0 Fault | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H0 run4_r2 | 81.66% | 5.77% | 4.58% | 0.85% | −0.675 | — | — | baseline |
| C-A r1 | 81.32% | 5.09% | 4.58% | 1.70% | −0.640 | −11.8% | 0.0% | FAIL |
| **C-A r2** | 81.83% | **4.58%** | **3.74%** | 1.19% | −0.682 | **−20.6%** | **−18.3%** | **PASS** |
| C-B r1 | 81.49% | 4.24% | 3.23% | 1.36% | −0.711 | −26.5% | −29.5% | PASS |
| **C-B r2** | 82.85% | **4.41%** | **3.23%** | 1.70% | −0.687 | **−23.6%** | **−29.5%** | **PASS** |

### 6.3 r2 主表 · vs PPO（同权重）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward | Gate（≤1.68%） |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| H0 run4_r2 | 89.64% | 0.68% | 0.68% | 0.34% | −0.649 | baseline |
| C-A r1 | 90.32% | 0.00% | 0.00% | 0.34% | −0.643 | PASS |
| **C-A r2** | 89.64% | **0.85%** | 0.34% | 0.34% | −0.667 | **PASS** |
| C-B r1 | 89.98% | 0.17% | 0.00% | 0.34% | −0.632 | PASS |
| **C-B r2** | 89.30% | **0.68%** | 0.51% | 0.51% | −0.644 | **PASS** |

### 6.4 r2 碰撞构成（vs IDM）

| Run | 碰撞 maps | Lateral | Front | Rear | Stopped | At-fault / 碰撞 | fault Lateral | fault Front |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 run4_r2 | 34 | 25 | 7 | 1 | 1 | 27/34 | 20 | 7 |
| C-A r1 | 30 | 14 | 14 | 0 | 2 | 27/30 | 12 | 14 |
| C-A r2 | 27 | 14 | 10 | 1 | 2 | 22/27 | 11 | 10 |
| C-B r1 | 25 | 18 | 5 | 1 | 1 | 19/25 | 14 | 5 |
| C-B r2 | 26 | 16 | 5 | 2 | 3 | 19/26 | 12 | 5 |

### 6.5 r1 / r2 汇总与更新结论

对两 seed 取算术平均（百分点）：

| 方法 | IDM Coll 均值 | IDM Fault 均值 | PPO Coll 均值 | 两跑均过 IDM 门槛？ | 两跑均过 PPO 门槛？ |
| --- | ---: | ---: | ---: | --- | --- |
| C-A | (5.09+4.58)/2 = **4.84%** | (4.58+3.74)/2 = **4.16%** | (0.00+0.85)/2 = **0.43%** | 否（仅 r2） | 是 |
| C-B | (4.24+4.41)/2 = **4.33%** | (3.23+3.23)/2 = **3.23%** | (0.17+0.68)/2 = **0.43%** | **是** | **是** |

相对 H0（Coll 5.77% / Fault 4.58%）：

| 方法 | 均值 Coll 相对变化 | 均值 Fault 相对变化 | 相对 −15% 门槛（用均值） |
| --- | ---: | ---: | --- |
| C-A | −16.2% | −9.2% | Coll 过、Fault 不过 → 按「或」规则均值可过，但 **r1 单跑未过** |
| C-B | −25.0% | −29.5% | **Coll 与 Fault 均过** |

**更新结论（在 §2.3 / §4 之上）：**

1. **C-B 更稳健**：r1/r2 均过 IDM（Coll 与 Fault）与 PPO 门槛；Fault 两跑同为 3.23%，Front 碰撞稳定在 5。  
2. **C-A 有运行间方差**：r1 未过关，r2 过关（Coll 4.58%、Fault 3.74%）；均值 Coll 刚过 −15%，但 Fault 均值仍未到 −15%，且 r1 Front 偏高的模式在 r2 有所缓解（14→10）却未消失。  
3. **vs PPO**：两方法两跑均 ≤1.68%；r2 自博弈 Coll 高于各自 r1（A: 0→0.85%；B: 0.17→0.68%），仍在门槛内，说明存在自博弈方差，但未塌到「只会打 IDM」。  
4. **综合**：仍推荐 **C-B 为主配方**；C-A 不宜仅凭单次 r1 否定，也不宜仅凭 r2 宣称稳定成功——至少需报告双跑范围。  

Goal 两跑均维持 ~81–83%（IDM）/ ~89–90%（PPO），无明显能力牺牲。

---

## 7. 复跑碰撞地图重叠与类型分析（vs IDM，追加）

> 数据：各 run `collision_snapshots.json` 的**首碰地图** `map_id`（一张图计一次）。  
> 机器可读明细：同目录 `collision_map_overlap_vs_idm.json`。  
> 基线对照：H0 run4_r1/r2（`problem/test_IDM/...`）。

### 7.1 碰撞地图集合大小与类型构成

| Run | 碰撞 maps | Lateral | Front | Rear | Stopped | Lateral 占比 | Front 占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 r2 | 34 | 25 | 7 | 1 | 1 | **74%** | 21% |
| C-A r1 | 30 | 14 | 14 | 0 | 2 | 47% | **47%** |
| C-A r2 | 27 | 14 | 10 | 1 | 2 | 52% | 37% |
| C-B r1 | 25 | 18 | 5 | 1 | 1 | **72%** | **20%** |
| C-B r2 | 26 | 16 | 5 | 2 | 3 | **62%** | **19%** |

要点：

- **H0 / C-B**：始终是 Lateral 主导（约 60–75%），Front 约 20%。  
- **C-A**：Front 占比显著更高（r1 近半、r2 仍 37%），与 §3「A 压 Lateral、抬 Front」一致，且 **r2 复跑仍保留该结构**（虽 Front 张数 14→10）。  
- **C-B**：两跑 Front 张数均为 **5**，类型谱最接近「把 H0 的 Front 压住、总图数下降」。

### 7.2 同方法 r1 ∩ r2：地图重叠

定义：Jaccard = \(|A\cap B|\,/\,|A\cup B|\)。同质 H0 两跑 Jaccard≈0.32，可作「运行间方差」参照。

| 对比 | \|X\| | \|Y\| | \|∩\| | 仅 X | 仅 Y | Jaccard | ∩ 上类型一致 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 r1 vs H0 r2 | 32 | 34 | 16 | 16 | 18 | 0.320 | — |
| **C-A r1 vs r2** | 30 | 27 | **14** | 16 | 13 | **0.326** | 9/14（64%） |
| **C-B r1 vs r2** | 25 | 26 | **14** | 11 | 12 | **0.378** | **13/14（93%）** |

解读：

1. **重叠率不高（约 1/3）**：与 H0 双跑同量级 → 碰撞地图对训练噪声敏感，单跑「撞在哪」不宜过度解读；应看**类型结构 + 总率**。  
2. **C-B 比 C-A 更「定型」**：交集上类型一致性 93% vs 64%；B 的失败模式跨 seed 更稳（多为 Lateral）。  
3. **C-A 交集上类型漂移**：14 张共有图中 5 张 r1/r2 分型不同（Front↔Lateral 互换为主），说明 A 的失败边界更抖。

**A r1∩r2 共有 map_id（14）**：  
`9, 97, 115, 142, 315, 346, 358, 474, 625, 707, 734, 762, 787, 790`

**B r1∩r2 共有 map_id（14）**：  
`97, 115, 223, 315, 358, 380, 438, 522, 618, 625, 707, 772, 816, 986`

其中 **A 与 B 的「双跑稳定失败」交集**（同时 ∈ A∩r1r2 且 ∈ B∩r1r2）共 **6** 张：  
`97, 115, 315, 358, 625, 707`。

更严的「H0 r2 + A r1/r2 + B r1/r2 皆撞」亦为这 **6** 张（见 JSON）。

### 7.3 相对 H0 r2 的地图重叠（方法是否「换了一批撞点」）

| 对比 | \|∩\| | Jaccard | 含义 |
| --- | ---: | ---: | --- |
| H0 r2 ∩ A r1 | 16 | 0.333 | 约一半 H0 撞点仍在 A |
| H0 r2 ∩ A r2 | 16 | 0.356 | 同上 |
| H0 r2 ∩ B r1 | 13 | 0.283 | B 消掉更多 H0 撞点 |
| H0 r2 ∩ B r2 | 17 | 0.395 | 仍保留一批「硬核」侧向图 |

**H0 r2 ∩ B r1 ∩ B r2**（H0 有、且 B 两跑仍撞）共 **11** 张：  
`97, 115, 223, 315, 358, 380, 438, 618, 625, 707, 816`  
以 Lateral 为主——即 B **降低了撞图总数与 Front，但未消除一批顽固 Lateral 场景**。

**跨方法 r1**：A r1 ∩ B r1 仅 10 张（Jaccard 0.222），说明 A/B 改动的失败集合并不相同：A 更偏 Front 子集，B 更偏「缩减后的 Lateral 子集」。

### 7.4 交集上的类型交叉（r1 → r2）

**C-A（∩=14）**

| r1 \ r2 | Lateral | Front | Stopped |
| --- | ---: | ---: | ---: |
| Lateral | 3 | 2 | 0 |
| Front | 3 | 4 | 0 |
| Stopped | 0 | 0 | 2 |

→ 存在明显的 Front↔Lateral 标签漂移（几何边界附近或首碰判定敏感）。

**C-B（∩=14）**

| r1 \ r2 | Lateral | Front | Stopped |
| --- | ---: | ---: | ---: |
| Lateral | **10** | 0 | 0 |
| Front | 1 | **2** | 0 |
| Stopped | 0 | 0 | **1** |

→ 几乎锁在 Lateral；仅 1 张 Front→Lateral。类型结论可复现。

### 7.5 本节结论（给报告用）

1. **复跑重叠中等（Jaccard≈0.33–0.38）**，与 H0 自身双跑相当 → 碰撞地图 ID 的 seed 方差大；比较方法应优先看 **Coll/Fault 率 + Lateral/Front 结构**，不要只盯单张 map。  
2. **C-B 类型结构跨 r1/r2 稳定**（Front 恒为 5；交集类型一致率 93%），支撑「B 主要压 Front、总撞图下降」的机制叙述。  
3. **C-A 类型不稳定且 Front 偏高**（双跑皆然），与「硬约束横向 → 纵向贴挤」叙事一致。  
4. 仍存在一小撮 **跨 H0/A/B 的顽固 map**（`97, 115, 315, 358, 625, 707`），适合作为后续 case study / C-AB 或更强纵向 shaping 的探针，而不是否定 B 的总体增益。
