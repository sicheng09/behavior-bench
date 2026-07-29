# run_repaet · 优化器对齐混智复训 · vs IDM 结果汇总

> 评测 ego：标准 `Drive` + `Recurrent(256,256)`  
> 协议：`pufferinter` · `MAP_IDS=all`（有效图 589）· traffic=`idm`  
> 基线：同质 Drive+LSTM 500M **H0 run4_r2**（[`u9ymfcqr`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/u9ymfcqr)）  
> 更新：2026-07-23（已补齐 MIXMOE stratified vs-IDM）  
> 机器可读表：同目录 `metrics_summary.csv`

**加粗规则**：相对 H0 r2，Goal↑ / Reward↑ 更高加粗；Collision↓ / At-fault↓ / Offroad↓ 更低加粗。

---

## 1. 实验设定

本轮修正旧混智实验「`batch`/`steps`×3 但 `minibatch` 未放大」的优化器几何偏差，使每策略对齐 H0 500M：

| 量 | H0 同质 | 本轮 3 路 1:1:1 |
| --- | ---: | ---: |
| `batch_size` | 524,288 | **1,572,864**（×3） |
| `total_timesteps` | 500M | **1.5B**（×3） |
| shared `minibatch` | 32,768 | **98,304**（×3） |
| stratified 每路 | — | **mb 32,768 × 16 steps** |
| 每策略环境步 / 外层 epoch | ≈500M / ≈954 | **同左** → ckpt `000954` |

两采样模式对照：

| 模式 | 几何控制 |
| --- | --- |
| **shared** | 全局 `minibatch=98304`，共享池均匀采样 |
| **stratified** | 每策略独立逻辑池；`policy_mb=32768` × `update_steps=16` |

### 1.1 三组混智配方

| 系列 | 策略组成 | 评测槽位 | W&B group |
| --- | --- | ---: | --- |
| **MIXMOE** | `Drive, DriveMoE, DriveMoE3` + `Recurrent×3` | policy **0** | `run_repaet-mixmoe` |
| **DriveLite A** | `Drive, Drive, DriveLite` · rnn `Recurrent,None,None` | policy **0** | `run_repaet-drivelite` |
| **DriveLite B** | `Drive, DriveLite, Drive` · rnn `Recurrent,Recurrent,None` | policy **0** | `run_repaet-drivelite` |
| **Perception** | `DrivePerceptionLow, Mid, Drive` + `Recurrent×3` | policy **2**（high） | `run_repaet-perception` |

### 1.2 Run 索引

| Run | Sampling | W&B | Checkpoint |
| --- | --- | --- | --- |
| MIXMOE shared | shared | [`ncn8qxko`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/ncn8qxko) | `experiments/puffer_drive_ncn8qxko/model_policy_0_…_000954.pt` |
| MIXMOE strat | stratified | [`o4trtoxc`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/o4trtoxc) | `experiments/puffer_drive_o4trtoxc/model_policy_0_…_000954.pt` |
| DriveLite A shared | shared | [`uhnxy2lw`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/uhnxy2lw) | `…uhnxy2lw/model_policy_0_…_000954.pt` |
| DriveLite A strat | stratified | [`r0jht7bf`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/r0jht7bf) | `…r0jht7bf/model_policy_0_…_000954.pt` |
| DriveLite B shared | shared | [`d9fq08yq`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/d9fq08yq) | `…d9fq08yq/model_policy_0_…_000954.pt` |
| DriveLite B strat | stratified | [`qgytyxt4`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/qgytyxt4) | `…qgytyxt4/model_policy_0_…_000954.pt` |
| Perception shared | shared | [`yj5u9i7h`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/yj5u9i7h) | `…yj5u9i7h/model_policy_2_…_000954.pt` |
| Perception strat | stratified | [`yov2516v`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/yov2516v) | `…yov2516v/model_policy_2_…_000954.pt` |

---

## 2. 主结果：vs IDM（pufferinter）

| Run | Sampling | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward ↑ | ΔColl vs H0 | ΔFault vs H0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 run4_r2（baseline） | — | 81.66% | 5.77% | 4.58% | 0.85% | −0.675 | — | — |
| MIXMOE shared | shared | 81.15% | 6.11% | 4.92% | 1.02% | **−0.652** | +5.9% | +7.4% |
| MIXMOE strat | stratified | 80.98% | 5.94% | 4.92% | 1.19% | −0.722 | +2.9% | +7.4% |
| DriveLite A shared | shared | 81.66% | **4.07%** | **3.57%** | 1.53% | **−0.578** | **-29.5%** | **-22.1%** |
| DriveLite A strat | stratified | **82.68%** | **2.72%** | **1.70%** | 1.02% | **−0.502** | **-52.9%** | **-62.9%** |
| DriveLite B shared | shared | **82.51%** | **4.41%** | **3.57%** | 1.70% | −0.676 | **-23.6%** | **-22.1%** |
| DriveLite B strat | stratified | 81.15% | **5.09%** | **4.07%** | 1.70% | −0.693 | **-11.8%** | **-11.1%** |
| Perception shared | shared | 80.48% | **3.57%** | **2.89%** | 1.02% | **−0.596** | **-38.1%** | **-36.9%** |
| Perception strat | stratified | 81.15% | **2.72%** | **2.38%** | 1.53% | **−0.542** | **-52.9%** | **-48.0%** |

### 2.1 shared（含 baseline）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward ↑ | ΔColl vs H0 | ΔFault vs H0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 run4_r2（baseline） | 81.66% | 5.77% | 4.58% | 0.85% | −0.675 | — | — |
| MIXMOE shared | 81.15% | 6.11% | 4.92% | 1.02% | **−0.652** | +5.9% | +7.4% |
| DriveLite A shared | 81.66% | **4.07%** | **3.57%** | 1.53% | **−0.578** | **-29.5%** | **-22.1%** |
| DriveLite B shared | **82.51%** | **4.41%** | **3.57%** | 1.70% | −0.676 | **-23.6%** | **-22.1%** |
| Perception shared | 80.48% | **3.57%** | **2.89%** | 1.02% | **−0.596** | **-38.1%** | **-36.9%** |

### 2.2 stratified（含 baseline）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward ↑ | ΔColl vs H0 | ΔFault vs H0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 run4_r2（baseline） | 81.66% | 5.77% | 4.58% | 0.85% | −0.675 | — | — |
| MIXMOE strat | 80.98% | 5.94% | 4.92% | 1.19% | −0.722 | +2.9% | +7.4% |
| DriveLite A strat | **82.68%** | **2.72%** | **1.70%** | 1.02% | **−0.502** | **-52.9%** | **-62.9%** |
| DriveLite B strat | 81.15% | **5.09%** | **4.07%** | 1.70% | −0.693 | **-11.8%** | **-11.1%** |
| Perception strat | 81.15% | **2.72%** | **2.38%** | 1.53% | **−0.542** | **-52.9%** | **-48.0%** |

### 2.3 按系列速览

| 系列 | 最优 Coll | 最优 Fault | 相对 H0 Coll | 备注 |
| --- | ---: | ---: | ---: | --- |
| DriveLite A | **2.72%**（strat） | **1.70%**（strat） | **−52.9%** | 本轮最强；Goal 亦升至 **82.68%** |
| Perception | **2.72%**（strat） | **2.38%**（strat） | **−52.9%** | Coll 与 LiteA strat 持平；Goal 略低于 H0 |
| DriveLite B | **4.41%**（shared） | **3.57%**（shared） | **−23.6%** | shared > strat；strat 仅小幅降撞 |
| MIXMOE | 5.94%（strat） | 4.92%（shared/strat） | +2.9% | shared/strat 均未优于 H0；strat Coll 略好于 shared |

---

## 3. 分系列对照

### 3.1 MIXMOE（结构混智）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 r2 | 81.66% | 5.77% | 4.58% | 0.85% | −0.675 |
| shared `ncn8qxko` | 81.15% | 6.11% | 4.92% | 1.02% | **−0.652** |
| strat `o4trtoxc` | 80.98% | 5.94% | 4.92% | 1.19% | −0.722 |

解读：对齐优化器后，**结构混智 shared / stratified 均未改善 vs-IDM 安全指标**。strat 相对 shared 仅把 Coll 从 6.11% 微降到 5.94%，仍略差于 H0（5.77%）；Fault 同为 4.92%。Goal 与 Reward 也未超过同质。可判定：在本对齐预算下，**Drive/DriveMoE/DriveMoE3 结构混智对 IDM 泛化无效**。

### 3.2 DriveLite（智力层级混智）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 r2 | 81.66% | 5.77% | 4.58% | 0.85% | −0.675 |
| A shared | 81.66% | **4.07%** | **3.57%** | 1.53% | **−0.578** |
| **A strat** | **82.68%** | **2.72%** | **1.70%** | 1.02% | **−0.502** |
| B shared | **82.51%** | **4.41%** | **3.57%** | 1.70% | −0.676 |
| B strat | 81.15% | **5.09%** | **4.07%** | 1.70% | −0.693 |

解读：

1. **A strat 全面领先**：Coll 5.77→2.72（−52.9%）、Fault 4.58→1.70（−62.9%），Goal 与 Reward 同步提升。
2. **A 上 stratified ≫ shared**；B 则 **shared > strat**（配方/槽位对采样模式敏感）。
3. Offroad 普遍略高于 H0（+0.17～+0.85 pp），安全收益主要来自车–车碰撞下降。

### 3.3 Perception（感知混智）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 r2 | 81.66% | 5.77% | 4.58% | 0.85% | −0.675 |
| shared | 80.48% | **3.57%** | **2.89%** | 1.02% | **−0.596** |
| **strat** | 81.15% | **2.72%** | **2.38%** | 1.53% | **−0.542** |

解读：感知混智在 **Coll/Fault/Reward** 上稳定优于 H0；**strat 再降一档碰撞**（与 LiteA strat 同为 2.72% Coll）。代价是 Goal 略低于同质（shared −1.18 pp / strat −0.51 pp）。

---

## 4. 碰撞构成（vs IDM 首碰）

来源：各 run `collision_snapshots.json`（`collision_classifier`）。

| Run | 碰撞 maps | Lateral | Front | Rear | Stopped | At-fault / 碰撞 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 r2 | 34 | **25** | 7 | 1 | 1 | 27/34（79%） |
| MIXMOE shared | 36 | 23 | 7 | 4 | 2 | 29/36（81%） |
| MIXMOE strat | 35 | 22 | 11 | 0 | 2 | 29/35（83%） |
| DriveLite A shared | 24 | 12 | 9 | 0 | 3 | 21/24（88%） |
| DriveLite A strat | **16** | **4** | 7 | 3 | 2 | 10/16（63%） |
| DriveLite B shared | 26 | 12 | 11 | 1 | 2 | 21/26（81%） |
| DriveLite B strat | 30 | 10 | **17** | 0 | 3 | 24/30（80%） |
| Perception shared | 21 | 10 | 8 | 1 | 2 | 17/21（81%） |
| Perception strat | **16** | 8 | 5 | 0 | 3 | 14/16（88%） |

要点：

1. H0 主因仍是 **Lateral**（25/34）。LiteA strat 将 Lateral 压到 **4**，撞图总数 34→16。
2. LiteB strat 反而抬高 **Front**（17），与 A strat 形成鲜明对比。
3. Perception strat 撞图同样压到 16，Lateral/Front 同步下降。
4. MIXMOE shared/strat 撞图数与 H0 接近（36/35 vs 34），Lateral 仍占主导，**未改变失败模式**。

---

## 5. shared vs stratified

| 系列 | Coll：shared → strat | Fault：shared → strat | 谁更好 |
| --- | ---: | ---: | --- |
| DriveLite A | 4.07 → **2.72** | 3.57 → **1.70** | **stratified** |
| DriveLite B | **4.41** → 5.09 | **3.57** → 4.07 | **shared** |
| Perception | 3.57 → **2.72** | 2.89 → **2.38** | **stratified** |
| MIXMOE | 6.11 → **5.94** | 4.92 → 4.92 | 均差于 H0；strat 略降 Coll |

结论：**分层池并不总是更优**——对 DriveLite A / Perception 帮助很大，对 DriveLite B 反而略差；对 MIXMOE 几乎无实质增益。后续应用按配方分别选型，而不是默认一边倒。

---

## 6. 总判与建议

| 优先级 | 结论 |
| --- | --- |
| 1 | **DriveLite A + stratified** 是本轮 vs-IDM 最优：Coll **2.72%**、Fault **1.70%**、Goal **82.68%**。 |
| 2 | **Perception + stratified** 安全指标并列最强 Coll，Fault 次优；Goal 略牺牲。 |
| 3 | DriveLite B 建议用 **shared**。 |
| 4 | **MIXMOE shared / strat 均未超过同质 H0**；结构混智在本对齐设定下对 IDM 泛化无效。 |
| 5 | 对齐 minibatch 后，混智收益主要体现在 **智力/感知混智**，而非结构 MoE 混智。 |

---

## 7. 产物路径

| Run | summary.csv 目录 |
| --- | --- |
| MIXMOE shared | `.logs/train/run_repaet/trainMOE/test_IDM/mix_moe_drive_moe_moe3_mb98k_1500m_run_repaet/Drive_Recurrent/20260723_001400_be0cbf/` |
| MIXMOE strat | `.logs/train/run_repaet/trainMOE/test_IDM/mix_moe_drive_moe_moe3_stratified_1500m_run_repaet/Drive_Recurrent/20260723_012134_3971dc/` |
| DriveLite A shared | `…/trainLite/test_IDM/drivelite_mixA_mb98k_shared_1500m_run_repaet/Drive_Recurrent/20260722_225102_503849/` |
| DriveLite A strat | `…/trainLite/test_IDM/drivelite_mixA_stratified_1500m_run_repaet/Drive_Recurrent/20260722_225801_eb6b8c/` |
| DriveLite B shared | `…/trainLite/test_IDM/drivelite_mixB_mb98k_shared_1500m_run_repaet/Drive_Recurrent/20260722_230201_7bf012/` |
| DriveLite B strat | `…/trainLite/test_IDM/drivelite_mixB_stratified_1500m_run_repaet/Drive_Recurrent/20260722_230001_25bd9f/` |
| Perception shared | `…/trainPerception/test_IDM/perception_mix_low_mid_high_mb98k_shared_1500m_run_repaet/Drive_Recurrent/20260722_232702_5ee103/` |
| Perception strat | `…/trainPerception/test_IDM/perception_mix_low_mid_high_stratified_1500m_run_repaet/Drive_Recurrent/20260722_232602_b6e010/` |

H0 基线说明：`.logs/train/run4/problem/test_IDM/homogeneous_drive_lstm_500m_results.md`  
对齐文档：`.logs/train/run4/try_IDM/train/README_optimizer_alignment.md`
