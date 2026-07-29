# run_repaet · 优化器对齐混智复训 · vs SMART 结果汇总

> 评测 ego：标准 `Drive` + `Recurrent(256,256)`  
> 协议：`pufferinter` · `MAP_IDS=all`（有效图 589）· traffic=`smart`  
> SMART 权重：`weights/SMART_epoch_030.pt`（greedy · `repredict_interval=5`）  
> 基线：同质 Drive+LSTM 500M **H0 run4_r2**（[`u9ymfcqr`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/u9ymfcqr)）vs SMART  
> 对照：同目录 `run_repaet_vs_idm_results.md` / `run_repaet_vs_expert_results.md`  
> 更新：2026-07-23  
> 机器可读表：同目录 `metrics_summary_smart.csv`

**加粗规则**：相对 H0 r2（vs SMART），Goal↑ / Reward↑ 更高加粗；Collision↓ / At-fault↓ / Offroad↓ 更低加粗。

---

## 1. 实验设定

与 vs-IDM / vs-Expert 汇总共用同一批对齐复训权重。本文件仅汇总 **traffic=`smart`**（学习型轨迹预测他车）。

| 项目 | 取值 |
| --- | --- |
| split / maps | `pufferinter` · `all`（有效 **589**） |
| traffic | **smart**（1× PPO ego + 其余 SMART） |
| SMART ckpt | `weights/SMART_epoch_030.pt` · `greedy=True` · `repredict_interval=5` |
| ego | MIXMOE/DriveLite=`policy_0`；Perception=`policy_2` |
| 结果根 | `.logs/train/run_repaet/{trainMOE,trainLite,trainPerception}/test_Smart/` |
| 启动 | `launch_all_eval_smart.sh`（原 train GPU 0–7 并行） |

### 1.1 Run 索引

| Run | Sampling | W&B | Checkpoint |
| --- | --- | --- | --- |
| MIXMOE shared | shared | [`ncn8qxko`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/ncn8qxko) | `…ncn8qxko/model_policy_0_…_000954.pt` |
| MIXMOE strat | stratified | [`o4trtoxc`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/o4trtoxc) | `…o4trtoxc/model_policy_0_…_000954.pt` |
| DriveLite A shared | shared | [`uhnxy2lw`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/uhnxy2lw) | `…uhnxy2lw/model_policy_0_…_000954.pt` |
| DriveLite A strat | stratified | [`r0jht7bf`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/r0jht7bf) | `…r0jht7bf/model_policy_0_…_000954.pt` |
| DriveLite B shared | shared | [`d9fq08yq`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/d9fq08yq) | `…d9fq08yq/model_policy_0_…_000954.pt` |
| DriveLite B strat | stratified | [`qgytyxt4`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/qgytyxt4) | `…qgytyxt4/model_policy_0_…_000954.pt` |
| Perception shared | shared | [`yj5u9i7h`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/yj5u9i7h) | `…yj5u9i7h/model_policy_2_…_000954.pt` |
| Perception strat | stratified | [`yov2516v`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/yov2516v) | `…yov2516v/model_policy_2_…_000954.pt` |

H0 SMART 基线说明：`.logs/train/run4/problem/test_Smart/homogeneous_drive_lstm_500m_results.md`

---

## 2. 主结果：vs SMART（pufferinter）

| Run | Sampling | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward ↑ | ΔColl vs H0 | ΔFault vs H0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 run4_r2（baseline） | — | 80.48% | 6.79% | 4.92% | 0.85% | −0.741 | — | — |
| MIXMOE shared | shared | 78.78% | 8.15% | 6.96% | 0.85% | −0.786 | +20.0% | +41.5% |
| MIXMOE strat | stratified | **82.51%** | **4.24%** | **3.57%** | 1.70% | −0.757 | **-37.6%** | **-27.4%** |
| DriveLite A shared | shared | **83.70%** | **4.24%** | **3.57%** | 0.85% | **−0.690** | **-37.6%** | **-27.4%** |
| DriveLite A strat | stratified | **84.21%** | **2.89%** | **2.21%** | **0.68%** | **−0.597** | **-57.4%** | **-55.1%** |
| DriveLite B shared | shared | **82.68%** | **4.41%** | **4.07%** | 1.36% | −0.771 | **-35.1%** | **-17.3%** |
| DriveLite B strat | stratified | 80.48% | **5.09%** | **4.58%** | 2.55% | −0.760 | **-25.0%** | **-6.9%** |
| Perception shared | shared | **81.83%** | **4.41%** | **3.40%** | 1.36% | **−0.699** | **-35.1%** | **-30.9%** |
| Perception strat | stratified | **81.32%** | **4.07%** | **2.55%** | 1.19% | **−0.637** | **-40.1%** | **-48.2%** |

### 2.1 shared（含 baseline）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward ↑ | ΔColl vs H0 | ΔFault vs H0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 run4_r2（baseline） | 80.48% | 6.79% | 4.92% | 0.85% | −0.741 | — | — |
| MIXMOE shared | 78.78% | 8.15% | 6.96% | 0.85% | −0.786 | +20.0% | +41.5% |
| DriveLite A shared | **83.70%** | **4.24%** | **3.57%** | 0.85% | **−0.690** | **-37.6%** | **-27.4%** |
| DriveLite B shared | **82.68%** | **4.41%** | **4.07%** | 1.36% | −0.771 | **-35.1%** | **-17.3%** |
| Perception shared | **81.83%** | **4.41%** | **3.40%** | 1.36% | **−0.699** | **-35.1%** | **-30.9%** |

### 2.2 stratified（含 baseline）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward ↑ | ΔColl vs H0 | ΔFault vs H0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 run4_r2（baseline） | 80.48% | 6.79% | 4.92% | 0.85% | −0.741 | — | — |
| MIXMOE strat | **82.51%** | **4.24%** | **3.57%** | 1.70% | −0.757 | **-37.6%** | **-27.4%** |
| DriveLite A strat | **84.21%** | **2.89%** | **2.21%** | **0.68%** | **−0.597** | **-57.4%** | **-55.1%** |
| DriveLite B strat | 80.48% | **5.09%** | **4.58%** | 2.55% | −0.760 | **-25.0%** | **-6.9%** |
| Perception strat | **81.32%** | **4.07%** | **2.55%** | 1.19% | **−0.637** | **-40.1%** | **-48.2%** |

### 2.3 按系列速览

| 系列 | 最优 Coll | 最优 Fault | 相对 H0 Coll | 备注 |
| --- | ---: | ---: | ---: | --- |
| DriveLite A | **2.89%**（strat） | **2.21%**（strat） | **−57.4%** | 本轮 SMART 最强；Goal **84.21%** |
| Perception | **4.07%**（strat） | **2.55%**（strat） | **−40.1%** | Fault 次优；Goal 略升 |
| MIXMOE | **4.24%**（strat） | **3.57%**（strat） | **−37.6%** | **strat 大幅优于 shared**（shared 反差于 H0） |
| DriveLite B | **4.41%**（shared） | **4.07%**（shared） | **−35.1%** | shared > strat（与 IDM 上 B 一致） |

---

## 3. 分系列对照

### 3.1 MIXMOE（结构混智）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 r2 | 80.48% | 6.79% | 4.92% | 0.85% | −0.741 |
| shared `ncn8qxko` | 78.78% | 8.15% | 6.96% | 0.85% | −0.786 |
| **strat** `o4trtoxc` | **82.51%** | **4.24%** | **3.57%** | 1.70% | −0.757 |

解读：相对 IDM/Expert 上「结构混智几乎无效」，SMART 上出现 **分层采样翻转**：strat Coll 6.79→4.24（−37.6%），shared 反而更差（8.15%）。说明对学习型他车，独立池更新对 Drive/MoE 混智更关键。

### 3.2 DriveLite（智力层级混智）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 r2 | 80.48% | 6.79% | 4.92% | 0.85% | −0.741 |
| A shared | **83.70%** | **4.24%** | **3.57%** | 0.85% | **−0.690** |
| **A strat** | **84.21%** | **2.89%** | **2.21%** | **0.68%** | **−0.597** |
| B shared | **82.68%** | **4.41%** | **4.07%** | 1.36% | −0.771 |
| B strat | 80.48% | **5.09%** | **4.58%** | 2.55% | −0.760 |

解读：

1. **A strat 全面领先**：Coll/Fault/Goal/Reward/Offroad 均优于 H0，撞图 40→17。
2. A 上 stratified ≫ shared；B 上 **shared > strat**（与 IDM 一致，与 Expert 不同）。
3. 智力混智在 SMART 上收益幅度接近甚至超过 IDM（A strat Coll −57% vs IDM −53%）。

### 3.3 Perception（感知混智）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 r2 | 80.48% | 6.79% | 4.92% | 0.85% | −0.741 |
| shared | **81.83%** | **4.41%** | **3.40%** | 1.36% | **−0.699** |
| **strat** | **81.32%** | **4.07%** | **2.55%** | 1.19% | **−0.637** |

解读：感知混智在 SMART 上 Coll/Fault/Reward 稳定优于 H0；strat 再降一档 Fault（2.55%）。Goal 两者均略高于同质。

---

## 4. 碰撞构成（vs SMART 首碰）

来源：各 run `collision_snapshots.json`。

| Run | 碰撞 maps | Lateral | Front | Rear | Stopped | At-fault / 碰撞 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 r2 | 40 | **23** | 9 | 7 | 1 | 29/40（73%） |
| MIXMOE shared | 48 | **32** | 11 | 4 | 1 | 41/48（85%） |
| MIXMOE strat | **25** | 17 | 6 | 1 | 1 | 21/25（84%） |
| DriveLite A shared | 25 | 13 | 9 | 2 | 1 | 21/25（84%） |
| DriveLite A strat | **17** | 11 | **2** | 2 | 2 | 13/17（76%） |
| DriveLite B shared | 26 | 11 | **14** | 1 | 0 | 24/26（92%） |
| DriveLite B strat | 30 | 10 | 13 | 3 | 4 | 27/30（90%） |
| Perception shared | 26 | 14 | 6 | 5 | 1 | 20/26（77%） |
| Perception strat | 24 | 11 | 5 | 7 | 1 | 15/24（63%） |

要点：

1. H0 SMART 主因仍是 **Lateral**（23/40），与 IDM 类似、不同于 Expert 的 Stopped 主导。
2. A strat / MIXMOE strat 显著压低撞图（40→17/25）；A strat 的 Front 仅 2。
3. MIXMOE shared 抬高 Lateral（32）→ Coll 变差；分层后 Lateral 回落到 17。

---

## 5. shared vs stratified

| 系列 | Coll：shared → strat | Fault：shared → strat | 谁更好（SMART） |
| --- | ---: | ---: | --- |
| DriveLite A | 4.24 → **2.89** | 3.57 → **2.21** | **stratified** |
| DriveLite B | **4.41** → 5.09 | **4.07** → 4.58 | **shared** |
| Perception | 4.41 → **4.07** | 3.40 → **2.55** | **stratified** |
| MIXMOE | 8.15 → **4.24** | 6.96 → **3.57** | **stratified**（翻转） |

结论：SMART 上 **A / Perception / MIXMOE 均偏好 stratified**；仅 DriveLite B 偏好 shared。MIXMOE 的采样模式敏感度在三协议中最高。

---

## 6. 与 vs-IDM / vs-Expert 对照（同权重）

| Run | IDM Coll | SMART Coll | Expert Coll | IDM Fault | SMART Fault |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 r2 | 5.77% | 6.79% | 16.81% | 4.58% | 4.92% |
| DriveLite A strat | **2.72%** | **2.89%** | 14.43% | **1.70%** | **2.21%** |
| DriveLite B strat | 5.09% | 5.09% | **14.09%** | 4.07% | 4.58% |
| Perception strat | **2.72%** | 4.07% | 15.62% | 2.38% | 2.55% |
| MIXMOE strat | 5.94% | **4.24%** | 16.81% | 4.92% | **3.57%** |
| MIXMOE shared | 6.11% | 8.15% | 16.13% | 4.92% | 6.96% |

解读：

1. 难度阶梯大致为：PPO ≪ IDM ≈ SMART ≪ Expert（H0 Coll 5.8 / 6.8 / 16.8）。
2. **A strat** 在 IDM 与 SMART 上同为最优安全档；Expert 上改由 B strat 领跑。
3. **MIXMOE strat** 是 SMART 特有亮点（IDM/Expert 上几乎无增益）。

---

## 7. 总判与建议

| 优先级 | 结论 |
| --- | --- |
| 1 | SMART 上 **DriveLite A + stratified** 全面最优：Coll **2.89%**、Fault **2.21%**、Goal **84.21%**。 |
| 2 | **MIXMOE 必须用 stratified**：shared 差于 H0，strat 则 Coll −37.6%。 |
| 3 | Perception strat 安全次优；DriveLite B 仍建议 **shared**。 |
| 4 | 跨协议稳健首选仍是 **DriveLite A strat**（IDM+SMART 双强）；Expert 需另看 B strat。 |
| 5 | 结构混智并非普遍无效——在学习型 SMART 交通上，分层更新可解锁明显收益。 |

---

## 8. 产物路径

| Run | summary.csv 目录 |
| --- | --- |
| MIXMOE shared | `.logs/train/run_repaet/trainMOE/test_Smart/mix_moe_drive_moe_moe3_mb98k_1500m_run_repaet/Drive_Recurrent/20260723_045454_db6c91/` |
| MIXMOE strat | `…/trainMOE/test_Smart/mix_moe_drive_moe_moe3_stratified_1500m_run_repaet/Drive_Recurrent/20260723_045459_5c3b9d/` |
| DriveLite A shared | `…/trainLite/test_Smart/drivelite_mixA_mb98k_shared_1500m_run_repaet/Drive_Recurrent/20260723_045505_158c55/` |
| DriveLite A strat | `…/trainLite/test_Smart/drivelite_mixA_stratified_1500m_run_repaet/Drive_Recurrent/20260723_045510_c2c37c/` |
| DriveLite B shared | `…/trainLite/test_Smart/drivelite_mixB_mb98k_shared_1500m_run_repaet/Drive_Recurrent/20260723_045516_1bb166/` |
| DriveLite B strat | `…/trainLite/test_Smart/drivelite_mixB_stratified_1500m_run_repaet/Drive_Recurrent/20260723_045521_af3a21/` |
| Perception shared | `…/trainPerception/test_Smart/perception_mix_low_mid_high_mb98k_shared_1500m_run_repaet/Drive_Recurrent/20260723_045527_6bb609/` |
| Perception strat | `…/trainPerception/test_Smart/perception_mix_low_mid_high_stratified_1500m_run_repaet/Drive_Recurrent/20260723_045532_c77a78/` |

H0 SMART 基线：`.logs/train/run4/problem/test_Smart/homogeneous_drive_lstm_500m_results.md`  
同权重 IDM / Expert 汇总：`run_repaet_vs_idm_results.md` / `run_repaet_vs_expert_results.md`
