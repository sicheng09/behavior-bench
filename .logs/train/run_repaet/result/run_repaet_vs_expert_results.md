# run_repaet · 优化器对齐混智复训 · vs Expert 结果汇总

> 评测 ego：标准 `Drive` + `Recurrent(256,256)`  
> 协议：`pufferinter` · `MAP_IDS=all`（有效图 589）· traffic=`expert`  
> 基线：同质 Drive+LSTM 500M **H0 run4_r2**（[`u9ymfcqr`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/u9ymfcqr)）vs Expert  
> 对照：同目录 `run_repaet_vs_idm_results.md`  
> 更新：2026-07-23  
> 机器可读表：同目录 `metrics_summary_expert.csv`

**加粗规则**：相对 H0 r2（vs Expert），Goal↑ / Reward↑ 更高加粗；Collision↓ / At-fault↓ / Offroad↓ 更低加粗。

---

## 1. 实验设定

与 vs-IDM 汇总共用同一批对齐复训权重（见 `run_repaet_vs_idm_results.md` §1）。本文件仅汇总 **traffic=`expert`** 评测。

| 项目 | 取值 |
| --- | --- |
| split / maps | `pufferinter` · `all`（有效 **589**） |
| traffic | **expert**（开环人类回放） |
| ego | MIXMOE/DriveLite=`policy_0`；Perception=`policy_2` |
| 结果根 | `.logs/train/run_repaet/{trainMOE,trainLite,trainPerception}/test_Expert/` |
| 启动 | `launch_all_eval_expert.sh`（原 train GPU 0–7 并行） |

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

H0 Expert 基线说明：`.logs/train/run4/problem/test_Expert/homogeneous_drive_lstm_500m_results.md`

---

## 2. 主结果：vs Expert（pufferinter）

| Run | Sampling | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward ↑ | ΔColl vs H0 | ΔFault vs H0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 run4_r2（baseline） | — | 71.31% | 16.81% | 15.28% | 0.68% | −0.817 | — | — |
| MIXMOE shared | shared | **72.67%** | **16.13%** | **14.43%** | 0.68% | **−0.789** | **-4.0%** | **-5.6%** |
| MIXMOE strat | stratified | **71.99%** | 16.81% | **14.77%** | 1.36% | −0.898 | +0.0% | **-3.3%** |
| DriveLite A shared | shared | **72.50%** | **15.79%** | **14.43%** | 0.85% | **−0.783** | **-6.1%** | **-5.6%** |
| DriveLite A strat | stratified | **73.85%** | **14.43%** | **13.41%** | 1.02% | **−0.701** | **-14.2%** | **-12.2%** |
| DriveLite B shared | shared | **72.67%** | **14.94%** | **14.26%** | **0.34%** | −0.885 | **-11.1%** | **-6.7%** |
| DriveLite B strat | stratified | **72.50%** | **14.09%** | **12.22%** | 1.87% | **−0.794** | **-16.2%** | **-20.0%** |
| Perception shared | shared | 71.14% | **16.13%** | **14.77%** | 0.85% | **−0.781** | **-4.0%** | **-3.3%** |
| Perception strat | stratified | **71.65%** | **15.62%** | **13.58%** | 0.68% | **−0.707** | **-7.1%** | **-11.1%** |

### 2.1 shared（含 baseline）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward ↑ | ΔColl vs H0 | ΔFault vs H0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 run4_r2（baseline） | 71.31% | 16.81% | 15.28% | 0.68% | −0.817 | — | — |
| MIXMOE shared | **72.67%** | **16.13%** | **14.43%** | 0.68% | **−0.789** | **-4.0%** | **-5.6%** |
| DriveLite A shared | **72.50%** | **15.79%** | **14.43%** | 0.85% | **−0.783** | **-6.1%** | **-5.6%** |
| DriveLite B shared | **72.67%** | **14.94%** | **14.26%** | **0.34%** | −0.885 | **-11.1%** | **-6.7%** |
| Perception shared | 71.14% | **16.13%** | **14.77%** | 0.85% | **−0.781** | **-4.0%** | **-3.3%** |

### 2.2 stratified（含 baseline）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward ↑ | ΔColl vs H0 | ΔFault vs H0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 run4_r2（baseline） | 71.31% | 16.81% | 15.28% | 0.68% | −0.817 | — | — |
| MIXMOE strat | **71.99%** | 16.81% | **14.77%** | 1.36% | −0.898 | +0.0% | **-3.3%** |
| DriveLite A strat | **73.85%** | **14.43%** | **13.41%** | 1.02% | **−0.701** | **-14.2%** | **-12.2%** |
| DriveLite B strat | **72.50%** | **14.09%** | **12.22%** | 1.87% | **−0.794** | **-16.2%** | **-20.0%** |
| Perception strat | **71.65%** | **15.62%** | **13.58%** | 0.68% | **−0.707** | **-7.1%** | **-11.1%** |

### 2.3 按系列速览

| 系列 | 最优 Coll | 最优 Fault | 相对 H0 Coll | 备注 |
| --- | ---: | ---: | ---: | --- |
| DriveLite B | **14.09%**（strat） | **12.22%**（strat） | **−16.2%** | Expert 上 Coll/Fault 最强 |
| DriveLite A | **14.43%**（strat） | **13.41%**（strat） | **−14.2%** | Goal 最高 **73.85%**；Reward 亦最优 |
| Perception | **15.62%**（strat） | **13.58%**（strat） | **−7.1%** | 中等改善；Goal 接近 H0 |
| MIXMOE | **16.13%**（shared） | **14.43%**（shared） | **−4.0%** | 小幅优于 H0；strat Coll 持平 |

---

## 3. 分系列对照

### 3.1 MIXMOE（结构混智）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 r2 | 71.31% | 16.81% | 15.28% | 0.68% | −0.817 |
| shared `ncn8qxko` | **72.67%** | **16.13%** | **14.43%** | 0.68% | **−0.789** |
| strat `o4trtoxc` | **71.99%** | 16.81% | **14.77%** | 1.36% | −0.898 |

解读：相对 vs-IDM「结构混智无效」，Expert 上 **shared 有小幅正收益**（Coll −4.0%、Fault −5.6%、Goal↑）。strat 几乎不降 Coll，Reward 更差。整体改善远小于 DriveLite。

### 3.2 DriveLite（智力层级混智）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 r2 | 71.31% | 16.81% | 15.28% | 0.68% | −0.817 |
| A shared | **72.50%** | **15.79%** | **14.43%** | 0.85% | **−0.783** |
| **A strat** | **73.85%** | **14.43%** | **13.41%** | 1.02% | **−0.701** |
| B shared | **72.67%** | **14.94%** | **14.26%** | **0.34%** | −0.885 |
| **B strat** | **72.50%** | **14.09%** | **12.22%** | 1.87% | **−0.794** |

解读：

1. **B strat** 在 Expert 上 Coll/Fault 最优（与 IDM 上「B 偏好 shared」相反）。
2. **A strat** Goal/Reward 最优，安全指标次优；与 IDM 上 A strat 全面领先形成呼应。
3. A/B 的 shared 也均优于 H0，说明智力混智对 Expert 有跨采样稳健收益。

### 3.3 Perception（感知混智）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 r2 | 71.31% | 16.81% | 15.28% | 0.68% | −0.817 |
| shared | 71.14% | **16.13%** | **14.77%** | 0.85% | **−0.781** |
| strat | **71.65%** | **15.62%** | **13.58%** | 0.68% | **−0.707** |

解读：Perception 在 Expert 上改善幅度小于 IDM（IDM Coll −53% vs Expert Coll −7%）。strat 仍优于 shared；Goal 基本持平 H0。

---

## 4. 碰撞构成（vs Expert 首碰）

来源：各 run `collision_snapshots.json`。Expert 协议下撞图基数显著高于 IDM（H0：99 vs 34）。

| Run | 碰撞 maps | Lateral | Front | Rear | Stopped | At-fault / 碰撞 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 r2 | 99 | 33 | 27 | 9 | 30 | 90/99（91%） |
| MIXMOE shared | 95 | 31 | 25 | 5 | 34 | 85/95（89%） |
| MIXMOE strat | 99 | 30 | 29 | 6 | 34 | 87/99（88%） |
| DriveLite A shared | 93 | 25 | 26 | 5 | 37 | 85/93（91%） |
| DriveLite A strat | **85** | **19** | 23 | 4 | 39 | 79/85（93%） |
| DriveLite B shared | 88 | 22 | 32 | 2 | 32 | 84/88（95%） |
| DriveLite B strat | **83** | **19** | 28 | 6 | 30 | 72/83（87%） |
| Perception shared | 95 | 29 | 28 | 5 | 33 | 87/95（92%） |
| Perception strat | 92 | **18** | 28 | 11 | 35 | 80/92（87%） |

要点：

1. Expert 失败模式含大量 **Stopped**（开环静止/缓行专家），与 IDM「Lateral 主导」不同。
2. DriveLite A/B strat 将撞图压到 **83–85**，Lateral 同步下降（33→19）。
3. At-fault 占比仍普遍 ≥87%——撞了大多判责，与 H0 Expert 文档一致。

---

## 5. shared vs stratified

| 系列 | Coll：shared → strat | Fault：shared → strat | 谁更好（Expert） |
| --- | ---: | ---: | --- |
| DriveLite A | 15.79 → **14.43** | 14.43 → **13.41** | **stratified** |
| DriveLite B | 14.94 → **14.09** | 14.26 → **12.22** | **stratified** |
| Perception | 16.13 → **15.62** | 14.77 → **13.58** | **stratified** |
| MIXMOE | **16.13** → 16.81 | **14.43** → 14.77 | **shared** |

结论：Expert 上 **DriveLite / Perception 均偏好 stratified**；MIXMOE 仍是 shared 略好。注意与 IDM 上「B 偏好 shared」不一致——协议切换会改变采样模式排序。

---

## 6. 与 vs-IDM 对照（同权重）

| Run | IDM Coll | Expert Coll | IDM Fault | Expert Fault |
| --- | ---: | ---: | ---: | ---: |
| H0 r2 | 5.77% | 16.81% | 4.58% | 15.28% |
| DriveLite A strat | **2.72%** | 14.43% | **1.70%** | 13.41% |
| DriveLite B strat | 5.09% | **14.09%** | 4.07% | **12.22%** |
| Perception strat | **2.72%** | 15.62% | 2.38% | 13.58% |
| MIXMOE shared | 6.11% | 16.13% | 4.92% | 14.43% |

解读：

1. 协议难度阶梯仍在：同权下 Expert Coll ≈ IDM ×2.5–5。
2. IDM 最优（A strat / Perc strat）在 Expert 上仍优于 H0，但 **相对降幅收窄**。
3. Expert 最优安全指标落到 **B strat**，与 IDM 排名不完全一致 → 需联合看两协议，不能只盯 IDM。

---

## 7. 总判与建议

| 优先级 | 结论 |
| --- | --- |
| 1 | Expert 上 **DriveLite B strat** Coll/Fault 最优（14.09% / 12.22%）；**A strat** Goal/Reward 最优。 |
| 2 | 智力混智（DriveLite）在 Expert 上全面优于 H0；感知混智中等；结构 MIXMOE 仅 shared 小幅正收益。 |
| 3 | 与 IDM 不同：Expert 上 DriveLite B **偏好 stratified**。 |
| 4 | Expert 撞图仍高（≥83），Stopped/Lateral/Front 并存——混智缓解有限，未接近「过关」。 |
| 5 | 选型建议：若主指标是 IDM → A strat；若同时看 Expert 安全 → 并列关注 B strat。 |

---

## 8. 产物路径

| Run | summary.csv 目录 |
| --- | --- |
| MIXMOE shared | `.logs/train/run_repaet/trainMOE/test_Expert/mix_moe_drive_moe_moe3_mb98k_1500m_run_repaet/Drive_Recurrent/20260723_023049_990915/` |
| MIXMOE strat | `…/trainMOE/test_Expert/mix_moe_drive_moe_moe3_stratified_1500m_run_repaet/Drive_Recurrent/20260723_023055_0a2799/` |
| DriveLite A shared | `…/trainLite/test_Expert/drivelite_mixA_mb98k_shared_1500m_run_repaet/Drive_Recurrent/20260723_023101_a9706d/` |
| DriveLite A strat | `…/trainLite/test_Expert/drivelite_mixA_stratified_1500m_run_repaet/Drive_Recurrent/20260723_023106_dedef7/` |
| DriveLite B shared | `…/trainLite/test_Expert/drivelite_mixB_mb98k_shared_1500m_run_repaet/Drive_Recurrent/20260723_023111_1f90ef/` |
| DriveLite B strat | `…/trainLite/test_Expert/drivelite_mixB_stratified_1500m_run_repaet/Drive_Recurrent/20260723_023116_7e30d0/` |
| Perception shared | `…/trainPerception/test_Expert/perception_mix_low_mid_high_mb98k_shared_1500m_run_repaet/Drive_Recurrent/20260723_023122_5b6b9c/` |
| Perception strat | `…/trainPerception/test_Expert/perception_mix_low_mid_high_stratified_1500m_run_repaet/Drive_Recurrent/20260723_023127_a6ec96/` |

H0 Expert 基线：`.logs/train/run4/problem/test_Expert/homogeneous_drive_lstm_500m_results.md`  
同权重 IDM 汇总：`.logs/train/run_repaet/result/run_repaet_vs_idm_results.md`
