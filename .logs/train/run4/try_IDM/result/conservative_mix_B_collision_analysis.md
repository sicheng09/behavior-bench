# Run4 try_IDM · ConservativeMix B 碰撞分析（vs IDM）

> 策略：ego `Drive` + `Recurrent(256,256)`；partner = reward shaping（Phase B）  
> 评测：`pufferinter` / `map-ids=all`（有效 **589** maps）；traffic = IDM（`v0=15`, `T=1.5`, `min_gap=1.0`, `a=1.0`, `b=3.0`）  
> 数据：各 run 的 `collision_snapshots.json`（首碰 + `collision_classifier`）  
> 更新时间：2026-07-19  
> 原始评测：`.logs/train/run4/try_IDM/test/cons_mix_B_*`  
> H0 对照：`.logs/train/run4/problem/homogeneous_drive_lstm_500m_results.md` §4

---

## 1. 覆盖的 Mix B 跑次

| 标签 | 采样 | W&B | Checkpoint（policy 0） | vs IDM 目录 |
| --- | --- | --- | --- | --- |
| **C-B mb65k** | `shared`，mb=65536 | [`plvd8yph`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/plvd8yph) | `…_plvd8yph/…_000954.pt` | `…/20260718_211540_45275f/` |
| **C-B mb65k r2** | `shared`，mb=65536 | [`5dljtpvh`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/5dljtpvh) | `…_5dljtpvh/…_000954.pt` | `…/20260719_062816_34e597/` |
| **C-B strat** | `stratified`，policy_mb=32768×2，update_steps=16×2 | [`u2qysy6b`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/u2qysy6b) | `…_u2qysy6b/…_000954.pt` | `…/20260719_025234_8b12d8/` |
| **C-B strat r2** | 同上 | [`0t9jeeso`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/0t9jeeso) | `…_0t9jeeso/…_000954.pt` | `…/20260719_062716_1e1cdc/` |

门槛（相对 H0 run4_r2）：vs IDM **Collision ≤ 4.90%**、**At-fault ≤ 3.89%**（各相对 −15%）。

---

## 2. 总览：碰撞率与构成

### 2.1 宏观指标（vs IDM）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad | 门槛 |
| --- | ---: | ---: | ---: | ---: | --- |
| H0 run4_r2 | 81.66% | 5.77% | 4.58% | 0.85% | 基线 |
| **C-B mb65k** | **83.02%** | **2.55%** | **1.87%** | 1.02% | **PASS** |
| C-B mb65k r2 | 82.68% | 4.41% | 3.23% | 1.53% | **PASS** |
| C-B strat | 81.83% | 4.92% | 3.74% | 1.36% | Coll 卡边 FAIL；Fault PASS |
| C-B strat r2 | 81.32% | 5.43% | 4.75% | 1.87% | **FAIL** |

要点：

1. **只有 shared mb65k 的首跑把撞图压到 15（2.55%）**；同配方 r2 回升到 26（4.41%），仍过门槛但明显更噪。  
2. **stratified 两路未再现 mb65k 的大幅降碰**；strat r2 的 Fault（4.75%）甚至略差于 H0。  
3. Goal 四路均 ≥ H0；Offroad 相对 H0 略升（+0.17～+1.02 pp）。

### 2.2 首碰分型计数

| Run | 碰撞 maps | Lateral | Front | Rear | Stopped | At-fault / 碰撞 | fault Lateral | fault Front |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 run4_r2 | 34 | **25** | 7 | 1 | 1 | 27/34（79%） | 20 | 7 |
| **C-B mb65k** | **15** | **9** | **3** | 1 | 2 | 11/15（73%） | 7 | 3 |
| C-B mb65k r2 | 26 | **19** | **1** | 3 | 3 | 19/26（73%） | 16 | 1 |
| C-B strat | 29 | 16 | **9** | 2 | 2 | 22/29（76%） | 12 | 9 |
| C-B strat r2 | 32 | 19 | **9** | 2 | 2 | 28/32（88%） | 18 | 9 |

分型含义（与 H0 §4 一致）：

| 类型 | 含义 | 责任规则 | 行为解读 |
| --- | --- | --- | --- |
| **ACTIVE_LATERAL** | 非正前/正后的侧方接触 | 仅当 ego **跨多车道** 才 at-fault | 超车/并线横向净空不够；IDM 钉车道不躲 |
| **ACTIVE_FRONT** | 他车在前方且 ego 在靠近 | **一律 at-fault** | 跟车过近 / 对 IDM 减速反应晚 |
| **ACTIVE_REAR** | 他车在后方 | 通常非责 | 后方贴上；不是主矛盾 |
| **STOPPED** | 静止或近零速接触 | 视几何 | 含起步邻车重叠等边角 |

---

## 3. 碰撞是怎么引发的？

### 3.1 共同结构：仍以 Lateral 为主

四路 Mix B **无一例外以 Lateral 为最大桶**（9–19），与 H0（25）同型，只是绝对次数不同：

```text
H0          █████████████████████████ 25 Lateral / 7 Front
C-B mb65k   █████████ 9 / ███ 3          ← 两侧同步压低（最佳）
C-B mb65k r2███████████████████ 19 / █ 1 ← Front 几乎消掉，Lateral 反弹
C-B strat   ████████████████ 16 / █████████ 9  ← Front 回到甚至超过 H0
C-B strat r2███████████████████ 19 / █████████ 9 ← Lateral+Front 双高
```

### 3.2 Lateral 几何（vs IDM）

| Run | n | 平均 \|lat\| | 平均 \|lon\| | ego 速 | other 速 | 侧向轨迹跨度 | fault / Lateral |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 r2 | 25 | 2.73 m | 2.22 m | 13.7 | 9.0 | 0.61 m | 20/25（80%） |
| C-B mb65k | 9 | 2.86 m | 2.10 m | 14.0 | 8.9 | 0.47 m | 7/9（78%） |
| C-B mb65k r2 | 19 | 2.79 m | 1.72 m | **15.7** | 9.9 | 0.59 m | 16/19（84%） |
| C-B strat | 16 | 3.02 m | 2.48 m | 13.9 | 9.1 | 0.47 m | 12/16（75%） |
| C-B strat r2 | 19 | 2.77 m | 2.87 m | **15.7** | 9.6 | 0.53 m | **18/19（95%）** |

解读：

- 接触瞬间的横向间距稳定在 **~2.7–3.0 m**（约一车宽量级贴边），说明失败模式仍是「贴着 IDM 侧向挤过去」，不是分类噪声。  
- **mb65k r2 / strat r2** 的 ego 速度更高（~15.7 vs H0 13.7）→ 侧向责碰占比升高，是 Fault 抬升的主因。  
- mb65k 首跑把 Lateral 从 25→9，几何尺度几乎不变 → **少撞同一类场景**，而非改成另一类事故。

### 3.3 Front 几何（次因，但几乎全责）

| Run | n | 平均 \|lat\| | 平均 \|lon\| | ego 速 | other 速 | Δv (ego−oth) | fault |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 r2 | 7 | 2.00 m | 3.73 m | 17.3 | 8.8 | +8.4 | 7/7 |
| C-B mb65k | 3 | 2.26 m | 3.06 m | **7.0** | 4.4 | +2.6 | 3/3 |
| C-B mb65k r2 | 1 | 1.99 m | 3.76 m | 8.3 | 7.0 | +1.3 | 1/1 |
| C-B strat | 9 | 1.87 m | 4.06 m | 17.4 | 11.4 | +6.0 | 9/9 |
| C-B strat r2 | 9 | 1.38 m | 4.46 m | **21.5** | 12.3 | **+9.2** | 9/9 |

解读：

- **shared mb65k**：Front 压到 1–3，且剩余 Front 多为低速/近停场景（ego≈7 m/s），不是高速追尾。  
- **stratified**：Front 回到 9（≥ H0），且 strat r2 的 Δv≈+9 m/s、ego≈21.5 → **高速跟车不足**重新成为 Fault 大户（9 次全责）。  
- 这解释了为何 stratified 的 At-fault 明显高于 mb65k 首跑：Lateral 未压够 + Front 全责回流。

### 3.4 稳定难图（≥3/4 路 Mix B 均撞）

| map_id | 出现次数 | 主要类型 | 是否在 H0 撞 |
| ---: | ---: | --- | --- |
| 115, 380, 438, 625, 707 | 4/4 | Lateral（707 偶发 Rear） | 是 |
| 315 | 4/4 | Lateral / Front / Rear 边界 | 是 |
| 358 | 4/4 | Stopped（step=0 邻车重叠） | 是 |
| 618 | 3/4 | Lateral / Rear | 是 |
| 351, 816 | 3/4 | Lateral / Front | 是 |
| 408, 504 | 3/4 | Lateral / Front | 否（Mix B 新坑） |
| 142 | 3/4 | Stopped | 否 |

**最稳的 Lateral 难图**：`115`, `380`, `438`, `625`（四路全中）。  
`358` 为 step=0 静止重叠，各策略几乎都会记一笔，对 Coll 率贡献固定约 1/589≈0.17 pp，不宜当作 shaping 成败信号。

### 3.5 碰撞地图重叠（Jaccard）

| | H0 | mb65k | mb65k r2 | strat | strat r2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **H0** | — | 0.256 | 0.333 | 0.286 | 0.294 |
| **mb65k** | | — | 0.323 | 0.294 | 0.205 |
| **mb65k r2** | | | — | 0.375 | 0.289 |
| **strat** | | | | — | 0.326 |

两两 Jaccard 约 **0.20–0.38**（中低重叠）：有一批稳定难图，但多数撞图随种子/采样变化——与 H0 三路种子间重叠量级相近。  
**mb65k 首跑**撞图最少，与其它路重叠也偏低（尤其 vs strat r2 仅 0.205），说明它真正「清掉」了一批 H0 会撞的图，而不是简单重排类型。

---

## 4. vs PPO（同质自碰，对照）

| Run | Goal | Collision | At-fault | 碰撞 maps | 构成 |
| --- | ---: | ---: | ---: | ---: | --- |
| C-B mb65k | 89.30% | 0.68% | 0.34% | 4 | Lateral×4 |
| C-B mb65k r2 | 89.64% | 0.34% | 0.34% | 2 | Lat×1, Stopped×1 |
| C-B strat | 89.98% | 0.34% | 0.17% | 2 | Front×1, Rear×1 |
| C-B strat r2 | 89.47% | 0.51% | 0.17% | 3 | Lat/Rear/Stopped |

PPO 自碰均很低（≤0.68%），**Mix B 的 IDM 碰撞问题不是「策略本身不稳定」**，而是对不躲让的 IDM 交通的泛化。

---

## 5. 综合解读

```text
Mix B（reward shaping partner）
  ├─ shared + mb65k（plvd8yph）
  │    → Lateral 25→9、Front 7→3；Coll 2.55% / Fault 1.87%  【当前最强】
  ├─ shared + mb65k r2（5dljtpvh）
  │    → Front≈消掉，但 Lateral 反弹到 19；仍过门槛、方差大
  ├─ stratified r1（u2qysy6b）
  │    → Lateral 中等，Front 回到 9；Coll 卡在 4.92% 门槛边
  └─ stratified r2（0t9jeeso）
       → Lateral 责碰 18/19 + Front×9 高速；Fault 4.75% 差于 H0
```

1. **成功模式（mb65k）**：在保持 Lateral 几何形态不变的前提下，**同时减少侧向贴超与前方追尾次数**；剩余 Front 偏慢速，Fault 密度下降。  
2. **失败回流（strat / 部分 seed）**：Lateral 压不净时，若 Front 以高 Δv 回流，At-fault 会快速恶化——strat r2 即此路径。  
3. **种子方差**：同配方 mb65k 两 seed 的 Coll 相差 **1.86 pp**（2.55% vs 4.41%）；写论文/定主结果需至少 2–3 seed，不宜只报最佳一跑。  
4. **相对 H0**：行为含义不变——IDM 保距但不横向配合；Mix B 的收益来自「更少去挤 IDM」，而非改变 IDM。

---

## 6. 文件索引

| 类型 | 路径 |
| --- | --- |
| 本分析 | `.logs/train/run4/try_IDM/result/conservative_mix_B_collision_analysis.md` |
| A/B mb65k 总汇总 | `.logs/train/run4/try_IDM/result/conservative_mix_A_B_mb65k_results.md` |
| 指标 CSV | `.logs/train/run4/try_IDM/result/metrics_summary.csv` |
| 原始 collision 快照 | `.logs/train/run4/try_IDM/test/cons_mix_B_*/Drive_Recurrent/*/collision_snapshots.json` |
| H0 碰撞分析 | `.logs/train/run4/problem/homogeneous_drive_lstm_500m_results.md` §4 |
| Stratified 训练说明 | `.logs/train/run4/try_IDM/train/stratified_mix_training.md` |

---

## 7. 一句话

**Mix B 在 shared·mb65k 首跑上把 IDM 碰撞从 H0 的 34 图压到 15（Coll 2.55% / Fault 1.87%），主因仍是 ACTIVE_LATERAL，但次数与 Front 同步下降；stratified 与 mb65k r2 显示种子方差大，Front 高速回流会迅速抬高 At-fault——主结果应报 mb65k，并附多 seed 区间而非单点。**
