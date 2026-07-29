# Run4 · 同质 Drive+LSTM 500M · vs SMART 结果汇总

> 策略：`Drive` + `Recurrent(256,256)`（标准同质）  
> 交通：SMART 轨迹预测控制器（`traffic.type=smart`，权重 `SMART_epoch_030.pt`）  
> 更新时间：2026-07-22  
> 对照文档：`.logs/train/run4/problem/test_IDM/homogeneous_drive_lstm_500m_results.md`  
> 原始产物：`.logs/train/run4/problem/test_Smart/`  
> 启动记录：`launch_h0_vs_smart_20260721.md`

---

## 1. 实验配置与评测协议

三路 ego 权重与 vs IDM / vs Expert / vs PPO 完全相同（标准同质 500M，最终 ckpt `000954`）。三路评测均已 **Evaluation complete**（有效 589 maps）。

| 项目 | 取值 |
| --- | --- |
| ego 策略 | `Drive` + `Recurrent(256,256)` · PPO |
| split | `pufferinter` |
| maps | `all`（manifest 1000；有效 **589**） |
| traffic | **smart**（其余激活车辆由 SMART 控） |
| SMART 权重 | `weights/SMART_epoch_030.pt`（1M 档，epoch=30） |
| 结果根 | `.logs/train/run4/problem/test_Smart/<run>/Drive_Recurrent/` |

### SMART 交通是什么？

**SMART** 在本仓库中是 **监督预训练的运动 token 预测模型**（非 RL），评测时封装为 planner/traffic：预测周边智能体未来轨迹 → 比例控制转成 `(accel, steer)`。  
与 IDM（规则跟驰）/ Expert（开环回放）/ PPO（学习策略）都不同：

| | SMART 交通 | IDM 交通 | Expert 交通 | PPO 交通（同权） |
| --- | --- | --- | --- | --- |
| 决策 | 预测模型 + P 控制 | 公式跟驰 + 车道跟随 | WOMD 真值开环 | 学习策略 |
| 是否反应场景 | 是（基于观测重预测） | 纵向对前车反应；横向钝 | **否** | 是 |
| 是否需权重 | 是（预测 ckpt） | 否 | 否 | 是（策略 ckpt） |
| 本实验用途 | 学习型他车 held-out | 规则交通主协议 | 人类开环 | 训练分布内 |

训练入口为 `pufferlib.prediction.puffer_prediction pretrain`，不是 `puffer train`。本机权重约 **1.23M** 参数、`hidden_dim=64`；ckpt 含 `epoch=30`（config `max_epochs=64`，**未训满**），属仓库默认 baseline，非多 seed 最优保证。

### 本仓库 vs SMART 部署

三路 `config.json` 中 `[traffic.smart]` **一致**：

| 参数 | 取值 | 含义 |
| --- | --- | --- |
| `weights_path` | `weights/SMART_epoch_030.pt` | 预测权重 |
| `device` | `cuda` | |
| `temperature` | `1.0` | 采样温度 |
| `greedy` | `True` | 贪心解码（确定性更强） |
| `repredict_interval` | `5` | 每 5 step 重跑预测 |

角色：**1× PPO ego** + **其余 active agents → SMART**。  
控制路径：`SMARTPlanner` / traffic SMART → 推理轨迹 → P-controller → `move_dynamics`（与 IDM 的 `move_idm`、Expert 的 `move_expert` 不同）。

> 说明：SMART planner 源码里留有逐步 `log.info` 调试输出，wrapper 日志可能很长，但不影响指标。

### 权重一览

| 标签 | W&B | Ego Checkpoint | vs SMART 目录 |
| --- | --- | --- | --- |
| **run1 对照** | `fdfw3v5e` | `experiments/puffer_drive_fdfw3v5e/…_000954.pt` | `…/20260721_203634_1f9040/` |
| **run4_r1** | `1ytcyfuk` | `experiments/puffer_drive_1ytcyfuk/…_000954.pt` | `…/20260721_203634_1aa05f/` |
| **run4_r2** | `u9ymfcqr` | `experiments/puffer_drive_u9ymfcqr/…_000954.pt` | `…/20260721_203634_94c143/` |

评测日志：`eval_*_Drive_vs_smart_gpu{0,5,6}_20260721_203632.log`。

---

## 2. 分权重结果表

训练末期与 vs IDM / vs PPO 沿用 IDM 汇总；本表新增 **vs SMART**。  
Goal / Coll / At-fault / Offroad 为 **589** 有效 map 跨图均值（`summary.csv`）。

### 2.1 run1 对照 · `fdfw3v5e`

| 阶段 | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| **训练（末期）** | 95.60% | 0.92% | — | 0.80% | 0.935（return） |
| **评测 vs IDM** | 81.32% | 5.94% | 3.23% | 1.36% | −0.724 |
| **评测 vs PPO** | 89.98% | 0.00% | 0.00% | 0.51% | −0.658 |
| **评测 vs SMART** | **80.65%** | **6.79%** | **4.92%** | 1.02% | −0.836 |

### 2.2 run4_r1 · `1ytcyfuk`

| 阶段 | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| **训练（末期）** | 96.48% | 0.82% | — | 0.80% | 0.946 |
| **评测 vs IDM** | 81.32% | 5.43% | 4.24% | 1.19% | −0.690 |
| **评测 vs PPO** | 89.47% | 0.51% | 0.17% | 0.51% | −0.640 |
| **评测 vs SMART** | **79.63%** | **7.47%** | **5.94%** | 0.85% | −0.828 |

### 2.3 run4_r2 · `u9ymfcqr`

| 阶段 | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| **训练（末期）** | 96.19% | 0.95% | — | 0.80% | 0.941 |
| **评测 vs IDM** | 81.66% | 5.77% | 4.58% | 0.85% | −0.676 |
| **评测 vs PPO** | 89.64% | 0.68% | 0.68% | 0.34% | −0.649 |
| **评测 vs SMART** | **80.48%** | **6.79%** | **4.92%** | 0.85% | −0.741 |

---

## 3. 横向对照

### 3.1 评测 vs SMART（pufferinter）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward | 碰撞 maps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| run1 对照 | **80.65%** | **6.79%** | **4.92%** | 1.02% | −0.836 | 40 |
| run4_r1 | 79.63% | 7.47% | 5.94% | **0.85%** | −0.828 | 44 |
| run4_r2 | 80.48% | **6.79%** | **4.92%** | **0.85%** | **−0.741** | 40 |

三路 Goal≈**80%**，Coll≈**6.8–7.5%**，Fault≈**4.9–5.9%**。run4_r1 略差（Coll/Fault 最高）；run1 与 run4_r2 几乎同 Coll/Fault。

### 3.2 同一权重：SMART vs IDM vs Expert vs PPO

以 run4_r2 为例：

| Traffic | Goal ↑ | Collision ↓ | At-fault ↓ |
| --- | ---: | ---: | ---: |
| PPO（同权） | 89.64% | 0.68% | 0.68% |
| IDM | 81.66% | 5.77% | 4.58% |
| **SMART** | **80.48%** | **6.79%** | **4.92%** |
| Expert | 71.31% | 16.81% | 15.28% |

**难度阶梯（同质 ego）**：同权 PPO ≪ IDM ≲ **SMART** ≪ Expert。  
SMART 相对 IDM：Goal 略低（约 −1pp），Coll/Fault 略高（约 +1pp / +0.3–1.7pp）——比规则 IDM 稍难，远好于 Expert 开环人类流。

三路相对各自 vs IDM：

| Run | ΔGoal (SMART−IDM) | ΔColl | ΔFault |
| --- | ---: | ---: | ---: |
| run1 | −0.67 pp | +0.85 pp | **+1.69 pp** |
| run4_r1 | −1.69 pp | +2.04 pp | +1.70 pp |
| run4_r2 | −1.18 pp | +1.02 pp | +0.34 pp |

run1 在 IDM 上 Fault 本就最低（3.23%），换 SMART 后 Fault 抬升最明显；run4_r2 的 Fault 几乎与 IDM 持平。

---

## 4. SMART 测试碰撞分析：碰撞是怎么引发的？

数据：各 run `collision_snapshots.json`（首碰 + `collision_classifier`）。  
设定：**1× PPO ego + 全员 SMART 交通**（`greedy=True`，`repredict_interval=5`）。

### 4.1 类型总览

| Run | 碰撞 maps | Lateral | Front | Rear | Stopped | At-fault / 碰撞 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| run1 | 40 | **19（48%）** | 11（28%） | 9（22%） | 1 | 29/40（**72%**） |
| run4_r1 | 44 | **32（73%）** | 6（14%） | 4（9%） | 2 | 35/44（**80%**） |
| run4_r2 | 40 | **23（58%）** | 9（22%） | 7（18%） | 1 | 29/40（**72%**） |

与 vs IDM / vs Expert 对照：

| 特征 | vs IDM | vs SMART | vs Expert |
| --- | --- | --- | --- |
| 主桶 | Lateral | **Lateral** | Stopped + Front |
| Rear 占比 | 很低（1–3） | **明显更高（4–9）** | 低 |
| Stopped | 少 | 极少（1–2） | **最大桶** |
| At-fault / 碰撞 | ~54–79% | ~72–80% | ~91% |
| 碰撞 maps | ~32–35 | ~40–44 | ~85–99 |

**SMART 与 IDM 同属「侧向主导」**，但 SMART 交通会带来更多 **Rear**（他车从后方贴上）——预测/控制他车可能比 IDM 更「贴」或速度分布更激进，而不只是钉车道跟驰。

### 4.2 各类型机制

| 类型 | 责任规则（简） | 在 SMART 设定下的成因 |
| --- | --- | --- |
| **ACTIVE_LATERAL** | 跨多车道才责 | ego 并线/超车时与 SMART 他车横向摩擦；他车会重预测但仍可能占道 |
| **ACTIVE_FRONT** | **一律 at-fault** | 对前方 SMART 减速/变轨反应不足 |
| **ACTIVE_REAR** | 通常非责 | SMART 他车从后方逼近（相对 IDM 更常见） |
| **STOPPED_*** | track 静止常责 | 极少；不是主矛盾 |

#### A. 侧向擦碰（主因）

| 特征 | run1 | run4_r1 | run4_r2 |
| --- | ---: | ---: | ---: |
| 次数 | 19 | **32** | 23 |
| at-fault | 17/19（89%） | 27/32（84%） | 19/23（83%） |
| 中位 ego / SMART 速度 | 15.0 / 10.4 | **18.6 / 13.1** | 15.8 / 9.7 |
| ego 明显更快 | 15/19 | 24/32 | 16/23 |
| 他车多在侧后（fwd 中位） | −2.2 m | −2.5 m | −2.3 m |
| 同向（\|Δheading\|&lt;30°） | 12/19 | 14/32 | 15/23 |

**因果链**与 IDM 类似：同质 ego 偏快、横向贴 → 侧向摩擦。  
run4_r1 的 Lateral **32** 次、ego 中位速度更高（18.6），对应其最高 Coll/Fault。侧向责碰比例（约 84–89%）高于 IDM run1（约 28% 的 Lateral 责碰），接近 IDM run4 的激进侧向模式。

#### B. 前方碰撞（次因，全责）

| 特征 | run1 | run4_r1 | run4_r2 |
| --- | ---: | ---: | ---: |
| 次数 | 11 | 6 | 9 |
| at-fault | 11/11 | 6/6 | 9/9 |
| 中位 ego / other | 17.9 / 3.4 | 16.0 / 5.8 | 15.5 / 3.9 |
| 同向追尾为主 | 10/11 | 5/6 | 7/9 |

Front 次数与 IDM 同量级（IDM 约 7–12）；SMART 前车也可突然按预测减速，ego 闭合不足则全责。

#### C. 后方碰撞（SMART 相对 IDM 的增量）

| 特征 | run1 | run4_r1 | run4_r2 |
| --- | ---: | ---: | ---: |
| 次数 | **9** | 4 | **7** |
| at-fault | 0/9 | 0/4 | 0/7 |
| 中位 ego / other | 14.3 / **17.6** | 19.0 / 20.7 | 10.7 / 11.2 |

Rear **全部非责**，但抬高总 Collision。说明 SMART 交通并非纯「保守钉车道」：部分他车会从后方逼近，把 Coll 从「纯 ego 责」里掺入不可责事件。  
这与 Expert（开环）/ IDM（低 rear）都不同——评读 SMART 的 Coll 时需同时看 At-fault。

#### D. 静止（可忽略）

每 run 仅 1–2 起，对总率影响小（与 Expert 形成鲜明对比）。

### 4.3 和评测部署的关系

```text
训练：周围多是会博弈的同质 PPO
   ↓
评测：周围是 SMART 预测交通（会动、会重规划，但行为分布 ≠ 训练同伴）
   ↓
侧向贴超 → Lateral（主责碰）
跟车不足 → Front（全责）
SMART 后方逼近 → Rear（抬 Coll、不抬 Fault）
```

因此 Coll≈7% 主要是 **分布偏移下的侧向/跟车摩擦 + 少量后方非责**，不是 Expert 式「撞静止障碍」。

### 4.4 优化含义（针对 SMART 测试）

| 优先级 | 类型 | 理由 |
| --- | --- | --- |
| **高** | Lateral（尤其 at-fault） | 主桶；run4_r1 上占 73% 碰撞 |
| **高** | Front | 全责；直接打 At-fault |
| 中 | Rear | 抬 Coll、不抬 Fault；反映交通攻击性/贴行，需结合 Fault 读表 |
| 低 | Stopped | 样本极少 |

若只优化 IDM 侧向，SMART 上仍可能因 **更快 ego（run4_r1）** 或 **Rear 增量** 使 Coll 略差于 IDM。

### 4.5 碰撞地图 ID 分布相似度（vs SMART）

#### 集合规模与两两重叠

| 对比 | \|A\| | \|B\| | \|∩\| | Jaccard | Overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| run1 vs run4_r1 | 40 | 44 | **21** | **0.333** | 0.525 |
| run1 vs run4_r2 | 40 | 40 | **23** | **0.404** | 0.575 |
| run4_r1 vs run4_r2 | 44 | 40 | **22** | **0.355** | 0.550 |

并集 **73** 张：

| 出现次数 | 地图数 | 占并集 |
| --- | ---: | ---: |
| 三路都撞 | **15** | 21% |
| 恰两路撞 | 21 | 29% |
| 仅一路撞 | 37 | **51%** |
| ≥2 路共享 | 36 | 49% |

与 vs IDM（J≈0.30–0.32）接近，远低于 vs Expert（J≈0.53–0.64）。**失败场景中等可复现 + 大量种子长尾**。

#### 三路共撞的 15 张图

`82, 223, 259, 430, 557, 618, 712, 718, 730, 734, 818, 848, 866, 876, 999`

#### 按类型重叠

| 类型 | run1 vs r1 | run1 vs r2 | r1 vs r2 |
| --- | ---: | ---: | ---: |
| Lateral Jaccard | 0.214 | 0.355 | 0.222 |
| Front Jaccard | 0.062 | 0.250 | 0.250 |
| At-fault map Jaccard | 0.280 | 0.289 | 0.306 |

共享 ≥2 路的 36 张中，类型桶一致 **23/36（64%）**。

#### 解读

1. Jaccard≈0.33–0.40：与 IDM 类似，有一批稳定难图（15 张全种子），但半数以上碰撞图随种子变。  
2. run4_r1 的 Lateral 膨胀是其 Coll/Fault 偏高的主因，不一定换了一套全新难图。  
3. 优化需同时覆盖共有 15 图 + 种子特异侧向责碰。

<details>
<summary>两两交集 map_id</summary>

- run1 ∩ r1：`73, 82, 142, 187, 223, 259, 327, 430, 557, 561, 618, 712, 718, 730, 734, 772, 818, 848, 866, 876, 999`
- run1 ∩ r2：`7, 82, 201, 212, 223, 259, 275, 430, 557, 618, 712, 718, 730, 734, 790, 818, 848, 866, 876, 898, 918, 990, 999`
- r1 ∩ r2：`82, 196, 223, 259, 315, 359, 430, 557, 611, 618, 620, 707, 712, 718, 730, 734, 777, 818, 848, 866, 876, 999`

</details>

> 快照目录：  
> `test_Smart/.../run1/.../20260721_203634_1f9040/collision_snapshots.json`  
> `test_Smart/.../run4_r1/.../20260721_203634_1aa05f/collision_snapshots.json`  
> `test_Smart/.../run4_r2/.../20260721_203634_94c143/collision_snapshots.json`

---

## 5. 指标口径（简要）

与 IDM / Expert 文档相同。碰撞 map 数 = `collision_snapshots.json` 条数（40 / 44 / 40），与 `collision_rate × 589` 一致。

> SMART 权重为单次预训练 ckpt（epoch 30）；换更大/更晚 SMART 或关闭 greedy 可能改变本表，当前结论绑定 `SMART_epoch_030.pt` + 上表解码设置。

---

## 6. 结论

1. **vs SMART 已完整跑完**：三路同质 Drive+LSTM 在 pufferinter 上 Goal≈**80%**，Coll≈**7%**，Fault≈**5–6%**。  
2. **难度介于 IDM 与 Expert 之间，且更靠近 IDM**：相对 IDM 略难（Goal −1pp 级，Coll/Fault 小幅升高），远好于 Expert（Coll≈15%+）。  
3. **失败结构像 IDM（Lateral 主导），但 Rear 更多**：读 Coll 时需结合 At-fault；部分增量碰撞是后方非责。  
4. **种子间**：run4_r1 因 Lateral 偏多略差；run1 / run4_r2 几乎同 Coll/Fault。碰撞图 Jaccard≈0.33–0.40，与 IDM 类似。  
5. **含义**：同质策略对「学习型预测交通」没有额外崩盘，但也未优于规则 IDM；若宣称对多样交通鲁棒，SMART 可作为 IDM 与 Expert 之间的中间协议。

---

## 7. 状态

- [x] run1 / run4_r1 / run4_r2 · vs SMART 完成（`summary.csv` + `collision_snapshots.json`）  
- [x] 与 vs IDM / vs PPO / vs Expert 对照写入本文  
- [ ] （可选）换 `greedy=False` 或更晚 SMART ckpt 做敏感性  
- [ ] （可选）SMART ego vs 同质 PPO traffic 的交叉表（见 `.logs/train/run4/Smart/test/`）
