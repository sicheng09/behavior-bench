# Run4 · 同质 Drive+LSTM 500M 结果汇总

> 策略：`Drive` + `Recurrent(256,256)`（标准同质）  
> 更新时间：2026-07-18  
> 说明：训练 / vs IDM / vs PPO 指标均已补全（`summary.csv`）。

---

## 1. 实验配置与步数

三路训练共用同一套「标准同质 500M」配方（对齐 run1 对照 `fdfw3v5e`）。

| 项目 | 取值 |
| --- | --- |
| 环境 / 入口 | `puffer_drive` · `pufferlib/config/ocean/drive.ini` |
| 策略 | `Drive` + `Recurrent`（`rnn_input=256`, `rnn_hidden=256`） |
| 参数量 | 614.2K |
| `batch_size` | 524,288（batch1x） |
| `minibatch_size` | 32,768（ini 默认） |
| `bptt_horizon` | 32（ini 默认） |
| `total_timesteps` | **500,000,000** |
| 预计 updates | \(500\text{M} / 524288 \approx\) **954**（最终 ckpt `model_puffer_drive_000954.pt`） |
| `env.num_maps` | 10,000 |
| 交通混合 | `mix_traffic=False` · `ppo=1.0` · `idm=0.0` · `expert=0.0` |
| 训练内 eval | `split=validation` · `num_maps=20` · WOSAC/human-replay 关闭 |
| GPU | 单卡（run1 历史；run4_r1→GPU6，run4_r2→GPU7） |
| 数据根 | `resources/drive/binaries` |

### 评测协议（验证）

| 项目 | vs IDM | vs PPO |
| --- | --- | --- |
| split | `pufferinter` | `pufferinter` |
| maps | `all`（manifest 1000；有效图约 589） | 同左 |
| ego | 该 run 的 `…000954.pt` | 同左 |
| traffic | IDM（见下节） | **同一权重** PPO |
| 结果根目录 | `.logs/train/run4/problem/test/<run>/Drive_Recurrent/` | 同左 |

### IDM 模型是什么？

**IDM（Intelligent Driver Model，智能驾驶员模型）** 是交通流 / 自动驾驶仿真里常用的**解析跟驰模型**（Treiber et al.）：用一组固定公式根据「期望速度、与前车间距、相对速度」算出纵向加速度，**不学习、无神经网络**。

直觉上，它模拟一个「守规矩」的驾驶员：

1. **前方空旷**：朝期望速度 \(v_0\) 加速，但加速度有上限 \(a\)。  
2. **前方有车**：维持期望车距（由静止最小间距 \(s_0\) + 时距 \(T\cdot v\) 等组成）；间距偏小或逼近前车时按舒适减速度 \(b\) 刹车。  
3. **只做纵向控制**：标准 IDM 本身不管变道；本仓库评测里横向由**沿车道中心线 / 路径跟随**完成，因此他车表现为「钉在车道上跟车」，几乎不会主动让道或超车博弈。

与 PPO 交通的对比：

| | IDM 交通 | PPO 交通（同权重自博弈） |
| --- | --- | --- |
| 决策 | 公式跟驰 + 车道路径 | 学习策略网络 |
| 横向 | 车道跟随，偏「钝」 | 会变道 / 贴挤 / 博弈 |
| 可调「性格」 | \(v_0, T, a, b, s_0\) 等 | 权重与训练分布 |
| 本实验用途 | **主评测协议**（泛化到规则交通） | 对照（分布更接近训练） |

### 本仓库 vs IDM 测试中的 IDM 配置

部署方式（`pufferlib/ocean/benchmark`）：

| 项 | 取值 |
| --- | --- |
| `traffic.type` | `idm` |
| 角色划分 | **1 个 ego（PPO）** + **其余激活车辆全部 IDM** |
| 控制路径 | `IDMPlanner` 设置后由 C 侧 `move_idm` 积分；ego 走 `move_dynamics` |
| 参数来源 | `pufferlib/config/evaluation.ini` → `[traffic.idm]`（评测 `config.json` 已落盘确认） |

#### 每个场景 / 每路评测的参数是否一致？

**一致。** 标准 vs IDM 评测里，IDM 参数是**全局一套**，不会按 map / 场景单独改：

1. **代码路径**：`IDMPlanner` 在 init/reset 时调用 `vec_set_idm_target_velocity`（同一 `target_velocity` 赋给所有 traffic agent）和 `vec_set_idm_params`（`min_gap / headway_time / accel_max / decel_max` 写到**全部 sub-env**），无 per-map 覆盖。  
2. **落盘核对**：下列三路 vs IDM 的 `config.json` 中 `[traffic.idm]` **逐字段相同**：

| 评测 | `config.json` |
| --- | --- |
| run1 对照 | `test/.../run1/.../20260714_150238_c150a5/config.json` |
| run4_r1 | `test/.../run4_r1/.../20260717_235733_fb3a02/config.json` |
| run4_r2 | `test/.../run4_r2/.../20260717_235733_595d6f/config.json` |

> 不同场景仍会因路网、周围车初始状态不同而行为不同，但 **IDM 公式参数本身不变**。

动力学参数（全部场景、三路评测共用）：

| 参数 | 配置键 | 取值 | 含义 |
| --- | --- | ---: | --- |
| 期望速度 | `target_velocity` | **15.0** m/s | 自由流目标速度 \(v_0\) |
| 最小间距 | `min_gap` | **1.0** m | 静止时希望保持的间距 \(s_0\) |
| 时距 | `headway_time` | **1.5** s | 期望时间车头时距 \(T\) |
| 最大加速度 | `accel_max` | **1.0** m/s² | 舒适加速上限 \(a\) |
| 舒适减速度 | `decel_max` | **3.0** m/s² | 制动强度参数 \(b\) |

对应 ini 片段：

```ini
[traffic]
type = idm

[traffic.idm]
target_velocity = 15.0
min_gap = 1.0
headway_time = 1.5
accel_max = 1.0
decel_max = 3.0
```

行为含义（结合碰撞分析）：IDM 会为前车按 \(T=1.5\,\mathrm{s}\) 保距并减速，但**不会像 PPO 同伴那样横向配合**；因此同质 ego 的失败主要来自「比 IDM 快 + 横向贴超」与「对 IDM 刹车跟车不足」，而非 IDM 主动进攻。

### 权重一览

| 标签 | W&B run | Checkpoint | 训练日志 |
| --- | --- | --- | --- |
| **run1 对照** | [`fdfw3v5e`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/fdfw3v5e) | `experiments/puffer_drive_fdfw3v5e/model_puffer_drive_000954.pt` | `homogeneous_drive_lstm_500m_batch1x_run1_20260714_120855.log` |
| **run4_r1** | [`1ytcyfuk`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/1ytcyfuk) | `experiments/puffer_drive_1ytcyfuk/model_puffer_drive_000954.pt` | `homogeneous_drive_lstm_500m_batch1x_run4_r1_20260717_230749.log` |
| **run4_r2** | [`u9ymfcqr`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/u9ymfcqr) | `experiments/puffer_drive_u9ymfcqr/model_puffer_drive_000954.pt` | `homogeneous_drive_lstm_500m_batch1x_run4_r2_20260717_230749.log` |

---

## 2. 分权重结果表

每一张表对应**同一套权重**：训练末期指标 + IDM 评测 + PPO 评测。

- **训练**：同质多智能体环境上的最终 Evaluate 窗口（dashboard / W&B `environment/*`），**不是** pufferinter。  
- **vs IDM / vs PPO**：严格 PDM 评测；Goal / Coll / At-fault / Offroad 为跨 map 均值。  
- 训练行 At-fault 为 `—`：训练 dashboard 无严格 PDM at-fault 项。

### 2.1 run1 对照 · `fdfw3v5e`

| 阶段 | Goal / Completion ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Return / Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| **训练（末期）** | 95.60% | 0.92% | — | 0.80% | 0.935 |
| **评测 vs IDM** | 81.32% | 5.94% | 3.23% | 1.36% | −0.724 |
| **评测 vs PPO** | 89.98% | 0.00% | 0.00% | 0.51% | −0.658 |

<details>
<summary>路径</summary>

- IDM：`test/homogeneous_drive_lstm_500m_batch1x_run1/Drive_Recurrent/20260714_150238_c150a5/`
- PPO：`test/homogeneous_drive_lstm_500m_batch1x_run1/Drive_Recurrent/20260716_173747_4c5dc7/`

</details>

### 2.2 run4_r1 · `1ytcyfuk`

| 阶段 | Goal / Completion ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Return / Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| **训练（末期）** | 96.48% | 0.82% | — | 0.80% | 0.946 |
| **评测 vs IDM** | 81.32% | 5.43% | 4.24% | 1.19% | −0.690 |
| **评测 vs PPO** | 89.47% | 0.51% | 0.17% | 0.51% | −0.640 |

<details>
<summary>路径</summary>

- IDM：`test/.../run4_r1/Drive_Recurrent/20260717_235733_fb3a02/`
- PPO：`test/.../run4_r1/Drive_Recurrent/20260717_235949_ff8e6e/` · log `eval_*_vs_ppo_gpu0_20260717_235944.log`

</details>

### 2.3 run4_r2 · `u9ymfcqr`

| 阶段 | Goal / Completion ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Return / Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| **训练（末期）** | 96.19% | 0.95% | — | 0.80% | 0.941 |
| **评测 vs IDM** | 81.66% | 5.77% | 4.58% | 0.85% | −0.676 |
| **评测 vs PPO** | 89.64% | 0.68% | 0.68% | 0.34% | −0.649 |

<details>
<summary>路径</summary>

- IDM：`test/.../run4_r2/Drive_Recurrent/20260717_235733_595d6f/`
- PPO：`test/.../run4_r2/Drive_Recurrent/20260717_235950_d8b97b/` · log `eval_*_vs_ppo_gpu1_20260717_235944.log`

</details>

---

## 3. 横向对照

### 3.1 训练末期（同质环境）

| Run | W&B | Completion ↑ | Collision ↓ | Offroad ↓ | Episode Return ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| run1 对照 | fdfw3v5e | 95.60% | 0.92% | 0.80% | 0.935 |
| run4_r1 | 1ytcyfuk | 96.48% | 0.82% | 0.80% | 0.946 |
| run4_r2 | u9ymfcqr | 96.19% | 0.95% | 0.80% | 0.941 |

### 3.2 评测 vs IDM（pufferinter）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| run1 对照 | 81.32% | 5.94% | 3.23% | 1.36% | −0.724 |
| run4_r1 | 81.32% | 5.43% | 4.24% | 1.19% | −0.690 |
| run4_r2 | 81.66% | 5.77% | 4.58% | 0.85% | −0.676 |

### 3.3 评测 vs PPO（pufferinter，同权重）

| Run | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ | Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| run1 对照 | 89.98% | 0.00% | 0.00% | 0.51% | −0.658 |
| run4_r1 | 89.47% | 0.51% | 0.17% | 0.51% | −0.640 |
| run4_r2 | 89.64% | 0.68% | 0.68% | 0.34% | −0.649 |

---

## 4. IDM 测试碰撞分析：碰撞是怎么引发的？

数据来自各 run vs IDM 的 `collision_snapshots.json`（首碰快照 + `collision_classifier` 分型）。  
评测设定回顾：**1 个 PPO ego + 其余全员车道跟随 IDM**（`v₀≈15 m/s`，车距 `T=1.5 s`，横向几乎不主动让道/超车）。

### 4.1 类型总览

| Run | 碰撞 maps | Lateral | Front | Rear | 其它 | At-fault / 碰撞 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| run1 对照 | 35 | **18（51%）** | 12（34%） | 3 | 2 | 19/35（54%） |
| run4_r1 | 32 | **20（62%）** | 7（22%） | 2 | 3 | 25/32（78%） |
| run4_r2 | 34 | **25（74%）** | 7（21%） | 1 | 1 | 27/34（79%） |

共同结论：**主因是侧向（ACTIVE_LATERAL）**，其次是前方（ACTIVE_FRONT）；被追尾（REAR）很少。  
run4 两种子相对 run1：**Front 更少，但 Lateral 更多，且侧向里 at-fault 比例明显更高** → 总 Coll 略降/持平，但 **At-fault 升到 4.2–4.6%**（run1 为 3.23%）。

### 4.2 各类型如何引发（机制）

分类规则（`pufferlib/evaluation/collision_classifier.py`）：

| 类型 | 几何条件（简化） | 责任规则 | 在本设定下的典型成因 |
| --- | --- | --- | --- |
| **ACTIVE_LATERAL** | 非正前/正后的侧方接触 | 仅当 ego **跨多车道** 才 at-fault | ego 同向超车/并线时横向净空不够；IDM **钉在车道上不躲** |
| **ACTIVE_FRONT** | 他车大致在前方且 ego 在靠近 | **一律 at-fault** | 跟车过近 / 对 IDM 减速反应晚（追尾式） |
| **ACTIVE_REAR** | 他车在后方 | 通常非责 | 后方 IDM 贴上；次数少，不是主矛盾 |
| STOPPED_* | 一方近似静止 | track 静止→常责；ego 静止→常非责 | 个案（起步/静止障碍） |

#### A. 侧向擦碰（主因）

三路 Lateral 几何高度一致：

| 特征 | run1 | run4_r1 | run4_r2 |
| --- | ---: | ---: | ---: |
| 中位 ego 速度 | 14.4 m/s | 14.2 m/s | 12.4 m/s |
| 中位他车（IDM）速度 | 8.1 m/s | 8.6 m/s | 9.9 m/s |
| ego 明显更快（Δv≥1） | 16/18 | 18/20 | 20/25 |
| 近似同向（\|Δheading\|&lt;30°） | 14/18 | 12/20 | 18/25 |
| 他车多在 ego 侧后方（fwd 中位） | −1.5 m | −2.3 m | −1.7 m |
| 发生时刻（step 中位） | ~34 | ~32 | ~33 |
| **其中 at-fault** | **5/18** | **16/20** | **20/25** |

**因果链**：同质训练鼓励偏快、偏贴的横向占位 → 评测遇到更慢、车道稳定的 IDM → 超车/并线时侧向摩擦。  
run4 侧向更多被判跨多车道（at-fault↑），说明失败更常伴随**更激进的变道/占道**，而不只是「同车道轻擦」。

#### B. 前方碰撞（次因，但几乎全责）

| 特征 | run1 | run4_r1 | run4_r2 |
| --- | ---: | ---: | ---: |
| 次数 | 12 | 7 | 7 |
| at-fault | 12/12 | 7/7 | 7/7 |
| 中位 ego / IDM 速度 | 14.6 / 8.0 | 21.7 / 11.9 | 17.6 / 10.9 |
| 中位闭合速度 | ~7.7 m/s | ~10.7 m/s | ~8.7 m/s |
| 多数小航向差（同向追尾） | 10/12 | 4/7 | 5/7 |

**因果链**：前车 IDM 按 1.5 s 车距公式减速；ego 仍按「同伴会配合」的节奏压距离 → 跟车不足 / 刹晚。  
Front **全部计入 At-fault**；run4 虽把 Front 从 12 压到 7，但 Lateral 责碰增加，**At-fault 总量仍高于 run1**。

#### C. 后方 / 静止（次要）

- REAR：每 run 仅 1–3 起，多数非责；IDM 很少从后方「顶」ego。  
- STOPPED_*：个位数；对总率影响小。

### 4.3 和评测部署的关系（为什么会撞成这样）

```text
训练：周围多是同类 PPO（会横向博弈、一起挤）
   ↓ 分布偏移
评测：周围全是车道跟随 IDM（纵向保车距、横向钝、不主动让道）
   ↓
同质 ego 仍偏快 + 横向贴 → Lateral 为主
对 IDM 刹车不敏感 → Front 为辅（全责）
```

因此 Coll≈5–6% **不是**「IDM 主动攻击」，而是 **把进攻型同质策略放进保守车道交通里的摩擦 + 跟车误差**。

### 4.4 优化含义（针对 IDM 测试）

| 优先级 | 类型 | 理由 |
| --- | --- | --- |
| **高** | Front | 次数不多但 **全责**；直接打 At-fault；形态清晰（跟车/TTC） |
| **高（run4）** | Lateral（尤其 at-fault 侧向） | 次数最多；run4 侧向责碰占比已到 80% 级，是 At-fault 抬升的主因 |
| 低 | Rear / Stopped | 样本少，搬不动总表 |

### 4.5 碰撞地图 ID 分布相似度（vs IDM）

对三路 `collision_snapshots.json` 中的 **`map_id` 集合**做重叠分析（每 map 一次首碰）。  
相似度：\(\mathrm{Jaccard}=|A\cap B|/|A\cup B|\)；\(\mathrm{Overlap}=|A\cap B|/\min(|A|,|B|)\)。

#### 集合规模与两两重叠

| 对比 | \|A\| | \|B\| | \|∩\| | Jaccard | Overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| run1 vs run4_r1 | 35 | 32 | **16** | **0.314** | 0.500 |
| run1 vs run4_r2 | 35 | 34 | **16** | **0.302** | 0.471 |
| run4_r1 vs run4_r2 | 32 | 34 | **16** | **0.320** | 0.500 |

三路碰撞 map 并集 **64** 张；其中：

| 出现次数 | 地图数 | 占并集 |
| --- | ---: | ---: |
| 三路都撞 | **11** | 17% |
| 恰两路撞 | 15 | 23% |
| 仅一路撞 | 38 | 59% |
| ≥2 路共享 | 26 | **41%** |

**结论**：两两 Jaccard 仅约 **0.30–0.32**（中低重叠）——失败场景有一批稳定「难图」，但多数碰撞地图随种子变化；run4 两个种子之间也不比「对照 vs 新跑」更像（J≈0.32）。

#### 三路共撞的 11 张图（稳定难例）

`97, 115, 223, 315, 346, 358, 426, 611, 618, 625, 816`

其中类型在多路上完全一致的硬例包括：

- **Front（全责）**：`97`, `816`
- **Lateral**：`223`, `346`, `625`（及多数路上为 Lateral 的 `315/618`）
- **Stopped ego**：`358`（开局静止贴碰，三路一致）

#### 按碰撞类型的 map 重叠（更低）

| 类型 | run1 vs r1 | run1 vs r2 | r1 vs r2 |
| --- | ---: | ---: | ---: |
| Lateral Jaccard | 0.310 | 0.194 | 0.250 |
| Front Jaccard | 0.118 | 0.188 | 0.273 |
| At-fault map Jaccard | 0.222 | 0.243 | 0.268 |

Front 集合更碎（次数少 + 种子敏感）；**At-fault 地图重叠也偏低（J≈0.22–0.27）**，说明责碰并不锁死在同一小撮 map 上。

#### 多路共撞时类型是否一致

在 ≥2 路都发生碰撞的 26 张图中，**19/26（73%）** 各 run 记录的 `collision_type` 一致；不一致的 7 张（如 `115, 259, 426, 429, 482, 611, 736`）多为 Lateral↔Front/Rear 边界，几何接近分类阈值时会跳类。

#### 解读

1. **~1/3 Jaccard**：同质 Drive+LSTM 在 IDM 评测上既有**可复现难图**（11 张全种子共撞），也有**大量种子特异失败**（38 张仅单 run）。  
2. 优化若只盯共有 11 张，覆盖并集不到两成；要降 Coll/At-fault 需同时处理**共性难例 + 种子发散的侧向责碰**。  
3. run4_r1 / r2 彼此重叠并不显著高于各自与 run1 的重叠 → 新种子没有收敛到另一套完全不同的失败模式，而是「共享核心 + 各自长尾」。

<details>
<summary>两两交集 map_id 列表</summary>

- run1 ∩ r1：`46, 97, 115, 142, 223, 259, 315, 346, 358, 426, 611, 618, 625, 816, 940, 999`
- run1 ∩ r2：`97, 115, 223, 315, 346, 358, 426, 429, 482, 611, 618, 625, 730, 734, 816, 868`
- r1 ∩ r2：`97, 115, 223, 315, 346, 358, 380, 426, 453, 490, 611, 618, 625, 707, 736, 816`

</details>

> 快照目录：  
> `test/.../run1/.../20260714_150238_c150a5/collision_snapshots.json`  
> `test/.../run4_r1/.../20260717_235733_fb3a02/collision_snapshots.json`  
> `test/.../run4_r2/.../20260717_235733_595d6f/collision_snapshots.json`

---

## 5. 指标口径（简要）

| 列 | 训练 | 评测 |
| --- | --- | --- |
| Goal / Completion | 训练分布上的 `completion_rate` | 有效 map 上 `goal_reached_rate` |
| Collision | 训练 `collision_rate` | 跨 map 平均（与 log 一致） |
| At-fault | 训练无此严格项 → `—` | 该 map 是否发生责任碰撞的发生率 |
| Offroad | 训练 `offroad_rate` | ego 是否碰 `ROAD_EDGE` 的跨 map 均值 |
| Return / Reward | `episode_return` | map 级 `total_reward` 均值 |

> 训练指标与评测指标**不可直接横比绝对值**（分布不同：同质 PPO 同伴 vs 全 IDM / 全同权 PPO）。表内同列仅便于并排查看。

---

## 6. 状态

- [x] run4_r1 · vs PPO `summary.csv` → `20260717_235949_ff8e6e`
- [x] run4_r2 · vs PPO `summary.csv` → `20260717_235950_d8b97b`

相对 run1 对照的 vs PPO：Goal 略低约 0.3–0.5pp；Coll / At-fault 从 0 升到小个位数百分点（仍远低于 vs IDM）。
