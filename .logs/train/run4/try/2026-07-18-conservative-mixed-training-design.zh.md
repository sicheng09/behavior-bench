# 保守伙伴混合训练设计（ConservativeMix）

日期：2026-07-18  
状态：设计已确认；实施计划见同目录 `2026-07-18-conservative-mixed-training-plan.md`  
适用项目：BehaviorBench / PufferDrive  
问题证据：`.logs/train/run4/problem/homogeneous_drive_lstm_500m_results.md`

---

## 1. 背景与目标

### 1.1 现象

同质多智能体（`Drive + Recurrent`，`mix_traffic=False`，全 PPO 自博弈）训练出的策略，在 pufferinter **vs IDM** 评测上碰撞率明显上升（约 5–6%），而 **vs PPO**（同权重同伴）碰撞接近 0。

碰撞分型（首碰快照）：

| 优先级 | 类型 | 机制 |
| --- | --- | --- |
| 主因 | ACTIVE_LATERAL | ego 偏快/贴超；IDM 钉在车道上不让道 → 侧向摩擦 |
| 次因 | ACTIVE_FRONT | IDM 按公式保距减速；ego 跟车不足 → 全责前方碰撞 |

根因是**训练—评测分布偏移**：训练同伴会横向博弈/配合，评测 IDM 换道差、横向钝、纵向谨慎保距、不主动让道。

### 1.2 目标

设计一种**混合训练**方法，使训练交通覆盖「保守、少换道、不主动让道」等 IDM **相关行为特征**，从而缓解上述偏移；但**训练中不直接加入评测器 IDM**（避免「对着评测器过拟合」的弱故事）。

成功判据（预注册）：

- 相对同质基线 H0（run4），pufferinter vs IDM 的 Collision 或 At-fault **相对下降 ≥ 15%**；
- 同时 vs PPO 的 Collision **上升不超过 1 个百分点**。

### 1.3 非目标（第一版）

- 不在训练中启用 `mix_traffic` 的评测同款 IDM（`idm_fraction > 0` 仅作负对照实验，不进入主方法故事）；
- 不修改 C 侧动力学、`move_idm`、评测 `collision_classifier` / 严格 PDM at-fault 语义；
- 不修改默认 `puffer_drive` / `drive.ini` 的既有语义与启动命令；
- 不做 PSRO/对手池 curriculum（可后续扩展）；
- 不把 adversarial 零和奖励与本方法默认耦合。

---

## 2. 推荐方案总览

采用**统一 ConservativeMix 基础设施**，用配置切换三种伙伴诱导模式：

| `partner_mode` | 含义 | 实现阶段 |
| --- | --- | --- |
| `action_constraint` | **A**：对 partner 横向动作做 logits mask | **先做** |
| `reward_shaping` | **B**：仅改写 partner 奖励，软诱导保守行为 | A 稳定后 |
| `both` | A+B 同时启用 | 最后对照 |
| `off` | 关闭诱导（近同质 mix_ppo 对照） | 消融 |

**推荐实施顺序：方案基础设施 + 先 A 后 B。**

一句话故事：

> 用可训练的行为约束/塑形伙伴，在训练分布中覆盖「保守、少换道、不主动让道」等规则交通特征，缓解同质自博弈 → IDM 评测的分布偏移，且训练不接入评测器 IDM。

---

## 3. 与直接混 IDM 的区别

| 维度 | 直接混评测 IDM | ConservativeMix（本设计） |
| --- | --- | --- |
| 控制器 | 评测同款解析跟驰 + 车道跟随 | 仍为 PPO 网络（可学、可 ckpt） |
| 参数重叠 | 高（`T=1.5`, `v0=15`, …） | **无**评测 IDM 公式；B 的训练时距默认 **1.8s ≠ 1.5** |
| 覆盖的特征 | 公式级一致 | 行为级：少大转角、偏居中、偏保距、不主动让道 |
| 故事风险 | 「对着评测器过拟合」 | 分布鲁棒 / 行为域随机化 |
| 负对照 | — | 单独跑 `mix_traffic` IDM（Neg），承认其弱故事，不写入主贡献 |

---

## 4. 非侵入边界与入口

> **最高优先级约束（用户确认）：** 一切实现必须非侵入——不得改变现有功能与启动方式，不得影响既有结果复现，且整次改动必须可整包撤回。

### 4.1 入口

沿用 `AdversarialMixDrive` 的挂载模式，**新增独立 env**，不改默认路径：

```text
puffer train puffer_drive                       # 不变：行为与今日完全一致
puffer train puffer_drive_conservative_mix      # 新：仅此命令走新路径
```

### 4.2 硬边界（功能 / 启动 / 复现）

1. **默认路径零行为变化**：未显式选择 `puffer_drive_conservative_mix` 时，不 import `pufferlib.conservative`、不对 `Drive` / `pufferl` 热路径加 hook、无额外开销、无随机性变化。
2. **启动方式不变**：现有 `puffer train puffer_drive`、评测脚本、旧 ini、旧 CLI flag 的语义与默认值保持原样；禁止为「顺便重构」改名、改默认、改解析顺序。
3. **复现隔离**：不得改动会改变同质/旧实验轨迹的代码路径——包括但不限于 C `drive.h` 动力学、默认奖励、`mix_traffic`/`mix_ppo` 默认值、观测构造、seed 处理、评测 `collision_classifier` / PDM。同一旧命令 + 同一 seed + 同一权重，应仍可复现既有指标。
4. **奖励隔离**：Ego（policy 0）奖励始终为原 `Drive` 输出；A/B 只作用于 partner（policy 1）且仅在新 env 内。
5. **不训练评测 IDM**：主方法 ini 中 `idm_fraction = 0.0`；混 IDM 仅作可选负对照，走既有 `mix_traffic` 能力，不改其默认。

### 4.3 可整包撤回（Revertibility）

实现时把改动收成**可删除的增量包**，撤回后仓库回到「从未引入 ConservativeMix」的行为：

| 允许的改动（增量） | 撤回方式 |
| --- | --- |
| 新目录 `pufferlib/conservative/` | 删除目录 |
| 新 ini `drive_conservative_mix.ini`（及 config 镜像） | 删除文件 |
| `environment.py` **仅追加**一行/一项新 env 注册 | 删除该注册项 |
| 新测试 `tests/test_conservative_*.py` | 删除文件 |
| 本设计/计划/日志（`.logs/train/run4/try/`） | 按需保留或删除，不影响训练代码 |

| 禁止的改动（会破坏撤回/复现） | 原因 |
| --- | --- |
| 修改 `Drive.step` / 默认奖励 / C 核心逻辑「顺带支持」本功能 | 旧路径行为可能漂移 |
| 改 `drive.ini` 默认值或 `mix_ppo=False` 语义 | 复现与启动方式被污染 |
| 在 `pufferl.py` 热路径无条件 import/branch 本功能 | 默认训练多开销或隐性行为变化 |
| 改评测协议或 at-fault 定义来「配合」新方法 | 与历史表不可比 |
| 重构无关模块并混在同一 diff | 无法整包撤回 |

**撤回验收**：删除上表「允许的改动」后，`puffer train puffer_drive` 与既有评测命令行为与改动前一致（可用既有同质短训/烟测对照）。

### 4.4 实施禁令（清单）

- ❌ 改 `pufferlib/config/ocean/drive.ini` 的默认字段值  
- ❌ 改 C 侧 IDM / 动力学 / agent 创建以「复用」保守行为  
- ❌ 让 `mix_ppo=False` 的默认 `Drive` 走任何 conservative 代码  
- ❌ 为接 A/B 而改写已有 adversarial 的默认行为（二者独立 env）  
- ✅ 只新增包 + 新 env 名 + 新 ini + 新测试；注册表追加且惰性 import  

### 4.5 角色与数据流

```text
mix_ppo:
  policy 0 = ego       Drive+Recurrent，原奖励，无动作约束
  policy 1 = partner   Drive+Recurrent，按 partner_mode 施加 A/B

evaluate 循环:
  partner logits
    → [A] 非法 steer 维置 -inf → sample（logprob 正确）
  env.step(actions)
    → [B] 仅改写 partner slots 的 reward
  分策略 PPO update（现有 mix_ppo 路径）
```

动作空间保持离散 91 = 7 accel × 13 steer（与 `drive.h` 中 `ACCELERATION_VALUES` / `STEERING_VALUES` 一致）。

---

## 5. 沿用 mix_ppo 的全局角色分配

配置：

```ini
mix_ppo = True
mix_ppo_policy_mix = ego:0.5, partner:0.5
mix_ppo_policy_names = Drive,Drive
mix_ppo_rnn_names = Recurrent,Recurrent
mix_ppo_policy_trainable = True,True
```

沿用现有 `assign_policy_ids` 的全局 deficit 交错分配（与 adversarial 设计相同），保证：

- 全局比例接近 50:50；
- slot 级 policy owner 稳定，兼容 BPTT / RNN / 独立 optimizer / checkpoint；
- 自然保留少量 ego-only / partner-only 场景。

`ConservativeMixDrive` 用 `agent_offsets` 审计 `assignment/mixed_scene_rate` 等指标；默认仅告警，不因小场景同质而失败。

---

## 6. 模式 A：动作约束（优先实现）

### 6.1 目标行为

Partner 横向「钝」：限制大转角，减少变道意图，不主动给 ego 让出横向净空。纵向第一版不硬 mask（留给 B 或可选后续）。

### 6.2 作用点（PPO 正确性）

**禁止**在 `env.step` 之后改写 action 却保留原 logprob。

正确做法：对 policy 1 外包 `ConstrainedPartnerPolicy`：

```text
logits → 非法 steer 对应动作维置 -inf → Categorical.sample
→ logprob 对应当前执行动作 → 写入 buffer
```

Ego（policy 0）不包装。

### 6.3 合法动作集

```text
action = accel_idx * 13 + steer_idx
合法 ⟺ |STEERING_VALUES[steer_idx]| ≤ partner_max_abs_steer
```

| 配置 | `partner_max_abs_steer` | 合法 steer 档数 | 用途 |
| --- | ---: | ---: | --- |
| 默认 | **0.333** | 中心 5 档 | C-A 主实验 |
| 收紧 | 0.167 | 中心 3 档 | C-A-tight 消融 |
| 关闭 | ≥ 1.0 | 全部 13 档 | 约束 off |

可选（非 v1 默认）：`partner_constrain_accel` 限制过大正加速度。

### 6.4 A 的日志指标

| 指标 | 含义 |
| --- | --- |
| `partner/steer_abs_mean` | partner 执行转角绝对值均值 |
| `partner/large_steer_frac` | `\|steer\| > 0.333` 占比（约束开启后应 ≈ 0） |
| `partner/allowed_logit_mass` | 合法动作 softmax 质量（防止异常全 mask） |
| `assignment/mixed_scene_rate` | 场景内 ego+partner 共存率 |

### 6.5 A 的风险与缓解

| 风险 | 缓解 |
| --- | --- |
| 过紧 → partner 成「不动的墙」，ego 过拟合 | 默认 0.333；保留 50% ego–ego；vs PPO 守门；做 0.167/off 消融 |
| 过松 → Lateral 偏移补不够 | 看 `large_steer_frac` 与 vs IDM Lateral 分解 |
| 事后改 action 破坏 PPO | 只用采样前 logits mask |

---

## 7. 模式 B：奖励塑形（A 之后）

### 7.1 挂载方式

对齐 `AdversarialMixDrive`：在 `Drive.step` 之后，**仅替换 partner slots** 的 reward；ego 保持原值。奖励最终 clamp 到 `[-1, 1]`，以匹配 `PuffeRL.evaluate` 的 clamp。

### 7.2 Partner 塑形项（v1）

| 分量 | 方向 | 信号 | 说明 |
| --- | --- | --- | --- |
| \(r_{\text{center}}\) | + | `l_center` / 横向偏移 | 放大居中 |
| \(r_{\text{align}}\) | + | `l_align` | 放大航向对齐 |
| \(r_{\text{steer}}\) | − | 本步 `\|steer\|` | 惩罚大转角 |
| \(r_{\text{gap}}\) | +/− | 几何前车时距 | 软保距；**不用 IDM 公式** |
| collision / offroad | 沿用 base | 原分量 | 不改责任定义 |

合成：

```text
r_partner = clip(
  r_base
  + w_c * r_center + w_a * r_align
  - w_s * |steer|
  + w_g * gap_shaping(headway; T_train)
, -1, 1)
```

默认起点：

```text
w_c = 0.05
w_a = 0.05
w_s = 0.05
w_g = 0.10
partner_target_headway = 1.8   # 秒；刻意 ≠ 评测 IDM 的 1.5
```

短跑校准目标：partner `large_steer_frac` 下降，且 partner completion 仍可用。

### 7.3 B 的风险与缓解

| 风险 | 缓解 |
| --- | --- |
| 权重不够，partner 仍激进 | 监控 steer/gap 指标；必要时加大 `w_s`/`w_g` |
| `w_g` 过大 → 过度龟速 | 监控 partner 速度；vs PPO / 训练 completion 守门 |
| `both` 下 `r_steer` 与 A 冗余 | 用 C-AB vs C-A 消融决定是否保留 |

---

## 8. 文件布局

```text
pufferlib/conservative/
  __init__.py
  env.py                 # ConservativeMixDrive(Drive)
  config.py              # partner_mode / steer / shaping 权重
  action_constraint.py   # 合法 steer 集、logits mask（阶段 A）
  reward.py              # PartnerShapingEvaluator（阶段 B）
  policy.py              # ConstrainedPartnerPolicy

pufferlib/config/ocean/drive_conservative_mix.ini
config/ocean/drive_conservative_mix.ini

pufferlib/ocean/environment.py   # 仅注册新 env 名
tests/test_conservative_action_constraint.py
tests/test_conservative_reward_shaping.py
```

**明确不改：** 默认 `drive.ini`、C `drive.h` 动力学与评测 IDM 路径、`mix_traffic` 默认语义、评测 at-fault 定义。

---

## 9. 配置默认值

新文件 `drive_conservative_mix.ini`（阶段 A 可训默认）：

```ini
env_name = puffer_drive_conservative_mix

[env]
partner_mode = action_constraint
partner_max_abs_steer = 0.333
partner_constrain_accel = False
partner_target_headway = 1.8
mix_traffic = False
ppo_fraction = 1.0
idm_fraction = 0.0
expert_fraction = 0.0

[train]
mix_ppo = True
mix_ppo_policy_mix = ego:0.5, partner:0.5
mix_ppo_policy_names = Drive,Drive
mix_ppo_rnn_names = Recurrent,Recurrent
mix_ppo_policy_trainable = True,True
; 其余超参对齐 homogeneous 500M batch1x（batch_size=524288, total_timesteps=500M 等）
```

塑形权重可放在同 ini 的 `[env]` 或独立小节，由 `config.py` 解析；`partner_mode=action_constraint` 时忽略塑形权重。

---

## 10. 实验矩阵

预算对齐 run4：`Drive+Recurrent`，500M，batch1x（524288），除非另注。

| ID | 训练设置 | 阶段 |
| --- | --- | --- |
| H0 | 同质 Drive+LSTM（已有 run4） | 基线 |
| C-A | ConservativeMix，`action_constraint`，steer≤0.333 | **先做** |
| C-A-tight | 同上，steer≤0.167 | A 消融 |
| C-B | `reward_shaping` only | A 后 |
| C-AB | `both` | 最后 |
| Neg | `mix_traffic` 混评测 IDM（如已有 ppo_idm50） | 负对照（弱故事） |

评测协议（与 run4 一致）：

- split = `pufferinter`，maps = all（有效约 589）；
- vs IDM：主表；报告 Goal / Coll / At-fault / Offroad，并分解 Lateral vs Front；
- vs PPO：同权重同伴，守门泛化是否塌缩。

---

## 11. 实现顺序与验收

| 阶段 | 交付 | 验收 |
| --- | --- | --- |
| A1 | `action_constraint` + env 注册 + 单测 | 单测绿；短训 partner `large_steer_frac≈0`；默认 `puffer_drive` 烟测不受影响 |
| A2 | 500M C-A + vs IDM / vs PPO | 对比 H0；看 Lateral/Front 与成功判据 |
| B1 | `reward_shaping` + 单测 | mode 切换不破坏 A；仅 partner reward 变化 |
| B2 | C-B、C-AB、可选 C-A-tight | 填矩阵；写结果小结 |

---

## 12. 风险汇总

| 风险 | 缓解 |
| --- | --- |
| Ego 过拟合「粘车道墙」 | 50% ego–ego；vs PPO 守门；steer 消融 |
| 故事上像假 IDM | 不用评测公式/参数；T_train=1.8；Neg 单独标注 |
| A 破坏 PPO on-policy | 采样前 logits mask，不在 step 后改 action |
| B 无效或过龟 | 预注册判据；短跑调 `w_*`；监控速度/completion |
| 与 adversarial 配置冲突 | 独立 env 名与包；默认互不启用 |
| 非侵入 / 复现被破坏 | 遵守 §4.2–4.4；默认路径零开销、零语义变化；改动可整包删除撤回 |
| 撤回困难 | 禁止改旧热路径；仅追加注册 + 新包；撤回验收见 §4.3 |

---

## 13. 动机—做法—差异—风险（摘要卡）

**动机**  
同质自博弈同伴会横向配合；评测 IDM 横向钝、不让道、纵向保距 → Lateral 为主、Front 为辅的碰撞上升。

**做法**  
新增 `puffer_drive_conservative_mix`：`mix_ppo` 下 ego + 保守伙伴；先用横向 logits 约束（A），再用伙伴专用奖励塑形（B）；默认关闭，不影响原命令。

**与直接混 IDM 的区别**  
伙伴仍是 PPO；只编码抽象行为特征，不调用评测 IDM 控制器与参数；IDM 混训仅作负对照。

**主要风险**  
伙伴过呆导致过拟合慢车墙；B 调参失败；误实现离策略 action 改写。用 vs PPO 守门、消融与正确 mask 点位规避。

---

## 14. 设计确认记录

- 方案选择：统一 ConservativeMix（A/B/both），先 A 后 B — 已确认  
- §1 架构与非侵入边界 — 已确认  
- §2 A 动作约束 — 已确认  
- §3 B 塑形与实验矩阵 — 已确认  
- §4 文件布局 / 默认配置 / 实现顺序 / 风险 — 已确认  
- 非侵入 / 复现隔离 / 可整包撤回（§4.2–4.4）— 已确认为最高优先级约束  
- 文档落盘路径（用户指定）：`.logs/train/run4/try/`
- 可退回说明：`pufferlib/conservative/REVERT.md`（副本：`.logs/train/run4/try/conservative-mix-revert.md`）
