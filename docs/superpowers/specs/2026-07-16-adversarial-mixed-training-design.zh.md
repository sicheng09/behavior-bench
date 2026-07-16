# 集合对抗生成混合训练设计

日期：2026-07-16  
状态：设计已确认，待实施计划  
适用项目：BehaviorBench / PufferDrive

## 1. 背景与目标

当前同质策略自博弈容易形成策略共适应。在 `.logs/val` 的异构交通评测中，训练内较低的碰撞率可能在面对 IDM 等未见控制器时上升到 4% 以上。

本设计在保留原有训练模式、启动参数、`Drive` 奖励和 `mix_ppo` 功能的前提下，引入场景级混合的非对称零和训练：

- Ego 使用仓库默认 `Drive + Recurrent` 网络和原始奖励。
- Primary Opponent 第一版同样使用 `Drive + Recurrent`，但使用独立的对抗奖励。
- 每个场景内部同时包含多辆 Ego 和多辆 Primary Opponent。
- Ego–Ego 自博弈继续存在；同时产生 Ego–Opponent 和 Opponent–Opponent 交互。
- Ego 与 Primary Opponent 分别拥有独立参数、value function 和 optimizer，并在同一次 PPO 训练中联合更新。
- 第一版优先使用 Python 扩展，不修改 C 动力学、agent 创建、原始奖励计算和碰撞处理。
- 不启用新配置时，系统完全走原 `puffer_drive` 路径，不增加运行时开销。

## 2. 非目标

第一版不包含：

- 动态 PSRO/PFSP curriculum 更新；
- episode 级切换 trainable policy owner；
- IDM、Expert、PDM 或冻结 PPO 接管 Primary Opponent 槽位；
- C 侧对抗奖励或新增 C telemetry；
- 单场景内复杂的多对多因果图学习；
- 修改原 `Drive`、原 `mix_traffic` 或原 `mix_ppo` 的默认语义。

上述能力仅通过接口预留，待第一版验证核心假设后再扩展。

## 3. 已验证的现有约束

### 3.1 训练链路

训练入口为：

```text
puffer train puffer_drive
  -> pufferlib.pufferl.train
  -> load_config
  -> load_env
  -> load_policy / load_mixed_policies
  -> PuffeRL.evaluate
  -> PuffeRL.train
```

`Drive` 在 Python 中创建共享 buffer，并通过 C binding 完成地图加载、agent 创建、动力学、奖励和 reset/step。

### 3.2 mix_ppo 语义

`mix_ppo` 已支持：

- 多个 PPO 网络按 agent slot 路由；
- 每个网络独立 RNN 状态、loss、optimizer 和 checkpoint；
- `segment_policy_ids` 将 BPTT segment 路由到对应 policy；
- C 侧按 policy id 记录分策略训练指标。

现有 `assign_policy_ids` 面向整个 agent 数组做全局 deficit interleaving。它不是按角色分块，而是按累计配额缺口交错排列；因此能在 agent 数不可整除时保持全局比例，并自然允许少量同质场景。第一版直接沿用该机制，仅使用 `Drive.agent_offsets` 审计每个场景的实际角色构成。

### 3.3 BPTT 约束

默认 episode 长度为 91，BPTT horizon 为 32。现有 `segment_policy_ids` 假设一个 segment 只属于一个 policy。

现有 mix_ppo 的 policy id 对 agent slot 保持固定，因此不会产生 BPTT optimizer 路由错误。只有在额外实现 episode reset 时动态交换 Ego/Opponent owner，才可能让一个 BPTT segment 包含两个 policy 的 transition。第一版不做动态 owner 切换。

### 3.4 Reward clamp

`PuffeRL.evaluate` 会将环境 reward clamp 到 `[-1, 1]`。因此仅设置数值很大的 \(w_2\) 不能自动保证 fault penalty 相对正奖励占优；所有奖励分量必须先归一化，并显式约束正奖励上限。

## 4. 推荐架构

### 4.1 总体结构

新增 `AdversarialMixDrive`，继承 `Drive`，仅覆盖 Python 侧初始化、resample 和 step 周边逻辑：

```text
Drive.__init__
  -> C 创建各场景与 active agents
  -> 直接接收现有 mix_ppo 生成的 policy_log_ids
  -> policy 0 解释为 Ego，policy 1 解释为 Primary Opponent
  -> 使用 agent_offsets 审计每个 scene 的实际构成

Drive.resample_maps
  -> mix_ppo role ids 保持不变
  -> 使用新的 agent_offsets 重新审计场景混合覆盖

AdversarialMixDrive.step(actions)
  -> 缓存 step 前状态
  -> 调用 Drive.step(actions)
  -> 读取 step 后状态及原 reward components
  -> AsymmetricRewardEvaluator 计算 Opponent reward
  -> Ego reward 保持原值
  -> 仅替换 Opponent slots 的 reward
```

默认 `Drive` 不继承、不导入这些模块。

### 4.2 Role 与 Policy

第一版定义两个稳定角色：

- `EGO = 0`
- `PRIMARY_OPPONENT = 1`

对应现有 `mix_ppo` 的两个 policy：

- policy 0：`Drive + Recurrent`，使用原 reward；
- policy 1：`Drive + Recurrent`，使用对抗 reward。

两者均可独立配置初始化 checkpoint、学习率和 `mix_ppo_policy_trainable`。第一版默认两者都从随机初始化开始并联合训练。

角色决定：

- 使用哪个 policy 前向；
- 使用哪个 value function；
- transition 进入哪个 optimizer；
- 使用原 reward 还是对抗 reward；
- 分角色日志归属。

## 5. 沿用 mix_ppo 的全局角色分配

### 5.1 分配语义

第一版不新增 role allocator。配置直接使用：

```ini
mix_ppo_policy_mix = ego:0.5, primary_opponent:0.5
```

现有 `assign_policy_ids(total_agents, fractions)` 使用累计 deficit 进行确定性交错：

```text
50:50 -> E O E O E O ...
75:25 -> E E O E E E O E ...
```

该机制提供：

1. 全局角色数接近目标比例，整数误差不超过 slot 离散化误差；
2. agent 数不能被比例整除时仍有效；
3. 不会出现先分配一大块 Ego、再分配一大块 Opponent；
4. policy owner 对 slot 稳定，兼容现有 BPTT、RNN state、optimizer 和 checkpoint；
5. 少量 Ego-only 场景保留原自博弈分布；
6. 少量 Opponent-only 场景增加训练分布变化，但需要监控，不能占比过高。

50:50 时角色严格交替，因此任何包含至少两个连续 PPO slots 的场景都会自然包含两类角色。其他比例下，小场景可能同质，这是允许的分布特征，不通过强制场景内取整改变全局比例。

### 5.2 场景构成审计

`AdversarialMixDrive` 使用现有 `policy_log_ids` 作为 role ids，并按：

```text
[agent_offsets[s], agent_offsets[s + 1])
```

统计每个场景的：

- Ego 数量；
- Primary Opponent 数量；
- mixed / ego-only / opponent-only 类型；
- 按 controlled-agent 数分桶的混合率。

初始化及 `resample_maps` 后重新计算审计指标，但不改变 role ids。

配置提供软门槛：

```yaml
role_assignment:
  mode: global_deficit
  warn_mixed_scene_rate_below: 0.8
  fail_mixed_scene_rate_below: null
```

默认仅告警和记录，不因自然存在的小场景同质而停止训练。严格失败阈值为可选研究配置。

### 5.3 与 mix_ppo 的关系

现有 mix_ppo 继续完整负责：

- `assign_policy_ids`；
- `_mix_ppo_forward_eval`；
- `segment_policy_ids`；
- 两个 policy 的独立 RNN state；
- 两个 optimizer；
- 分 policy checkpoint 和日志。

新增模块只解释 policy 0/1 的角色含义，并在环境返回 reward 前按 role 替换 Opponent reward。它不是另一套多策略 PPO。

## 6. Python 非对称奖励

### 6.1 Reward routing

每一步先保存原奖励：

```python
base_rewards = self.rewards.copy()
```

计算完成后：

```text
Ego slots      <- base_rewards[Ego slots]
Opponent slots <- adversarial_rewards
```

Ego reward 必须与同状态、同动作下原 `Drive.step` 返回值逐元素一致。该性质通过自动化测试保证。

### 6.2 状态来源

第一版仅使用现有接口：

- `Drive.get_global_agent_state()`：位置、heading、尺寸、entity id/type；
- 连续两步 global state：推导速度、加速度和横向运动；
- `observations`：collision state、局部 partner 几何和自身速度；
- `actions`：classic dynamics 下的期望加速度与 steering；
- `reward_components_raw`：collision、offroad、comfort、reverse、speed-limit 等行为项；
- `agent_offsets` 与 role ids：限制在同一场景内部匹配。

不启用 `include_global_state`，因此 Ego 的默认 observation 维度与网络输入保持不变。

### 6.3 建议奖励

\[
R_{adv} =
\operatorname{clip}\left(
w_1 C_{ego}
- w_2 P_{adv\_fault}
- w_3 C_{kinematics}
+ w_4 N_{normal},
-1, 1
\right)
\]

其中：

#### \(C_{ego}\)

仅奖励可归因于当前 Opponent 的 Ego 困难行为：

- Ego 急刹；
- Ego 转向或横向运动突变；
- Ego–Opponent TTC 风险势能上升；
- 高置信度 Ego 全责追尾。

对于 Opponent \(j\)，从同场景 Ego 集合中取最大因果门控值，而不是求和：

\[
C_{ego}^{(j)} = \max_{i \in \mathcal{E}_{scene}}
g(i,j)\,c(i,j)
\]

其中 \(g(i,j)\) 要求：

- Opponent 位于 Ego 的有效风险扇区；
- pairwise closing speed 为正；
- TTC 或距离达到配置阈值；
- Opponent 风险行为在时间上先于 Ego 急刹/碰撞。

该门控防止 Opponent 从无关车辆造成的 Ego 急刹中获得奖励。

TTC 使用相对位置和沿视线方向的 closing speed 近似：

\[
\mathrm{TTC}_{ij} =
\frac{\max(d_{ij}-d_{safe},0)}
\max(v^{close}_{ij}, \epsilon)}
\]

风险项采用势能增量，而非持续低 TTC 的逐步常数奖励，减少长期尾随刷分：

\[
C_{ttc} =
\max\left(0,\Phi(\mathrm{TTC}_{t})-
\Phi(\mathrm{TTC}_{t-1})\right)
\]

其中 \(\Phi\) 随 TTC 降低而增大。

#### \(P_{adv\_fault}\)

第一版采用保守归责：

- Opponent 与正常行驶车辆发生主动前向或侧向碰撞：高惩罚；
- Opponent 在碰撞前存在明显横向切入、急转、急刹或超限运动：高惩罚；
- 无法可靠识别碰撞对象或责任：按 Opponent fault 处理；
- 仅当 Ego 从后方接近，且 Opponent 在 lookback 窗口内保持车道、没有急刹/切入/超限动作时，允许判定为 Ego 高置信度全责追尾。

该规则宁可漏掉一部分正向 adversarial credit，也不允许通过制造不可归责事故获得收益。

#### \(C_{kinematics}\)

使用 soft barrier 惩罚：

- 纵向加速度越界；
- 横向加速度越界；
- 纵/横向 jerk 越界；
- steering 或 steering rate 越界；
- offroad、reverse、speed-limit violation。

阈值和缩放全部来自 YAML，不写死在 evaluator 内。

#### \(N_{normal}\)

可选正常驾驶正则：

- 正向进度；
- 车道方向一致；
- 非无故静止。

其权重必须显著小于 fault penalty，防止 Opponent 通过完全静止规避风险，也不能覆盖安全惩罚。

### 6.4 数值约束

由于 trainer 会 clamp reward：

- 所有非事故正向项合计上限建议不超过 `0.25`；
- 普通运动学违规建议落在 `[-0.5, 0]`；
- Opponent 全责事故映射到 `-1.0`；
- 默认要求 `w2 >= 4 * w1`；
- 配置加载时验证上限关系，不满足时拒绝启动。

## 7. 配置设计

项目训练配置以 INI 为主，因此使用两层配置：

### 7.1 独立训练 INI

新增：

```text
pufferlib/config/ocean/drive_adversarial.ini
```

包含原 `drive.ini` 所需字段，并设置：

```ini
[base]
env_name = puffer_drive_adversarial
policy_name = Drive
rnn_name = Recurrent

[env]
adversarial_config_path = pufferlib/config/adversarial/opponent_mix.yaml

[train]
mix_ppo = True
mix_ppo_policy_mix = ego:0.5, primary_opponent:0.5
mix_ppo_policy_names = Drive,Drive
mix_ppo_rnn_names = Recurrent,Recurrent
mix_ppo_policy_trainable = True,True
```

### 7.2 对抗 YAML

新增：

```text
pufferlib/config/adversarial/opponent_mix.yaml
```

职责：

- 场景混合率告警/可选失败阈值；
- reward 权重；
- TTC、急刹、转向和运动学阈值；
- reward component 上限；
- 非有限 reward 事件的终止阈值；
- curriculum/sampler 的保留字段。

YAML 只由 `pufferlib.adversarial.config` 加载，不改变原 `load_config` 的 INI 解析行为。

### 7.3 回退保证

以下命令完全保持原行为：

```bash
puffer train puffer_drive
```

只有显式使用新 env/config 时才加载 adversarial 模块：

```bash
puffer train puffer_drive_adversarial \
  --config pufferlib/config/ocean/drive_adversarial.ini
```

原 `mix_ppo` 参数和功能继续可独立使用。

## 8. 文件变更

### 8.1 新增

```text
pufferlib/adversarial/__init__.py
pufferlib/adversarial/config.py
pufferlib/adversarial/registry.py
pufferlib/adversarial/sampler.py
pufferlib/adversarial/reward.py
pufferlib/adversarial/env.py
pufferlib/config/ocean/drive_adversarial.ini
pufferlib/config/adversarial/opponent_mix.yaml
tests/test_adversarial_assignment_audit.py
tests/test_adversarial_reward.py
tests/test_adversarial_mix_env.py
```

`registry.py` 第一版注册 `ego_drive_recurrent` 和 `primary_opponent_drive_recurrent` 两种角色语义及其 reward evaluator。实际神经网络仍由现有 `load_mixed_policies` 构建。IDM、Expert、冻结 PPO 的 schema 可以保留，但若第一版未实现对应 builder，加载时必须明确报错，不能静默降级。

`sampler.py` 第一版只定义未来 opponent-pool sampler 协议，不参与当前 role 分配。

### 8.2 修改

`pufferlib/ocean/environment.py`

- 注册 `puffer_drive_adversarial`；
- 原 `puffer_drive` 映射不变。

`setup.py`

- 声明 `PyYAML>=6.0` 运行时依赖，用于安全加载独立对抗配置。

第一版不修改：

```text
pufferlib/pufferl.py
pufferlib/ocean/drive/drive.h
pufferlib/ocean/drive/binding.c
```

## 9. 日志与指标

在现有 `mix_ppo/policy_0/*`、`mix_ppo/policy_1/*` 基础上新增 Python info：

- 每个 scene 的 Ego/Opponent 数量分布；
- `mixed_scene_rate`；
- `ego_only_scene_rate`；
- `opponent_only_scene_rate`；
- 按 controlled-agent 数分桶的混合率；
- `adv/cost_ego`；
- `adv/penalty_fault`；
- `adv/cost_kinematics`；
- `adv/normality`；
- `adv/reward_total`；
- `adv/min_ttc`；
- `adv/ego_hard_brake_events`；
- `adv/fault_collision_events`；
- `adv/ambiguous_collision_events`。

聚合日志不得在每个 step 传输大数组；wrapper 累积标量，在现有 report interval 上报。

## 10. 错误处理

启动时立即拒绝：

- role 比例总和不为正；
- `mix_ppo` 未开启或 policy 数量不是 2；
- policy/RNN 配置与两个 role 不匹配；
- reward 权重不满足安全上限；
- YAML 含未实现 strategy 且被启用；
- `agent_offsets` 非单调或越界。
- 场景混合率低于用户显式启用的严格失败阈值。

运行时：

- state 中出现无效位置时跳过相应 pair，并计数；
- TTC closing speed 非正时不产生 TTC 正奖励；
- 事故责任无法确认时使用保守 fault penalty；
- NaN/Inf reward 立即替换为 `-1` 并记录错误计数；超过阈值终止训练。

## 11. 测试与验收

### 11.1 mix_ppo 分配与场景审计

- 全局数量满足目标比例；
- 不出现“前几个场景全 Ego、后几个场景全 Opponent”的全局分段；
- 50:50 配置严格交替；
- agent 数不能被比例整除时误差受控；
- wrapper 使用的 role ids 与现有 `policy_log_ids` 一致；
- map resample 后 role ids 不变，并重新计算 mixed/ego-only/opponent-only 指标；
- 小场景允许同质，不修改原 role ids。

### 11.2 Reward

- Ego reward 与原 `Drive` bit-exact；
- Opponent reward 才被替换；
- Ego 无关急刹不向错误 Opponent 提供 credit；
- TTC 下降时风险势能增加；
- 非 closing pair 不产生 TTC credit；
- Opponent 主动侧撞得到 `-1`；
- 高置信度 Ego 追尾可向 Opponent 提供受限正奖励；
- 急切入后被追尾仍判 Opponent fault；
- 运动学越界持续产生负项；
- reward 始终有限且位于 `[-1, 1]`。

### 11.3 联合训练

- 两个 `Drive + Recurrent` policy 均完成一次小规模 PPO update；
- 两个 optimizer 的参数均变化；
- Ego policy 只消费原 reward；
- Opponent policy 只消费对抗 reward；
- 每 policy 日志和 checkpoint 正确；
- BPTT segment 的 role id 稳定。

### 11.4 向后兼容

- 原 `tests/test_mix_ppo.py` 全部通过；
- 原 `puffer train puffer_drive` 冒烟测试通过；
- 单策略和原 mix_ppo checkpoint 格式不变；
- 未启用新 config 时不调用 adversarial reward evaluator；
- 未启用新 config 时无新增 per-step Python/C 开销。

### 11.5 性能

新模式中 `get_global_agent_state` 与 pairwise 风险计算必须按 scene 向量化。禁止构造全局 \(N \times N\) 矩阵。

性能验收：

- 只在场景内部计算 pair；
- 内存复杂度为所有场景 pair 数之和；
- 与相同 agent 数的原 mix_ppo 相比，目标 SPS 下降不超过 10%；
- 若超过 10%，先优化 Python 批处理，再讨论只读 C telemetry。

## 12. 实验建议

第一阶段至少包含：

1. 原同质 `Drive_Recurrent` baseline；
2. 50:50 Ego/Opponent，Opponent 使用非对称奖励；
3. 75:25 与 25:75 比例消融；
4. 移除 fault penalty；
5. 移除 kinematics penalty；
6. 移除 TTC 势能；
7. 只做多 policy 同奖励的控制组，用于区分“网络分离”和“非对称奖励”的贡献。

评测保持现有 `.logs/val` 协议，至少报告：

- IDM traffic collision rate；
- at-fault collision rate；
- goal/completion；
- Ego self-play 训练性能是否退化；
- 不同 seed 的 paired bootstrap confidence interval。

## 13. 后续扩展接口

第一版稳定后再考虑：

- `OpponentSampler.sample(scene_context, history)`；
- 按 opponent type 维护 EMA collision/failure；
- FailureAware/PFSP/PSRO 权重更新；
- Primary Opponent 槽位由 IDM、Expert、冻结 PPO 接管；
- 在 terminal 与 BPTT 安全边界切换 behavior strategy；
- C 侧只读 collision partner telemetry，用于提高事故归责精度。

任何扩展都必须保持 role owner 与 behavior strategy 解耦，不能让一个 BPTT segment 被两个 optimizer 共同拥有。
