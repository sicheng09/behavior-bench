# Behavior-Aware LSTM Policy 设计文档

## 背景

当前 PPO 策略在测试车流接近训练群体时表现较好，但当测试车流切换到训练中未见过的 IDM 或 PDM 行为时，指标会明显退化，尤其是 collision rate 和 at-fault collision rate。目标是在尽量不侵入现有训练与评测框架的前提下，提高策略对未见车流行为的泛化能力。

当前代码库已经有几块可以复用的基础：

- `Drive` 负责把单帧 observation 编码成 embedding，并从 embedding 解码 action/value。
- `LSTMWrapper` 负责在 `Drive.encode_observations()` 和 `Drive.decode_actions()` 之间加入时序记忆。
- `DriveLatentWorldModel` 有 latent transition 辅助损失，但它的预测分支在评测时没有直接参与动作选择。
- reward-conditioned policy 可以通过 creward 输入表达不同底层驾驶风格。
- 评测框架已经支持 IDM、PDM、PPO、SMART 和 conditioned traffic。

本设计选择新增一条策略路径，而不是直接修改现有 `Drive` 或 `DriveLatentWorldModel` 的行为。

## 目标

新增一个真正基于 LSTM 的 behavior-aware policy。它从历史观测中学习一个交通行为 belief，预测下一步周围 partner 的运动，并把这个行为表示输入 actor/value 决策。

该策略的目标是提升对未见 IDM/PDM 参数、未见策略 ID 或未见车流组合的鲁棒性。它不应依赖测试车流拥有标准化的奖励系数。

## 非目标

- 第一版不实现多步 action reranking。
- 不要求 IDM/PDM traffic 有标准 reward coefficient。
- 不替换现有 `Drive`、`DriveConditioned` 或 `DriveLatentWorldModel` 路径。
- 在共享 LSTM 版本完成验证前，不额外加入第二套 recurrent 模块。

## 现有 LSTM 语义

原始 recurrent policy 的结构是：

```text
obs_t
  -> Drive.encode_observations(obs_t)
  -> LSTMWrapper LSTMCell
  -> Drive.decode_actions(h_t)
  -> action_t, value_t
```

`Drive` 本身是 feedforward policy。时序状态由 wrapper 维护，对应 `state["lstm_h"]` 和 `state["lstm_c"]`。评测时，每一步先编码当前 observation，再更新 LSTM cell，把新的 hidden state 写回 `state`，最后从 LSTM 输出解码 action/value。训练时，wrapper 沿 BPTT 序列逐步展开，并在 episode 边界重置 hidden state。

新策略保留这个分工：policy 负责 observation encoder 和各个 head；wrapper 负责 recurrent sequence handling。

## 建议架构

新增策略类，暂名 `DriveBehaviorAware`；新增 wrapper，暂名 `BehaviorAwareLSTMWrapper`。

整体流程：

```text
obs_t
  -> ego / road / partner encoders
  -> scene_embedding_t
  -> shared LSTM core
  -> h_t
  -> behavior_latent_t = behavior_head(h_t)

actor/value input:
  concat(h_t, behavior_latent_t)

prediction input:
  concat(behavior_latent_t, current_partner_tokens_t)
```

`h_t` 仍然是通用 recurrent memory。它可以包含 ego 历史、路线进度、道路上下文、动作惯性和交通上下文。`behavior_latent_t` 从 `h_t` 派生，但会被 next-partner prediction 明确约束。它表示的不是“车流 ID”，而是周围 agent 随时间表现出来的行为 belief。

actor/value 同时接收 `h_t` 和 `behavior_latent_t`。这样既保留原策略容量，也让行为 belief 可以直接影响动作选择。

## Observation 编码

第一版复用 `Drive` 的 observation 切分：

- ego block
- partner block
- road block

用于决策的特征：

- ego 用 MLP 编码
- road object 用 MLP 编码，加入 road type one-hot 后 max-pool
- partner object 用 MLP 编码，然后 max-pool
- 将 ego、road 和 pooled partner feature 拼接成 `scene_embedding_t`

用于预测的特征：

- 保留 max-pool 之前的 per-partner encoded tokens
- 第一版直接使用这些 encoded partner tokens
- 将这些 tokens 与 `behavior_latent_t` 拼接，用于预测下一步 partner delta

第一版只预测当前 observation 中可见的 partner slots，不预测场景中的所有 agent。

## Behavior Latent

`behavior_latent_t` 不等同于 LSTM hidden state `h_t`。

- `h_t` 是通用时序记忆，主要由 PPO 回报优化。
- `behavior_latent_t = MLP(h_t)` 是受约束的投影，训练目标是支持下一步 partner motion prediction。

这个设计给行为相关信息提供一个 bottleneck。actor/value 使用该 bottleneck，使预测性行为 belief 能够影响最终决策。

## 预测目标

主要辅助任务是 next-step partner motion prediction：

```text
target_t = partner_state_{t+1} - partner_state_t
prediction_t = f(behavior_latent_t, partner_token_t)
```

第一版只预测紧凑的连续字段：

- relative x delta
- relative y delta
- relative speed delta
- relative heading delta

这样避免预测完整 observation、road state 或 ego state，也避免对 IDM/PDM 引入 reward label 要求。

需要使用 validity mask，避免 padding partner slots 参与 loss。第一版可以把全零 partner slot 视为 invalid，这与当前 observation padding 约定一致。

## 训练损失

PPO loss 保持不变。wrapper 像 `LatentWorldModelWrapper` 一样暴露 `compute_auxiliary_loss()`：

```text
loss = ppo_loss + prediction_loss_coef * masked_mse(prediction, target)
```

辅助 loss 日志建议包含：

- `behavior_prediction_loss`
- `behavior_prediction_valid_fraction`

prediction loss 在 BPTT 训练阶段根据已存储的 `mb_obs[:, :-1]` 和 `mb_obs[:, 1:]` 计算。评测时不能使用未来 observation。

## 评测行为

评测时流程是：

```text
obs_t
  -> scene_embedding_t
  -> LSTM update
  -> behavior_latent_t
  -> actor/value
```

prediction head 可以用于诊断，但动作选择只能依赖当前和过去 observation，也就是只能通过 LSTM state 使用历史信息。不能使用未来 observation。

## 非侵入式集成

实现时新增路径：

- 在 `pufferlib/ocean/torch.py` 新增 `DriveBehaviorAware`
- 新增 `BehaviorAwareLSTMWrapper`，可以放在 `pufferlib/ocean/torch.py` 的策略附近；如果后续足够通用，再移入 `pufferlib/models.py`
- 在 `pufferlib/planning/policy.py` 增加 planner 支持，以便评测 checkpoint
- 在 registry 中增加 `behavior_aware`
- 训练配置尽量复用已有 policy-name 机制

现有 policy name 和 planner type 的行为必须保持不变。

## 训练分布

泛化目标要求训练中有足够的车流异质性。第一轮训练 sweep 应优先增加 behavior diversity，而不是只混合不同网络结构：

- 多种 creward profile 的 reward-conditioned PPO traffic
- 随机 target velocity、headway、minimum gap、acceleration、deceleration 的 IDM traffic，如果当前接口支持
- PDM traffic，或者至少把 PDM 作为 held-out evaluation target
- 如果当前训练稳定，加入 PPO opponent snapshots

初始实现可以先使用最容易配置的 traffic source，等策略路径跑通后再扩大车流分布。

## Ablation 计划

至少评测以下三组：

- `Drive + Recurrent`：原始 baseline
- `BehaviorAware no-fuse`：有 prediction loss，但 actor/value 只使用 `h_t`
- `BehaviorAware fused`：actor/value 使用 `concat(h_t, behavior_latent_t)`

主要对比对象应是未见或 held-out traffic：

- pufferinter vs IDM
- pufferinter vs PDM
- pufferrandom vs IDM
- pufferrandom vs PDM
- 可选：vs PPO 和 conditioned traffic

主要指标：

- collision rate
- at-fault collision rate
- goal completion
- offroad rate

期望结果不一定是在 PPO-vs-PPO 的 in-distribution 场景上更高，而是当 traffic 切换到未见 IDM/PDM 行为时，指标退化更小。

## 开放实现选择

第一版采用保守默认值：

- `behavior_latent_dim`：64 或 128
- `prediction_loss_coef`：从 0.05 或 0.1 开始，避免压过 PPO loss
- prediction target normalization：先使用 observation-space 中已经归一化的字段
- prediction mask：全零 partner slot 视为 invalid
- fusion：拼接 `h_t` 和 `behavior_latent_t`，经过一个小 MLP 后输入 actor/value

这些参数在第一版跑通后应变成可配置项。

## 风险

- auxiliary loss 权重过大可能伤害 PPO 驾驶性能。
- 一步预测目标可能只学到短期运动学，不一定学到策略性交互。
- 如果训练车流多样性不足，latent 仍可能过拟合已见 behavior。
- 如果 partner slot 排序变化明显，per-slot prediction 会有噪声。第一版接受这个风险，并只做 one-step prediction。

## 成功标准

如果 fused behavior-aware policy 在 held-out IDM/PDM traffic 上相对 `Drive + Recurrent` 有更小的 collision/fault 指标退化，同时 goal completion 保持在可接受范围内，则认为第一版设计成功。

