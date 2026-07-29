# PufferDrive 当前分支与 2.0 训练代码差异说明

本文档基于当前工作区 `/home/fanyuqi/wsc/behavior-bench` 的 `HEAD/main-behavior-bench` 与 `origin/2.0` 的代码对比，重点说明：

- 当前默认 PPO 训练使用的策略网络结构；
- 策略网络的输入、输出含义和维度；
- 当前分支领先 `origin/2.0` 的 3 个 commit 对训练逻辑做了哪些改动；
- 这些改动对训练数据、优势估计、LSTM 状态和 loss 计算的影响。

## 1. 当前默认策略网络

当前默认配置位于 `pufferlib/config/ocean/drive.ini`：

```ini
[base]
policy_name = Drive
rnn_name = Recurrent

[policy]
input_size = 64
hidden_size = 256

[rnn]
input_size = 256
hidden_size = 256

[env]
action_type = discrete
dynamics_model = classic
max_obs_partners = 31
```

因此，默认训练使用的是：

```text
Drive MLP encoder + Recurrent LSTMWrapper
```

也就是说，`Drive` 本身先把单帧 observation 编码成 256 维 embedding，然后外层 `Recurrent = pufferlib.models.LSTMWrapper` 再用 256 维 LSTM hidden state 做时序建模，最后用 actor/value head 输出动作 logits 和 value。

从训练日志 `.logs/train/runs/ppo_drive_default_2b_20260703_182513.log` 看，这次 `ppo_drive_default_2b` run 的 dashboard 显示：

```text
Params: 614.2K
Env:    puffer_drive
```

这个参数量与默认 `Drive + Recurrent` 结构一致，不是新增的 `DriveTransformer`、`DriveGameFormer`、`DriveConditionedPaper` 等变体。

## 2. 默认 observation 输入结构

默认环境是：

```ini
dynamics_model = classic
action_type = discrete
max_obs_partners = 31
```

相关常量在 `pufferlib/ocean/drive/drive.h` 和 `pufferlib/ocean/drive/drive.py`：

```c
#define EGO_FEATURES_CLASSIC 7
#define EGO_FEATURES_JERK 10
#define PARTNER_FEATURES 8
#define ROAD_FEATURES 7
#define MAX_ROAD_SEGMENT_OBSERVATIONS 128
#define MAX_OBS_PARTNERS 31
```

因此 classic 默认 observation 总维度是：

```text
ego:       7
partners: 31 * 8   = 248
roads:    128 * 7  = 896
total:    7 + 248 + 896 = 1151
```

即：

```text
observation shape = [B, 1151]
```

### 2.1 Ego 输入

classic dynamics 下，ego 有 7 个特征：

```text
0: rel_goal_x，目标点在 ego 坐标系下的 x，相对距离缩放后
1: rel_goal_y，目标点在 ego 坐标系下的 y，相对距离缩放后
2: signed_speed / MAX_SPEED
3: width / MAX_VEH_WIDTH
4: length / MAX_VEH_LEN
5: collision_state flag
6: respawn flag
```

如果 `dynamics_model = jerk`，ego 变成 10 维，额外包含：

```text
steering_angle
a_long
a_lat
respawn flag
```

当前默认不是 jerk，因此使用 7 维 ego。

### 2.2 Partner 输入

partner block 是：

```text
[B, 31 * 8] -> reshape -> [B, 31, 8]
```

每个 partner 8 维，大致包括：

```text
0: relative x
1: relative y
2: width
3: length
4: relative heading x / cos
5: relative heading y / sin
6: signed speed
7: entity type
```

当前分支的 C 环境会收集 candidate partners，并按距离排序，只写入最近的 `max_obs_partners=31` 个 partner。剩余 slot 用 0 padding。

这点与旧 2.0 的行为有差别：当前 observation 更明确是“最近 N 个 partner”，这会影响网络看到的 traffic context。

### 2.3 Road 输入

road block 是：

```text
[B, 128 * 7] -> reshape -> [B, 128, 7]
```

每个 road object 7 维，其中最后一维是 road/entity type。

进入网络时，`Drive.encode_observations()` 会把最后一维类别做 one-hot：

```python
road_continuous = road_objects[:, :, : self.road_features - 1]  # 6 dims
road_categorical = road_objects[:, :, self.road_features - 1]   # 1 dim
road_onehot = F.one_hot(road_categorical.long(), num_classes=7)
road_objects = torch.cat([road_continuous, road_onehot], dim=2)
```

因此 road encoder 实际看到的是：

```text
6 continuous road features + 7 one-hot road type = 13 dims
```

## 3. 默认 Drive 网络结构

默认策略实现在 `pufferlib/ocean/torch.py` 的 `class Drive`。

结构可以概括为：

```text
flat observation [B, 1151]
  ├── ego_obs [B, 7]
  │     Linear(7, 64)
  │     LayerNorm(64)
  │     Linear(64, 64)
  │
  ├── partner_obs [B, 31, 8]
  │     Linear(8, 64)
  │     LayerNorm(64)
  │     Linear(64, 64)
  │     max-pool over 31 partners -> [B, 64]
  │
  └── road_obs [B, 128, 13]
        Linear(13, 64)
        LayerNorm(64)
        Linear(64, 64)
        max-pool over 128 road objects -> [B, 64]

concat [ego, road, partner] -> [B, 192]
  GELU
  Linear(192, 256)
  ReLU
  -> Drive embedding [B, 256]

Recurrent LSTMWrapper
  LSTMCell / LSTM hidden size 256
  -> recurrent hidden [B, 256]

Actor head
  Linear(256, sum(atn_dim))

Value head
  Linear(256, 1)
```

注意几点：

- ego、partner、road 三个 encoder 都是很浅的 MLP。
- partner 和 road 都是 permutation-invariant 的 max pooling，没有 attention。
- `Drive` 的 actor 和 critic 共享前面的编码器和 LSTM hidden。
- 当前分支新增了一批更复杂的 policy class，但默认 `drive.ini` 没启用它们。

## 4. 默认动作输出

默认是：

```ini
action_type = discrete
dynamics_model = classic
```

在 `pufferlib/ocean/drive/drive.py`：

```python
self.single_action_space = gymnasium.spaces.MultiDiscrete([7 * 13])
```

所以 classic 离散动作空间是：

```text
MultiDiscrete([91])
```

`Drive.decode_actions()` 中：

```python
action = self.actor(flat_hidden)
action = torch.split(action, self.atn_dim, dim=1)
value = self.value_fn(flat_hidden)
return action, value
```

默认 `self.atn_dim = [91]`，因此 actor 输出：

```text
logits tuple: (Tensor[B, 91],)
value:        Tensor[B, 1]
```

真正的动作由 `pufferlib.pytorch.sample_logits()` 采样得到：

```text
sampled action shape = [B, 1]
action value range   = 0..90
```

如果使用 continuous action，则 actor 输出 Normal 分布参数：

```text
loc, scale -> Normal(loc, softplus(scale) + 1e-4)
```

但默认训练不是 continuous。

## 5. 与 `origin/2.0` 的默认网络差异

默认 `Drive` 网络主体和 2.0 非常接近：

- 2.0 也是 ego/partner/road 三路 MLP；
- partner/road 也是 per-object encoding 后 max pooling；
- 也是共享 embedding 后接 actor/value；
- 默认也包了 `Recurrent` LSTM。

主要差异：

1. 当前 `Drive` 用 `env.dynamics_model` 决定 ego dim：

   ```python
   self.ego_dim = 10 if env.dynamics_model == "jerk" else 7
   ```

   2.0 中是：

   ```python
   self.ego_dim = env.ego_features
   ```

2. 当前 `Drive.__init__()` 多了 `action_chunk_size` 参数，但默认路径没有真正改变动作输出逻辑。

3. 当前 `forward_eval()` 简化成直接调用 `forward()`，实际行为与原来的 encode/decode 等价。

4. 当前分支新增很多可选网络：

   ```text
   DriveConditioned
   DriveTransformer
   DriveGameFormer
   DriveConditionedPaper
   DrivePaper
   DriveGameFormerConditioned
   DriveLatentWorldModel
   LatentWorldModelWrapper
   DriveNoGoal
   ```

   这些是新增能力，但默认 `ppo_drive_default_2b` 没使用。

结论：默认策略网络结构本身不是当前分支与 2.0 最大差异；真正影响训练行为的是 PPO 训练循环、优势估计、环境终止/截断处理和数据分布。

## 6. 当前分支领先 3 个 commit 的整体情况

当前分支相对 `origin/2.0` 领先 3 个 commit：

```text
805fe04c Update title
a9cb828e Update README.md
53cc9c11 Initial commit
```

其中：

- `805fe04c` 只改 README 标题；
- `a9cb828e` 只改 README；
- 训练逻辑、环境、策略网络、配置的大规模变化基本都集中在 `53cc9c11 Initial commit`。

因此下面说的“当前分支相对 2.0 的训练差异”，主要就是 `53cc9c11` 引入的差异。

## 7. PPO 训练循环差异

核心文件：

```text
pufferlib/pufferl.py
pufferlib/models.py
pufferlib/extensions/pufferlib.cpp
pufferlib/extensions/cuda/pufferlib.cu
pufferlib/vector.py
```

### 7.1 `rollout_horizon` 改为 `bptt_horizon`

2.0 配置里使用：

```ini
rollout_horizon = 32
```

当前分支改为：

```ini
bptt_horizon = 32
```

默认 `pufferlib/config/default.ini` 也从：

```ini
rollout_horizon = 64
```

改成：

```ini
bptt_horizon = 64
```

语义上，当前代码把 horizon 明确视作 LSTM backprop-through-time 的长度，而不再用 rollout 和 BPTT 两个概念混用。

### 7.2 Experience buffer 改成保存 `horizon + 1`

2.0 中：

```python
self.observations = torch.zeros(segments, rollout_horizon, ...)
self.values = torch.zeros(segments, rollout_horizon)
self.rewards = torch.zeros(segments, rollout_horizon)
```

当前分支中：

```python
self.observations = torch.zeros(segments, horizon + 1, ...)
self.values = torch.zeros(segments, horizon + 1)
self.rewards = torch.zeros(segments, horizon + 1)

self.actions = torch.zeros(segments, horizon, ...)
self.logprobs = torch.zeros(segments, horizon)
self.terminals = torch.zeros(segments, horizon)
self.truncations = torch.zeros(segments, horizon)
```

影响：

- action/logprob/terminal/truncation 只有当前 step 的 `horizon` 个；
- observation/value/reward 多保存一个 next step；
- advantage 可以使用真正的 `V(s_{t+1})`；
- 最后一帧不再只能依赖 hack 或丢弃。

这是当前训练逻辑相对 2.0 的关键改动之一。

### 7.3 新增 next-state value 同步

当前 `PuffeRL.evaluate()` 在收集完一个 BPTT window 后，会额外调用：

```python
self.vecenv.sync_get_observations(agent_slice, timeout=30.0)
```

用来拿每个 agent 的下一帧 observation、reward、terminal、truncation，并填入 buffer 的 `l + 1` 位置：

```python
self.observations[agent_slice, l + 1] = o_device
self.rewards[agent_slice, l + 1] = r
self.values[agent_slice, l + 1] = value.flatten()
```

2.0 没有这个额外 sync loop，而是使用 truncation bootstrap hack。

当前做法更接近标准 TD/GAE 需要的数据布局：

```text
obs[0..T]
value[0..T]
reward[0..T]
action[0..T-1]
```

### 7.4 terminal 和 truncation 被分开处理

2.0 中，训练前向时基本使用：

```python
done_mask = (d + t).clamp(max=1)
self.terminals[...] = done_mask.float()
```

当前分支改为分别保存：

```python
self.terminals[batch_rows, l] = d.float()
self.truncations[batch_rows, l] = t.float()
```

优势计算也变成：

```python
compute_puff_advantage(
    values,
    rewards,
    terminals,
    truncations,
    ratio,
    advantages,
    gamma,
    gae_lambda,
    vtrace_rho_clip,
    vtrace_c_clip,
)
```

C++/CUDA kernel 也从接收 `dones` 改成接收 `terminations` 和 `truncations`。

影响：

- terminal 表示真实 episode termination，例如 agent 到达终点、碰撞、offroad 后被 remove；
- truncation 表示 rollout/window 或环境 resample 导致的截断；
- 两者对 GAE/V-trace 的含义不同，当前代码开始区分这两者。

### 7.5 advantage kernel 改动

2.0 的 CPU/CUDA advantage row 逻辑大致是：

```cpp
for (int t = horizon - 2; t >= 0; t--) {
    nextnonterminal = 1.0 - dones[t_next];
    delta = rewards[t_next] + gamma * values[t_next] * nextnonterminal - values[t];
    advantages[t] = ...
}
```

当前分支变成：

```cpp
for (int t = horizon - 1; t >= 0; t--) {
    if (truncations[t] == 1.0) {
        advantages[t] = 0.0;
        lastpufferlam = 0.0;
    } else {
        nextnonterminal = 1.0 - terminations[t_next];
        delta = rewards[t_next] + gamma * values[t_next] * nextnonterminal - values[t];
        advantages[t] = ...
    }
}
```

主要变化：

- 因为 value/reward 现在是 `horizon + 1`，所以循环可以从 `horizon - 1` 开始；
- truncation step 的 advantage 直接置 0；
- truncation 会重置 `lastpufferlam`，防止 GAE 跨 reset/window 泄漏；
- terminal 只影响 `nextnonterminal`，不会和 truncation 混在一起。

这会显著改变 value target 和 policy gradient，尤其是在 episode 中频繁 remove/reset/truncate 的场景。

### 7.6 删除 2.0 的 truncation bootstrap hack

2.0 中有逻辑：

```python
if l > 0:
    trunc_mask = (t > 0) & (d == 0)
    r = r + trunc_mask.to(r.dtype) * config["gamma"] * self.values[batch_rows, l - 1]
```

当前分支删除了这个 hack。

旧逻辑的问题是：Drive C 环境可能已经 reset，`V(s_t)` 对应的状态不一定是 pre-reset 状态，于是用前一个 value 作为 heuristic proxy。

当前分支通过额外保存 next observation/value，试图用更明确的数据结构替代这个 heuristic。

### 7.7 LSTM 训练 forward 改成逐 timestep reset

2.0 的 `LSTMWrapper.forward()` 使用整段 batched LSTM：

```python
hidden = hidden.reshape(B, TT, input_size)
hidden = hidden.transpose(0, 1)
hidden, (lstm_h, lstm_c) = self.lstm.forward(hidden, lstm_state)
...
```

当前分支改为逐 timestep 的 `LSTMCell` 循环：

```python
for t in range(TT):
    mask = trunc_or_term_before[:, t] == 1.0
    h = torch.where(mask.unsqueeze(-1), torch.zeros_like(h), h)
    c = torch.where(mask.unsqueeze(-1), torch.zeros_like(c), c)
    hidden = self.policy.encode_observations(x[:, t, :], state_timestep)
    hidden, c = self.cell(hidden, lstm_state)
```

训练调用时传入：

```python
trunc_or_term_before[:, 1:] = (
    mb_truncations[:, :-1].bool() | mb_terminals[:, :-1].bool()
)
```

影响：

- 如果上一步 terminal 或 truncation，当前 timestep 的 LSTM hidden/cell 会被清零；
- 避免 LSTM 状态跨 episode/reset 泄漏；
- 更适合当前 remove/reset 频繁的 Drive 环境；
- 代价是不能直接利用整段 `nn.LSTM` 的高效批量实现，训练 forward 可能更慢。

### 7.8 PPO loss 增加 invalid mask

当前分支在训练时构造 invalid mask：

```python
terminals = mb_terminals.bool()
truncations = mb_truncations.bool()
terminals_shifted = torch.cat([zeros, terminals[:, :-1]], dim=1)
invalid_mb_mask = terminals & terminals_shifted
invalid_mb_mask = invalid_mb_mask | truncations
invalid_mb_obs = (mb_obs[:, -1, 0] == -1000)
invalid_mb_mask[:, -1] = invalid_mb_mask[:, -1] | invalid_mb_obs
```

然后 policy loss、value loss、entropy 都用 mask 加权：

```python
loss_mask = ~invalid_mb_mask
pg_loss = (pg_loss_individual * loss_mask * mb_prio).sum() / denom
v_loss = ((0.5 * v_loss_individual * loss_mask) * mb_prio).sum() / denom
entropy_loss = (entropy * loss_mask).sum() / loss_mask.sum()
```

2.0 更接近直接 `.mean()`：

```python
pg_loss = torch.max(pg_loss1, pg_loss2).mean()
v_loss = 0.5 * torch.max(...).mean()
entropy_loss = entropy.mean()
```

影响：

- truncation step 不参与 loss；
- terminal 之后的无效 step 不参与 loss；
- `sync_get_observations()` 超时返回 sentinel observation `-1000` 时，不让这类假数据污染 loss；
- loss 数值和梯度尺度都会和 2.0 不同。

### 7.9 prioritized sampling 改动

2.0 中 prioritized sequence sampling 主要用：

```python
adv = advantages.abs().sum(axis=1)
prio_weights = adv ** alpha
idx = torch.multinomial(prio_probs, self.minibatch_segments)
```

当前分支先 mask 无效 step，并按有效 step 数归一：

```python
masked_advantages = advantages * ~invalid_mask
adv = masked_advantages.abs().sum(axis=1)
valid_steps = (~invalid_mask).sum(axis=1) + 1e-6
adv_avg = adv / valid_steps
prio_weights = adv_avg ** alpha
```

影响：

- 旧版长序列或无效 step 多的序列可能因为 sum 更大而被过度采样；
- 当前按有效 step 平均，更偏向“每个有效 step 的平均 advantage 大小”；
- 对 remove/truncation 较多的训练环境更合理。

### 7.10 advantage normalization 被弱化或移除

2.0 中：

```python
adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)
```

当前代码保留了相关注释，但实际 loss 使用：

```python
adv = mb_advantages
mb_prio = mb_prio / (mb_prio.max() + 1e-8)
```

也就是说，当前没有直接对 advantage 做 batch mean/std normalization。

影响：

- policy gradient 更依赖原始 advantage 尺度；
- reward/termination/truncation 改动会更直接影响 loss；
- 训练稳定性和学习率/clip/vf_coef 的关系可能与 2.0 不同。

### 7.11 移除 entropy annealing

2.0 配置有：

```ini
anneal_entropy = False
```

代码里也支持：

```python
current_ent_coef = cosine annealed ent coef
```

当前分支删除了 `ent_coef_initial` 和动态 entropy coefficient 逻辑，loss 直接使用：

```python
loss = pg_loss + vf_coef * v_loss - ent_coef * entropy_loss
```

因此当前默认 entropy coefficient 是固定的。

### 7.12 新增 advantage filtering

当前 `default.ini` 新增：

```ini
advantage_filtering = False
advantage_filter_threshold = 0.01
advantage_filter_ewma_beta = 0.25
```

默认关闭。

如果开启，当前代码会维护 advantage max 的 EWMA，并过滤掉绝对 advantage 低于阈值的 transition：

```python
threshold = advantage_filter_threshold * ewma_a_max
adv_filter_mask = (advantages.abs() >= threshold) & ~invalid_mask
```

这是 Gigaflow paper Alg. 1 风格的过滤逻辑。默认训练不启用，但训练框架已经支持。

### 7.13 新增 opponent pool

当前 `drive.ini` 新增：

```ini
opponent_pool = False
opponent_pool_fraction = 0.25
opponent_pool_snapshot_interval = 100
opponent_pool_max_snapshots = 20
opponent_pool_warmup_epochs = 100
```

默认关闭。

如果开启：

- 训练中会保存历史 policy snapshot；
- 每个 epoch 可以抽一部分 agent 用历史策略动作覆盖当前 learner 动作；
- opponent segment 会从 learner loss 中 mask 掉；
- 用来做历史策略对手池，增加鲁棒性。

默认 `ppo_drive_default_2b` 没启用。

### 7.14 新增 extra loss / auxiliary loss hook

当前 `PuffeRL` 有：

```python
self.extra_loss_fn = None
```

训练中：

```python
if self.extra_loss_fn is not None:
    extra_loss, extra_logs = self.extra_loss_fn(...)
    loss = loss + extra_loss

if hasattr(self.uncompiled_policy, "compute_auxiliary_loss"):
    aux_loss, aux_logs = self.uncompiled_policy.compute_auxiliary_loss()
    loss = loss + aux_loss
```

这为 BC KL、SMART reference KL、latent world model transition loss 等机制留了接口。

默认 `Drive + Recurrent` 没有 `compute_auxiliary_loss()`，所以默认不会额外加 loss。

## 8. 环境和数据分布差异

核心文件：

```text
pufferlib/config/ocean/drive.ini
pufferlib/ocean/drive/drive.py
pufferlib/ocean/drive/drive.h
pufferlib/ocean/drive/binding.c
```

### 8.1 地图路径从 `map_dir` 改为 `data_root + split`

2.0 默认配置：

```ini
map_dir = "resources/drive/binaries/training"
num_maps = 10000
```

当前默认配置：

```ini
split = "training"
num_maps = 79000
```

`Drive.__init__()` 中通过：

```python
data_root = os.environ["DRIVE_BINARIES_DATA_ROOT"]
split_dir = os.path.join(data_root, split)
```

读取数据。

影响：

- 当前训练默认从外部数据根目录下的 split 加载；
- 默认地图数量扩大到 79000；
- 训练数据分布与 2.0 默认资源目录明显不同。

### 8.2 默认 control mode 改变

2.0：

```ini
control_mode = "control_agents"
max_controlled_agents = 32
```

当前：

```ini
control_mode = "control_vehicles"
max_controlled_agents = -1
```

影响：

- 当前默认控制对象从 agents 转向 vehicles；
- 默认不再限制为 32 个 controlled agents；
- active agent 数、交通参与者组成、reward 统计都会变化。

### 8.3 默认 goal/collision/offroad behavior 改变

2.0 默认：

```ini
goal_behavior = 0
collision_behavior = 0
offroad_behavior = 0
```

当前默认：

```ini
goal_behavior = 3
collision_behavior = 2
offroad_behavior = 2
```

注释含义：

```text
goal_behavior:
  0 = respawn
  1 = generate_new_goals
  2 = stop
  3 = remove

collision/offroad behavior:
  0 = ignore
  1 = stop
  2 = remove
```

影响：

- 当前默认到达 goal 后会 remove；
- collision/offroad 后也会 remove；
- remove 后 agent 后续 step 需要被 mask；
- 这正是当前 PPO 训练中新增 terminal/truncation/mask 逻辑的重要原因。

### 8.4 新增更多 reward components

当前配置新增很多 reward 项：

```ini
reward_speed_limit = 0.0
reward_lane_alignment = 0.0
reward_lane_distance = 0.0
reward_velocity = 0.0
reward_comfort = 0.0
reward_l_align = 0.0
reward_l_align_vel = 0.5
reward_l_center = 0.0
reward_timestep = 0.0
collision_shrink = 0.7
```

`drive.py` 也新增：

```python
REWARD_COMPONENT_NAMES = (
    "collision", "offroad", "goal", "jerk_legacy", "velocity", "comfort",
    "l_align", "l_center", "timestep", "reverse", "speed_limit",
)
```

环境会写 `reward_components` 和 `reward_components_raw`，用于更细粒度评估和分析。

默认真正非零的新增项里，`reward_l_align_vel = 0.5` 比较重要。

### 8.5 新增 mixed traffic / IDM / Expert 支持

当前配置新增：

```ini
idm_others = False
mix_traffic = False
ppo_fraction = 1.0
idm_fraction = 0.0
expert_fraction = 0.0
idm_target_velocity = 15.0
idm_random_velocity = False
```

默认关闭 mixed traffic 和 idm others，但代码已经支持：

- 部分 agent 用 PPO；
- 部分 agent 用 IDM；
- 部分 agent replay expert；
- IDM 速度可以固定或随机。

这对后续鲁棒性训练、自博弈/混合交通训练有影响，但默认 run 不启用。

### 8.6 Partner observation 改为最近 N 个

当前 `compute_observations()` 会：

1. 收集周围 candidate partners；
2. 排除 removed、invalid、respawning entity；
3. 按距离排序；
4. 只写入最近 `max_obs_partners` 个。

这让 observation 更稳定地聚焦近邻交通参与者。

### 8.7 新增 reward conditioning observation

当前新增：

```c
#define CREWARD_FEATURES 10
```

如果 `env.reward_conditioning=True`，observation 会在 road block 后追加 10 维 creward conditioning features。

对应 policy 有：

```text
DriveConditioned
DriveConditionedPaper
DriveGameFormerConditioned
```

默认 `drive.ini` 没开启 reward conditioning，所以默认 observation 仍是 1151 维。

## 9. 评估、日志和 checkpoint 相关变化

当前训练逻辑还改了不少工程层面的东西：

### 9.1 训练中视频记录

`PuffeRL.evaluate()` 新增训练场景 video capture：

```python
_VIDEO_LOG_INTERVAL = 100_000_000
_VIDEO_CAPTURE_STEPS = 50
_VIDEO_NUM_SCENARIOS = 5
```

日志中可以看到：

```text
[VideoLogger] Recorded 50 frames across 5 scenarios on driver_env
```

这说明当前 run 的第一轮 evaluate 录制了训练场景视频。

### 9.2 checkpoint resume 增强

当前 `save_checkpoint()` 除保存模型，还保存：

```python
trainer_state.pt
```

包含：

```text
optimizer_state_dict
global_step
agent_step
update
model_name
run_id
```

`train()` 里如果从 checkpoint 加载，会恢复：

```python
pufferl.global_step
pufferl.epoch
```

并尝试参考 wandb summary 的 `_step`。

### 9.3 DDP logging 改动

当前代码显式处理 DDP rank：

- 只有 rank 0 创建 wandb/neptune logger；
- DDP policy 会 unwrap 到 `uncompiled_policy`；
- rank 非 0 不保存最终 checkpoint path。

## 10. 当前默认配置与 2.0 默认配置的关键差异摘要

| 项目 | `origin/2.0` 默认 | 当前分支默认 |
|---|---:|---:|
| policy | Drive | Drive |
| rnn | Recurrent | Recurrent |
| policy input size | 64 | 64 |
| policy hidden size | 256 | 256 |
| rnn hidden size | 256 | 256 |
| action type | discrete | discrete |
| dynamics | classic | classic |
| obs dim | 1151 | 1151 |
| action dim | 91 | 91 |
| horizon key | rollout_horizon | bptt_horizon |
| buffer obs/value/reward length | T | T+1 |
| terminal/truncation | 混合 done | 分开处理 |
| truncation bootstrap | heuristic reward hack | next-state value + truncation mask |
| LSTM train forward | batched LSTM | timestep LSTMCell + reset mask |
| loss | 大多直接 mean | invalid mask + priority weight |
| advantage normalization | 有 batch 标准化 | 默认不做标准化 |
| entropy annealing | 代码支持 | 移除，固定 ent_coef |
| map source | map_dir | data_root + split |
| num_maps | 10000 | 79000 |
| control_mode | control_agents | control_vehicles |
| goal behavior | respawn | remove |
| collision/offroad behavior | ignore | remove |
| optional mixed traffic | 较少 | IDM/Expert/PPO mix 支持 |
| optional policy variants | 少 | conditioning/Transformer/GameFormer/paper/world model |

## 11. 总体结论

当前默认 `ppo_drive_default_2b` 的策略网络仍然是 2.0 风格的轻量 MLP + LSTM：

```text
1151-dim obs -> Drive MLP encoder -> 256-dim embedding -> LSTM(256) -> actor logits [91] + value [1]
```

网络主体没有发生根本变化。

真正重要的变化在训练逻辑和环境语义：

1. PPO buffer 从 `T` 改为 `T+1`，支持最后一步 bootstrap；
2. terminal 和 truncation 分开进入 advantage kernel；
3. 移除了旧的 truncation reward bootstrap hack；
4. LSTM 训练前向可以在 terminal/truncation 后逐 timestep 清零状态；
5. loss 使用 invalid mask，排除 truncation、terminal 后和 sync timeout 的无效样本；
6. prioritized sampling 改为基于有效 step 的平均 advantage；
7. 默认环境从 respawn/ignore 更偏向 remove/terminate，训练样本分布明显变化；
8. 数据加载、地图数量、control mode、reward components、mixed traffic 能力都被扩展。

因此，如果要解释当前训练曲线与 PufferDrive 2.0 的差异，优先应关注：

```text
终止/截断语义 + T+1 bootstrap + LSTM reset + invalid mask + remove-style 环境行为
```

而不是默认策略网络结构本身。
