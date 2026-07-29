# Run1 混合智能实验代码改动参考

本文档面向后续 agent，用来复现或继续维护本次 run1 混合智能策略实验所需的关键代码改动。它只总结当前对话中为三等级/混合训练和验证所做的代码层改动，不包含完整实验结果分析；结果分析见：

```text
/home/fanyuqi/wsc/behavior-bench/.logs/try/run1_mixed_intelligence_summary.md
```

## 1. 改动目标

本次代码改动支持以下能力：

1. 新增低等级策略 `DriveLite`，保持原始 observation 输入和 action/value 输出接口不变，但使用更轻量的单帧策略网络。
2. 扩展 `mix_ppo`，允许每个混合策略单独指定是否使用 RNN，以及 RNN 的输入/隐藏维度。
3. 修复单策略无 LSTM 训练路径，避免 feedforward policy 被错误传入 LSTM wrapper 专用参数。
4. 扩展 benchmark eval 的 PPO planner，使其能正确加载：
   - `Drive + Recurrent(256,256)`
   - `Drive` no LSTM
   - `DriveLite` no LSTM
   - `DriveLite + Recurrent(224,224)`

这些改动都应保持非侵入：默认 `Drive + Recurrent`、旧 `mix_ppo`、旧 benchmark PPO eval 不显式设置新参数时，应继续按原行为运行。

## 2. 新增策略：`DriveLite`

文件：

```text
pufferlib/ocean/torch.py
```

新增类：

```python
class DriveLite(Drive):
    """Lightweight reactive Drive policy for low-level mixed-intelligence agents."""
```

### 2.1 设计含义

`DriveLite` 与 `Drive` 接收相同 observation：

```text
ego      = 7
partner  = 31 x 8
road     = 128 x 7
```

输出接口也与 `Drive` 相同：

```text
discrete logits: 91
value: 1
```

差异在网络复杂度：

```text
Drive:
  entity encoder: Linear -> LayerNorm -> Linear
  scene hidden:   256

DriveLite:
  entity encoder: Linear -> GELU
  entity width:   input_size // 2
  scene hidden:   hidden_size * 7 // 8
```

默认参数下：

```text
Drive base:     87,900
DriveLite base: 43,420
ratio:          49.4%
```

### 2.2 当前实现要点

`DriveLite.__init__()` 不调用 `Drive.__init__()`，而是直接用 `nn.Module.__init__(self)`，以便替换 encoder 宽度和 shared embedding 维度。

关键字段：

```python
lite_input_size = max(1, input_size // 2)
lite_hidden_size = max(1, (hidden_size * 7) // 8)
self.hidden_size = lite_hidden_size
```

这点很重要：如果把 `DriveLite` 包 LSTM，RNN input/hidden 应使用 `DriveLite.hidden_size`。默认配置下为：

```text
DriveLite hidden_size = 224
Recurrent input_size  = 224
Recurrent hidden_size = 224
```

## 3. `mix_ppo` 支持每个策略独立 RNN 配置

文件：

```text
pufferlib/pufferl.py
```

函数：

```python
load_mixed_policies(args, vecenv, env_name="", base_policy=None)
```

### 3.1 新增训练配置项

新增可选字段：

```text
train.mix_ppo_rnn_names
train.mix_ppo_rnn_input_sizes
train.mix_ppo_rnn_hidden_sizes
```

对应默认配置文件：

```text
pufferlib/config/ocean/drive.ini
```

新增默认空值：

```ini
mix_ppo_rnn_names =
mix_ppo_rnn_input_sizes =
mix_ppo_rnn_hidden_sizes =
```

### 3.2 兼容规则

如果不设置 `mix_ppo_rnn_names`：

```python
rnn_names = [args["rnn_name"]] * len(policy_names)
```

因此旧命令：

```bash
--train.mix-ppo-policy-names "Drive,DriveMoE,DriveMoE3"
```

仍会在默认 `rnn_name=Recurrent` 下给所有 mixed policies 包同一个 `Recurrent`。

如果设置：

```bash
--train.mix-ppo-rnn-names "Recurrent,None,None"
```

则：

```text
policy 0 -> Recurrent
policy 1 -> no RNN
policy 2 -> no RNN
```

`None`、`null`、空字符串都会被解释为 no RNN。

### 3.3 不同 RNN 尺寸

为了支持 `Drive + Recurrent(256,256)` 与 `DriveLite + Recurrent(224,224)` 同时混合，新增：

```bash
--train.mix-ppo-rnn-input-sizes "256,224,0"
--train.mix-ppo-rnn-hidden-sizes "256,224,0"
```

实现逻辑：

```python
rnn_kwargs = dict(args["rnn"])
if rnn_input_sizes:
    rnn_kwargs["input_size"] = int(rnn_input_sizes[i])
if rnn_hidden_sizes:
    rnn_kwargs["hidden_size"] = int(rnn_hidden_sizes[i])
policy = rnn_cls(vecenv.driver_env, policy, **rnn_kwargs)
```

对 `rnn_name=None` 的策略，`0` 只是占位，不会生效。

## 4. 混合训练中的 feedforward / recurrent 分支

文件：

```text
pufferlib/pufferl.py
```

### 4.1 `policy_uses_rnn`

在 `PuffeRL.__init__()` 中记录每个 mixed policy 是否带 RNN：

```python
self.policy_uses_rnn = [
    hasattr(p, "lstm") or hasattr(p, "cell")
    for p in self.uncompiled_policies
]
```

用途：

1. rollout eval 时只给 recurrent policy 传/保存 `lstm_h/lstm_c`。
2. PPO minibatch loss 里按 policy 类型选择时间维处理方式。
3. bootstrap value 时使用正确调用签名。

### 4.2 mixed rollout eval

函数：

```python
_mix_ppo_forward_eval(...)
_mix_ppo_forward_values(...)
```

关键逻辑：

```python
if self.policy_uses_rnn[policy_idx]:
    state["lstm_h"] = ...
    state["lstm_c"] = ...

logits, value = policy.forward_eval(o_device[policy_mask], state)

if self.policy_uses_rnn[policy_idx]:
    self.lstm_h[...] = state["lstm_h"]
    self.lstm_c[...] = state["lstm_c"]
```

feedforward policy 仍会收到 `state`，但 `Drive.forward_eval(x, state=None)` 能兼容并忽略 LSTM 字段。

### 4.3 mixed PPO loss

函数：

```python
_compute_ppo_minibatch_loss(..., policy_uses_rnn=None)
```

关键逻辑：

```python
uses_rnn = config["use_rnn"] if policy_uses_rnn is None else policy_uses_rnn

if uses_rnn:
    policy_obs = mb_obs[:, :-1]
    result = policy(policy_obs, state, trunc_or_term_before)
else:
    policy_obs = mb_obs[:, :-1].reshape(-1, *self.vecenv.single_observation_space.shape)
    result = policy(policy_obs, state)
```

bootstrap value：

```python
if uses_rnn:
    bootstrap_obs = mb_obs[:, -1:]
    result = policy(bootstrap_obs, state, mb_truncations[:, -1:], episode_ended=True)
else:
    bootstrap_obs = mb_obs[:, -1:].reshape(-1, *self.vecenv.single_observation_space.shape)
    result = policy(bootstrap_obs, state)
```

调用 mixed policy loss 时必须传入对应 policy 的 RNN 标记：

```python
self.policy_uses_rnn[policy_idx]
```

## 5. 修复单策略 no-LSTM 训练路径

文件：

```text
pufferlib/pufferl.py
```

问题：

`mix_ppo=False` 且 `--rnn-name None` 时，原单策略训练路径仍执行：

```python
self.policy(policy_obs, state, trunc_or_term_before)
```

这会让 feedforward `Drive.forward()` / `DriveLite.forward()` 收到过多参数，报错：

```text
TypeError: Drive.forward() takes from 2 to 3 positional arguments but 4 were given
```

修复：

```python
if config["use_rnn"]:
    result = self.policy(policy_obs, state, trunc_or_term_before)
else:
    result = self.policy(policy_obs, state)
```

bootstrap value 也同理：

```python
if config["use_rnn"]:
    result = self.policy(bootstrap_obs, state, mb_truncations[:, -1:], episode_ended=True)
else:
    result = self.policy(bootstrap_obs, state)
```

这个修复是同质 `Drive no LSTM` / `DriveLite no LSTM` 能训练的必要条件。

## 6. benchmark eval 支持 no-LSTM 与自定义 RNN 尺寸

涉及文件：

```text
pufferlib/planning/policy.py
pufferlib/planning/registry.py
pufferlib/config/evaluation.ini
tests/test_eval_planners.py
```

### 6.1 `PPOConfig` 新增字段

```python
rnn_name: str = "Recurrent"
rnn_input_size: int = 256
rnn_hidden_size: int = 256
```

### 6.2 `PPOPlanner` 构建 policy

原逻辑总是包 `LSTMWrapper`，这会导致 no-LSTM checkpoint 在 eval 时错误套一层随机 LSTM。

现在逻辑：

```python
rnn_name = config.rnn_name
if isinstance(rnn_name, str) and rnn_name.lower() in ("none", "null", ""):
    rnn_name = None

if rnn_name is None:
    self.policy = base_policy.to(self.device)
    self._policy_uses_rnn = False
else:
    rnn_cls = getattr(pt, rnn_name)
    self.policy = rnn_cls(
        policy_env, base_policy,
        input_size=config.rnn_input_size,
        hidden_size=config.rnn_hidden_size,
    ).to(self.device)
    self._policy_uses_rnn = True
```

`plan()` 中只有 recurrent policy 才初始化和保存 LSTM state：

```python
if self._policy_uses_rnn:
    self.lstm_h = torch.zeros(batch_size, self.config.rnn_hidden_size, device=self.device)
    self.lstm_c = torch.zeros(batch_size, self.config.rnn_hidden_size, device=self.device)
...
if self._policy_uses_rnn:
    self.lstm_h = state["lstm_h"]
    self.lstm_c = state["lstm_c"]
```

### 6.3 registry config builder

文件：

```text
pufferlib/planning/registry.py
```

`_build_ppo_config()` 新增：

```python
rnn_name=str(cfg.get("rnn_name", "Recurrent")),
rnn_input_size=int(cfg.get("rnn_input_size", cfg.get("hidden_size", 256))),
rnn_hidden_size=int(cfg.get("rnn_hidden_size", cfg.get("hidden_size", 256))),
```

### 6.4 evaluation.ini 默认值

文件：

```text
pufferlib/config/evaluation.ini
```

在 `[planner.ppo]` 和 `[traffic.ppo]` 中加入：

```ini
rnn_name = Recurrent
rnn_input_size = 256
rnn_hidden_size = 256
```

旧 eval 命令不设置这些字段时，仍然是默认 `Drive + Recurrent(256,256)`。

## 7. 典型训练命令

### 7.1 Mixed A：三等级主实验

```bash
puffer train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  --train.name mix_intelligence_3level_lstm_drive_drivelite_6b_batch3x_run1 \
  --train.batch-size 1572864 \
  --train.total-timesteps 6000000000 \
  --env.num-maps 10000 \
  --env.mix-traffic False \
  --train.mix-ppo True \
  --train.mix-ppo-policy-mix "l3:1,l2:1,l1:1" \
  --train.mix-ppo-policy-names "Drive,Drive,DriveLite" \
  --train.mix-ppo-rnn-names "Recurrent,None,None" \
  --train.mix-ppo-policy-paths ",," \
  --train.mix-ppo-policy-trainable "True,True,True"
```

对应策略：

```text
p0 = Drive + Recurrent(256,256)
p1 = Drive no LSTM
p2 = DriveLite no LSTM
```

### 7.2 Mixed B：DriveLite 保留 LSTM

```bash
puffer train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  --train.name mix_drive_lstm_drivelite_lstm224_drive_nolstm_6b_batch3x_run1 \
  --train.batch-size 1572864 \
  --train.total-timesteps 6000000000 \
  --env.num-maps 10000 \
  --env.mix-traffic False \
  --train.mix-ppo True \
  --train.mix-ppo-policy-mix "drive_lstm:1,drivelite_lstm:1,drive_ff:1" \
  --train.mix-ppo-policy-names "Drive,DriveLite,Drive" \
  --train.mix-ppo-rnn-names "Recurrent,Recurrent,None" \
  --train.mix-ppo-rnn-input-sizes "256,224,0" \
  --train.mix-ppo-rnn-hidden-sizes "256,224,0" \
  --train.mix-ppo-policy-paths ",," \
  --train.mix-ppo-policy-trainable "True,True,True"
```

对应策略：

```text
p0 = Drive + Recurrent(256,256)
p1 = DriveLite + Recurrent(224,224)
p2 = Drive no LSTM
```

### 7.3 同质 no-LSTM 训练

`Drive no LSTM`：

```bash
puffer train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  --train.name homogeneous_drive_nolstm_2b_batch1x_run1 \
  --policy-name Drive \
  --rnn-name None \
  --train.batch-size 524288 \
  --train.total-timesteps 2000000000 \
  --env.num-maps 10000
```

`DriveLite no LSTM`：

```bash
puffer train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  --train.name homogeneous_drivelite_nolstm_2b_batch1x_run1 \
  --policy-name DriveLite \
  --rnn-name None \
  --train.batch-size 524288 \
  --train.total-timesteps 2000000000 \
  --env.num-maps 10000
```

`DriveLite + LSTM224`：

```bash
puffer train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  --train.name homogeneous_drivelite_lstm224_2b_batch1x_run1 \
  --policy-name DriveLite \
  --rnn-name Recurrent \
  --rnn.input-size 224 \
  --rnn.hidden-size 224 \
  --train.batch-size 524288 \
  --train.total-timesteps 2000000000 \
  --env.num-maps 10000
```

## 8. 典型验证命令

### 8.1 Drive + Recurrent

```bash
python pufferlib/ocean/benchmark/eval.py \
  --output-dir .logs/val/run1/results/<run>/p0_Drive_Recurrent \
  --planner.type ppo \
  --planner.ppo.weights-path /path/to/model_policy_0_puffer_drive_003815.pt \
  --planner.ppo.device cuda \
  --planner.ppo.policy-class-name Drive \
  --planner.ppo.input-size 64 \
  --planner.ppo.hidden-size 256 \
  --planner.ppo.rnn-name Recurrent \
  --planner.ppo.rnn-input-size 256 \
  --planner.ppo.rnn-hidden-size 256 \
  --traffic.type idm \
  --eval.split pufferinter \
  --map-ids all
```

### 8.2 Drive no LSTM

```bash
python pufferlib/ocean/benchmark/eval.py \
  --output-dir .logs/val/run1/results/<run>/p1_Drive_NoLSTM \
  --planner.type ppo \
  --planner.ppo.weights-path /path/to/model_policy_1_puffer_drive_003815.pt \
  --planner.ppo.device cuda \
  --planner.ppo.policy-class-name Drive \
  --planner.ppo.input-size 64 \
  --planner.ppo.hidden-size 256 \
  --planner.ppo.rnn-name None \
  --planner.ppo.rnn-input-size 256 \
  --planner.ppo.rnn-hidden-size 256 \
  --traffic.type idm \
  --eval.split pufferinter \
  --map-ids all
```

### 8.3 DriveLite no LSTM

```bash
python pufferlib/ocean/benchmark/eval.py \
  --output-dir .logs/val/run1/results/<run>/p2_DriveLite_NoLSTM \
  --planner.type ppo \
  --planner.ppo.weights-path /path/to/model_policy_2_puffer_drive_003815.pt \
  --planner.ppo.device cuda \
  --planner.ppo.policy-class-name DriveLite \
  --planner.ppo.input-size 64 \
  --planner.ppo.hidden-size 256 \
  --planner.ppo.rnn-name None \
  --planner.ppo.rnn-input-size 224 \
  --planner.ppo.rnn-hidden-size 224 \
  --traffic.type idm \
  --eval.split pufferinter \
  --map-ids all
```

### 8.4 DriveLite + Recurrent224

```bash
python pufferlib/ocean/benchmark/eval.py \
  --output-dir .logs/val/run1/results/<run>/p1_DriveLite_Recurrent224 \
  --planner.type ppo \
  --planner.ppo.weights-path /path/to/model_policy_1_puffer_drive_003815.pt \
  --planner.ppo.device cuda \
  --planner.ppo.policy-class-name DriveLite \
  --planner.ppo.input-size 64 \
  --planner.ppo.hidden-size 256 \
  --planner.ppo.rnn-name Recurrent \
  --planner.ppo.rnn-input-size 224 \
  --planner.ppo.rnn-hidden-size 224 \
  --traffic.type idm \
  --eval.split pufferinter \
  --map-ids all
```

## 9. 验证与 smoke checks

本次实际做过的关键 smoke checks：

```text
1. DriveLite 参数量检查：
   Drive base     = 87,900
   DriveLite base = 43,420
   ratio          = 49.4%

2. mixed RNN size loading：
   policy classes = ['LSTMWrapper', 'LSTMWrapper', 'Drive']
   hidden sizes   = [256, 224, 256]
   params         = [614236, 446620, 87900]

3. single no-LSTM training smoke：
   Drive no LSTM      evaluate() + train() OK
   DriveLite no LSTM  evaluate() + train() OK

4. tiny mixed training smoke：
   ['LSTMWrapper', 'Drive', 'DriveLite']
   evaluate() + train() OK
```

环境里没有 `pytest` 模块，因此没有运行完整 pytest suite。至少应跑：

```bash
python -m py_compile \
  pufferlib/pufferl.py \
  pufferlib/ocean/torch.py \
  pufferlib/planning/policy.py \
  pufferlib/planning/registry.py
```

如果后续环境补齐 pytest，建议运行：

```bash
python -m pytest tests/test_mix_ppo.py tests/test_eval_planners.py -q
```

## 10. 注意事项

1. `DriveLite` 默认 hidden 为 224，不是 256。包 LSTM 时必须使用 `rnn.input_size=224` 和 `rnn.hidden_size=224`。
2. no-LSTM checkpoint 在 eval 时必须设置 `--planner.ppo.rnn-name None`，否则会错误套随机 LSTM。
3. mixed training 中如果多个策略 RNN 尺寸不同，必须同时设置：
   ```text
   mix_ppo_rnn_names
   mix_ppo_rnn_input_sizes
   mix_ppo_rnn_hidden_sizes
   ```
4. `mix_ppo_rnn_input_sizes` / `hidden_sizes` 的长度必须与 `mix_ppo_policy_mix` 项数一致；无 RNN 策略可填 `0` 占位。
5. 旧 `mix_ppo` 命令不设置新增字段时应保持旧行为，即所有策略沿用全局 `rnn_name` 和 `args["rnn"]`。
6. `env.num_agents=1024` 不能被 3 整除，因此 1:1:1 混合下每个策略 agent slot 数有极小整数误差，但 batch/total steps 已按 3 倍放大，近似保证每个策略与同质训练预算等价。
