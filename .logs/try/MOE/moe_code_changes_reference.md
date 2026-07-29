# MOE 相关代码改动与复现指南

本文档面向后续 agent，用来理解和复现历史 MOE / MIXMOE 实验。实验结果总结见：

```text
/home/fanyuqi/wsc/behavior-bench/.logs/try/MOE/moe_experiment_summary.md
```

## 1. 相关文件

MOE 实验主要涉及以下代码和配置：

| 文件 | 作用 |
|---|---|
| `pufferlib/ocean/torch.py` | 定义 `DriveMoE` / `DriveMoE3` 策略网络 |
| `pufferlib/policy_mix.py` | 解析 `mix_ppo_policy_mix`，分配 agent policy id，整理 mix_ppo 日志 |
| `pufferlib/pufferl.py` | 加载混合策略、rollout 分发、按策略分组训练、保存多策略 checkpoint |
| `pufferlib/config/ocean/drive.ini` | 默认 PPO / mix_ppo 配置 |
| `tests/test_mix_ppo.py` | mix_ppo 与 MoE 相关测试 |
| `.logs/train/train-muliti-ppo.md` | 历史 multi-PPO/MOE 训练模板与解释 |
| `.logs/val/ppo_policy_eval_summary.md` | 历史 MOE 验证结果汇总 |

## 2. `DriveMoE` / `DriveMoE3` 策略结构

文件：

```text
pufferlib/ocean/torch.py
```

### 2.1 `_DriveMoEExpert`

`_DriveMoEExpert` 是一个 Drive-style observation encoder。每个 expert 都独立编码同一个 observation：

```text
obs
  ├─ ego      -> Linear -> LayerNorm -> Linear
  ├─ partner  -> Linear -> LayerNorm -> Linear -> max pool
  └─ road     -> one-hot road type -> Linear -> LayerNorm -> Linear -> max pool

concat [ego, road, partner]
  -> GELU
  -> Linear -> hidden_size
```

默认配置下：

```text
input_size = 64
hidden_size = 256
```

expert 输出维度为 256，直接兼容默认 `Recurrent(input_size=256, hidden_size=256)`。

### 2.2 `DriveMoE`

`DriveMoE` 是 soft-router MoE：

```python
self.experts = nn.ModuleList(
    _DriveMoEExpert(env, input_size=input_size, hidden_size=hidden_size)
    for _ in range(self.num_experts)
)
self.router = nn.Sequential(
    Linear(observation_size, input_size),
    GELU(),
    Linear(input_size, num_experts),
)
```

forward 逻辑：

```python
expert_embeddings = torch.stack(
    [expert(observations) for expert in self.experts],
    dim=1,
)
router_logits = self.router(observations.float())
router_weights = torch.softmax(router_logits, dim=1)
embedding = (expert_embeddings * router_weights.unsqueeze(-1)).sum(dim=1)
```

然后复用 Drive-style actor/value：

```text
embedding -> actor -> 91 logits
embedding -> value -> 1
```

`last_router_weights` 会保存最近一次 forward 的 router 权重，便于诊断。

### 2.3 `DriveMoE3`

`DriveMoE3` 是 `DriveMoE` 的薄 wrapper：

```python
class DriveMoE3(DriveMoE):
    def __init__(self, env, **kwargs):
        super().__init__(env, num_experts=3, **kwargs)
```

对应：

```text
DriveMoE  = 2 experts
DriveMoE3 = 3 experts
```

### 2.4 参数量

默认 `drive.ini` 参数下：

| Policy | Base 参数量 | Recurrent 后参数量 |
|---|---:|---:|
| `Drive` | 87,900 | 614,236 |
| `DriveMoE` | 226,014 | 752,350 |
| `DriveMoE3` | 290,335 | 816,671 |

## 3. mix_ppo 数据分配与隔离

文件：

```text
pufferlib/policy_mix.py
pufferlib/pufferl.py
```

### 3.1 policy mix 解析

`parse_policy_mix()` 支持：

```text
"drive:1,moe:1,moe3:1"
"learner:0.5,transformer:0.25,gameformer:0.25"
```

输出：

```python
names, fractions
```

fractions 会归一化。

### 3.2 agent slot 分配

`assign_policy_ids()` 使用 deficit-based interleaving：

```text
fractions = [1/3, 1/3, 1/3]
agent slots:
  0, 1, 2, 3, 4, 5, ...
policy ids:
  0, 1, 2, 0, 1, 2, ...
```

这样避免连续块分配导致某些场景集中出现单一策略。

### 3.3 rollout 分发

`PuffeRL._mix_ppo_forward_eval()` 根据 `agent_policy_ids` 切 mask：

```python
local_policy_ids = self.agent_policy_ids[env_id]
for policy_idx, policy in enumerate(self.policies):
    policy_mask = local_policy_ids == policy_idx
    logits, value = policy.forward_eval(o_device[policy_mask], state)
```

每个 agent slot 的 action/value/logprob 由其所属策略计算。

### 3.4 训练隔离

采样写入 rollout buffer 时：

```python
self.segment_policy_ids[batch_rows] = self.agent_policy_ids[env_id]
```

训练时：

```python
mb_policy_ids = self.segment_policy_ids[idx]
policy_mask = mb_policy_ids == policy_idx
```

每个策略只使用自己的 segments 计算 PPO loss，不会用其他策略等级/结构的经验更新。

### 3.5 checkpoint 保存

混合训练会保存：

```text
model_policy_0_puffer_drive_<epoch>.pt
model_policy_1_puffer_drive_<epoch>.pt
model_policy_2_puffer_drive_<epoch>.pt
mix_model_puffer_drive_<epoch>.pt
trainer_state.pt
```

其中单策略验证通常使用 `model_policy_i_...pt`。

## 4. 配置项

文件：

```text
pufferlib/config/ocean/drive.ini
```

相关字段：

```ini
[base]
policy_name = Drive
rnn_name = Recurrent

[train]
mix_ppo = False
mix_ppo_policy_mix = learner:1.0
mix_ppo_policy_names =
mix_ppo_policy_paths =
mix_ppo_policy_trainable =
```

MOE 混合训练需要覆盖：

```bash
--train.mix-ppo True
--train.mix-ppo-policy-mix "drive:1,moe:1,moe3:1"
--train.mix-ppo-policy-names "Drive,DriveMoE,DriveMoE3"
--train.mix-ppo-policy-paths ",,"
--train.mix-ppo-policy-trainable "True,True,True"
```

历史 MOE 实验使用统一默认 `rnn_name=Recurrent`，因此每个策略都包默认 `Recurrent(256,256)`。

## 5. 典型训练命令

### 5.1 同质 DriveMoE

```bash
puffer train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  --policy-name DriveMoE \
  --rnn-name Recurrent \
  --train.total-timesteps 2000000000 \
  --train.batch-size 524288 \
  --env.num-maps 10000 \
  --wandb
```

### 5.2 同质 DriveMoE3

```bash
puffer train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  --policy-name DriveMoE3 \
  --rnn-name Recurrent \
  --train.total-timesteps 2000000000 \
  --train.batch-size 524288 \
  --env.num-maps 10000 \
  --wandb
```

### 5.3 MIXMOE 2B

```bash
puffer train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  --train.name mix_ppo_drive_moe_moe3_1to1to1_default_2b \
  --env.num-maps 10000 \
  --train.mix-ppo True \
  --train.mix-ppo-policy-mix "drive:1,moe:1,moe3:1" \
  --train.mix-ppo-policy-names "Drive,DriveMoE,DriveMoE3" \
  --train.mix-ppo-policy-paths ",," \
  --train.mix-ppo-policy-trainable "True,True,True" \
  --wandb
```

### 5.4 MIXMOE3x 6B / batch3x

为保证每个策略每次 update 约获得同质训练相同 batch，需要：

```text
batch_size = 3 * 524288 = 1572864
total_timesteps = 3 * 2B = 6B
```

命令：

```bash
puffer train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  --train.name mix_ppo_drive_moe_moe3_1to1to1_default_6b_batch3x \
  --train.batch-size 1572864 \
  --train.total-timesteps 6000000000 \
  --env.num-maps 10000 \
  --train.mix-ppo True \
  --train.mix-ppo-policy-mix "drive:1,moe:1,moe3:1" \
  --train.mix-ppo-policy-names "Drive,DriveMoE,DriveMoE3" \
  --train.mix-ppo-policy-paths ",," \
  --train.mix-ppo-policy-trainable "True,True,True" \
  --wandb
```

历史训练模板详见：

```text
/home/fanyuqi/wsc/behavior-bench/.logs/train/train-muliti-ppo.md
```

## 6. 典型验证命令

### 6.1 同质 DriveMoE vs IDM

```bash
python pufferlib/ocean/benchmark/eval.py \
  --output-dir .logs/val/runs/2MOE.1 \
  --planner.type ppo \
  --planner.ppo.weights-path /home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_aoljsqfo.pt \
  --planner.ppo.device cuda \
  --planner.ppo.policy-class-name DriveMoE \
  --planner.ppo.input-size 64 \
  --planner.ppo.hidden-size 256 \
  --traffic.type idm \
  --eval.split pufferinter \
  --map-ids all
```

### 6.2 MIXMOE 单 policy checkpoint vs IDM

```bash
python pufferlib/ocean/benchmark/eval.py \
  --output-dir .logs/val/runs/MIXMOE.1/p1 \
  --planner.type ppo \
  --planner.ppo.weights-path /home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_x9earvkh/model_policy_1_puffer_drive_003815.pt \
  --planner.ppo.device cuda \
  --planner.ppo.policy-class-name DriveMoE \
  --planner.ppo.input-size 64 \
  --planner.ppo.hidden-size 256 \
  --traffic.type idm \
  --eval.split pufferinter \
  --map-ids all
```

对应关系：

```text
p0 = Drive
p1 = DriveMoE
p2 = DriveMoE3
```

## 7. 测试与 smoke checks

相关测试文件：

```text
tests/test_mix_ppo.py
tests/test_policy_pool.py
```

历史测试覆盖：

1. `parse_policy_mix()` 解析命名比例。
2. `assign_policy_ids()` 按比例交错分配 agent slots。
3. dashboard 中 p0/p1/p2 loss/stats 显示。
4. mix_ppo checkpoint 为每个 policy 单独保存权重。
5. `DriveMoE` / `DriveMoE3` 前向 shape 与默认 `Recurrent` 兼容。
6. 小规模 `mix_ppo` 训练能跑过一次 `evaluate()` + `train()`。

建议验证：

```bash
python -m py_compile \
  pufferlib/ocean/torch.py \
  pufferlib/policy_mix.py \
  pufferlib/pufferl.py \
  tests/test_mix_ppo.py
```

如果环境有 pytest：

```bash
python -m pytest tests/test_mix_ppo.py tests/test_policy_pool.py -q
```

## 8. 注意事项

1. `DriveMoE` 和 `DriveMoE3` 的输出 hidden size 必须与 RNN input size 一致。默认都是 256。
2. `DriveMoE3` 不是单独新架构，只是 `DriveMoE(num_experts=3)`。
3. `mix_ppo` 与 `opponent_pool` 当前不兼容，代码中会直接报错。
4. 历史 `MIXMOE` 2B 版本没有把 batch 放大 3 倍，因此每个 policy 每次 update 的样本量约为同质训练的 1/3。
5. `MIXMOE3x` 用 batch3x 和 6B steps 才与同质训练的每策略 batch/总样本量近似等价。
6. eval 时必须按 checkpoint 对应 policy class 设置 `planner.ppo.policy-class-name`，例如：
   ```text
   model_policy_1 from MIXMOE -> DriveMoE
   model_policy_2 from MIXMOE -> DriveMoE3
   ```
7. `last_router_weights` 可以用于后续诊断 router 是否学到了有意义的 expert 选择，但历史总结中尚未做 router 行为分析。

## 9. 与 DriveLite 实验的关系

MOE 实验主要回答：

```text
结构异质是否能提升泛化？
```

DriveLite / 混合智能等级实验主要回答：

```text
策略智能等级异质是否能提升泛化？
```

历史结果显示，MOE 结构异质本身没有稳定超过 Drive baseline。这也是后续转向“Drive + LSTM / Drive no LSTM / DriveLite no LSTM”这种可解释智能等级划分的重要动机。
