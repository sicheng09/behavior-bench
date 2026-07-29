# Perception 混合智能代码改动与复现指南

本文档总结 run2 感知混智实验的代码改动、训练命令、验证命令和注意事项。实验结果见：

```text
/home/fanyuqi/wsc/behavior-bench/.logs/try/Perception/perception_experiment_summary.md
```

## 1. 改动目标

在当前 `behavior-bench` 中非侵入复现 `pufferdrive_re` 的 low/mid/high 感知混智核心机制：

```text
low  = 前方 60 度扇形，50m
mid  = 前方 120 度扇形，50m
high = 原始全局 observation
```

不修改：

```text
默认 Drive
环境 / C binding
旧训练命令
旧评估命令
```

只新增 perception-masked policy class，并复用当前 `mix_ppo` 的 agent-slot 固定分配、策略隔离训练和 checkpoint 拆分能力。

## 2. 新增策略类

文件：

```text
pufferlib/ocean/torch.py
```

新增：

```text
DrivePerceptionMasked
DrivePerceptionLow
DrivePerceptionMid
DrivePerceptionHigh
```

### 2.1 外部输入不变

所有策略外部 observation 仍为：

```text
obs_dim = 1151
ego = 7
partner = 31 * 8
road = 128 * 7
```

不改变环境输出，不减少真实对象，只在 policy 内部 mask 不可见对象。

### 2.2 mask 逻辑

`DrivePerceptionMasked.encode_observations()`：

```python
masked = self.apply_perception_mask(observations)
return super().encode_observations(masked, state=state)
```

`apply_perception_mask()`：

```text
if high:
  return observations

partner objects:
  visible -> 保留
  invisible -> 整行置零

road objects:
  visible -> 保留
  invisible -> 整行置零

ego:
  永远保留
```

可见性：

```text
normalized_radius = perception_radius * 0.02
dist = sqrt(rel_x^2 + rel_y^2)
angle = atan2(rel_y, rel_x)
visible = non_empty and dist <= normalized_radius and abs(angle) <= sector_degrees / 2
```

默认：

```text
perception_radius = 50.0
normalized_radius = 1.0
```

### 2.3 三个 wrapper

```python
class DrivePerceptionLow(DrivePerceptionMasked):
    perception_level = "low"
    sector_degrees = 60.0

class DrivePerceptionMid(DrivePerceptionMasked):
    perception_level = "mid"
    sector_degrees = 120.0

class DrivePerceptionHigh(DrivePerceptionMasked):
    perception_level = "high"
    sector_degrees = 360.0
```

实际训练中 high 可以直接使用 `Drive`。

## 3. 与 pufferdrive_re 的差异

`pufferdrive_re` 使用：

```text
DriveMixedPolicy
MixedRecurrent
group_ids
repeat_policy_samples
```

当前实现使用：

```text
mix_ppo
policy_id / segment_policy_ids
每个 policy 独立 Recurrent
batch/steps 放大 3 倍
```

对应关系：

| `pufferdrive_re` | 当前实现 |
|---|---|
| `group_id=0/1/2` | `policy_id=0/1/2` |
| `DriveMixedPolicy.level_policies` | `mix_ppo` 的多个 policy |
| `MixedRecurrent` 每 level 一个 LSTM | 每个 policy 自己包 `Recurrent` |
| repeat=3 | 3 倍 batch 和 3 倍 total steps |

当前实现更非侵入，且直接增加 raw rollout，而不是重复同一批样本。

## 4. 训练命令

### 4.1 同质 Low

```bash
puffer train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  --train.name perception_low_homogeneous_2b_batch1x_run2 \
  --policy-name DrivePerceptionLow \
  --rnn-name Recurrent \
  --train.batch-size 524288 \
  --train.total-timesteps 2000000000 \
  --env.num-maps 10000 \
  --env.mix-traffic False \
  --wandb
```

### 4.2 同质 Mid

```bash
puffer train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  --train.name perception_mid_homogeneous_2b_batch1x_run2 \
  --policy-name DrivePerceptionMid \
  --rnn-name Recurrent \
  --train.batch-size 524288 \
  --train.total-timesteps 2000000000 \
  --env.num-maps 10000 \
  --env.mix-traffic False \
  --wandb
```

### 4.3 同质 High

```bash
puffer train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  --train.name perception_high_drive_homogeneous_2b_batch1x_run2 \
  --policy-name Drive \
  --rnn-name Recurrent \
  --train.batch-size 524288 \
  --train.total-timesteps 2000000000 \
  --env.num-maps 10000 \
  --env.mix-traffic False \
  --wandb
```

### 4.4 Mixed Single GPU

```bash
puffer train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  --train.name perception_mixed_low_mid_high_6b_batch3x_run2 \
  --train.batch-size 1572864 \
  --train.total-timesteps 6000000000 \
  --env.num-maps 10000 \
  --env.mix-traffic False \
  --train.mix-ppo True \
  --train.mix-ppo-policy-mix "low:1,mid:1,high:1" \
  --train.mix-ppo-policy-names "DrivePerceptionLow,DrivePerceptionMid,Drive" \
  --train.mix-ppo-rnn-names "Recurrent,Recurrent,Recurrent" \
  --train.mix-ppo-policy-paths ",," \
  --train.mix-ppo-policy-trainable "True,True,True" \
  --wandb
```

### 4.5 Mixed DDP3

DDP3 与单卡 6B/batch3x 全局预算等价：

```text
per-rank batch_size = 524288
per-rank total_timesteps = 2000000000
global batch ~= 1572864
global steps ~= 6000000000
```

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=3 \
  -m pufferlib.pufferl train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  --train.name perception_mixed_low_mid_high_ddp3_global6b_run2 \
  --train.batch-size 524288 \
  --train.total-timesteps 2000000000 \
  --env.num-maps 10000 \
  --env.mix-traffic False \
  --train.mix-ppo True \
  --train.mix-ppo-policy-mix "low:1,mid:1,high:1" \
  --train.mix-ppo-policy-names "DrivePerceptionLow,DrivePerceptionMid,Drive" \
  --train.mix-ppo-rnn-names "Recurrent,Recurrent,Recurrent" \
  --train.mix-ppo-policy-paths ",," \
  --train.mix-ppo-policy-trainable "True,True,True" \
  --wandb
```

## 5. 验证命令

所有验证使用：

```text
split = pufferinter
traffic = idm
map_ids = all
```

### 5.1 Low

```bash
python pufferlib/ocean/benchmark/eval.py \
  --output-dir .logs/val/run2/results/<run>/DrivePerceptionLow_Recurrent \
  --planner.type ppo \
  --planner.ppo.weights-path /path/to/model_policy_0_puffer_drive_003815.pt \
  --planner.ppo.device cuda \
  --planner.ppo.policy-class-name DrivePerceptionLow \
  --planner.ppo.input-size 64 \
  --planner.ppo.hidden-size 256 \
  --planner.ppo.rnn-name Recurrent \
  --planner.ppo.rnn-input-size 256 \
  --planner.ppo.rnn-hidden-size 256 \
  --traffic.type idm \
  --eval.split pufferinter \
  --map-ids all
```

### 5.2 Mid

```bash
python pufferlib/ocean/benchmark/eval.py \
  --output-dir .logs/val/run2/results/<run>/DrivePerceptionMid_Recurrent \
  --planner.type ppo \
  --planner.ppo.weights-path /path/to/model_policy_1_puffer_drive_003815.pt \
  --planner.ppo.device cuda \
  --planner.ppo.policy-class-name DrivePerceptionMid \
  --planner.ppo.input-size 64 \
  --planner.ppo.hidden-size 256 \
  --planner.ppo.rnn-name Recurrent \
  --planner.ppo.rnn-input-size 256 \
  --planner.ppo.rnn-hidden-size 256 \
  --traffic.type idm \
  --eval.split pufferinter \
  --map-ids all
```

### 5.3 High

```bash
python pufferlib/ocean/benchmark/eval.py \
  --output-dir .logs/val/run2/results/<run>/Drive_Recurrent \
  --planner.type ppo \
  --planner.ppo.weights-path /path/to/model_policy_2_puffer_drive_003815.pt \
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

## 6. 自动验证脚本

run2 使用过这些脚本：

```text
.logs/val/run2/run_one_policy_eval.sh
.logs/val/run2/run_one_policy_eval_1000.sh
.logs/val/run2/wait_and_eval_run2_homo_first.sh
.logs/val/run2/wait_and_eval_run2_1000_homo_first.sh
.logs/val/run2/wait_and_eval_run2_ddp3.sh
.logs/val/run2/wait_and_eval_run2_ddp3_1000.sh
```

输出：

```text
.logs/val/run2/results
.logs/val/run2/results1000
```

## 7. 测试与 smoke checks

新增 tests：

```text
tests.test_mix_ppo.TestMixPPO.test_drive_perception_low_masks_objects_outside_front_sector
tests.test_mix_ppo.TestMixPPO.test_drive_perception_mid_keeps_wider_front_sector_than_low
tests.test_mix_ppo.TestMixPPO.test_mix_ppo_loads_perception_level_policies
```

运行：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench
python -m unittest \
  tests.test_mix_ppo.TestMixPPO.test_drive_perception_low_masks_objects_outside_front_sector \
  tests.test_mix_ppo.TestMixPPO.test_drive_perception_mid_keeps_wider_front_sector_than_low \
  tests.test_mix_ppo.TestMixPPO.test_mix_ppo_loads_perception_level_policies
```

结果：

```text
Ran 3 tests in 0.034s
OK
```

也跑过 tiny mixed training smoke：

```text
['DrivePerceptionLow', 'DrivePerceptionMid', 'Drive']
perception mix tiny train smoke ok
```

## 8. 注意事项

1. `DrivePerceptionHigh` 与 `Drive` 感知等价，实验中 high 直接用 `Drive`。
2. low/mid/high 都是完整 `Drive + Recurrent`，差异只在 observation mask。
3. mask 是 policy 内部行为，不影响环境真实对象。
4. 当前实现使用 `mix_ppo` 固定 agent slot 分配，不是 pufferdrive_re 的 `DriveMixedPolicy` 内部 group routing。
5. 当前样本公平通过 3 倍 batch/steps 实现，不使用 repeat 同一批样本。
6. DDP3 的 per-rank batch/steps 要按同质设置，三卡合计才与单卡 batch3x 等价。
