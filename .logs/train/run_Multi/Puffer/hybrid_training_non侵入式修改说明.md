# 混合训练非侵入式修改说明

## 目标

本次改动借鉴 `pufferdrive_re` 的混智思路，但采用增量接入方式：

- 现有 `puffer train puffer_drive` 的配置、策略类和训练路径保持不变；
- 混智功能通过新策略名 `Hybrid*`、独立配置 `puffer_drive_hybrid` 和新脚本启用；
- 不改变已有 `Drive`、`DriveLite`、`DriveMoE3` 的参数结构、默认 forward 行为或 checkpoint key；
- 不修改 `mix_ppo` 的原有训练语义，只复用已有的多策略加载、冻结和 stratified sampling 能力。

## 智能等级与感知划分

新增策略如下：

| 新策略 | 基础网络 | 感知 | 时序设计 |
|---|---|---|---|
| `HybridDriveLiteLow` | `DriveLite` | 前方 60°、50m | 无 LSTM |
| `HybridDriveMid` | `Drive` | 前方 120°、50m | `Recurrent` |
| `HybridDriveMoE3High` | `DriveMoE3` | 全局感知 | `Recurrent` |
| `HybridDriveOriginal` | `Drive` | 全局感知 | `Recurrent` |

感知 mask 在具体策略自己的 encoder 之前执行。对于 `DriveMoE3`，router 和所有 expert 都接收 mask 后的 observation，避免 router 通过原始输入泄漏高等级感知信息。

mask 是无参数操作，不增加模型参数。High/Original 路径保持原始输入不变。

## 关键代码改动

### 1. 新增策略类

在 `pufferlib/ocean/torch.py` 追加 `_HybridPerceptionMixin` 和四个 `Hybrid*` 策略类。

旧类没有被重写：

```text
Drive
DriveLite
DriveMoE3
Recurrent
```

仍然沿用原来的实现。

### 2. 新增混智配置与环境别名

新增独立配置 `pufferlib/config/ocean/drive_hybrid.ini`：

```ini
env_name = puffer_drive_hybrid
```

环境注册只追加 `drive_hybrid -> Drive`，复用原有 Drive 环境动力学；原来的
`drive.ini` 不增加 alias，避免旧配置解析结果变化。

### 3. 新增 target-only 启动脚本

脚本：

```text
scripts/train_drive_hybrid_target_only.sh
```

默认实验比例：

```text
Original 70%
Low      10%
Mid      10%
High     10%
```

训练语义：

- Original 是唯一可训练 policy；
- Low/Mid/High 通过 checkpoint 加载并冻结；
- Low/Mid/High 只作为周围交通参与者；
- `mix_traffic` 仍由现有环境层控制，未与 PPO policy mix 混淆。

启动前设置：

```bash
export MID_WEIGHTS=/path/to/mid.pt
export HIGH_WEIGHTS=/path/to/high_moe3.pt
bash scripts/train_drive_hybrid_target_only.sh
```

脚本默认将 checkpoint、trainer state 和训练输出写入：

```text
.logs/train/run_Multi/Puffer/
```

也可以通过 `LOG_DIR=/path/to/output` 覆盖输出目录。该目录只影响新增混智
启动脚本，不改变 `drive.ini` 或其他既有训练的 `data_dir`。

如果未设置 `DRIVE_BINARIES_DATA_ROOT`，新增脚本默认使用仓库内的：

```text
pufferlib/resources/drive/binaries
```

这条命令是新增入口；现有训练命令不需要改参数。

## 当前已知边界

1. 目前新增脚本是 target-only 混智入口，不代表已经完成正式 500M 公平预算审计。
2. 如果要严格复现 homogeneous 与 mixed 的目标 own-policy sample budget，需要进一步根据 Original 比例设置总 rollout batch，并审计 `policy_0` 的有效样本数、PPO update 数和 optimizer steps。
3. 当前使用已有 `mix_ppo` 的 policy-ID 分配逻辑。后续若需要按 scene 强制交织、统计 mixed-scene rate，应新增 assignment/audit 功能，不能改变默认 assignment 语义。

## 回退方案

### 方案 A：只停用新功能（推荐）

不启动新脚本即可：

```bash
puffer train puffer_drive
puffer eval puffer_drive --load-model-path <existing-checkpoint>
```

旧路径不依赖 `Hybrid*` 策略，也不依赖 `puffer_drive_hybrid`。

### 方案 B：删除本次增量

在确认没有需要保留的混智实验后，删除新增文件：

```bash
rm -f scripts/train_drive_hybrid_target_only.sh
rm -f tests/test_hybrid_perception.py
rm -f pufferlib/config/ocean/drive_hybrid.ini
rm -f '.logs/train/run_Multi/Puffer/hybrid_training_non侵入式修改说明.md'
```

然后从 `pufferlib/ocean/environment.py` 删除新增的
`"drive_hybrid": "Drive"` 条目。

本次修改没有改变 `Drive`、`DriveLite`、`DriveMoE3` 的原始定义，因此不需要回退这些类，也不应删除已有 checkpoint。

### 方案 C：使用 git 回退

先查看本次改动：

```bash
git diff -- pufferlib/ocean/torch.py pufferlib/ocean/environment.py \
  pufferlib/config/ocean/drive_hybrid.ini \
  scripts/train_drive_hybrid_target_only.sh tests/test_hybrid_perception.py \
  '.logs/train/run_Multi/Puffer/hybrid_training_non侵入式修改说明.md'
```

新增文件可以直接删除；但 `pufferlib/ocean/environment.py` 当前可能包含
其他功能的既有改动，因此不要对它执行整文件 `git restore`。如果确认该文件
没有并行修改，再手动删除本次新增的 `drive_hybrid` 条目。

```bash
git restore pufferlib/ocean/torch.py
rm -f scripts/train_drive_hybrid_target_only.sh
rm -f tests/test_hybrid_perception.py
rm -f pufferlib/config/ocean/drive_hybrid.ini
rm -f '.logs/train/run_Multi/Puffer/hybrid_training_non侵入式修改说明.md'
```

如果工作区存在无关改动，不要使用宽范围 `git restore .`；应只按上面的文件列表回退，避免误伤现有训练和实验文件。

## 回退后验收

```bash
python -c "from pufferlib.pufferl import load_config; print(load_config('puffer_drive')['env_name'])"
python -m unittest tests.test_drive_config -q
python -m unittest tests.test_mix_ppo -q
```

预期：

```text
load_config('puffer_drive')['env_name'] == 'puffer_drive'
```

## 本次实际启动命令（2026-07-24）

### 非侵入审计结论

本次混智改动没有改写既有策略的默认实现：

- `Drive`、`DriveLite`、`DriveMoE3` 原有类体未修改；只在
  `pufferlib/ocean/torch.py` 文件末尾追加 `Hybrid*` 类。
- `puffer_drive` 仍使用原来的 `drive.ini` 和 `Drive`/`Recurrent` 路径。
- `pufferlib/ocean/environment.py` 只追加 `drive_hybrid` 环境入口；旧的
  `drive` 入口不变。
- `pufferlib/config/ocean/drive.ini` 中现有的 `mix_ppo_sampling` 等字段是本次
  之前工作区已有改动；本次已恢复没有追加 alias 或改变 `env_name`。
- `.logs/train/run_repaet/` 在本次启动前后没有被写入；本次所有输出都在
  `.logs/train/run_Multi/Puffer/train/` 下。

因此，原有 `run_repaet` 中的训练和测试仍应使用原命令复现。本次新方法只
通过新的 `Hybrid*` 策略、`drive_hybrid.ini` 和新的输出目录启用。

### 运行环境

```bash
cd /home/fanyuqi/wsc/behavior-bench
export DRIVE_BINARIES_DATA_ROOT=/home/fanyuqi/wsc/behavior-bench/pufferlib/resources/drive/binaries
```

仓库内 training split 包含 10,000 个 binary maps。

### 辅助策略训练命令

Low：窄感知 + DriveLite + 无 LSTM，使用 GPU 1：

```bash
script -q -f /home/fanyuqi/wsc/behavior-bench/.logs/train/run_Multi/Puffer/train/pretrain_low/terminal.log -c \
"env CUDA_VISIBLE_DEVICES=1 DRIVE_BINARIES_DATA_ROOT=$DRIVE_BINARIES_DATA_ROOT \
/home/fanyuqi/miniconda3/envs/behavior-bench/bin/puffer train puffer_drive \
  --train.name hybrid_pretrain_low \
  --train.data-dir /home/fanyuqi/wsc/behavior-bench/.logs/train/run_Multi/Puffer/train/pretrain_low \
  --policy-name HybridDriveLiteLow \
  --rnn-name None \
  --policy.input-size 16 \
  --policy.hidden-size 80 \
  --env.num-maps 10000 \
  --train.total-timesteps 500000000"
```

Mid：中等感知 + Drive + LSTM，使用 GPU 2：

```bash
script -q -f /home/fanyuqi/wsc/behavior-bench/.logs/train/run_Multi/Puffer/train/pretrain_mid/terminal.log -c \
"env CUDA_VISIBLE_DEVICES=2 DRIVE_BINARIES_DATA_ROOT=$DRIVE_BINARIES_DATA_ROOT \
/home/fanyuqi/miniconda3/envs/behavior-bench/bin/puffer train puffer_drive \
  --train.name hybrid_pretrain_mid \
  --train.data-dir /home/fanyuqi/wsc/behavior-bench/.logs/train/run_Multi/Puffer/train/pretrain_mid \
  --policy-name HybridDriveMid \
  --rnn-name Recurrent \
  --policy.input-size 16 \
  --policy.hidden-size 80 \
  --rnn.input-size 80 \
  --rnn.hidden-size 80 \
  --env.num-maps 10000 \
  --train.total-timesteps 500000000"
```

High：全局感知 + DriveMoE3 + LSTM，使用 GPU 3：

```bash
script -q -f /home/fanyuqi/wsc/behavior-bench/.logs/train/run_Multi/Puffer/train/pretrain_high/terminal.log -c \
"env CUDA_VISIBLE_DEVICES=3 DRIVE_BINARIES_DATA_ROOT=$DRIVE_BINARIES_DATA_ROOT \
/home/fanyuqi/miniconda3/envs/behavior-bench/bin/puffer train puffer_drive \
  --train.name hybrid_pretrain_high \
  --train.data-dir /home/fanyuqi/wsc/behavior-bench/.logs/train/run_Multi/Puffer/train/pretrain_high \
  --policy-name HybridDriveMoE3High \
  --rnn-name Recurrent \
  --policy.input-size 16 \
  --policy.hidden-size 80 \
  --rnn.input-size 80 \
  --rnn.hidden-size 80 \
  --env.num-maps 10000 \
  --train.total-timesteps 500000000"
```

每个目录完成后，选择其中最终的 `model_puffer_drive_*.pt` 作为对应的
辅助权重；不要使用旧的 64/256 checkpoint 或没有相同感知定义的 checkpoint。

### Original 目标混合训练命令

辅助权重完成后执行：

```bash
export LOW_WEIGHTS=/home/fanyuqi/wsc/behavior-bench/.logs/train/run_Multi/Puffer/train/pretrain_low/<final-low-checkpoint>.pt
export MID_WEIGHTS=/home/fanyuqi/wsc/behavior-bench/.logs/train/run_Multi/Puffer/train/pretrain_mid/<final-mid-checkpoint>.pt
export HIGH_WEIGHTS=/home/fanyuqi/wsc/behavior-bench/.logs/train/run_Multi/Puffer/train/pretrain_high/<final-high-checkpoint>.pt
export LOG_DIR=/home/fanyuqi/wsc/behavior-bench/.logs/train/run_Multi/Puffer/train/target_original_o70_l10_m10_h10

DRIVE_BINARIES_DATA_ROOT=/home/fanyuqi/wsc/behavior-bench/pufferlib/resources/drive/binaries \
bash scripts/train_drive_hybrid_target_only.sh
```

该目标训练使用：

```text
Original 70%：可训练，Drive + LSTM
Low      10%：冻结，DriveLite + 窄感知，无 LSTM
Mid      10%：冻结，Drive + 中等感知 + LSTM
High     10%：冻结，DriveMoE3 + 全局感知 + LSTM
```

只有 Original 进入 PPO optimizer；辅助策略只控制周围车辆。目标训练启动前
必须先确认三个辅助 checkpoint 都存在并能被严格加载。

### 当前启动状态

本次已经启动三个辅助预训练任务，实时日志位于：

```text
train/pretrain_low/terminal.log
train/pretrain_mid/terminal.log
train/pretrain_high/terminal.log
```

Original 混合训练尚未使用不匹配权重提前启动；待上述三个任务完成后，按上面
命令启动。
