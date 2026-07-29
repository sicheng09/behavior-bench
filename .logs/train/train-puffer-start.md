# BehaviorBench PufferDrive 训练启动指南

本文档面向“环境已配置、数据已下载，准备通过 SSH 开始训练”的场景。建议先按第一节的标准模板跑一次小规模训练；后续章节解释每个参数、W&B、GPU、日志、评估和常见问题。

配置来源以当前仓库为准：

- PPO / PufferDrive 主配置：`pufferlib/config/ocean/drive.ini`
- 全局默认配置：`pufferlib/config/default.ini`
- SMART 预测模型配置：`pufferlib/config/prediction/smart.ini`
- CLI 入口：`pyproject.toml` 中的 `puffer = pufferlib.pufferl:main`

## SSH / tmux 标准训练流程

训练通常会跑很久，SSH 断开会导致普通前台进程退出。推荐每次训练都在 `tmux` 里启动，并把输出保存到 `.logs/train/runs/`。

### 1. 新建 tmux 会话

```bash
ssh <user>@<server>
tmux new -s puffer-train
```

如果会话已经存在：

```bash
tmux attach -t puffer-train
```

如果确认要删除旧会话：

```bash
tmux kill-session -t puffer-train
```

### 2. 标准小规模训练模板

先复制这一整段，在 `参数填写区` 改 GPU、run 名、训练步数、地图数等参数。这个模板默认跑小规模 PPO 训练，适合第一次验证训练链路。

```bash
cd "$HOME/wsc/behavior-bench"
conda activate behavior-bench
# 如果你用 uv 环境，改成：
# source .venv/bin/activate

# ===== 路径与数据 =====
export REPO_ROOT="$(pwd)"
export DRIVE_DATA_ROOT="$REPO_ROOT/data/processed"
export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/resources/drive/binaries"
export LOG_DIR="$REPO_ROOT/.logs/train/runs"
mkdir -p "$LOG_DIR"

# ===== 参数填写区：按需修改 =====
export GPU_IDS="0"                         # 单卡例子："0"；多卡例子："0,1,2,3"
export RUN_NAME="ppo_drive_small_5m"
export CONFIG_PATH="pufferlib/config/ocean/drive.ini"
export TOTAL_TIMESTEPS="5000000"
export ENV_NUM_MAPS="1000"
export VEC_NUM_WORKERS="4"
export VEC_NUM_ENVS="4"
export TRAIN_BATCH_SIZE="131072"
export TRAIN_MINIBATCH_SIZE="8192"
export TRAIN_MAX_MINIBATCH_SIZE="8192"
export TRAIN_CHECKPOINT_INTERVAL="20"

# 训练内评估开关：当前代码的训练内评估子进程参数不完整，保持关闭；
# 训练完成后按“评估训练结果”章节单独运行 puffer eval 或 benchmark eval。
export EVAL_SPLIT="validation"
export EVAL_NUM_MAPS="20"
export ENABLE_WOSAC_REALISM="False"
export ENABLE_HUMAN_REPLAY="False"

# W&B：如果不想上传，删除训练命令里的 --wandb 三行即可
export WANDB_PROJECT="behavior-bench"
export WANDB_GROUP="ppo-small"
# export WANDB_ENTITY="<your-wandb-entity>"   # 可选：上传到指定个人账号或团队

# ===== 日志与 GPU =====
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TRAIN_LOG="$LOG_DIR/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
set -o pipefail

# ===== 启动训练 =====
puffer train puffer_drive \
  --config "$CONFIG_PATH" \
  --train.name "$RUN_NAME" \
  --train.total-timesteps "$TOTAL_TIMESTEPS" \
  --env.num-maps "$ENV_NUM_MAPS" \
  --vec.num-workers "$VEC_NUM_WORKERS" \
  --vec.num-envs "$VEC_NUM_ENVS" \
  --train.batch-size "$TRAIN_BATCH_SIZE" \
  --train.minibatch-size "$TRAIN_MINIBATCH_SIZE" \
  --train.max-minibatch-size "$TRAIN_MAX_MINIBATCH_SIZE" \
  --train.checkpoint-interval "$TRAIN_CHECKPOINT_INTERVAL" \
  --eval.split "$EVAL_SPLIT" \
  --eval.num-maps "$EVAL_NUM_MAPS" \
  --eval.wosac-realism-eval "$ENABLE_WOSAC_REALISM" \
  --eval.human-replay-eval "$ENABLE_HUMAN_REPLAY" \
  --wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-group "$WANDB_GROUP" \
  2>&1 | tee -a "$TRAIN_LOG"
```

### 3. 正式默认训练模板

小规模训练确认正常后，可以使用默认 `drive.ini` 的正式配置。默认训练规模是 `2_000_000_000` timesteps、`79000` maps、`batch_size=524288`。

```bash
cd "$HOME/wsc/behavior-bench"
conda activate behavior-bench

export REPO_ROOT="$(pwd)"
export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/resources/drive/binaries"
export LOG_DIR="$REPO_ROOT/.logs/train/runs"
mkdir -p "$LOG_DIR"

export GPU_IDS="0"
export RUN_NAME="ppo_drive_default_2b"
export CONFIG_PATH="pufferlib/config/ocean/drive.ini"
export ENV_NUM_MAPS="10000"                 # 必须 <= 当前 split 下实际 map_*.bin 数量；完整默认配置是 79000
export EVAL_SPLIT="validation"
export EVAL_NUM_MAPS="20"
# 训练内评估保持关闭：当前 utils.py 会给 WOSAC 子进程传不存在的
# --eval.wosac-num-agents，并访问缺失的 human_replay_num_agents。
# 需要评估时，训练结束后单独运行下方“评估训练结果”命令。
export ENABLE_WOSAC_REALISM="False"
export ENABLE_HUMAN_REPLAY="False"
export WANDB_PROJECT="behavior-bench"
export WANDB_GROUP="ppo-default"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TRAIN_LOG="$LOG_DIR/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
set -o pipefail

puffer train puffer_drive \
  --config "$CONFIG_PATH" \
  --train.name "$RUN_NAME" \
  --env.num-maps "$ENV_NUM_MAPS" \
  --eval.split "$EVAL_SPLIT" \
  --eval.num-maps "$EVAL_NUM_MAPS" \
  --eval.wosac-realism-eval "$ENABLE_WOSAC_REALISM" \
  --eval.human-replay-eval "$ENABLE_HUMAN_REPLAY" \
  --wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-group "$WANDB_GROUP" \
  2>&1 | tee -a "$TRAIN_LOG"
```

### 4. 指定策略网络训练模板

默认 `drive.ini` 使用：

```ini
policy_name = Drive
rnn_name = Recurrent
```

如果要指定其他 PPO-like policy，例如新添加的 `DriveMoE`，在训练命令中显式传入 `--policy-name` 和 `--rnn-name`。只要该策略定义在 `pufferlib/ocean/torch.py` 中，并且输出 embedding 维度与 `[rnn] input_size` 对齐，就可以这样启动。

下面示例启动 `DriveMoE + Recurrent` 同质 PPO 训练：

```bash
cd "$HOME/wsc/behavior-bench"
conda activate behavior-bench

export REPO_ROOT="$(pwd)"
export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/resources/drive/binaries"
export LOG_DIR="$REPO_ROOT/.logs/train/runs"
mkdir -p "$LOG_DIR"

export GPU_IDS="1"
export RUN_NAME="ppo_drive_moe_2b"
export CONFIG_PATH="pufferlib/config/ocean/drive.ini"
export POLICY_NAME="DriveMoE"
export RNN_NAME="Recurrent"
export ENV_NUM_MAPS="10000"
export EVAL_SPLIT="validation"
export EVAL_NUM_MAPS="20"
export ENABLE_WOSAC_REALISM="False"
export ENABLE_HUMAN_REPLAY="False"
export WANDB_PROJECT="behavior-bench"
export WANDB_GROUP="ppo-moe"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TRAIN_LOG="$LOG_DIR/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
set -o pipefail

puffer train puffer_drive \
  --config "$CONFIG_PATH" \
  --policy-name "$POLICY_NAME" \
  --rnn-name "$RNN_NAME" \
  --train.name "$RUN_NAME" \
  --env.num-maps "$ENV_NUM_MAPS" \
  --eval.split "$EVAL_SPLIT" \
  --eval.num-maps "$EVAL_NUM_MAPS" \
  --eval.wosac-realism-eval "$ENABLE_WOSAC_REALISM" \
  --eval.human-replay-eval "$ENABLE_HUMAN_REPLAY" \
  --wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-group "$WANDB_GROUP" \
  2>&1 | tee -a "$TRAIN_LOG"
```

可替换策略示例：

```text
Drive
DriveNoGoal
DriveTransformer
DriveGameFormer
DriveMoE
DriveLatentWorldModel
```

注意：

- `DriveMoE + Recurrent` 默认参数量约 `752.4K`，应明显高于 `Drive + Recurrent` 的 `614.2K`。
- 如果使用 `DrivePaper` / `DriveConditionedPaper` 这类输出 512 维 hidden 的策略，默认 `rnn.input_size=256` 不兼容，需要单独调整 RNN 配置或关闭 `rnn_name`。
- conditioned 系列策略需要同步开启 `--env.reward-conditioning True`。

### 5. 多卡 torchrun 模板

多卡训练时，`CUDA_VISIBLE_DEVICES` 里有几张卡，`--nproc-per-node` 就设成几。

```bash
cd "$HOME/wsc/behavior-bench"
conda activate behavior-bench

export REPO_ROOT="$(pwd)"
export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/resources/drive/binaries"
export LOG_DIR="$REPO_ROOT/.logs/train/runs"
mkdir -p "$LOG_DIR"

export GPU_IDS="0,1,2,3"
export NPROC_PER_NODE="4"
export RUN_NAME="ppo_drive_ddp_4gpu"
export CONFIG_PATH="pufferlib/config/ocean/drive.ini"
export ENV_NUM_MAPS="10000"                 # 必须 <= 当前 split 下实际 map_*.bin 数量
export WANDB_PROJECT="behavior-bench"
export WANDB_GROUP="ppo-ddp"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TRAIN_LOG="$LOG_DIR/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
set -o pipefail

torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC_PER_NODE" \
  -m pufferlib.pufferl train puffer_drive \
  --config "$CONFIG_PATH" \
  --train.name "$RUN_NAME" \
  --env.num-maps "$ENV_NUM_MAPS" \
  --wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-group "$WANDB_GROUP" \
  2>&1 | tee -a "$TRAIN_LOG"
```

### 6. tmux 查看和离开

训练运行中临时离开：

```text
Ctrl-b 然后按 d
```

重新进入：

```bash
tmux attach -t puffer-train
```

查看历史输出：

```text
Ctrl-b 然后按 [
```

进入滚动模式后，用方向键、`PageUp`、`PageDown` 滚动，按 `q` 退出。

实时查看训练日志：

```bash
tail -f "$TRAIN_LOG"
```

查看所有训练日志：

```bash
ls -lh .logs/train/runs
```

常用窗口操作：

```text
Ctrl-b 然后按 c    新建窗口
Ctrl-b 然后按 n    下一个窗口
Ctrl-b 然后按 p    上一个窗口
Ctrl-b 然后按 ,    重命名当前窗口
exit              退出当前窗口 shell
```

建议一个 `tmux` 会话里至少保留两个窗口：一个跑训练，一个查看 `nvidia-smi`、日志或数据状态。

## 启动前检查

### 环境和数据

进入仓库根目录并激活环境：

```bash
cd "$HOME/wsc/behavior-bench"
conda activate behavior-bench
# 或者：source .venv/bin/activate
```

设置数据路径：

```bash
export REPO_ROOT="$(pwd)"
export DRIVE_DATA_ROOT="$REPO_ROOT/data/processed"
export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/resources/drive/binaries"
```

`DRIVE_BINARIES_DATA_ROOT` 必须指向 split 的上一级目录，训练时会读取：

```text
$DRIVE_BINARIES_DATA_ROOT/training/map_000000.bin
$DRIVE_BINARIES_DATA_ROOT/validation/map_000000.bin
```

检查数据：

```bash
python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["DRIVE_BINARIES_DATA_ROOT"])
for split in ["training", "validation", "pufferhard", "pufferinter"]:
    path = root / split
    if path.exists():
        n = len(list(path.glob("map_*.bin")))
        print(f"{split}: {n} bin files at {path}")
    else:
        print(f"{split}: missing ({path})")
PY
```

如果 C 扩展或本地 binding 有问题，重新构建：

```bash
python setup.py build_ext --inplace --force
```

### CPU smoke test

先跑仓库内置的端到端测试，确认包、C 扩展、数据路径、环境加载和训练循环都能工作：

```bash
python tests/test_drive_train.py
```

它会在 CPU 上使用极小配置运行到约 50K global steps。它不是正式训练，只用于排错。

## 参数填写区说明

### 路径参数

| 变量 | 作用 |
| --- | --- |
| `REPO_ROOT` | 当前仓库根目录，用 `pwd` 动态生成，避免写个人绝对路径 |
| `DRIVE_DATA_ROOT` | 原始/处理后数据目录，主要用于转换流程 |
| `DRIVE_BINARIES_DATA_ROOT` | 训练和评估读取 `.bin` 的根目录 |
| `LOG_DIR` | 训练终端日志保存目录，建议固定为 `.logs/train/runs` |
| `TRAIN_LOG` | 当前 run 的终端日志文件 |

### 训练规模参数

| 变量 / CLI | 作用 | 小规模建议 | 默认正式值 |
| --- | --- | --- | --- |
| `RUN_NAME` / `--train.name` | 实验名，也会用于 W&B run name | `ppo_drive_small_5m` | 自定义 |
| `CONFIG_PATH` / `--config` | 额外加载的训练配置文件 | `pufferlib/config/ocean/drive.ini` | 同左 |
| `TOTAL_TIMESTEPS` / `--train.total-timesteps` | 训练总环境步数 | `5000000` | `2_000_000_000` |
| `ENV_NUM_MAPS` / `--env.num-maps` | 加载地图数 | `1000` 或更少 | `79000` |
| `VEC_NUM_WORKERS` / `--vec.num-workers` | 并行 worker 数 | `4` | `16` |
| `VEC_NUM_ENVS` / `--vec.num-envs` | 每个 worker 的 env 数 | `4` | `16` |
| `TRAIN_BATCH_SIZE` / `--train.batch-size` | 每次 PPO update 的 transition 数 | `131072` | `524288` |
| `TRAIN_MINIBATCH_SIZE` | 梯度更新 minibatch | `8192` | `32768` |
| `TRAIN_MAX_MINIBATCH_SIZE` | 梯度累积前单次最大 minibatch | `8192` | `32768` |
| `TRAIN_CHECKPOINT_INTERVAL` | checkpoint 间隔 | `20` | `1000` |
| `EVAL_SPLIT` / `--eval.split` | 训练内评估使用的数据 split | `validation` | `training` |
| `EVAL_NUM_MAPS` / `--eval.num-maps` | 每次训练内评估使用的地图数 | `20` | `20` |
| `ENABLE_WOSAC_REALISM` / `--eval.wosac-realism-eval` | 训练内 WOSAC realism 开关；当前代码中请保持 `False`，训练后单独评估 | `False` | `False` |
| `ENABLE_HUMAN_REPLAY` / `--eval.human-replay-eval` | 训练内 human replay 开关；当前代码中请保持 `False`，训练后单独评估 | `False` | `False` |

当前默认 batch 关系：

```text
batch_size = num_agents * num_workers * bptt_horizon
524288 = 1024 * 16 * 32
```

修改 batch 时注意：

- `minibatch_size` 必须能被 `bptt_horizon` 整除。
- `batch_size` 必须大于等于 `minibatch_size`。
- OOM 时优先降 `minibatch_size` / `max_minibatch_size`，再降 `batch_size`、`vec.num-workers` 或 `vec.num-envs`。

## GPU 选择

先查看机器上的 GPU 编号和占用：

```bash
nvidia-smi
```

只用物理 GPU 3：

```bash
export CUDA_VISIBLE_DEVICES="3"
puffer train puffer_drive --config pufferlib/config/ocean/drive.ini
```

注意：设置 `CUDA_VISIBLE_DEVICES=3` 后，程序内部看到的是 `cuda:0`，但实际使用的是物理 GPU 3。这是 CUDA 的正常重新编号。

使用物理 GPU 2、3、4、5 做 DDP：

```bash
export CUDA_VISIBLE_DEVICES="2,3,4,5"
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  -m pufferlib.pufferl train puffer_drive \
  --config pufferlib/config/ocean/drive.ini
```

同一台服务器上并行跑多个实验时，建议每个实验一个 `tmux` 会话，并指定不同 GPU：

```bash
tmux new -s puffer-gpu0
CUDA_VISIBLE_DEVICES=0 puffer train puffer_drive --train.name exp_gpu0

tmux new -s puffer-gpu1
CUDA_VISIBLE_DEVICES=1 puffer train puffer_drive --train.name exp_gpu1
```

## W&B 设置

只要训练命令带 `--wandb`，W&B 就需要 API key。

### 使用当前机器的全局登录账号

如果你愿意把当前 shell 默认账号设成某个 W&B 账号：

```bash
wandb login
wandb status
```

这种方式会改变当前用户的默认 W&B 登录状态，多人共用服务器时不一定合适。

### 不改变全局登录，临时上传到自己的账号

如果不想改变已有的 `wandb login` 账号，推荐在当前 `tmux` 会话里临时设置环境变量。不要把 API key 写进 Markdown、脚本或 git 文件。

```bash
read -s WANDB_API_KEY
export WANDB_API_KEY
export WANDB_ENTITY="<your-wandb-entity>"
```

执行 `read -s WANDB_API_KEY` 后，粘贴你自己的 W&B API key 并回车。`-s` 会隐藏输入，避免 key 出现在屏幕和 shell 历史里。

含义：

- `WANDB_API_KEY` 决定用哪个账号的权限上传。
- `WANDB_ENTITY` 决定 run 上传到哪个个人账号或团队空间，通常填你的 W&B username 或 team entity。
- `--wandb-project behavior-bench` 决定项目名。
- `--wandb-group ppo-small` 决定 run 分组。
- 当前仓库训练入口有 `--wandb-project` 和 `--wandb-group`，没有单独的 `--wandb-entity` 参数，因此 entity 用环境变量指定。

如果先不上传，删除训练命令里的 `--wandb`、`--wandb-project`、`--wandb-group`。也可以离线记录：

```bash
export WANDB_MODE=offline
```

之后上传离线 run：

```bash
wandb sync <offline-run-dir>
```

## 日志保存

标准训练模板会保存终端输出到：

```text
.logs/train/runs/<RUN_NAME>_<YYYYMMDD_HHMMSS>.log
```

相关命令：

```bash
export LOG_DIR="$REPO_ROOT/.logs/train/runs"
export TRAIN_LOG="$LOG_DIR/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"
set -o pipefail

puffer train puffer_drive ... 2>&1 | tee -a "$TRAIN_LOG"
```

说明：

- `2>&1` 会把 stderr 也写进日志。
- `tee -a` 会同时显示终端输出并追加保存到日志文件。
- `set -o pipefail` 避免训练命令失败时被 `tee` 掩盖。
- checkpoint 和训练状态默认仍写在 `experiments/` 下，`.logs/train/runs/` 保存的是终端日志。

## PPO 默认配置

配置加载优先级：

```text
CLI 参数 > --config 指定的 ini > env_name 对应 ini > pufferlib/config/default.ini
```

CLI 覆盖规则是 `--section.key value`，配置里的下划线在命令行里变成短横线：

```bash
puffer train puffer_drive \
  --train.learning-rate 0.001 \
  --env.num-maps 1000 \
  --vec.num-workers 8
```

### `[base]`

| 参数 | 当前默认值 |
| --- | --- |
| `env_name` | `puffer_drive` |
| `policy_name` | `Drive` |
| `rnn_name` | `Recurrent` |

### `[env]`

| 参数 | 当前默认值 | 说明 |
| --- | --- | --- |
| `num_agents` | `1024` | 并行控制/仿真的 agent 数 |
| `action_type` | `discrete` | 动作空间，另可设 `continuous` |
| `dynamics_model` | `classic` | 动力学模型，另可设 `jerk` |
| `dt` | `0.1` | 仿真步长，单位秒 |
| `episode_length` | `91` | 每个场景 episode 长度 |
| `split` | `training` | 默认读取 `$DRIVE_BINARIES_DATA_ROOT/training` |
| `num_maps` | `79000` | 默认加载地图数 |
| `init_steps` | `0` | 初始 warmup 步 |
| `control_mode` | `control_vehicles` | 控制车辆实体 |
| `init_mode` | `create_all_valid` | 初始化所有 valid agent |
| `goal_behavior` | `3` | 到达目标后 remove |
| `goal_radius` | `2.0` | 到达目标判定半径 |
| `goal_speed` | `100.0` | 目标速度上限 |
| `collision_behavior` | `2` | 碰撞后 remove |
| `offroad_behavior` | `2` | 出路后 remove |
| `reward_vehicle_collision` | `-0.5` | 车辆碰撞惩罚 |
| `reward_offroad_collision` | `-0.5` | 出路惩罚 |
| `reward_goal` | `1.0` | 到达目标奖励 |
| `mix_traffic` | `False` | 是否混合 PPO / IDM / Expert 交通 |
| `ppo_fraction` | `1.0` | 混合交通中 PPO 比例 |
| `idm_fraction` | `0.0` | 混合交通中 IDM 比例 |
| `expert_fraction` | `0.0` | 混合交通中 expert replay 比例 |

常用 `split` 包括 `training`、`validation`、`testing`、`pufferhard`、`pufferinter`、`nuplan_test14`。

### `[vec]`

| 参数 | 当前默认值 |
| --- | --- |
| `backend` | `Multiprocessing`，来自 `default.ini` |
| `num_workers` | `16` |
| `num_envs` | `16` |
| `batch_size` | `4` |
| `zero_copy` | `True`，来自 `default.ini` |

### `[train]`

| 参数 | 当前默认值 | 来源 |
| --- | --- | --- |
| `device` | `cuda` | `default.ini` |
| `optimizer` | `muon` | `default.ini` |
| `precision` | `float32` | `default.ini` |
| `seed` | `42` | `drive.ini` / `default.ini` |
| `total_timesteps` | `2_000_000_000` | `drive.ini` |
| `learning_rate` | `0.003` | `drive.ini` |
| `anneal_lr` | `True` | `drive.ini` |
| `batch_size` | `524288` | `drive.ini` |
| `minibatch_size` | `32768` | `drive.ini` |
| `max_minibatch_size` | `32768` | `drive.ini` |
| `bptt_horizon` | `32` | `drive.ini` |
| `update_epochs` | `1` | `drive.ini` |
| `gamma` | `0.98` | `drive.ini` |
| `gae_lambda` | `0.95` | `drive.ini` |
| `clip_coef` | `0.2` | `drive.ini` |
| `vf_coef` | `2` | `drive.ini` |
| `vf_clip_coef` | `0.2` | `drive.ini` |
| `ent_coef` | `0.005` | `drive.ini` |
| `max_grad_norm` | `1` | `drive.ini` |
| `checkpoint_interval` | `1000` | `drive.ini` |
| `data_dir` | `experiments` | `default.ini` |
| `render` | `False` | `drive.ini` |
| `kl_coef` | `0.1` | `drive.ini` |
| `opponent_pool` | `False` | `drive.ini` |

### `[policy]` / `[rnn]`

| 段 | 参数 | 当前默认值 |
| --- | --- | --- |
| `[policy]` | `input_size` | `64` |
| `[policy]` | `hidden_size` | `256` |
| `[rnn]` | `input_size` | `256` |
| `[rnn]` | `hidden_size` | `256` |

### `[eval]`

| 参数 | 当前默认值 |
| --- | --- |
| `eval_interval` | `1000` |
| `split` | `training` |
| `num_maps` | `20` |
| `backend` | `PufferEnv` |
| `wosac_realism_eval` | `False` |
| `wosac_num_rollouts` | `32` |
| `wosac_init_steps` | `10` |
| `human_replay_eval` | `False` |

当前代码中不要在 `puffer train` 命令里开启 WOSAC realism 或 Human replay：

```bash
puffer train puffer_drive \
  --eval.wosac-realism-eval False \
  --eval.human-replay-eval False
```

原因是训练循环会通过 `pufferlib/utils.py` 启动评估子进程；当前实现会给 WOSAC 子进程传入 parser 不支持的 `--eval.wosac-num-agents`，并且 Human replay 子进程会读取配置中不存在的 `human_replay_num_agents`。在不改代码的前提下，训练命令应保持这两个开关为 `False`，确保传入训练入口和评估子进程的参数都是当前版本支持的。

推荐流程：

- 小规模 smoke / 调参阶段：保持 `False`。
- 正式训练：也保持 `False`，避免训练结束时触发评估子进程报错。
- 需要完整评估报告时：训练后单独用 `puffer eval` 或 benchmark eval 框架跑，避免拖慢训练主循环，也避免训练内子进程参数不兼容。

## 常用训练变体

### 调小机器资源

```bash
puffer train puffer_drive \
  --env.num-maps 1000 \
  --vec.num-workers 4 \
  --vec.num-envs 4 \
  --train.batch-size 131072 \
  --train.minibatch-size 8192 \
  --train.max-minibatch-size 8192
```

### 只用小数据快速迭代

```bash
puffer train puffer_drive \
  --train.total-timesteps 5000000 \
  --env.num-maps 100 \
  --train.checkpoint-interval 20
```

### 切换数据 split

```bash
puffer train puffer_drive \
  --env.split validation \
  --env.num-maps 1000
```

### 使用 jerk dynamics

```bash
puffer train puffer_drive \
  --env.dynamics-model jerk \
  --env.action-type discrete
```

### 混合交通训练

```bash
puffer train puffer_drive \
  --env.mix-traffic True \
  --env.ppo-fraction 0.5 \
  --env.idm-fraction 0.5 \
  --env.expert-fraction 0.0
```

## 从 checkpoint 恢复

```bash
export REPO_ROOT="$(pwd)"
export LOG_DIR="$REPO_ROOT/.logs/train/runs"
export RUN_NAME="ppo_drive_resume"
export TRAIN_LOG="$LOG_DIR/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"
set -o pipefail

puffer train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  --load-model-path experiments/puffer_drive_<run_id>/model_puffer_drive_<step>.pt \
  --train.name "$RUN_NAME" \
  --wandb \
  --wandb-project behavior-bench \
  --wandb-group resume \
  2>&1 | tee -a "$TRAIN_LOG"
```

## 评估训练结果

### PufferLib 内置 eval

```bash
puffer eval puffer_drive \
  --load-model-path experiments/puffer_drive_<run_id>/model_puffer_drive_<step>.pt
```

WOSAC realism：

```bash
puffer eval puffer_drive \
  --load-model-path experiments/puffer_drive_<run_id>/model_puffer_drive_<step>.pt \
  --eval.wosac-realism-eval True \
  --eval.human-replay-eval False \
  --eval.split validation \
  --eval.num-maps 20
```

Human replay：

```bash
puffer eval puffer_drive \
  --load-model-path experiments/puffer_drive_<run_id>/model_puffer_drive_<step>.pt \
  --eval.wosac-realism-eval False \
  --eval.human-replay-eval True \
  --eval.split validation \
  --eval.num-maps 20
```

### Benchmark eval 框架

默认 PDM ego vs IDM traffic：

```bash
python pufferlib/ocean/benchmark/eval.py --map-ids 0-10
```

PPO ego vs IDM traffic：

```bash
python pufferlib/ocean/benchmark/eval.py \
  --planner.type ppo \
  --planner.ppo.weights-path experiments/puffer_drive_<run_id>/model_puffer_drive_<step>.pt \
  --traffic.type idm \
  --eval.split pufferhard \
  --map-ids 0-50
```

评估框架默认读取 `pufferlib/config/evaluation.ini`，结果写入 `experiments/<timestamp>_<uuid>/`。

## SMART 预测模型训练

如果目标是训练 SMART 轨迹预测模型，而不是 PPO policy，需要先构建 prediction cache。

### 构建 cache

```bash
python scripts/build_prediction_cache.py \
  --config pufferlib/config/prediction/smart.ini \
  --splits training validation \
  --num-workers 32
```

小规模 overfit / 调试：

```bash
python scripts/build_prediction_cache.py \
  --config pufferlib/config/prediction/smart_overfit.ini \
  --splits training \
  --num-workers 4
```

### 单卡训练

```bash
python -m pufferlib.prediction.puffer_prediction pretrain \
  --config pufferlib/config/prediction/smart.ini
```

### 多卡训练

```bash
torchrun --nproc_per_node=8 \
  -m pufferlib.prediction.puffer_prediction pretrain \
  --config pufferlib/config/prediction/smart.ini
```

### SMART 默认参数

| 段 | 参数 | 当前默认值 |
| --- | --- | --- |
| `[data]` | `train_split` | `training` |
| `[data]` | `val_split` | `validation` |
| `[data]` | `batch_size` | `4` |
| `[data]` | `num_workers` | `4` |
| `[data]` | `num_historical_steps` | `11` |
| `[data]` | `num_future_steps` | `80` |
| `[data]` | `shift` | `5` |
| `[model]` | `hidden_dim` | `128` |
| `[model]` | `num_actions` | `2048` |
| `[model]` | `num_agent_layers` | `6` |
| `[model]` | `num_heads` | `8` |
| `[model]` | `dropout` | `0.1` |
| `[train]` | `device` | `cuda` |
| `[train]` | `max_epochs` | `100` |
| `[train]` | `learning_rate` | `5e-4` |
| `[train]` | `weight_decay` | `0.01` |
| `[train]` | `max_grad_norm` | `0.5` |
| `[train]` | `checkpoint_interval` | `5` |
| `[train]` | `checkpoint_dir` | `experiments/prediction/checkpoints` |

训练后可用 realism eval：

```bash
python pufferlib/ocean/benchmark/eval_realism.py \
  --planner.type smart \
  --planner.smart.weights-path experiments/prediction/checkpoints/epoch_100.pt \
  --map-ids 0-228
```

## 超参搜索和对照实验

`drive.ini` 已定义 sweep 空间：

| 参数 | 分布 | min | mean | max |
| --- | --- | --- | --- | --- |
| `train.learning_rate` | `log_normal` | `0.001` | `0.003` | `0.005` |
| `train.ent_coef` | `log_normal` | `0.001` | `0.005` | `0.03` |
| `train.gamma` | `log_normal` | `0.97` | `0.98` | `0.999` |
| `train.gae_lambda` | `log_normal` | `0.95` | `0.98` | `0.999` |

运行 sweep 需要 W&B 或 Neptune：

```bash
puffer sweep puffer_drive \
  --wandb \
  --wandb-project behavior-bench \
  --wandb-group sweep \
  --max-runs 20
```

对照实验当前配置包含：

```text
controlled_exp.train.goal_speed = [10, 20, 30, 3]
controlled_exp.train.ent_coef = [0.001, 0.005, 0.01]
controlled_exp.train.seed = [42, 55, 1]
```

运行：

```bash
puffer controlled_exp puffer_drive \
  --wandb \
  --wandb-project behavior-bench \
  --wandb-group controlled
```

## 常见问题

### `DRIVE_BINARIES_DATA_ROOT is not set`

设置环境变量，并确认其下存在 split 子目录：

```bash
export REPO_ROOT="$(pwd)"
export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/resources/drive/binaries"
```

### 找不到 split 目录

报错类似：

```text
Split directory .../training not found
```

说明 `DRIVE_BINARIES_DATA_ROOT` 指错了。它应该指向包含 `training/`、`validation/` 等子目录的根目录，而不是某个 split 本身。

### `num_maps exceeds available maps`

报错类似：

```text
ValueError: num_maps (79000) exceeds available maps in directory (10000).
```

说明当前配置要求加载的地图数超过了实际下载/转换好的 `.bin` 数量。`drive.ini` 默认是 `env.num_maps = 79000`，但如果当前只有 10k training maps，就需要显式覆盖：

```bash
puffer train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  --env.num-maps 10000
```

也可以先检查当前 split 有多少地图：

```bash
python - <<'PY'
import os
from pathlib import Path
root = Path(os.environ["DRIVE_BINARIES_DATA_ROOT"])
split = "training"
print(len(list((root / split).glob("map_*.bin"))))
PY
```

### 找不到 `map_000000.bin`

当前 C binding 使用六位编号文件名：

```text
map_000000.bin
map_000001.bin
...
```

如果你的数据是 `map_000.bin` 这类旧格式，需要重新转换或重命名为六位格式。

### CUDA OOM

优先尝试：

```bash
--train.minibatch-size 8192 --train.max-minibatch-size 8192
```

如果仍然 OOM，再降低：

```bash
--train.batch-size 131072 --vec.num-workers 4 --vec.num-envs 4
```

### 训练很慢或 worker 卡住

先跑：

```bash
python tests/test_drive_train.py
```

然后用小配置复现：

```bash
puffer train puffer_drive \
  --train.total-timesteps 100000 \
  --env.num-maps 1 \
  --vec.num-workers 1 \
  --vec.num-envs 1 \
  --train.batch-size 128 \
  --train.minibatch-size 128 \
  --train.max-minibatch-size 128 \
  --train.bptt-horizon 8 \
  --train.device cpu
```
