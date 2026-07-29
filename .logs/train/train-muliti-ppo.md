# Multi-PPO 正式默认训练模板

下面模板用于启动新的 `mix_ppo` 训练方式：在 PPO-controlled agents 内部按比例混合多个 PPO-like 策略，并让这些策略一起参与训练。训练规模保持 `pufferlib/config/ocean/drive.ini` 的正式默认配置：`total_timesteps=2_000_000_000`、`num_maps=79000`、`batch_size=524288`、`minibatch_size=32768`、`bptt_horizon=32`。

```bash
cd "$HOME/wsc/behavior-bench"

# 使用固定 tmux session 名，方便后续 attach / stop / cleanup。
# 日志文件名仍然带时间戳，不会被覆盖。
export TMUX_SESSION="mix-ppo-default"

# 如果同名训练还在跑，先停止旧 session，避免重复占用同一张 GPU。
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
tmux new-session -d -s "$TMUX_SESSION" "bash -lc '
set -euo pipefail

cd \"$HOME/wsc/behavior-bench\"
source \"$HOME/miniconda3/etc/profile.d/conda.sh\"
conda activate behavior-bench

export REPO_ROOT=\"\$(pwd)\"
export DRIVE_BINARIES_DATA_ROOT=\"\$REPO_ROOT/resources/drive/binaries\"
export LOG_DIR=\"\$REPO_ROOT/.logs/train/runs\"
mkdir -p \"\$LOG_DIR\"

# ===== 正式默认训练参数 =====
export GPU_IDS=\"3\"
export RUN_NAME=\"mix_ppo_drive_default_2b\"
export CONFIG_PATH=\"pufferlib/config/ocean/drive.ini\"
export ENV_NUM_MAPS=\"10000\"

# ===== mix_ppo 参数 =====
export ENABLE_MIX_PPO=\"True\"
export MIX_PPO_POLICY_MIX=\"learner:0.5,transformer:0.25,gameformer:0.25\"
export MIX_PPO_POLICY_NAMES=\"Drive,DriveTransformer,DriveGameFormer\"
export MIX_PPO_POLICY_PATHS=\",,\"
export MIX_PPO_POLICY_TRAINABLE=\"True,True,True\"

# ===== mix_traffic 参数 =====
# mix_ppo 作用在 PPO-controlled agents 内部。
# 如果 MIX_TRAFFIC=False，则所有 active agents 都属于 PPO-controlled agents。
# 如果 MIX_TRAFFIC=True，则只有 ppo_fraction 对应的那部分 agents 会再被 mix_ppo 混合。
export ENABLE_MIX_TRAFFIC=\"False\"
export PPO_FRACTION=\"1.0\"
export IDM_FRACTION=\"0.0\"
export EXPERT_FRACTION=\"0.0\"

# ===== 训练内评估保持关闭 =====
# 当前训练内 WOSAC / human replay eval 子进程参数不完整，正式训练时保持关闭；
# 训练结束后再单独运行 puffer eval。
export EVAL_SPLIT=\"validation\"
export EVAL_NUM_MAPS=\"20\"
export ENABLE_WOSAC_REALISM=\"False\"
export ENABLE_HUMAN_REPLAY=\"False\"

# ===== W&B =====
export WANDB_PROJECT=\"behavior-bench\"
export WANDB_GROUP=\"mix-ppo-default\"

# ===== 日志与 GPU =====
export CUDA_VISIBLE_DEVICES=\"\$GPU_IDS\"
export TRAIN_LOG=\"\$LOG_DIR/\${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log\"

puffer train puffer_drive \
  --config \"\$CONFIG_PATH\" \
  --train.name \"\$RUN_NAME\" \
  --env.num-maps \"\$ENV_NUM_MAPS\" \
  --env.mix-traffic \"\$ENABLE_MIX_TRAFFIC\" \
  --env.ppo-fraction \"\$PPO_FRACTION\" \
  --env.idm-fraction \"\$IDM_FRACTION\" \
  --env.expert-fraction \"\$EXPERT_FRACTION\" \
  --train.mix-ppo \"\$ENABLE_MIX_PPO\" \
  --train.mix-ppo-policy-mix \"\$MIX_PPO_POLICY_MIX\" \
  --train.mix-ppo-policy-names \"\$MIX_PPO_POLICY_NAMES\" \
  --train.mix-ppo-policy-paths \"\$MIX_PPO_POLICY_PATHS\" \
  --train.mix-ppo-policy-trainable \"\$MIX_PPO_POLICY_TRAINABLE\" \
  --eval.split \"\$EVAL_SPLIT\" \
  --eval.num-maps \"\$EVAL_NUM_MAPS\" \
  --eval.wosac-realism-eval \"\$ENABLE_WOSAC_REALISM\" \
  --eval.human-replay-eval \"\$ENABLE_HUMAN_REPLAY\" \
  --wandb \
  --wandb-project \"\$WANDB_PROJECT\" \
  --wandb-group \"\$WANDB_GROUP\" \
  2>&1 | tee -a \"\$TRAIN_LOG\"
'"

echo "Started tmux session: $TMUX_SESSION"
echo "Attach with: tmux attach -t $TMUX_SESSION"
```

### 3. 3-GPU DDP 等价全局 6B 版本

这个版本使用 `GPU 4,5,6` 三张卡。当前 DDP 语义沿用同质 PPO：每张卡各自收集本地 batch，backward 时同步梯度。因此为了对齐单卡 `batch3x + 6B` 的全局训练预算：

```text
per-rank batch_size = 524288
global batch/update = 524288 * 3 = 1572864
per-rank total_timesteps = 2000000000
global total agent-steps ~= 2000000000 * 3 = 6000000000
updates ~= 2000000000 / 524288 ~= 3815
```

```bash
cd "$HOME/wsc/behavior-bench"

export TMUX_SESSION="mix-ppo-drive-moe-moe3-ddp3-gpu456-$(date +%Y%m%d_%H%M%S)"

tmux new-session -d -s "$TMUX_SESSION" "bash -lc '
set -euo pipefail

cd \"$HOME/wsc/behavior-bench\"
source \"$HOME/miniconda3/etc/profile.d/conda.sh\"
conda activate behavior-bench

export REPO_ROOT=\"\$(pwd)\"
export DRIVE_BINARIES_DATA_ROOT=\"\$REPO_ROOT/resources/drive/binaries\"
export LOG_DIR=\"\$REPO_ROOT/.logs/train/runs\"
mkdir -p \"\$LOG_DIR\"

export GPU_IDS=\"4,5,6\"
export NPROC_PER_NODE=\"3\"
export RUN_NAME=\"mix_ppo_drive_moe_moe3_1to1to1_ddp3_global6b\"
export CONFIG_PATH=\"pufferlib/config/ocean/drive.ini\"
export ENV_NUM_MAPS=\"10000\"
export TRAIN_BATCH_SIZE=\"524288\"
export TOTAL_TIMESTEPS=\"2000000000\"
export ENABLE_MIX_PPO=\"True\"
export MIX_PPO_POLICY_MIX=\"drive:1,moe:1,moe3:1\"
export MIX_PPO_POLICY_NAMES=\"Drive,DriveMoE,DriveMoE3\"
export MIX_PPO_POLICY_PATHS=\",,\"
export MIX_PPO_POLICY_TRAINABLE=\"True,True,True\"
export ENABLE_MIX_TRAFFIC=\"False\"
export PPO_FRACTION=\"1.0\"
export IDM_FRACTION=\"0.0\"
export EXPERT_FRACTION=\"0.0\"
export EVAL_SPLIT=\"validation\"
export EVAL_NUM_MAPS=\"20\"
export ENABLE_WOSAC_REALISM=\"False\"
export ENABLE_HUMAN_REPLAY=\"False\"
export WANDB_PROJECT=\"behavior-bench\"
export WANDB_GROUP=\"mix-ppo-ddp\"
export CUDA_VISIBLE_DEVICES=\"\$GPU_IDS\"
export TRAIN_LOG=\"\$LOG_DIR/\${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log\"

echo \"TRAIN_LOG=\$TRAIN_LOG\"
echo \"GPU_IDS=\$GPU_IDS\"
echo \"NPROC_PER_NODE=\$NPROC_PER_NODE\"
echo \"TRAIN_BATCH_SIZE=\$TRAIN_BATCH_SIZE\"
echo \"TOTAL_TIMESTEPS=\$TOTAL_TIMESTEPS\"

torchrun --standalone --nnodes=1 --nproc-per-node=\"\$NPROC_PER_NODE\" \
  -m pufferlib.pufferl train puffer_drive \
  --config \"\$CONFIG_PATH\" \
  --train.name \"\$RUN_NAME\" \
  --train.batch-size \"\$TRAIN_BATCH_SIZE\" \
  --train.total-timesteps \"\$TOTAL_TIMESTEPS\" \
  --env.num-maps \"\$ENV_NUM_MAPS\" \
  --env.mix-traffic \"\$ENABLE_MIX_TRAFFIC\" \
  --env.ppo-fraction \"\$PPO_FRACTION\" \
  --env.idm-fraction \"\$IDM_FRACTION\" \
  --env.expert-fraction \"\$EXPERT_FRACTION\" \
  --train.mix-ppo \"\$ENABLE_MIX_PPO\" \
  --train.mix-ppo-policy-mix \"\$MIX_PPO_POLICY_MIX\" \
  --train.mix-ppo-policy-names \"\$MIX_PPO_POLICY_NAMES\" \
  --train.mix-ppo-policy-paths \"\$MIX_PPO_POLICY_PATHS\" \
  --train.mix-ppo-policy-trainable \"\$MIX_PPO_POLICY_TRAINABLE\" \
  --eval.split \"\$EVAL_SPLIT\" \
  --eval.num-maps \"\$EVAL_NUM_MAPS\" \
  --eval.wosac-realism-eval \"\$ENABLE_WOSAC_REALISM\" \
  --eval.human-replay-eval \"\$ENABLE_HUMAN_REPLAY\" \
  --wandb \
  --wandb-project \"\$WANDB_PROJECT\" \
  --wandb-group \"\$WANDB_GROUP\" \
  2>&1 | tee -a \"\$TRAIN_LOG\"
'"

echo "Started tmux session: $TMUX_SESSION"
echo "Attach with: tmux attach -t $TMUX_SESSION"
```

## 干净停止与清理

如果训练通过上面的模板启动，不要只关闭本地终端窗口；关闭终端或 detach tmux 不会停止后台训练。优先按下面顺序清理。

### 1. 优雅停止当前模板启动的训练

```bash
cd "$HOME/wsc/behavior-bench"
export TMUX_SESSION="mix-ppo-default"

# 进入训练窗口后按 Ctrl-C，让 Python / W&B 有机会正常退出。
tmux attach -t "$TMUX_SESSION"
```

如果无法 attach，直接向 session 发送 Ctrl-C：

```bash
tmux send-keys -t "$TMUX_SESSION" C-c
sleep 10
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
```

### 2. 确认并清理残留 puffer 进程

```bash
# 查看当前 GPU 占用和 PID。注意：CUDA_VISIBLE_DEVICES=3 时，PyTorch 报错里的 GPU 0 对应物理 GPU 3。
nvidia-smi

# 查看仍在运行的本项目训练进程。
pgrep -af "puffer train puffer_drive|pufferlib.pufferl.*train.*puffer_drive" || true

# 先温和停止本项目训练进程。
pkill -TERM -f "puffer train puffer_drive|pufferlib.pufferl.*train.*puffer_drive" || true
sleep 10

# 如果仍然存在，再强制停止。
pgrep -af "puffer train puffer_drive|pufferlib.pufferl.*train.*puffer_drive" || true
pkill -KILL -f "puffer train puffer_drive|pufferlib.pufferl.*train.*puffer_drive" || true
```

### 3. GPU 仍显示显存占用但没有进程

如果 `nvidia-smi` 显示某张 GPU 仍占用大量显存，但 Processes 表为空，可能是容器/驱动层残留 context 或不可见进程。先检查设备文件占用：

```bash
# 例：检查物理 GPU 3
sudo fuser -v /dev/nvidia3 /dev/nvidiactl /dev/nvidia-uvm
```

如果确认没有其他用户或任务在用这张卡，并且你有权限，可以重置该 GPU：

```bash
sudo nvidia-smi --gpu-reset -i 3
```

如果 reset 失败，通常需要管理员清理对应宿主机进程或重启 GPU driver / 节点。

## 关键参数说明

- `ENABLE_MIX_PPO=True`：开启新的 PPO 内部混合训练；关闭时走原来的单策略训练路径。
- `MIX_PPO_POLICY_MIX`：定义 PPO-controlled agents 内部的策略比例。示例中 50% agents 使用 `learner`，25% 使用 `transformer`，25% 使用 `gameformer`。比例会归一化，agent 在 rollout 内固定归属一个 policy。
- `MIX_PPO_POLICY_NAMES`：每个混合槽对应的 policy class，必须与 `MIX_PPO_POLICY_MIX` 的项数一致。可用现有 PPO-like 网络，例如 `Drive`、`DriveTransformer`、`DriveGameFormer`。
- `MIX_PPO_POLICY_PATHS`：每个 policy 的 checkpoint 路径，空表示从随机初始化开始训练。示例 `",,"` 表示三个 policy 都不加载 checkpoint。
- `MIX_PPO_POLICY_TRAINABLE`：每个 policy 是否参与训练更新。示例全部为 `True`，表示三个 PPO-like policy 都一起训练。
- `ENABLE_MIX_TRAFFIC / PPO_FRACTION / IDM_FRACTION / EXPERT_FRACTION`：保持原有 `mix_traffic` 语义。`mix_ppo` 只会混合 `ppo_fraction` 对应的 PPO-controlled agents，不改变 C 层 IDM / Expert 分配逻辑。
- `ENV_NUM_MAPS=79000`：使用正式默认训练地图数量，要求 `$DRIVE_BINARIES_DATA_ROOT/training` 下实际 `.bin` 数量足够。
- 当前 `mix_ppo` 不支持 `torchrun DDP`、`torch.compile` 或同时开启 `opponent_pool`；正式模板使用单进程单卡后台 tmux 启动。

## 训练初始化与更新流程

训练主循环在 `pufferlib/pufferl.py::train()` 中反复执行 `pufferl.evaluate()` 和 `pufferl.train()`：前者从环境采样并填满 rollout buffer，后者用这批样本做 PPO 更新。

### 1. 真实 map 中车辆如何初始化

`pufferlib/ocean/drive/drive.py::Drive.__init__()` 会先调用 C binding 的 `shared()`，从 `$DRIVE_BINARIES_DATA_ROOT/<split>` 中采样真实 `.bin` map，逐个统计每个 map 可初始化的 PPO-controlled agents，并用 `agent_offsets` 把多个 map 的 agent slots 串接起来。默认 `env.num_agents=1024`，所以一个 Python `Drive` 环境通常包含多个真实 map，而不是一个 map。

每个真实 map 再通过 `binding.env_init(..., map_id=map_ids[i], max_agents=nxt-cur, ...)` 初始化成 C 层 `Drive` env。C 侧 `set_active_agents()` 会遍历 map 里的 entities：跳过 `init_steps` 时无效的轨迹，根据 `init_mode/control_mode` 判断是否创建车辆，再用 `should_control_agent()` 判断是否进入 PPO-controlled agents。若 `mix_traffic=True`，它还会按 deficit-based interleaving 把候选车辆分成 PPO / IDM / Expert；否则满足控制条件的车辆进入 PPO，其他可作为 expert/static 背景车。

### 2. mix_ppo 如何作用在 PPO 车辆上

`mix_ppo` 不决定车辆是否存在，也不改变 C 层 movement mode。它只在 PPO-controlled agent slots 内部分配不同 Python policy。当前 `assign_policy_ids()` 按输入比例做 deficit-based 交错分配，例如 `0.5/0.25/0.25` 会生成类似 `[0, 1, 2, 0, 0, 1, 2, 0, ...]` 的 policy id pattern。rollout 前向时，`_mix_ppo_forward_eval()` 根据 `self.agent_policy_ids[env_id]` 切 mask，把不同 agents 送入不同 policy 网络。

### 3. 如何攒够一个 PPO batch

默认 `vec.num_workers=16`，`vec.batch_size=4`，一次 `vecenv.recv()` 会从 16 个 worker 中取 4 个 ready worker 的一步数据。每个 worker 对应一个 Python `Drive` 环境，默认约 `1024` 个 PPO slots，所以一次环境 step 大约收集 `4 * 1024 = 4096` 个 agent-step。

`evaluate()` 会反复执行：`recv()` 取 observation/reward/done，policy forward 得到 action/value/logprob，把数据写入 `self.observations/actions/logprobs/rewards/values/terminals/truncations`，再 `send(action)` 推进环境。每个 agent 的连续轨迹长度达到 `bptt_horizon=32` 后形成一个 segment。默认 `train.batch_size=524288`，因此 `segments = 524288 / 32 = 16384`，刚好对应 `16 * 1024` 个 agent slots 各跑满 32 步。

### 4. PPO 如何更新以及下一轮如何继续

rollout buffer 填满后，`train()` 先用 rewards/values/terminals/truncations 计算 advantage，再按 segment 采样 minibatch。普通 PPO 直接用一个 policy 更新；`mix_ppo=True` 时，会根据 `segment_policy_ids` 把 minibatch 分给各个 policy，每个 policy 只用自己的 segments 计算 loss、反传并执行自己的 optimizer step。

一次 `train()` 完成后，主循环不会强制重建所有环境，而是继续调用下一轮 `evaluate()` 从当前环境状态采样。只有某个真实 map 到达 episode end、termination、truncation 或 resample 条件时，C env 才 reset/resample；新的 map 进入同一批 agent slots 后，会继续沿用这些 slots 对应的 policy id pattern。

## mix_ppo 策略分配粒度说明

当前代码中，`mix_ppo` 的策略比例不是按 `train.batch_size` 切分，也不是按一次 minibatch 切分，而是先按 **一个 Python `Drive` 环境内打包后的 PPO-controlled agent slots** 生成一个固定的 policy id pattern，然后把这个 pattern 复制到外层并行 vector env。

这里要区分两个概念：

- 真实 map / scenario：一个 `.bin` 场景，C 层 `Drive` env 会在这个 map 里找有效车辆并初始化 active agents。
- Python `pufferlib.ocean.drive.Drive` 环境：会把多个真实 map 打包在一起，直到累计 PPO-controlled agents 达到 `env.num_agents` 目标值。默认 `env.num_agents=1024`，所以一个 Python `Drive` 环境通常不是一个 map，而是多个 map 的连续 agent slots。

实际调用链如下：

1. `pufferlib/pufferl.py::load_env()` 读取 `MIX_PPO_POLICY_MIX`，计算一个 Python `Drive` 环境内需要分配的 `ppo_agents`：
   - `ENABLE_MIX_TRAFFIC=False` 时，`ppo_agents = env.num_agents`。
   - `ENABLE_MIX_TRAFFIC=True` 且 `PPO_FRACTION < 1.0` 时，`ppo_agents = int(env.num_agents * PPO_FRACTION)`。
2. `assign_policy_ids(ppo_agents, fractions)` 根据比例生成 `mix_ppo_local_policy_ids`。
3. Python `Drive` 环境初始化时，`binding.shared()` 会逐个采样真实 map，统计每个 map 可初始化的 PPO-controlled agents，并用 `agent_offsets` 把多个 map 的 agent slots 串接起来，直到累计达到 `env.num_agents`。
4. `Drive._set_policy_log_ids()` / C 侧 `vec_set_policy_log_ids()` 会把 `mix_ppo_local_policy_ids` 按 `agent_offsets` 顺序切给各个真实 map。也就是说，单个真实 map 拿到的是这个 1024 长度 pattern 中的一个连续切片。
5. 外层 `PuffeRL.__init__()` 读取 `mix_ppo_local_policy_ids`，将它重复到 `vecenv.num_agents` 长度，得到全局 `self.agent_policy_ids`。
6. rollout 前向时按 `self.agent_policy_ids[env_id]` 对当前 batch 的 agent 切 mask，不同 mask 送入不同 policy。
7. 训练更新时，`self.segment_policy_ids` 记录每个 rollout segment 属于哪个 policy，然后各 policy 只用自己的 segment 计算 PPO loss。

在本模板默认参数下：

```text
env.num_agents = 1024
vec.num_envs = 16
vec.num_workers = 16
vec.batch_size = 4
train.batch_size = 524288
bptt_horizon = 32
MIX_PPO_POLICY_MIX = learner:0.5,transformer:0.25,gameformer:0.25
```

因此一个 Python `Drive` 环境内生成的本地 policy id pattern 长度是 `1024`，不是 `vec.batch_size=4`，也不是 `train.batch_size=524288`。当前实现采用和 `mix_traffic` 思路一致的 deficit-based 逐 slot 分配：每放置一个 PPO-controlled agent slot，就选择相对目标比例最“欠账”的 policy。

```text
目标总数仍然是：
policy0: 512 agent slots
policy1: 256 agent slots
policy2: 256 agent slots

但顺序是交错的，例如前 8 个 slots：
[0, 1, 2, 0, 0, 1, 2, 0]
```

然后这段 1024 长度的交错 pattern 会复制到外层 16 个并行 Python `Drive` 环境上。真实 map 的 agents 会按照 `agent_offsets` 拿到这个 1024 pattern 的连续切片：

```text
map A: local slots [0, active_count_A)
map B: local slots [active_count_A, active_count_A + active_count_B)
...
```

由于 pattern 已经交错，单个真实 map 即使只拿到其中一小段连续切片，也会更接近输入比例。它不像旧连续块实现那样让 `[0,512)` 范围内的 map 全是 policy0、`[512,768)` 范围内的 map 全是 policy1。当前实现仍然保证一个 Python `Drive` 环境打包后的 1024 个 PPO slots 满足输入比例，同时尽量让任意前缀/局部切片也接近该比例。

`vec.batch_size=4` 的含义是一次从 16 个 worker/env 中取 4 个 worker/env 的数据组成当前 env step batch；它不会改变单个 env 内的 policy 分配块长度。`train.batch_size=524288` 是 PPO rollout buffer 的总样本规模；结合 `bptt_horizon=32` 会得到 `segments=524288/32=16384`，刚好等于默认 `1024 agents/env * 16 envs`，也不是 policy 连续块长度。

因此，当前代码的 `mix_ppo` 分配方式已经和 `mix_traffic` 的核心思想一致：都使用“谁相对目标比例最欠账，就把下一个 agent 分给谁”的 deficit-based interleaving。区别只是作用层级不同：`mix_traffic` 在 C 层真实 map 内分配 PPO/IDM/Expert；`mix_ppo` 在 Python 侧对 PPO-controlled agent slots 分配不同 PPO-like policy，不改变 C 层 movement mode。

## 当前支持的 PPO-like 策略

下面统计基于当前默认 `pufferlib/config/ocean/drive.ini`：

```text
observation_dim = 1151
ego_dim = 7
partner = 31 objects * 8 features = 248
road = 128 objects * 7 features = 896
road one-hot 后每个 road token = 13
action_dim = 91
policy.input_size = 64
policy.hidden_size = 256
rnn.input_size = 256
rnn.hidden_size = 256
```

默认模板 `rnn_name=Recurrent`，所以表中“默认参数量”包含 policy 本身和 LSTM wrapper。若只看 policy 本体，见“base 参数量”。

| Policy class | 默认可直接用于当前模板 | base 参数量 | 默认 Recurrent 后参数量 | 主要结构 |
| --- | --- | ---: | ---: | --- |
| `Drive` | 是 | 87,900 | 614,236 | MLP + max-pool |
| `DriveNoGoal` | 是 | 87,772 | 614,108 | `Drive` 去掉 ego goal 输入 |
| `DriveTransformer` | 是 | 88,796 | 615,132 | ego query cross-attention |
| `DriveGameFormer` | 是 | 189,340 | 715,676 | GameFormer-style encoder/decoder |
| `DriveMoE` | 是 | 226,014 | 752,350 | 2-expert soft-router MoE |
| `DriveMoE3` | 是 | 290,335 | 816,671 | 3-expert soft-router MoE |
| `DriveLatentWorldModel` | 可前向使用 | 243,292 | 769,628 | `Drive` encoder + latent transition model |
| `DriveConditioned` | 需 `env.reward_conditioning=True` | 109,276 | 635,612 | `Drive` + Creward 分支 |
| `DriveGameFormerConditioned` | 需 `env.reward_conditioning=True` | 194,396 | 720,732 | `DriveGameFormer` + Creward token |
| `DrivePaper` | 默认 Recurrent 不兼容 | 415,324 | 不兼容 | actor/critic 独立 tower |
| `DriveConditionedPaper` | 默认 Recurrent 不兼容，且需 reward conditioning | 458,076 | 不兼容 | conditioned actor/critic 独立 tower |

### 统一输入切分

默认 observation 是一个 1151 维向量，所有普通 Drive policy 都先按下面方式切分：

```text
obs[0:7]       ego:      7
obs[7:255]     partners: 31 * 8 = 248
obs[255:1151]  roads:    128 * 7 = 896
```

其中 road token 的最后一维是类别 id，进入网络前会 one-hot 到 7 类：

```text
road token: 7 raw features -> 6 continuous + 7 one-hot = 13
```

所有默认可直接混用的策略最终都输出一个 256 维 policy embedding，交给 `Recurrent`：

```text
policy embedding 256 -> LSTM/Recurrent 256 -> actor logits 91 + value 1
```

### 默认可直接混用的策略

#### `Drive`

最基础的 MLP + permutation-invariant max-pool 结构。

```text
Observation 1151
├─ ego: 7
│  └─ Linear 7 -> 64 -> LayerNorm -> Linear 64 -> 64
├─ partners: 31 x 8
│  └─ per partner Linear 8 -> 64 -> LayerNorm -> Linear 64 -> 64
│     └─ max over 31 partners -> 64
└─ roads: 128 x 7
   └─ road type one-hot: 7 -> 13
   └─ per road Linear 13 -> 64 -> LayerNorm -> Linear 64 -> 64
      └─ max over 128 roads -> 64

concat [ego64, partner64, road64] -> 192
GELU -> Linear 192 -> 256
ReLU -> policy embedding 256
Recurrent LSTMCell 256 -> 256
├─ actor Linear 256 -> 91 discrete logits
└─ value Linear 256 -> 1
```

#### `DriveNoGoal`

与 `Drive` 相同，但 ego 分支不看前两个 goal features。适合做“无显式目标向量”的对照策略。

```text
Observation 1151
├─ ego: 7, 使用 obs[2:7] -> 5
│  └─ Linear 5 -> 64 -> LayerNorm -> Linear 64 -> 64
├─ partners: 31 x 8 -> 同 Drive -> 64
└─ roads: 128 x 7 -> one-hot 后 128 x 13 -> 同 Drive -> 64

concat 192 -> GELU -> Linear 192 -> 256
Recurrent 256 -> 256
├─ actor 256 -> 91
└─ value 256 -> 1
```

#### `DriveTransformer`

先把 ego、partners、roads 都编码成 token，再用 ego token 作为 query 对全场景做 cross-attention。

```text
Observation 1151
├─ ego token: 7 -> 64 -> 64, shape B x 1 x 64
├─ partner tokens: 31 x (8 -> 64 -> 64), shape B x 31 x 64
└─ road tokens: 128 x (13 -> 64 -> 64), shape B x 128 x 64

add type embeddings
tokens = [ego, partners, roads], shape B x 160 x 64
query = ego token, shape B x 1 x 64

CrossAttention(query=ego, key/value=tokens)
FFN 64 -> 128 -> 64
squeeze ego token -> 64
Linear 64 -> 256 + GELU
Recurrent 256 -> 256
├─ actor 256 -> 91
└─ value 256 -> 1
```

#### `DriveGameFormer`

更重的 token encoder/decoder 结构，先做全场景 self-attention，再让 agent tokens 解码并回看 scene tokens。

```text
Observation 1151
├─ ego token: 1 x 64
├─ partner tokens: 31 x 64
└─ road tokens: 128 x 64

tokens = 160 x 64
TransformerEncoder x 2 layers
├─ d_model = 64
├─ heads = 4
└─ FFN = 64 -> 128 -> 64

agent tokens = ego + partners = 32 x 64
Decoder step 1: self-attention among 32 agent tokens
Decoder step 2: cross-attention agent tokens -> 160 scene tokens
take ego token -> 64
LayerNorm -> Linear 64 -> 256 -> GELU
Recurrent 256 -> 256
├─ actor 256 -> 91
└─ value 256 -> 1
```

#### `DriveMoE`

`DriveMoE` 是 2-expert 的 soft-router mixture-of-experts。每个 expert 分支都是 `Drive` 同款 observation encoder；router 读取原始 observation，为每个 agent 输出两个 expert 权重，然后对两个 expert embedding 做加权求和。

```text
Observation 1151
├─ Expert 0: Drive-style encoder
│  ├─ ego: 7 -> 64 -> 64
│  ├─ partners: 31 x 8 -> per-token 64 -> max-pool -> 64
│  ├─ roads: 128 x 13 -> per-token 64 -> max-pool -> 64
│  └─ concat 192 -> Linear 192 -> 256
├─ Expert 1: Drive-style encoder
│  └─ same as Expert 0 -> 256
└─ Router:
   └─ raw observation 1151 -> Linear 1151 -> 64 -> GELU -> Linear 64 -> 2
      -> softmax weights [w0, w1]

fused embedding = w0 * expert0_256 + w1 * expert1_256
Recurrent 256 -> 256
├─ actor 256 -> 91
└─ value 256 -> 1
```

默认 expert 数量为 2，可通过 `num_experts` 扩展。为了直接兼容当前 Multi-PPO 模板，`DriveMoE` 的融合输出保持 256 维，与默认 `rnn.input_size=256` 对齐。

`DriveMoE3` 是 `DriveMoE` 的 3-expert wrapper，结构完全相同，只是：

```text
Expert 数: 3
Router 输出: Linear 64 -> 3
fused embedding = w0 * expert0_256 + w1 * expert1_256 + w2 * expert2_256
```

在 `mix_ppo` 中可直接通过 `MIX_PPO_POLICY_NAMES` 使用：

```text
Drive,DriveMoE,DriveMoE3
```

#### `DriveLatentWorldModel`

主干和 `Drive` 类似，但额外学习一个 latent transition model，用于从上一步 latent 和 action 预测下一步 latent。

```text
Observation 1151
├─ ego/partner/road encoders 同 Drive
└─ concat 192 -> Linear 192 -> 256

latent z_t = 256
actor 256 -> 91
value 256 -> 1

transition branch:
previous latent z_{t-1}: 256
previous action one-hot: 91
concat: 347
Linear 347 -> 256 -> LayerNorm -> GELU -> Linear 256 -> 256
```

当前默认 `Recurrent` 包装下可前向使用；如果要真正训练其 imagination / transition auxiliary loss，需要额外确认训练 loop 是否接入该 loss。

### 需要额外配置的策略

#### `DriveConditioned`

需要 `env.reward_conditioning=True`，observation 末尾增加 10 维 `Creward`：

```text
conditioned observation_dim = 1161
普通 obs: 1151
Creward: 10
```

结构是在 `Drive` 的三分支外增加一个 reward-condition 分支：

```text
Observation 1161
├─ ego: 7 -> 64
├─ partners: 31 x 8 -> max-pool -> 64
├─ roads: 128 x 13 -> max-pool -> 64
└─ creward: 10 -> 64 -> 64

concat [ego64, partner64, road64, creward64] -> 256
GELU -> Linear 256 -> 256
Recurrent 256 -> 256
├─ actor 256 -> 91
└─ value 256 -> 1
```

#### `DriveGameFormerConditioned`

需要 `env.reward_conditioning=True`。相比 `DriveGameFormer`，额外把 `Creward` 编成一个 token 放入 scene token 序列。

```text
Observation 1161
├─ ego token: 1 x 64
├─ partner tokens: 31 x 64
├─ road tokens: 128 x 64
└─ creward token: 1 x 64

tokens = 161 x 64
TransformerEncoder x 2 layers
agent tokens = 32 x 64
Decoder self-attention among agents
Decoder cross-attention agents -> 161 scene tokens
ego token -> Linear 64 -> 256
Recurrent 256 -> 256
├─ actor 256 -> 91
└─ value 256 -> 1
```

#### `DrivePaper` / `DriveConditionedPaper`

Paper 系列采用 actor / critic 完全独立的 tower，不共享 encoder 和 backbone。它不是默认 Multi-PPO 模板中最方便的选择，因为输出隐藏维度是 512。

```text
Observation
├─ actor tower
│  ├─ ego encoder -> 64
│  ├─ partner encoder + max-pool -> 64
│  ├─ road encoder + max-pool -> 64
│  └─ concat -> backbone 192 or 256 -> 256 -> 256 -> 256
└─ critic tower
   ├─ same structure, independent parameters
   └─ output -> 256

encode_observations output:
actor_h 256 + critic_h 256 = 512
```

因此默认 `rnn.input_size=256, rnn.hidden_size=256` 不兼容。若要配合 Recurrent 使用，需要专门把 RNN 输入和 hidden 改成 512；实测参数量为：

```text
DrivePaper + Recurrent(input=512, hidden=512): 2,516,572
DriveConditionedPaper + Recurrent(input=512, hidden=512): 2,559,324
```

更简单的方式是使用 `rnn_name=None`，让它们作为 feedforward actor/critic tower 使用；这需要单独确认训练配置和 mix_ppo 的 RNN 状态管理逻辑。

## 环境/训练开关说明

### 1. `env.reward_conditioning=True`

`env.reward_conditioning=True` 是一种“奖励条件化”环境开关。开启后，环境会在每个 agent 的 observation 里额外加入一段 `Creward` 条件向量，让 policy 不只看到道路、车辆和 ego 状态，还能看到“当前希望它按什么奖励偏好驾驶”。

在当前默认配置下，普通 observation 是：

```text
observation_dim = 1151
ego: 7
partners: 31 * 8
roads: 128 * 7
```

如果开启 `reward_conditioning=True`，observation 会额外增加：

```text
Creward dim = 10
conditioned observation_dim = 1161
```

这 10 维条件向量对应不同 reward / behavior 偏好的配置，例如 collision、boundary/offroad、comfort、lane alignment、center bias、reverse、goal speed 等。它主要服务于 conditioned 系列策略：

```text
DriveConditioned
DriveGameFormerConditioned
DriveConditionedPaper
```

这些策略会把额外的 `Creward` 向量作为单独分支或 token 编码进去。例如 `DriveConditioned` 的结构是：

```text
ego branch
partner branch
road branch
creward branch
-> concat
-> policy/value
```

默认 `drive.ini` 里是关闭的，所以普通 Multi-PPO 模板不需要它。只有想训练“可根据奖励偏好条件切换驾驶风格”的 policy 时，才需要打开：

```bash
--env.reward-conditioning True
```

如果不开启这个开关却使用 `DriveConditioned` / `DriveGameFormerConditioned` 等 conditioned policy，policy 会期待 observation 里有 `Creward` 维度，但实际 observation 没有，前向时会出现维度不匹配。

### 2. `train.opponent_pool=True`

`opponent_pool=True` 是训练层面的“历史对手池”开关，用来让当前 policy 在 rollout 时和自己过去的快照共同出现在环境中，从而增加训练对手/交通行为的多样性。它不是环境 observation 开关，也不是新的 policy class。

当前默认配置：

```text
opponent_pool = False
opponent_pool_fraction = 0.25
opponent_pool_snapshot_interval = 100
opponent_pool_max_snapshots = 20
opponent_pool_warmup_epochs = 100
```

代码逻辑在 `pufferlib/pufferl.py` 中：

1. 初始化 trainer 时，如果 `opponent_pool=True`，会维护一个历史 policy 快照队列，并创建一个 frozen shadow policy：

```text
opponent_pool_snapshots: 最多保存 opponent_pool_max_snapshots 个历史 state_dict
_opponent_pool_shadow: 当前 policy 的冻结拷贝，用来加载历史快照并生成动作
opponent_segments: 标记哪些 rollout segments 属于历史对手，不参与当前 policy 更新
```

2. 每隔 `opponent_pool_snapshot_interval` 个 epoch，把当前 policy 的参数快照存入队列：

```text
epoch % opponent_pool_snapshot_interval == 0
-> 保存当前 uncompiled_policy.state_dict()
```

3. 当训练达到 `opponent_pool_warmup_epochs` 且已有历史快照后，每轮采样时随机选一份历史快照加载到 `_opponent_pool_shadow`，并随机抽取一部分 agents 作为 opponent：

```text
n_opp = total_agents * opponent_pool_fraction
随机选择 n_opp 个 agent slots
这些 slots 的 action 由历史 shadow policy 生成
其余 slots 仍由当前 policy 生成
```

4. 写入 rollout buffer 时，会把 opponent 对应的 segments 标记到 `opponent_segments`。训练更新时，这些 opponent segments 会被 mask 掉，不参与当前 policy 的 PPO loss：

```text
masked_advantages = masked_advantages * ~opponent_segments
```

也就是说，opponent pool 的作用是：**用历史 policy 作为环境中的一部分对手/交通参与者，但只训练当前 policy 控制的那部分样本**。

重要限制：当前代码明确禁止 `mix_ppo` 和 `opponent_pool` 同时开启：

```text
if self.mix_ppo and self.opponent_pool:
    raise APIUsageError("mix_ppo cannot be combined with opponent_pool")
```

原因是二者都需要改写 rollout 中“哪个 agent 由哪个 policy 控制”的逻辑。`mix_ppo` 要按 `agent_policy_ids` 在多个当前 PPO-like policies 之间分配；`opponent_pool` 要把一部分 agents 的 action 替换成历史 policy，并在训练时屏蔽对应 segments。当前实现还没有定义这两套 mask 同时存在时的优化语义。

## 3 倍 batch 的 Drive/DriveMoE/DriveMoE3 Multi-PPO 训练模板

下面模板用于 `Drive,DriveMoE,DriveMoE3 = 1:1:1` 的异质 PPO 训练，并把全局 `train.batch-size` 从默认 `524288` 扩大到 `1572864`。这样每个 policy 在每次全局 update 中大约获得：

```text
1572864 / 3 = 524288 agent-steps
```

也就是接近同质 PPO 每次 update 的数据量。

### 1. 2B steps 版本

这个版本保持总采样量 `2B` 不变，因此总 update 次数约为默认训练的 `1/3`。

```bash
cd "$HOME/wsc/behavior-bench"

export TMUX_SESSION="mix-ppo-drive-moe-moe3-batch3x-gpu7-$(date +%Y%m%d_%H%M%S)"

tmux new-session -d -s "$TMUX_SESSION" "bash -lc '
set -euo pipefail

cd \"$HOME/wsc/behavior-bench\"
source \"$HOME/miniconda3/etc/profile.d/conda.sh\"
conda activate behavior-bench

export REPO_ROOT=\"\$(pwd)\"
export DRIVE_BINARIES_DATA_ROOT=\"\$REPO_ROOT/resources/drive/binaries\"
export LOG_DIR=\"\$REPO_ROOT/.logs/train/runs\"
mkdir -p \"\$LOG_DIR\"

export GPU_IDS=\"7\"
export RUN_NAME=\"mix_ppo_drive_moe_moe3_1to1to1_default_2b_batch3x\"
export CONFIG_PATH=\"pufferlib/config/ocean/drive.ini\"
export ENV_NUM_MAPS=\"10000\"
export TRAIN_BATCH_SIZE=\"1572864\"
export ENABLE_MIX_PPO=\"True\"
export MIX_PPO_POLICY_MIX=\"drive:1,moe:1,moe3:1\"
export MIX_PPO_POLICY_NAMES=\"Drive,DriveMoE,DriveMoE3\"
export MIX_PPO_POLICY_PATHS=\",,\"
export MIX_PPO_POLICY_TRAINABLE=\"True,True,True\"
export ENABLE_MIX_TRAFFIC=\"False\"
export PPO_FRACTION=\"1.0\"
export IDM_FRACTION=\"0.0\"
export EXPERT_FRACTION=\"0.0\"
export EVAL_SPLIT=\"validation\"
export EVAL_NUM_MAPS=\"20\"
export ENABLE_WOSAC_REALISM=\"False\"
export ENABLE_HUMAN_REPLAY=\"False\"
export WANDB_PROJECT=\"behavior-bench\"
export WANDB_GROUP=\"mix-ppo-default\"
export CUDA_VISIBLE_DEVICES=\"\$GPU_IDS\"
export TRAIN_LOG=\"\$LOG_DIR/\${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log\"

echo \"TRAIN_LOG=\$TRAIN_LOG\"
echo \"TRAIN_BATCH_SIZE=\$TRAIN_BATCH_SIZE\"

puffer train puffer_drive \
  --config \"\$CONFIG_PATH\" \
  --train.name \"\$RUN_NAME\" \
  --train.batch-size \"\$TRAIN_BATCH_SIZE\" \
  --env.num-maps \"\$ENV_NUM_MAPS\" \
  --env.mix-traffic \"\$ENABLE_MIX_TRAFFIC\" \
  --env.ppo-fraction \"\$PPO_FRACTION\" \
  --env.idm-fraction \"\$IDM_FRACTION\" \
  --env.expert-fraction \"\$EXPERT_FRACTION\" \
  --train.mix-ppo \"\$ENABLE_MIX_PPO\" \
  --train.mix-ppo-policy-mix \"\$MIX_PPO_POLICY_MIX\" \
  --train.mix-ppo-policy-names \"\$MIX_PPO_POLICY_NAMES\" \
  --train.mix-ppo-policy-paths \"\$MIX_PPO_POLICY_PATHS\" \
  --train.mix-ppo-policy-trainable \"\$MIX_PPO_POLICY_TRAINABLE\" \
  --eval.split \"\$EVAL_SPLIT\" \
  --eval.num-maps \"\$EVAL_NUM_MAPS\" \
  --eval.wosac-realism-eval \"\$ENABLE_WOSAC_REALISM\" \
  --eval.human-replay-eval \"\$ENABLE_HUMAN_REPLAY\" \
  --wandb \
  --wandb-project \"\$WANDB_PROJECT\" \
  --wandb-group \"\$WANDB_GROUP\" \
  2>&1 | tee -a \"\$TRAIN_LOG\"
'"

echo "Started tmux session: $TMUX_SESSION"
echo "Attach with: tmux attach -t $TMUX_SESSION"
```

### 2. 6B steps 版本

这个版本同时把 `train.total-timesteps` 扩大到 `6B`，用于让 `batch3x` 训练的 update 次数接近默认 `2B / 524288` 的数量级。

```bash
cd "$HOME/wsc/behavior-bench"

export TMUX_SESSION="mix-ppo-drive-moe-moe3-batch3x-6b-gpu0-$(date +%Y%m%d_%H%M%S)"

tmux new-session -d -s "$TMUX_SESSION" "bash -lc '
set -euo pipefail

cd \"$HOME/wsc/behavior-bench\"
source \"$HOME/miniconda3/etc/profile.d/conda.sh\"
conda activate behavior-bench

export REPO_ROOT=\"\$(pwd)\"
export DRIVE_BINARIES_DATA_ROOT=\"\$REPO_ROOT/resources/drive/binaries\"
export LOG_DIR=\"\$REPO_ROOT/.logs/train/runs\"
mkdir -p \"\$LOG_DIR\"

export GPU_IDS=\"0\"
export RUN_NAME=\"mix_ppo_drive_moe_moe3_1to1to1_default_6b_batch3x\"
export CONFIG_PATH=\"pufferlib/config/ocean/drive.ini\"
export ENV_NUM_MAPS=\"10000\"
export TRAIN_BATCH_SIZE=\"1572864\"
export TOTAL_TIMESTEPS=\"6000000000\"
export ENABLE_MIX_PPO=\"True\"
export MIX_PPO_POLICY_MIX=\"drive:1,moe:1,moe3:1\"
export MIX_PPO_POLICY_NAMES=\"Drive,DriveMoE,DriveMoE3\"
export MIX_PPO_POLICY_PATHS=\",,\"
export MIX_PPO_POLICY_TRAINABLE=\"True,True,True\"
export ENABLE_MIX_TRAFFIC=\"False\"
export PPO_FRACTION=\"1.0\"
export IDM_FRACTION=\"0.0\"
export EXPERT_FRACTION=\"0.0\"
export EVAL_SPLIT=\"validation\"
export EVAL_NUM_MAPS=\"20\"
export ENABLE_WOSAC_REALISM=\"False\"
export ENABLE_HUMAN_REPLAY=\"False\"
export WANDB_PROJECT=\"behavior-bench\"
export WANDB_GROUP=\"mix-ppo-default\"
export CUDA_VISIBLE_DEVICES=\"\$GPU_IDS\"
export TRAIN_LOG=\"\$LOG_DIR/\${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log\"

echo \"TRAIN_LOG=\$TRAIN_LOG\"
echo \"TRAIN_BATCH_SIZE=\$TRAIN_BATCH_SIZE\"
echo \"TOTAL_TIMESTEPS=\$TOTAL_TIMESTEPS\"

puffer train puffer_drive \
  --config \"\$CONFIG_PATH\" \
  --train.name \"\$RUN_NAME\" \
  --train.batch-size \"\$TRAIN_BATCH_SIZE\" \
  --train.total-timesteps \"\$TOTAL_TIMESTEPS\" \
  --env.num-maps \"\$ENV_NUM_MAPS\" \
  --env.mix-traffic \"\$ENABLE_MIX_TRAFFIC\" \
  --env.ppo-fraction \"\$PPO_FRACTION\" \
  --env.idm-fraction \"\$IDM_FRACTION\" \
  --env.expert-fraction \"\$EXPERT_FRACTION\" \
  --train.mix-ppo \"\$ENABLE_MIX_PPO\" \
  --train.mix-ppo-policy-mix \"\$MIX_PPO_POLICY_MIX\" \
  --train.mix-ppo-policy-names \"\$MIX_PPO_POLICY_NAMES\" \
  --train.mix-ppo-policy-paths \"\$MIX_PPO_POLICY_PATHS\" \
  --train.mix-ppo-policy-trainable \"\$MIX_PPO_POLICY_TRAINABLE\" \
  --eval.split \"\$EVAL_SPLIT\" \
  --eval.num-maps \"\$EVAL_NUM_MAPS\" \
  --eval.wosac-realism-eval \"\$ENABLE_WOSAC_REALISM\" \
  --eval.human-replay-eval \"\$ENABLE_HUMAN_REPLAY\" \
  --wandb \
  --wandb-project \"\$WANDB_PROJECT\" \
  --wandb-group \"\$WANDB_GROUP\" \
  2>&1 | tee -a \"\$TRAIN_LOG\"
'"

echo "Started tmux session: $TMUX_SESSION"
echo "Attach with: tmux attach -t $TMUX_SESSION"
```
