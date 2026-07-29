# 三等级混合智能策略训练设计

## 1. 总体思路

当前目标是让训练环境中的 PPO-controlled agents 不再全部来自同一种策略智能等级，而是同时包含高、中、低三类驾驶策略。这样主策略在训练时会持续遇到不同决策能力的交通参与者，从而缓解“只在同类 PPO 交通中表现好，切换到未见 IDM/PDM 车流后 collision rate 明显退化”的问题。

这次设计不改变感知输入，也不改变环境 observation。三个等级都接收同样的 ego、partner、road 信息；差异只来自策略决策能力：

```text
Level 3: Drive + Recurrent
  完整单帧场景编码 + LSTM 时序 belief
  代表具备历史记忆和交互趋势推理的最高等级策略

Level 2: Drive
  完整单帧场景编码，但没有 LSTM
  代表只基于当前帧做 reactive decision 的中等级策略

Level 1: DriveLite
  同样输入，但使用更轻量的 feedforward reactive policy，且没有 LSTM
  代表没有时序 belief、策略映射复杂度更低的低等级策略
```

论文叙事上，这不是简单的“性格差异”或“调小 hidden size”，而是两条策略智能能力逐级移除：

```text
Temporal reasoning:
  Level 3 有 LSTM，可以从历史轨迹中形成 traffic behavior belief。
  Level 2/1 没有 LSTM，只能根据当前 observation 反应。

Decision complexity:
  Level 2 仍使用完整 Drive 单帧策略网络。
  Level 1 使用 DriveLite，决策模块更浅，更接近低复杂度 reactive controller。
```

## 2. 网络结构可视化

### Level 3: `Drive + Recurrent`

```text
obs_t (1151)
  ├─ ego:      7
  ├─ partners: 31 x 8
  └─ roads:    128 x 7

Drive encoder:
  ego      -> Linear 7  -> 64 -> LayerNorm -> Linear 64 -> 64
  partner  -> Linear 8  -> 64 -> LayerNorm -> Linear 64 -> 64 -> max pool
  road     -> one-hot road type, 13 dims
            -> Linear 13 -> 64 -> LayerNorm -> Linear 64 -> 64 -> max pool

concat [ego64, partner64, road64] = 192
  -> GELU
  -> Linear 192 -> 256
  -> ReLU
  -> scene embedding h_enc_t

Recurrent:
  h_enc_t -> LSTMCell(256, 256) -> h_lstm_t

Heads:
  h_lstm_t -> actor Linear 256 -> 91 discrete action logits
  h_lstm_t -> value Linear 256 -> 1
```

参数量：

```text
Drive base:        87,900
Drive + Recurrent: 614,236
```

智能等级含义：

```text
Full temporal policy.
它能用历史状态推断其他车的加减速趋势、让行意图和交互模式。
```

### Level 2: `Drive` without LSTM

```text
obs_t (1151)
  -> same Drive encoder as Level 3
  -> scene embedding h_enc_t

Heads:
  h_enc_t -> actor Linear 256 -> 91 discrete action logits
  h_enc_t -> value Linear 256 -> 1
```

参数量：

```text
Drive base: 87,900
```

智能等级含义：

```text
Reactive joint policy.
它看见同样的当前帧信息，但没有 recurrent memory，因此缺少时序 belief。
```

### Level 1: `DriveLite` without LSTM

```text
obs_t (1151)
  ├─ ego:      7
  ├─ partners: 31 x 8
  └─ roads:    128 x 7

DriveLite encoder:
  ego      -> Linear 7  -> 32 -> GELU
  partner  -> Linear 8  -> 32 -> GELU -> max pool
  road     -> one-hot road type, 13 dims
            -> Linear 13 -> 32 -> GELU -> max pool

concat [ego32, partner32, road32] = 96
  -> Linear 96 -> 224
  -> scene embedding h_lite_t

Heads:
  h_lite_t -> actor Linear 224 -> 91 discrete action logits
  h_lite_t -> value Linear 224 -> 1
```

参数量：

```text
DriveLite: 43,420
DriveLite / Drive base: 49.4%
```

智能等级含义：

```text
Low-complexity reactive policy.
它接收同样的 observation，但没有时序 belief，且单帧策略映射更浅，代表低复杂度驾驶决策。
```

## 3. 训练等价性设计

同质 Drive + Recurrent 默认训练配置为：

```text
train.batch_size      = 524,288
train.total_timesteps = 2,000,000,000
train.bptt_horizon    = 32
train.update_epochs   = 1
```

三等级混合训练中，如果仍使用 `batch_size=524288`，三个策略平均每个 update 只能各拿到约 `174762` agent-steps，不等价于同质训练。为了让每个策略在每次 PPO update 中都获得与同质训练相同的样本量，应使用：

```text
mixed batch_size = 3 * 524,288 = 1,572,864
policy mix       = l3:1,l2:1,l1:1
```

如果还希望每个策略的总训练步数也等价于同质训练的 2B agent-steps，则总采样量也需要放大 3 倍：

```text
mixed total_timesteps = 3 * 2,000,000,000 = 6,000,000,000
```

这样每个策略大约获得：

```text
per-policy samples per update = 1,572,864 / 3 = 524,288
per-policy total samples      = 6,000,000,000 / 3 = 2,000,000,000
```

这保证每个等级策略在 batch size、总样本量、PPO update 轮数上都与同质训练对齐。

## 4. 隔离性与轨迹切分保证

### 4.1 Agent 到策略的绑定

`mix_ppo` 在 `load_env()` 阶段根据 `mix_ppo_policy_mix` 生成 `mix_ppo_local_policy_ids`。对于 `l3:1,l2:1,l1:1`，每个 Python `Drive` 环境内的 PPO-controlled agent slots 会被 deficit-based interleaving 分配到三个策略。

示意：

```text
agent slot ids:
  0, 1, 2, 3, 4, 5, ...

policy ids:
  0, 1, 2, 0, 1, 2, ...

policy 0 -> Level 3: Drive + Recurrent
policy 1 -> Level 2: Drive
policy 2 -> Level 1: DriveLite
```

这个 policy id 在 rollout 中绑定到 agent slot。采样时 `_mix_ppo_forward_eval()` 根据 `self.agent_policy_ids[env_id]` 做 mask，只把某个策略对应的 agents 送入该策略网络。

因此不会出现 Level 1 agent 的 action 由 Level 3 policy 生成，或 Level 2 agent 的 rollout 被 Level 1 policy 控制。

### 4.2 Trajectory segment 到策略的绑定

rollout buffer 写入时，代码会把当前 agent slot 的 policy id 写入：

```text
self.segment_policy_ids[batch_rows] = self.agent_policy_ids[env_id]
```

训练时 `PuffeRL.train()` 使用：

```text
mb_policy_ids = self.segment_policy_ids[idx]
policy_mask = mb_policy_ids == policy_idx
```

然后每个 policy 只接收自己的 `mb_obs`、`mb_actions`、`mb_logprobs`、`mb_rewards`、`mb_values` 和 advantage。

因此不会出现一个策略等级使用另一个策略等级的经验训练。trajectory segment 的策略归属在采样写入时已经固定，训练时按 `segment_policy_ids` 分组。

### 4.3 Recurrent 与 feedforward 的时间维处理

这次新增 `mix_ppo_rnn_names` 后，混合训练可以同时包含 recurrent 和 feedforward policy：

```text
mix_ppo_policy_names = Drive,Drive,DriveLite
mix_ppo_rnn_names    = Recurrent,None,None
```

训练代码会为每个 policy 记录是否使用 RNN：

```text
policy_uses_rnn[0] = True
policy_uses_rnn[1] = False
policy_uses_rnn[2] = False
```

对于 Level 3 recurrent policy：

```text
mb_obs shape:
  (segments, bptt_horizon + 1, obs_dim)

policy input:
  mb_obs[:, :-1]
  shape = (segments, bptt_horizon, obs_dim)

LSTM wrapper 沿时间维展开。
```

对于 Level 2/1 feedforward policy：

```text
mb_obs shape:
  (segments, bptt_horizon + 1, obs_dim)

policy input:
  mb_obs[:, :-1].reshape(-1, obs_dim)
  shape = (segments * bptt_horizon, obs_dim)

policy 输出后再 reshape 回原始 logprob/value 的 segment-time 结构。
```

bootstrap value 也按 policy 类型分别处理：

```text
recurrent:
  bootstrap_obs = mb_obs[:, -1:]
  policy(bootstrap_obs, state, last_truncation, episode_ended=True)

feedforward:
  bootstrap_obs = mb_obs[:, -1:].reshape(-1, obs_dim)
  policy(bootstrap_obs, state)
```

这避免了 feedforward `Drive` / `DriveLite` 收到 LSTM-only 参数，也避免了 recurrent policy 丢失时间维。

### 4.4 旧功能兼容性

`mix_ppo_rnn_names` 是可选参数。不设置时，所有 mixed policies 仍然沿用全局 `rnn_name`。

旧命令：

```text
--train.mix-ppo-policy-names "Drive,DriveMoE,DriveMoE3"
```

在默认 `rnn_name=Recurrent` 下仍等价于：

```text
--train.mix-ppo-rnn-names "Recurrent,Recurrent,Recurrent"
```

因此之前的 MIXMOE / MIXMOE3x 训练命令不会因为本次改动改变行为。

## 5. 推荐训练启动命令

### 5.1 单卡 6B 等价训练版本

这个版本用于保证每个等级策略获得与同质 Drive + Recurrent 训练相同的 batch size 和总样本量。

```bash
cd "$HOME/wsc/behavior-bench"

export TMUX_SESSION="mix-intelligence-3level-6b-gpu0-$(date +%Y%m%d_%H%M%S)"

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
export RUN_NAME=\"mix_intelligence_3level_lstm_drive_drivelite_6b_batch3x\"
export CONFIG_PATH=\"pufferlib/config/ocean/drive.ini\"
export ENV_NUM_MAPS=\"10000\"
export TRAIN_BATCH_SIZE=\"1572864\"
export TOTAL_TIMESTEPS=\"6000000000\"

export ENABLE_MIX_PPO=\"True\"
export MIX_PPO_POLICY_MIX=\"l3:1,l2:1,l1:1\"
export MIX_PPO_POLICY_NAMES=\"Drive,Drive,DriveLite\"
export MIX_PPO_RNN_NAMES=\"Recurrent,None,None\"
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
export WANDB_GROUP=\"mix-intelligence-3level\"
export CUDA_VISIBLE_DEVICES=\"\$GPU_IDS\"
export TRAIN_LOG=\"\$LOG_DIR/\${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log\"

echo \"TRAIN_LOG=\$TRAIN_LOG\"
echo \"TRAIN_BATCH_SIZE=\$TRAIN_BATCH_SIZE\"
echo \"TOTAL_TIMESTEPS=\$TOTAL_TIMESTEPS\"
echo \"MIX_PPO_POLICY_NAMES=\$MIX_PPO_POLICY_NAMES\"
echo \"MIX_PPO_RNN_NAMES=\$MIX_PPO_RNN_NAMES\"

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
  --train.mix-ppo-rnn-names \"\$MIX_PPO_RNN_NAMES\" \
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

### 5.2 快速 smoke 版本

这个版本只用于确认命令和数据流跑通，不用于正式结论。

```bash
cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

export DRIVE_BINARIES_DATA_ROOT="$PWD/resources/drive/binaries"
export PUFFER_DISABLE_VIDEO=1

puffer train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  --train.name mix_intelligence_3level_smoke \
  --train.device cpu \
  --train.total-timesteps 16 \
  --train.batch-size 16 \
  --train.minibatch-size 16 \
  --train.max-minibatch-size 16 \
  --train.bptt-horizon 4 \
  --train.update-epochs 1 \
  --train.checkpoint-interval 999999 \
  --env.num-agents 4 \
  --env.num-maps 1 \
  --env.episode-length 2 \
  --env.resample-frequency 1000 \
  --env.init-mode create_all_valid \
  --env.control-mode control_agents \
  --train.mix-ppo True \
  --train.mix-ppo-policy-mix "l3:0.5,l2:0.25,l1:0.25" \
  --train.mix-ppo-policy-names "Drive,Drive,DriveLite" \
  --train.mix-ppo-rnn-names "Recurrent,None,None" \
  --eval.wosac-realism-eval False \
  --eval.human-replay-eval False
```

## 6. 训练后评估建议

正式训练完成后，建议分别导出或使用保存的三个 policy 权重做 held-out traffic 评估：

```text
Level 3 checkpoint:
  model_policy_0_*.pt

Level 2 checkpoint:
  model_policy_1_*.pt

Level 1 checkpoint:
  model_policy_2_*.pt
```

主要评估组合：

```text
pufferinter + IDM
pufferinter + PDM
pufferrandom + IDM
pufferrandom + PDM
```

核心比较不是三等级策略彼此谁最好，而是：

```text
同质 Drive + Recurrent 训练出的高等级策略
vs
三等级混合训练中的 Level 3 策略
```

如果混合训练中的 Level 3 在 held-out IDM/PDM 上 collision/fault collision 退化更小，同时 goal completion 保持可接受，则说明异质智能训练提升了高等级策略对未见交通行为的泛化能力。

## 7. 已做的实现级检查

当前代码层面已经做过以下 smoke checks：

```text
1. 三等级 mixed policy 加载结果：
   ['LSTMWrapper', 'Drive', 'DriveLite']

2. tiny mixed training smoke:
   完成一次 evaluate() + train()
   三个策略均有独立 p0/p1/p2 loss 和 stats

3. legacy mixed policy 默认行为：
   不设置 mix_ppo_rnn_names 时，旧 mix_ppo 仍全部使用全局 Recurrent wrapper

4. Python 编译检查：
   pufferlib/pufferl.py
   pufferlib/ocean/torch.py
   tests/test_mix_ppo.py
```

注意：当前环境没有可用的 `pytest` 模块，因此尚未运行完整 pytest 测试套件。
