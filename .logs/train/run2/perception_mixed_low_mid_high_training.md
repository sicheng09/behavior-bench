# Run2 感知层混合智能训练设计

## 1. 实验目标

本次 run2 目标是在当前 `behavior-bench` 代码库中，非侵入复现 `/home/fanyuqi/pufferdrive_re/docs/exp` 里的感知层混智核心思想。

核心思路是：不改变环境、不改变 C binding、不改变默认 `Drive` 策略，而是新增三个 perception-masked policy class，并通过当前仓库已有的 `mix_ppo` 机制进行多策略混合训练。

本次迁移的不是 population / candidate / semantic enhancement 全套机制，而是其中最核心的 low/mid/high 感知等级：

| 等级 | 策略类 | 感知定义 | 训练语义 |
|---:|---|---|---|
| 0 | `DrivePerceptionLow` | 前方 60 度扇形，半径 50m | 低感知智能体 |
| 1 | `DrivePerceptionMid` | 前方 120 度扇形，半径 50m | 中感知智能体 |
| 2 | `Drive` 或 `DrivePerceptionHigh` | 原始全局 observation | 高感知智能体 |

## 2. 非侵入实现方式

已新增策略类：

```text
pufferlib/ocean/torch.py
  DrivePerceptionMasked
  DrivePerceptionLow
  DrivePerceptionMid
  DrivePerceptionHigh
```

这些策略的外部 observation 维度与 `Drive` 完全一致：

```text
obs_dim = 1151
ego      = 7
partners = 31 * 8
roads    = 128 * 7
```

差异只发生在 policy 内部：

```text
observations
  -> apply_perception_mask()
  -> Drive.encode_observations()
  -> Recurrent
  -> actor/value
```

因此：

- 默认 `Drive` 行为不变。
- 旧训练命令不变。
- 旧评估命令不变。
- 不修改环境物理状态。
- 不删除真实车辆，只把 policy 不可见的 observation object 特征置零。

## 3. Perception Mask 规则

高感知：

```text
DrivePerceptionHigh 或 Drive:
  不做 mask，保留原始 observation。
```

低/中感知：

```text
ego features: 保留
partner objects: 超出可见扇形/半径则整行置零
road objects: 超出可见扇形/半径则整行置零
padding objects: 保持零
```

可见性计算：

```text
normalized_radius = perception_radius * 0.02
dist = sqrt(rel_x^2 + rel_y^2)
angle = atan2(rel_y, rel_x)
visible = non_empty and dist <= normalized_radius and abs(angle) <= sector_degrees / 2
```

当前默认：

```text
perception_radius = 50.0
normalized_radius = 1.0
low sector = 60 deg  -> +/-30 deg
mid sector = 120 deg -> +/-60 deg
high = unmasked
```

## 4. 与 pufferdrive_re 的关系

`pufferdrive_re` 的正式混智增强使用：

```text
DriveMixedPolicy
MixedRecurrent
group_ids
repeat_policy_samples = True
repeat_policy_factors = [3,3,3]
```

当前仓库不直接搬这套内部 group/rnn 框架，而是用已有 `mix_ppo` 复现核心语义：

```text
policy_id 0 -> DrivePerceptionLow
policy_id 1 -> DrivePerceptionMid
policy_id 2 -> Drive / DrivePerceptionHigh
```

二者对应关系：

| pufferdrive_re | 当前 behavior-bench |
|---|---|
| `group_id` | `policy_id` / `segment_policy_ids` |
| `DriveMixedPolicy.level_policies` | `mix_ppo` 的多 policy ModuleList |
| `MixedRecurrent` 每 level 一个 LSTM | 每个 mixed policy 自己包 `Recurrent` |
| `repeat=3` 使 optimizer-seen 样本等价 | batch size 和 total timesteps 放大 3 倍 |

当前方式更非侵入，也更贴近 PPO on-policy 采样：不是重复同一批样本，而是为三个策略采集更多 raw rollout。

## 5. 标准训练命令

本次 run2 建议使用当前仓库的标准大规模训练方式：单卡、`6B` 总步数、`batch3x`，使每个感知等级策略获得接近同质训练的 batch 和总样本量。

```bash
cd "$HOME/wsc/behavior-bench"

export TMUX_SESSION="perception-mixed-low-mid-high-run2-gpu0-$(date +%Y%m%d_%H%M%S)"

tmux new-session -d -s "$TMUX_SESSION" "bash -lc '
set -euo pipefail

cd \"$HOME/wsc/behavior-bench\"
source \"$HOME/miniconda3/etc/profile.d/conda.sh\"
conda activate behavior-bench

export REPO_ROOT=\"\$(pwd)\"
export DRIVE_BINARIES_DATA_ROOT=\"\$REPO_ROOT/resources/drive/binaries\"
export LOG_DIR=\"\$REPO_ROOT/.logs/train/run2\"
mkdir -p \"\$LOG_DIR\"

export GPU_IDS=\"0\"
export RUN_NAME=\"perception_mixed_low_mid_high_6b_batch3x_run2\"
export CONFIG_PATH=\"pufferlib/config/ocean/drive.ini\"
export ENV_NUM_MAPS=\"10000\"
export TRAIN_BATCH_SIZE=\"1572864\"
export TOTAL_TIMESTEPS=\"6000000000\"

export ENABLE_MIX_PPO=\"True\"
export MIX_PPO_POLICY_MIX=\"low:1,mid:1,high:1\"
export MIX_PPO_POLICY_NAMES=\"DrivePerceptionLow,DrivePerceptionMid,Drive\"
export MIX_PPO_RNN_NAMES=\"Recurrent,Recurrent,Recurrent\"
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
export WANDB_GROUP=\"perception-mixed-run2\"
export CUDA_VISIBLE_DEVICES=\"\$GPU_IDS\"
export TRAIN_LOG=\"\$LOG_DIR/\${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log\"

echo \"TRAIN_LOG=\$TRAIN_LOG\"
echo \"GPU_IDS=\$GPU_IDS\"
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

## 6. 参数说明

### 6.1 混合策略参数

```text
--train.mix-ppo True
```

开启当前仓库已有的多策略 PPO-controlled agent 混合训练。

```text
--train.mix-ppo-policy-mix "low:1,mid:1,high:1"
```

表示三个感知等级按 1:1:1 分配 agent slots。

```text
--train.mix-ppo-policy-names "DrivePerceptionLow,DrivePerceptionMid,Drive"
```

三种策略分别为：

| Slot | Policy | 感知 |
|---|---|---|
| p0 | `DrivePerceptionLow` | 前方 60 度，50m |
| p1 | `DrivePerceptionMid` | 前方 120 度，50m |
| p2 | `Drive` | 全局 observation |

```text
--train.mix-ppo-rnn-names "Recurrent,Recurrent,Recurrent"
```

每个感知等级都保留独立 LSTM。这里复现 pufferdrive_re 的“每个 level 一个 LSTM”语义。

### 6.2 样本量等价参数

同质训练默认：

```text
batch_size = 524288
total_timesteps = 2000000000
```

三等级混合训练使用：

```text
batch_size = 1572864 = 3 * 524288
total_timesteps = 6000000000 = 3 * 2B
```

这样每个感知等级策略约获得：

```text
per-policy batch ~= 524288
per-policy total samples ~= 2B
```

注意：`env.num_agents=1024` 不能被 3 整除，因此每个策略的 agent slot 数会有约 0.1% 的整数误差，但整体训练预算近似等价。

### 6.3 环境与 traffic 参数

```text
--env.mix-traffic False
--env.ppo-fraction 1.0
--env.idm-fraction 0.0
--env.expert-fraction 0.0
```

表示本实验的异质性来自 PPO-controlled agents 内部的感知等级混合，而不是 C 层 IDM/Expert traffic 混合。

## 7. 验证建议

训练完成后，每个 mixed policy 会保存独立权重：

```text
model_policy_0_puffer_drive_003815.pt  # low
model_policy_1_puffer_drive_003815.pt  # mid
model_policy_2_puffer_drive_003815.pt  # high
```

建议在 held-out traffic 上分别评估：

```text
pufferinter + idm
pufferinter + pdm
pufferrandom + idm
pufferrandom + pdm
```

其中首轮最小验证可以只跑：

```text
pufferinter + idm
```

示例：

```bash
python pufferlib/ocean/benchmark/eval.py \
  --output-dir .logs/val/run2/results/perception_mixed_low_mid_high/p0_DrivePerceptionLow \
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

对应：

```text
p0 -> DrivePerceptionLow
p1 -> DrivePerceptionMid
p2 -> Drive
```

## 8. 已完成的本地检查

新增 focused tests：

```text
tests.test_mix_ppo.TestMixPPO.test_drive_perception_low_masks_objects_outside_front_sector
tests.test_mix_ppo.TestMixPPO.test_drive_perception_mid_keeps_wider_front_sector_than_low
tests.test_mix_ppo.TestMixPPO.test_mix_ppo_loads_perception_level_policies
```

执行：

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

同时已跑 tiny training smoke：

```text
['DrivePerceptionLow', 'DrivePerceptionMid', 'Drive']
perception mix tiny train smoke ok
```

## 9. 注意事项

1. `DrivePerceptionHigh` 与 `Drive` 在感知上等价；训练命令中可以直接使用 `Drive` 作为 high。
2. low/mid 不改变真实交通状态，只改变策略可见 observation。
3. 当前实现使用 `mix_ppo` 的 agent-slot 固定分配，不使用 pufferdrive_re 的 `group_ids` 内部路由。
4. 当前实现使用 batch/steps 放大 3 倍来保证样本量近似等价，不使用 pufferdrive_re 的 `repeat_policy_samples=True`。
5. 如果后续要完全复现 pufferdrive_re 的 population/candidate 机制，需要额外设计，不建议作为 run2 首版目标。
