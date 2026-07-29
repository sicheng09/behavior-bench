# Run3: Adversarial Mixed Training (Ego + Primary Opponent)

参考 `.logs/train/train-muliti-ppo.md` 的正式默认训练模板，启动对抗混合训练。

## 角色

- policy 0 `ego`：原 Drive 奖励 + Drive/Recurrent
- policy 1 `primary_opponent`：非对称对抗奖励 + Drive/Recurrent

## 预算对齐（相对同质 Drive+Recurrent）

| 项目 | 同质 | 单卡 batch2x + 4B（2 策略 50:50） |
|------|------|----------------------------------|
| 全局 `batch_size` | 524288 | **1048576** (`batch2x`) |
| 全局 `minibatch_size` | 32768 | **65536** (`mb2x`) |
| `total_timesteps` | 2B | **4B** |
| 每策略每 update 样本量 | ~524288 | ~524288 |
| update 次数 | ~3815 | ~3815 |
| 每策略总样本 | ~2B | ~2B |

## 单卡 batch2x + 4B

单进程把 batch / total 放大 2 倍，使每个策略仍看到同质默认的每 update 样本量与总样本：

```bash
bash /home/fanyuqi/wsc/behavior-bench/.logs/train/run3/start_adversarial_mix_batch2x_4b.sh
```

- 地图：`resources/drive/binaries`，`num_maps=10000`
- GPU：`0`
- W&B group：`adversarial-mix-run3`

```bash
tmux attach -t adv-mix-drive-lstm-batch2x-4b-gpu0
tail -f /home/fanyuqi/wsc/behavior-bench/.logs/train/run3/adv_mix_ego_opponent_drive_lstm_batch2x_4b_run3_*.log
```

## 2-GPU DDP（推荐，对齐单卡 batch2x + 4B）

2 策略 50:50 用双卡更自然：每卡默认预算，全局刚好等于上面的单卡 run。

| 项目 | 单卡 batch2x + 4B | DDP2 每卡 | 全局（×2） |
|------|-------------------|-----------|------------|
| `batch_size` | 1048576 | **524288** | 1048576 |
| `minibatch_size` | 65536 | **32768** | — |
| `total_timesteps` | 4B | **2B** | ≈4B |
| updates | ~3815 | ~3815 | ~3815 |

```bash
bash /home/fanyuqi/wsc/behavior-bench/.logs/train/run3/start_adversarial_mix_ddp2_batch2x_4b.sh
```

- GPU：`1,2`
- W&B group：`adversarial-mix-run3-ddp`

```bash
tmux attach -t adv-mix-drive-lstm-ddp2-gpu12
tail -f /home/fanyuqi/wsc/behavior-bench/.logs/train/run3/adv_mix_ego_opponent_drive_lstm_ddp2_batch2x_4b_run3_*.log
```

### 短训 500M（双卡）

每卡 `total_timesteps=500M`（全局 ≈1B，updates ≈953），GPU `3,4`：

```bash
bash /home/fanyuqi/wsc/behavior-bench/.logs/train/run3/start_adversarial_mix_ddp2_500m.sh
```

```bash
tmux attach -t adv-mix-drive-lstm-ddp2-500m-gpu34
tail -f /home/fanyuqi/wsc/behavior-bench/.logs/train/run3/adv_mix_ego_opponent_drive_lstm_ddp2_500m_run3_*.log
```

### 消融：弱目标项 500M（双卡）

相对上面的 baseline（`opponent_mix.yaml`，`goal=0`），仅把对手奖励改为
`opponent_mix_weak_goal.yaml`（`goal=0.15` 稀疏到达奖励，仍受 `positive_reward_cap=0.25` 约束），GPU `5,6`：

```bash
bash /home/fanyuqi/wsc/behavior-bench/.logs/train/run3/start_adversarial_mix_ddp2_500m_weak_goal.sh
```

```bash
tmux attach -t adv-mix-drive-lstm-ddp2-500m-weak-goal-gpu56
tail -f /home/fanyuqi/wsc/behavior-bench/.logs/train/run3/adv_mix_ego_opponent_drive_lstm_ddp2_500m_weak_goal_run3_*.log
```

旧的三卡脚本 `start_adversarial_mix_ddp3_batch2x_4b.sh` 不推荐用于本 2 策略配方（全局会变成 batch3x / 6B）。

## 验证（500M baseline → ego vs IDM）

训练 `adv-mix-drive-lstm-ddp2-500m-gpu34` 结束后，自动在 `.logs/val/run3` 评测
（pufferinter + IDM，格式同 run1/run2）：

```bash
bash /home/fanyuqi/wsc/behavior-bench/.logs/val/run3/wait_and_eval_ddp2_500m.sh
```

结果目录：`.logs/val/run3/results/adv_mix_ego_opponent_drive_lstm_ddp2_500m_run3/p0_ego_Drive_Recurrent/`

弱目标消融结束后同样评测：

```bash
bash /home/fanyuqi/wsc/behavior-bench/.logs/val/run3/wait_and_eval_ddp2_500m_weak_goal.sh
```

结果目录：`.logs/val/run3/results/adv_mix_ego_opponent_drive_lstm_ddp2_500m_weak_goal_run3/p0_ego_Drive_Recurrent/`

### 消融：Adv + 50% IDM 背景车（500M 双卡）

相对 baseline（`mix_traffic=False`），打开混合交通：
`mix_traffic=True`, `ppo_fraction=0.5`, `idm_fraction=0.5`。
PPO 半边内仍 `ego:opponent = 0.5:0.5`；奖励权重同默认，但 YAML 用
`opponent_mix_idm50.yaml`（仅把 `warn_mixed_scene_rate_below` 降到 0.5，
避免 IDM 稀释后的 mixed_scene_rate 警告被 `pufferl` 当成错误）。GPU `0,1`：

```bash
bash /home/fanyuqi/wsc/behavior-bench/.logs/train/run3/start_adversarial_mix_ddp2_500m_idm50.sh
```

```bash
tmux attach -t adv-mix-drive-lstm-ddp2-500m-idm50-gpu01
tail -f /home/fanyuqi/wsc/behavior-bench/.logs/train/run3/adv_mix_ego_opponent_drive_lstm_ddp2_500m_idm50_run3_*.log
```

### 对照：纯 PPO + 50% IDM（无对抗，单卡 500M）

`puffer_drive` + Drive/Recurrent，`mix_traffic=True`，`ppo/idm=0.5/0.5`，**无** `mix_ppo` / 无对手。
预算对齐同质单卡：`batch=524288`，`total_timesteps=500M`。GPU `2`：

```bash
bash /home/fanyuqi/wsc/behavior-bench/.logs/train/run3/start_ppo_idm50_500m.sh
```

```bash
tmux attach -t ppo-idm50-drive-lstm-500m-gpu2
tail -f /home/fanyuqi/wsc/behavior-bench/.logs/train/run3/ppo_idm50_drive_lstm_500m_run3_*.log
```

旧的双卡误开局日志已归档为
`ppo_idm50_drive_lstm_ddp2_500m_run3_*_CANCELLED.log`（勿用作结果）。
