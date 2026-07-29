# MOE 异质策略训练与验证总结

## 1. 实验思路

MOE 系列实验的目标是验证“网络结构异质性”是否能提升 PPO 策略在未见 traffic policy 上的泛化能力，尤其是 `pufferinter + IDM` 下的 collision / at-fault collision 表现。

该系列与后续 DriveLite 三等级实验不同：MOE 实验主要改变 **策略网络结构**，而不是显式定义“智能等级”。训练中混合的策略是：

| Policy Slot | 策略 | 结构含义 |
|---|---|---|
| p0 | `Drive` | 基础 Drive encoder，MLP + per-object max-pool |
| p1 | `DriveMoE` | 2-expert soft-router MoE，每个 expert 是 Drive-style encoder |
| p2 | `DriveMoE3` | 3-expert soft-router MoE |

`DriveMoE` 的核心想法是：每个 agent 根据当前 observation 通过 router 选择/组合多个 Drive-style expert embedding：

```text
obs
  -> expert_0(obs) -> hidden_0
  -> expert_1(obs) -> hidden_1
  -> router(obs)   -> weights [w0, w1]

mixed_hidden = w0 * hidden_0 + w1 * hidden_1
mixed_hidden -> actor/value
```

`DriveMoE3` 与 `DriveMoE` 相同，只是 expert 数从 2 个变成 3 个。

## 2. 策略结构与参数量

基于默认 `pufferlib/config/ocean/drive.ini`：

```text
observation_dim = 1151
policy.input_size = 64
policy.hidden_size = 256
rnn.input_size = 256
rnn.hidden_size = 256
action_dim = 91
```

| Policy | Base 参数量 | Recurrent 后参数量 | 主要结构 |
|---|---:|---:|---|
| `Drive` | 87,900 | 614,236 | 单个 Drive-style encoder |
| `DriveMoE` | 226,014 | 752,350 | 2 个 Drive-style experts + soft router |
| `DriveMoE3` | 290,335 | 816,671 | 3 个 Drive-style experts + soft router |

## 3. 训练实验

### 3.1 同质训练

同质训练分别训练单一策略：

| 实验 | Policy | 权重 | 训练日志 |
|---|---|---|---|
| Drive 同质 PPO repeat 1 | `Drive` | `experiments/puffer_drive_xt5biagd.pt` | `.logs/train/runs/ppo_drive_default_2b_20260703_182513.log` |
| Drive 同质 PPO repeat 2 | `Drive` | `experiments/puffer_drive_e3anj5yt.pt` | `.logs/train/runs/ppo_drive_default_2b_20260707_013818.log` |
| DriveMoE 同质 PPO repeat 1 | `DriveMoE` | `experiments/puffer_drive_aoljsqfo.pt` | `.logs/train/runs/ppo_drive_moe_2b_20260707_121524.log` |
| DriveMoE 同质 PPO repeat 2 | `DriveMoE` | `experiments/puffer_drive_jhme2lc6.pt` | `.logs/train/runs/ppo_drive_moe_2b_repeat_20260708_024753.log` |
| DriveMoE3 同质 PPO repeat 1 | `DriveMoE3` | `experiments/puffer_drive_vm8y7nks.pt` | `.logs/train/runs/ppo_drive_moe3_2b_20260708_031630.log` |
| DriveMoE3 同质 PPO repeat 2 | `DriveMoE3` | `experiments/puffer_drive_8qo9cvm7.pt` | `.logs/train/runs/ppo_drive_moe3_2b_repeat_20260708_031631.log` |

### 3.2 MIXMOE 训练

`MIXMOE` 是三策略混合训练：

```text
mix_ppo_policy_mix   = drive:1,moe:1,moe3:1
mix_ppo_policy_names = Drive,DriveMoE,DriveMoE3
```

| 实验 | 训练方式 | 权重目录 | 训练日志 |
|---|---|---|---|
| MIXMOE.1 | 2B, 1:1:1 混合 | `experiments/puffer_drive_x9earvkh/` | `.logs/train/runs/mix_ppo_drive_moe_moe3_1to1to1_default_2b_20260708_140532.log` |
| MIXMOE.2 | 2B, repeat | `experiments/puffer_drive_7u0evx5k/` | `.logs/train/runs/mix_ppo_drive_moe_moe3_1to1to1_default_2b_repeat2_20260708_140533.log` |
| MIXMOE3x.1 | 6B/batch3x fixed | `experiments/puffer_drive_nj7x0eeu/` | `.logs/train/runs/mix_ppo_drive_moe_moe3_1to1to1_default_6b_batch3x_fixed_20260710_025848.log` |
| MIXMOE3x.2 | 3-GPU DDP global6B | `experiments/puffer_drive_hilzy188/` | `.logs/train/runs/mix_ppo_drive_moe_moe3_1to1to1_ddp3_global6b_20260710_163222.log` |

## 4. pufferinter + IDM 验证结果

数据来源：

```text
/home/fanyuqi/wsc/behavior-bench/.logs/val/ppo_policy_eval_summary.md
```

### 4.1 同质策略

| 实验 | Repeat | Policy | 有效 maps | Goal | Goal% | Reward | Coll | Fault | Offroad |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Drive 同质 PPO | 1 | `Drive` | 589 | 499/589 | 84.7% | -0.70 | 4.1% | 2.4% | 0.7% |
| Drive 同质 PPO | 2 | `Drive` | 589 | 494/589 | 83.9% | -0.66 | 4.4% | 3.1% | 1.0% |
| DriveMoE 同质 PPO | 1 | `DriveMoE` | 589 | 492/589 | 83.5% | -0.57 | 3.4% | 1.5% | 0.8% |
| DriveMoE 同质 PPO | 2 | `DriveMoE` | 589 | 487/589 | 82.7% | -0.70 | 5.3% | 3.7% | 0.5% |
| DriveMoE3 同质 PPO | 1 | `DriveMoE3` | 589 | 489/589 | 83.0% | -0.70 | 4.2% | 3.1% | 0.3% |
| DriveMoE3 同质 PPO | 2 | `DriveMoE3` | 589 | 477/589 | 81.0% | -0.74 | 6.6% | 4.2% | 0.8% |

### 4.2 MIXMOE 最终 checkpoint

| 实验 | Repeat | Slot | Policy | 有效 maps | Goal | Goal% | Reward | Coll | Fault | Offroad |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| MIXMOE | 1 | p0 | `Drive` | 589 | 479/589 | 81.3% | -0.70 | 6.5% | 3.9% | 1.7% |
| MIXMOE | 1 | p1 | `DriveMoE` | 589 | 477/589 | 81.0% | -0.55 | 5.4% | 3.2% | 0.7% |
| MIXMOE | 1 | p2 | `DriveMoE3` | 589 | 486/589 | 82.5% | -0.66 | 5.8% | 3.9% | 0.8% |
| MIXMOE | 2 | p0 | `Drive` | 589 | 481/589 | 81.7% | -0.68 | 6.6% | 4.9% | 1.4% |
| MIXMOE | 2 | p1 | `DriveMoE` | 589 | 478/589 | 81.2% | -0.63 | 4.6% | 2.7% | 1.9% |
| MIXMOE | 2 | p2 | `DriveMoE3` | 589 | 478/589 | 81.2% | -0.63 | 4.8% | 3.2% | 1.2% |

### 4.3 MIXMOE3x 最终 checkpoint

| 实验 | Repeat | Slot | Policy | 训练方式 | 有效 maps | Goal | Goal% | Reward | Coll | Fault | Offroad |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Drive 同质 PPO | 1 | - | `Drive` | 单策略同质 | 589 | 499/589 | 84.7% | -0.70 | 4.1% | 2.4% | 0.7% |
| Drive 同质 PPO | 2 | - | `Drive` | 单策略同质 | 589 | 494/589 | 83.9% | -0.66 | 4.4% | 3.1% | 1.0% |
| MIXMOE3x | 1 | p0 | `Drive` | 单卡 batch3x fixed | 589 | 487/589 | 82.7% | -0.62 | 4.2% | 2.7% | 0.3% |
| MIXMOE3x | 1 | p1 | `DriveMoE` | 单卡 batch3x fixed | 589 | 479/589 | 81.3% | -0.60 | 4.1% | 2.5% | 0.7% |
| MIXMOE3x | 1 | p2 | `DriveMoE3` | 单卡 batch3x fixed | 589 | 472/589 | 80.1% | -0.50 | 3.6% | 2.0% | 1.0% |
| MIXMOE3x | 2 | p0 | `Drive` | 3 卡 DDP global6B | 589 | 484/589 | 82.2% | -0.71 | 5.6% | 3.6% | 0.8% |
| MIXMOE3x | 2 | p1 | `DriveMoE` | 3 卡 DDP global6B | 589 | 483/589 | 82.0% | -0.81 | 6.5% | 4.9% | 0.8% |
| MIXMOE3x | 2 | p2 | `DriveMoE3` | 3 卡 DDP global6B | 589 | 486/589 | 82.5% | -0.75 | 5.6% | 4.2% | 1.5% |

### 4.4 MIXMOE3x 1000 epoch

| 实验 | Repeat | Slot | Policy | 训练方式 | 有效 maps | Goal | Goal% | Reward | Coll | Fault | Offroad |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Drive 同质 PPO | 1 | - | `Drive` | 单策略同质 | 589 | 470/589 | 79.8% | -0.76 | 6.5% | 4.9% | 1.5% |
| Drive 同质 PPO | 2 | - | `Drive` | 单策略同质 | 589 | 479/589 | 81.3% | -0.67 | 4.9% | 3.2% | 1.5% |
| MIXMOE3x | 1 | p0 | `Drive` | 单卡 batch3x fixed | 589 | 463/589 | 78.6% | -0.80 | 9.0% | 7.1% | 1.9% |
| MIXMOE3x | 1 | p1 | `DriveMoE` | 单卡 batch3x fixed | 589 | 444/589 | 75.4% | -0.80 | 9.0% | 7.6% | 5.4% |
| MIXMOE3x | 1 | p2 | `DriveMoE3` | 单卡 batch3x fixed | 589 | 454/589 | 77.1% | -0.80 | 8.0% | 5.9% | 3.9% |
| MIXMOE3x | 2 | p0 | `Drive` | 3 卡 DDP global6B | 589 | 469/589 | 79.6% | -0.78 | 7.8% | 5.8% | 2.0% |
| MIXMOE3x | 2 | p1 | `DriveMoE` | 3 卡 DDP global6B | 589 | 447/589 | 75.9% | -0.91 | 9.7% | 8.8% | 3.6% |
| MIXMOE3x | 2 | p2 | `DriveMoE3` | 3 卡 DDP global6B | 589 | 459/589 | 77.9% | -0.89 | 9.2% | 6.8% | 2.9% |

## 5. 重要日志与权重索引

| 实验 | 权重 | 典型 pufferinter + IDM 日志 |
|---|---|---|
| DriveMoE 同质 repeat 1 | `experiments/puffer_drive_aoljsqfo.pt` | `.logs/val/runs/2MOE.1/eval_pufferinter_moeppo_final_vs_idm_gpu2_20260707_181554.log` |
| DriveMoE 同质 repeat 2 | `experiments/puffer_drive_jhme2lc6.pt` | `.logs/val/runs/2MOE.2/eval_pufferinter_moeppo_repeat_vs_idm_gpu0_20260708_134804.log` |
| DriveMoE3 同质 repeat 1 | `experiments/puffer_drive_vm8y7nks.pt` | `.logs/val/runs/3MOE.1/eval_pufferinter_moe3ppo_vs_idm_gpu0_20260708_173012.log` |
| DriveMoE3 同质 repeat 2 | `experiments/puffer_drive_8qo9cvm7.pt` | `.logs/val/runs/3MOE.2/eval_pufferinter_moe3ppo_repeat_vs_idm_gpu6_20260708_173409.log` |
| MIXMOE.1 p0 | `experiments/puffer_drive_x9earvkh/model_policy_0_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.1/p0/eval_mixmoe_p0_pufferinter_Drive_vs_idm_gpu1_20260708_221827.log` |
| MIXMOE.1 p1 | `experiments/puffer_drive_x9earvkh/model_policy_1_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.1/p1/eval_mixmoe_p1_pufferinter_DriveMoE_vs_idm_gpu5_20260708_221831.log` |
| MIXMOE.1 p2 | `experiments/puffer_drive_x9earvkh/model_policy_2_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.1/p2/eval_mixmoe_p2_pufferinter_DriveMoE3_vs_idm_gpu3_20260708_233710.log` |
| MIXMOE.2 p0 | `experiments/puffer_drive_7u0evx5k/model_policy_0_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.2/p0/eval_mixmoe2_p0_pufferinter_Drive_vs_idm_gpu2_20260709_003222.log` |
| MIXMOE.2 p1 | `experiments/puffer_drive_7u0evx5k/model_policy_1_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.2/p1/eval_mixmoe2_p1_pufferinter_DriveMoE_vs_idm_gpu7_20260709_003226.log` |
| MIXMOE.2 p2 | `experiments/puffer_drive_7u0evx5k/model_policy_2_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.2/p2/eval_mixmoe2_p2_pufferinter_DriveMoE3_vs_idm_gpu3_20260709_003559.log` |
| MIXMOE3x.1 p0 | `experiments/puffer_drive_nj7x0eeu/model_policy_0_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE3x.1/p0/eval_mixmoe3x_p0_pufferinter_Drive_vs_idm_gpu7_20260710_232705.log` |
| MIXMOE3x.1 p1 | `experiments/puffer_drive_nj7x0eeu/model_policy_1_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE3x.1/p1/eval_mixmoe3x_p1_pufferinter_DriveMoE_vs_idm_gpu7_20260710_232705.log` |
| MIXMOE3x.1 p2 | `experiments/puffer_drive_nj7x0eeu/model_policy_2_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE3x.1/p2/eval_mixmoe3x_p2_pufferinter_DriveMoE3_vs_idm_gpu7_20260710_232705.log` |
| MIXMOE3x.2 p0 | `experiments/puffer_drive_hilzy188/model_policy_0_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE3x.2/p0/eval_mixmoe3x2_p0_pufferinter_Drive_vs_idm_gpu7_20260711_005059.log` |
| MIXMOE3x.2 p1 | `experiments/puffer_drive_hilzy188/model_policy_1_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE3x.2/p1/eval_mixmoe3x2_p1_pufferinter_DriveMoE_vs_idm_gpu7_20260711_005059.log` |
| MIXMOE3x.2 p2 | `experiments/puffer_drive_hilzy188/model_policy_2_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE3x.2/p2/eval_mixmoe3x2_p2_pufferinter_DriveMoE3_vs_idm_gpu7_20260711_005059.log` |

## 6. 结论

### 6.1 单策略 MoE 没有稳定优于 Drive baseline

同质 `DriveMoE` repeat 1 在 `pufferinter + IDM` 上 collision/fault 较好：

```text
DriveMoE r1: Coll 3.4%, Fault 1.5%
Drive r1:    Coll 4.1%, Fault 2.4%
```

但 `DriveMoE` repeat 2 退化明显：

```text
DriveMoE r2: Coll 5.3%, Fault 3.7%
Drive r2:    Coll 4.4%, Fault 3.1%
```

说明 MoE 单策略存在较大 seed / 训练轨迹方差，不能仅凭一次 repeat 得出稳定提升。

### 6.2 结构异质 MIXMOE 没有带来明确泛化收益

`MIXMOE` 两个 repeat 在 `pufferinter + IDM` 上的 p0/p1/p2 均未超过同质 Drive baseline，尤其是 p0 Drive 在混合训练后 collision 从 baseline 的 4.1%/4.4% 上升到 6.5%/6.6%。

这说明单纯混合 `Drive / DriveMoE / DriveMoE3` 这种“结构异质”并不必然改善对 IDM traffic 的泛化。

### 6.3 batch3x 修正后，MIXMOE3x 有改善但仍未超越 Drive baseline

`MIXMOE3x.1` 使用 3 倍 batch，使每个策略每次 update 获得接近同质训练的样本量。最终 checkpoint 下：

```text
MIXMOE3x.1 p0 Drive:     Coll 4.2%, Fault 2.7%
MIXMOE3x.1 p1 DriveMoE:  Coll 4.1%, Fault 2.5%
MIXMOE3x.1 p2 DriveMoE3: Coll 3.6%, Fault 2.0%
```

它比早期 1000 epoch 明显改善，也比普通 MIXMOE 好，但 Goal% 仍低于 Drive 同质 baseline：

```text
Drive baseline r1/r2 Goal: 84.7% / 83.9%
MIXMOE3x.1 Goal:           82.7% / 81.3% / 80.1%
```

### 6.4 结论倾向

MOE 实验给出的结论更偏负面：

```text
结构异质性本身不足以稳定提升未见 IDM 车流泛化。
```

它帮助确认了后续 DriveLite / 混合智能等级实验的必要性：相比“不同网络结构”，更可解释、更接近论文叙事的异质性应来自策略能力等级，例如是否具备 LSTM 时序 belief、是否使用低复杂度 reactive policy。

### 6.5 后续建议

如果后续还继续做 MoE，可以考虑：

1. 不再单独扩大 expert 数，而是把 MoE 作为高等级策略内部结构，与行为预测/behavior-aware latent 结合。
2. 对 router 权重做诊断，看它是否真的区分不同交通场景，还是退化成平均 ensemble。
3. 做多 seed 均值/方差，而不是单次 repeat。
4. 与 DriveLite 结果对照，强调“结构异质”和“智能等级异质”的区别。
