# PPO 策略 Benchmark 验证结果汇总

生成时间：2026-07-09 18:56:20

数据来源：`.logs/val/runs/` 下各次 benchmark eval wrapper 日志的最终 `PufferDrive Eval` 汇总块。

说明：
- `有效 maps` 是日志最终显示的 `PufferDrive Eval <n>/1000 maps`，由于 active-agent 过滤，通常小于 1000。
- `Goal%` 由 `Goal Reached a/b` 计算。
- `Traffic=ppo` 时 planner 与 traffic 都显式加载同一路径权重；`Traffic=idm` 时 traffic 使用 IDM。
- `MIXMOE` 中 `p0=Drive`，`p1=DriveMoE`，`p2=DriveMoE3`，均使用异质训练最终 `003815` 单策略权重。

## pufferinter + IDM 指标对比

| 实验 | Repeat | Policy Slot | Policy | 有效 maps | Goal | Goal% | Reward | Coll | Fault | Offroad |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| Drive 同质 PPO | 1 | - | Drive | 589 | 499/589 | 84.7% | -0.70 | 4.1% | 2.4% | 0.7% |
| Drive 同质 PPO | 2 | - | Drive | 589 | 494/589 | 83.9% | -0.66 | 4.4% | 3.1% | 1.0% |
| DriveMoE 同质 PPO | 1 | - | DriveMoE | 589 | 492/589 | 83.5% | -0.57 | 3.4% | 1.5% | 0.8% |
| DriveMoE 同质 PPO | 2 | - | DriveMoE | 589 | 487/589 | 82.7% | -0.70 | 5.3% | 3.7% | 0.5% |
| DriveMoE3 同质 PPO | 1 | - | DriveMoE3 | 589 | 489/589 | 83.0% | -0.70 | 4.2% | 3.1% | 0.3% |
| DriveMoE3 同质 PPO | 2 | - | DriveMoE3 | 589 | 477/589 | 81.0% | -0.74 | 6.6% | 4.2% | 0.8% |
| MIXMOE | 1 | p0 | Drive | 589 | 479/589 | 81.3% | -0.70 | 6.5% | 3.9% | 1.7% |
| MIXMOE | 1 | p1 | DriveMoE | 589 | 477/589 | 81.0% | -0.55 | 5.4% | 3.2% | 0.7% |
| MIXMOE | 1 | p2 | DriveMoE3 | 589 | 486/589 | 82.5% | -0.66 | 5.8% | 3.9% | 0.8% |
| MIXMOE | 2 | p0 | Drive | 589 | 481/589 | 81.7% | -0.68 | 6.6% | 4.9% | 1.4% |
| MIXMOE | 2 | p1 | DriveMoE | 589 | 478/589 | 81.2% | -0.63 | 4.6% | 2.7% | 1.9% |
| MIXMOE | 2 | p2 | DriveMoE3 | 589 | 478/589 | 81.2% | -0.63 | 4.8% | 3.2% | 1.2% |

## pufferinter + PPO 指标对比

| 实验 | Repeat | Policy Slot | Policy | 有效 maps | Goal | Goal% | Reward | Coll | Fault | Offroad |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| Drive 同质 PPO | 1 | - | Drive | 589 | 532/589 | 90.3% | -0.64 | 0.2% | 0.2% | 0.3% |
| Drive 同质 PPO | 2 | - | Drive | 589 | 533/589 | 90.5% | -0.63 | 0.2% | 0.2% | 0.2% |
| DriveMoE 同质 PPO | 1 | - | DriveMoE | 589 | 529/589 | 89.8% | -0.65 | 0.5% | 0.5% | 0.3% |
| DriveMoE 同质 PPO | 2 | - | DriveMoE | 589 | 534/589 | 90.7% | -0.61 | 0.0% | 0.0% | 0.2% |
| DriveMoE3 同质 PPO | 1 | - | DriveMoE3 | 589 | 530/589 | 90.0% | -0.63 | 0.7% | 0.5% | 0.2% |
| DriveMoE3 同质 PPO | 2 | - | DriveMoE3 | 589 | 533/589 | 90.5% | -0.62 | 0.2% | 0.0% | 0.2% |
| MIXMOE | 1 | p0 | Drive | 589 | 527/589 | 89.5% | -0.65 | 1.0% | 0.5% | 0.2% |
| MIXMOE | 1 | p1 | DriveMoE | 589 | 528/589 | 89.6% | -0.62 | 0.7% | 0.3% | 0.3% |
| MIXMOE | 1 | p2 | DriveMoE3 | 589 | 531/589 | 90.2% | -0.64 | 0.2% | 0.0% | 0.5% |
| MIXMOE | 2 | p0 | Drive | 589 | 527/589 | 89.5% | -0.64 | 0.5% | 0.3% | 0.8% |
| MIXMOE | 2 | p1 | DriveMoE | 589 | 526/589 | 89.3% | -0.63 | 0.2% | 0.2% | 0.8% |
| MIXMOE | 2 | p2 | DriveMoE3 | 589 | 531/589 | 90.2% | -0.65 | 0.3% | 0.2% | 0.3% |

## 同质策略结果

| 实验 | Repeat | Policy | Split | Traffic | 有效 maps | Goal | Goal% | Reward | Coll | Fault | Offroad | 状态 |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Drive 同质 PPO | 1 | Drive | pufferinter | idm | 589 | 499/589 | 84.7% | -0.70 | 4.1% | 2.4% | 0.7% | 完成 |
| Drive 同质 PPO | 1 | Drive | pufferinter | ppo | 589 | 532/589 | 90.3% | -0.64 | 0.2% | 0.2% | 0.3% | 完成 |
| Drive 同质 PPO | 1 | Drive | pufferrandom | idm | 620 | 535/620 | 86.3% | -0.25 | 2.1% | 1.1% | 1.1% | 完成 |
| Drive 同质 PPO | 1 | Drive | pufferrandom | ppo | 620 | 560/620 | 90.3% | -0.16 | 0.0% | 0.0% | 0.5% | 完成 |
| Drive 同质 PPO | 2 | Drive | pufferinter | idm | 589 | 494/589 | 83.9% | -0.66 | 4.4% | 3.1% | 1.0% | 完成 |
| Drive 同质 PPO | 2 | Drive | pufferinter | ppo | 589 | 533/589 | 90.5% | -0.63 | 0.2% | 0.2% | 0.2% | 完成 |
| Drive 同质 PPO | 2 | Drive | pufferrandom | idm | 620 | 542/620 | 87.4% | -0.19 | 1.5% | 0.6% | 1.1% | 完成 |
| Drive 同质 PPO | 2 | Drive | pufferrandom | ppo | 620 | 560/620 | 90.3% | -0.15 | 0.3% | 0.3% | 0.5% | 完成 |
| DriveMoE 同质 PPO | 1 | DriveMoE | pufferinter | idm | 589 | 492/589 | 83.5% | -0.57 | 3.4% | 1.5% | 0.8% | 完成 |
| DriveMoE 同质 PPO | 1 | DriveMoE | pufferinter | ppo | 589 | 529/589 | 89.8% | -0.65 | 0.5% | 0.5% | 0.3% | 完成 |
| DriveMoE 同质 PPO | 1 | DriveMoE | pufferrandom | idm | 620 | 538/620 | 86.8% | -0.16 | 2.3% | 1.6% | 1.1% | 完成 |
| DriveMoE 同质 PPO | 1 | DriveMoE | pufferrandom | ppo | 620 | 561/620 | 90.5% | -0.15 | 0.0% | 0.0% | 0.5% | 完成 |
| DriveMoE 同质 PPO | 2 | DriveMoE | pufferinter | idm | 589 | 487/589 | 82.7% | -0.70 | 5.3% | 3.7% | 0.5% | 完成 |
| DriveMoE 同质 PPO | 2 | DriveMoE | pufferinter | ppo | 589 | 534/589 | 90.7% | -0.61 | 0.0% | 0.0% | 0.2% | 完成 |
| DriveMoE 同质 PPO | 2 | DriveMoE | pufferrandom | idm | 620 | 535/620 | 86.3% | -0.23 | 3.2% | 1.9% | 0.6% | 完成 |
| DriveMoE 同质 PPO | 2 | DriveMoE | pufferrandom | ppo | 620 | 561/620 | 90.5% | -0.15 | 0.2% | 0.2% | 0.5% | 完成 |
| DriveMoE3 同质 PPO | 1 | DriveMoE3 | pufferinter | idm | 589 | 489/589 | 83.0% | -0.70 | 4.2% | 3.1% | 0.3% | 完成 |
| DriveMoE3 同质 PPO | 1 | DriveMoE3 | pufferinter | ppo | 589 | 530/589 | 90.0% | -0.63 | 0.7% | 0.5% | 0.2% | 完成 |
| DriveMoE3 同质 PPO | 1 | DriveMoE3 | pufferrandom | idm | 620 | 532/620 | 85.8% | -0.22 | 3.1% | 2.1% | 0.6% | 完成 |
| DriveMoE3 同质 PPO | 1 | DriveMoE3 | pufferrandom | ppo | 620 | 561/620 | 90.5% | -0.16 | 0.3% | 0.0% | 0.3% | 完成 |
| DriveMoE3 同质 PPO | 2 | DriveMoE3 | pufferinter | idm | 589 | 477/589 | 81.0% | -0.74 | 6.6% | 4.2% | 0.8% | 完成 |
| DriveMoE3 同质 PPO | 2 | DriveMoE3 | pufferinter | ppo | 589 | 533/589 | 90.5% | -0.62 | 0.2% | 0.0% | 0.2% | 完成 |
| DriveMoE3 同质 PPO | 2 | DriveMoE3 | pufferrandom | idm | 620 | 535/620 | 86.3% | -0.21 | 2.7% | 1.8% | 0.8% | 完成 |
| DriveMoE3 同质 PPO | 2 | DriveMoE3 | pufferrandom | ppo | 620 | 558/620 | 90.0% | -0.16 | 0.5% | 0.0% | 0.6% | 完成 |

## 异质 MIXMOE 单策略结果

| Repeat | Policy Slot | Policy | Split | Traffic | 有效 maps | Goal | Goal% | Reward | Coll | Fault | Offroad | 状态 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | p0 | Drive | pufferinter | idm | 589 | 479/589 | 81.3% | -0.70 | 6.5% | 3.9% | 1.7% | 完成 |
| 1 | p0 | Drive | pufferinter | ppo | 589 | 527/589 | 89.5% | -0.65 | 1.0% | 0.5% | 0.2% | 完成 |
| 1 | p0 | Drive | pufferrandom | idm | 620 | 540/620 | 87.1% | -0.22 | 2.6% | 1.8% | 0.6% | 完成 |
| 1 | p0 | Drive | pufferrandom | ppo | 620 | 560/620 | 90.3% | -0.16 | 0.3% | 0.3% | 0.5% | 完成 |
| 1 | p1 | DriveMoE | pufferinter | idm | 589 | 477/589 | 81.0% | -0.55 | 5.4% | 3.2% | 0.7% | 完成 |
| 1 | p1 | DriveMoE | pufferinter | ppo | 589 | 528/589 | 89.6% | -0.62 | 0.7% | 0.3% | 0.3% | 完成 |
| 1 | p1 | DriveMoE | pufferrandom | idm | 620 | 536/620 | 86.5% | -0.13 | 1.5% | 0.8% | 0.8% | 完成 |
| 1 | p1 | DriveMoE | pufferrandom | ppo | 620 | 559/620 | 90.2% | -0.15 | 0.3% | 0.2% | 0.5% | 完成 |
| 1 | p2 | DriveMoE3 | pufferinter | idm | 589 | 486/589 | 82.5% | -0.66 | 5.8% | 3.9% | 0.8% | 完成 |
| 1 | p2 | DriveMoE3 | pufferinter | ppo | 589 | 531/589 | 90.2% | -0.64 | 0.2% | 0.0% | 0.5% | 完成 |
| 1 | p2 | DriveMoE3 | pufferrandom | idm | 620 | 536/620 | 86.5% | -0.19 | 2.6% | 1.3% | 0.5% | 完成 |
| 1 | p2 | DriveMoE3 | pufferrandom | ppo | 620 | 562/620 | 90.6% | -0.15 | 0.0% | 0.0% | 0.3% | 完成 |
| 2 | p0 | Drive | pufferinter | idm | 589 | 481/589 | 81.7% | -0.68 | 6.6% | 4.9% | 1.4% | 完成 |
| 2 | p0 | Drive | pufferinter | ppo | 589 | 527/589 | 89.5% | -0.64 | 0.5% | 0.3% | 0.8% | 完成 |
| 2 | p0 | Drive | pufferrandom | idm | 620 | 541/620 | 87.3% | -0.19 | 2.4% | 1.6% | 0.5% | 完成 |
| 2 | p0 | Drive | pufferrandom | ppo | 620 | 562/620 | 90.6% | -0.16 | 0.2% | 0.0% | 0.3% | 完成 |
| 2 | p1 | DriveMoE | pufferinter | idm | 589 | 478/589 | 81.2% | -0.63 | 4.6% | 2.7% | 1.9% | 完成 |
| 2 | p1 | DriveMoE | pufferinter | ppo | 589 | 526/589 | 89.3% | -0.63 | 0.2% | 0.2% | 0.8% | 完成 |
| 2 | p1 | DriveMoE | pufferrandom | idm | 620 | 536/620 | 86.5% | -0.19 | 2.4% | 1.6% | 0.6% | 完成 |
| 2 | p1 | DriveMoE | pufferrandom | ppo | 620 | 560/620 | 90.3% | -0.15 | 0.3% | 0.2% | 0.5% | 完成 |
| 2 | p2 | DriveMoE3 | pufferinter | idm | 589 | 478/589 | 81.2% | -0.63 | 4.8% | 3.2% | 1.2% | 完成 |
| 2 | p2 | DriveMoE3 | pufferinter | ppo | 589 | 531/589 | 90.2% | -0.65 | 0.3% | 0.2% | 0.3% | 完成 |
| 2 | p2 | DriveMoE3 | pufferrandom | idm | 620 | 536/620 | 86.5% | -0.21 | 2.3% | 1.1% | 0.5% | 完成 |
| 2 | p2 | DriveMoE3 | pufferrandom | ppo | 620 | 562/620 | 90.6% | -0.17 | 0.3% | 0.2% | 0.2% | 完成 |

## MIXMOE3x pufferinter + IDM 与 Drive 同质对比

`MIXMOE3x.1` 使用单卡 `batch3x fixed` 训练最终 `nj7x0eeu/003815` 权重；`MIXMOE3x.2` 使用三卡 DDP `global6b` 训练最终 `hilzy188/003815` 权重。

| 实验 | Repeat | Policy Slot | Policy | 训练方式 | 有效 maps | Goal | Goal% | Reward | Coll | Fault | Offroad |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Drive 同质 PPO | 1 | - | Drive | 单策略同质 | 589 | 499/589 | 84.7% | -0.70 | 4.1% | 2.4% | 0.7% |
| Drive 同质 PPO | 2 | - | Drive | 单策略同质 | 589 | 494/589 | 83.9% | -0.66 | 4.4% | 3.1% | 1.0% |
| MIXMOE3x | 1 | p0 | Drive | 单卡 batch3x fixed | 589 | 487/589 | 82.7% | -0.62 | 4.2% | 2.7% | 0.3% |
| MIXMOE3x | 1 | p1 | DriveMoE | 单卡 batch3x fixed | 589 | 479/589 | 81.3% | -0.60 | 4.1% | 2.5% | 0.7% |
| MIXMOE3x | 1 | p2 | DriveMoE3 | 单卡 batch3x fixed | 589 | 472/589 | 80.1% | -0.50 | 3.6% | 2.0% | 1.0% |
| MIXMOE3x | 2 | p0 | Drive | 3 卡 DDP global6b | 589 | 484/589 | 82.2% | -0.71 | 5.6% | 3.6% | 0.8% |
| MIXMOE3x | 2 | p1 | DriveMoE | 3 卡 DDP global6b | 589 | 483/589 | 82.0% | -0.81 | 6.5% | 4.9% | 0.8% |
| MIXMOE3x | 2 | p2 | DriveMoE3 | 3 卡 DDP global6b | 589 | 486/589 | 82.5% | -0.75 | 5.6% | 4.2% | 1.5% |

初步观察：`MIXMOE3x` 两组在 pufferinter + IDM 下的 Goal% 仍低于 Drive 同质 PPO；三卡 DDP 版本的 p2 Goal% 接近同质 Drive repeat2，但 collision/fault/offroad 更高。

## MIXMOE3x pufferinter + IDM 1000 epoch 对比

本表与上一节使用相同评估对象和配置，但统一选取 `001000` checkpoint。

| 实验 | Repeat | Policy Slot | Policy | 训练方式 | 有效 maps | Goal | Goal% | Reward | Coll | Fault | Offroad |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Drive 同质 PPO | 1 | - | Drive | 单策略同质 | 589 | 470/589 | 79.8% | -0.76 | 6.5% | 4.9% | 1.5% |
| Drive 同质 PPO | 2 | - | Drive | 单策略同质 | 589 | 479/589 | 81.3% | -0.67 | 4.9% | 3.2% | 1.5% |
| MIXMOE3x | 1 | p0 | Drive | 单卡 batch3x fixed | 589 | 463/589 | 78.6% | -0.80 | 9.0% | 7.1% | 1.9% |
| MIXMOE3x | 1 | p1 | DriveMoE | 单卡 batch3x fixed | 589 | 444/589 | 75.4% | -0.80 | 9.0% | 7.6% | 5.4% |
| MIXMOE3x | 1 | p2 | DriveMoE3 | 单卡 batch3x fixed | 589 | 454/589 | 77.1% | -0.80 | 8.0% | 5.9% | 3.9% |
| MIXMOE3x | 2 | p0 | Drive | 3 卡 DDP global6b | 589 | 469/589 | 79.6% | -0.78 | 7.8% | 5.8% | 2.0% |
| MIXMOE3x | 2 | p1 | DriveMoE | 3 卡 DDP global6b | 589 | 447/589 | 75.9% | -0.91 | 9.7% | 8.8% | 3.6% |
| MIXMOE3x | 2 | p2 | DriveMoE3 | 3 卡 DDP global6b | 589 | 459/589 | 77.9% | -0.89 | 9.2% | 6.8% | 2.9% |

初步观察：1000 epoch 时 `MIXMOE3x` 各策略在 pufferinter + IDM 下均未超过 Drive 同质 PPO；与最终 `003815` checkpoint 相比，早期 checkpoint 的 collision/fault/offroad 明显更高。

## 备注与初步观察

- 同质 `DriveMoE` 两个 repeat 在 benchmark 上差异明显，说明单次训练权重存在较大 seed/训练轨迹方差。
- `MIXMOE` 表中每个 policy slot 是从异质训练中拆出的单独最终权重，不代表完整三策略同时评估。
- 如需最终结论，建议以同配置两个 repeat 的均值/方差为主，而不是只看单个 checkpoint。

## 权重与日志索引

| 实验 | Repeat | Policy Slot | Policy | 权重路径 | 日志 |
|---|---:|---|---|---|---|
| Drive 同质 PPO | 1 | - | Drive | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_xt5biagd.pt` | `.logs/val/runs/Drive.1/eval_pufferinter_ppo_vs_idm_gpu1_20260706_163822.log` |
| Drive 同质 PPO | 1 | - | Drive | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_xt5biagd.pt` | `.logs/val/runs/Drive.1/eval_pufferinter_ppo_vs_ppo_gpu1_20260706_203303.log` |
| Drive 同质 PPO | 1 | - | Drive | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_xt5biagd.pt` | `.logs/val/runs/Drive.1/eval_pufferrandom_ppo_vs_idm_gpu2_20260706_164531.log` |
| Drive 同质 PPO | 1 | - | Drive | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_xt5biagd.pt` | `.logs/val/runs/Drive.1/eval_pufferrandom_ppo_vs_ppo_gpu4_20260706_203304.log` |
| Drive 同质 PPO | 2 | - | Drive | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_e3anj5yt.pt` | `.logs/val/runs/Drive.2/eval_pufferinter_ppo_vs_idm_gpu2_20260707_115909.log` |
| Drive 同质 PPO | 2 | - | Drive | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_e3anj5yt.pt` | `.logs/val/runs/Drive.2/eval_pufferinter_ppo_vs_ppo_gpu5_20260708_030626.log` |
| Drive 同质 PPO | 2 | - | Drive | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_e3anj5yt.pt` | `.logs/val/runs/Drive.2/eval_pufferrandom_ppo_vs_idm_gpu7_20260708_030628.log` |
| Drive 同质 PPO | 2 | - | Drive | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_e3anj5yt.pt` | `.logs/val/runs/Drive.2/eval_pufferrandom_ppo_vs_ppo_gpu6_20260708_030627.log` |
| DriveMoE 同质 PPO | 1 | - | DriveMoE | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_aoljsqfo.pt` | `.logs/val/runs/2MOE.1/eval_pufferinter_moeppo_final_vs_idm_gpu2_20260707_181554.log` |
| DriveMoE 同质 PPO | 1 | - | DriveMoE | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_aoljsqfo.pt` | `.logs/val/runs/2MOE.1/eval_pufferinter_moeppo_vs_moeppo_gpu0_20260707_185315.log` |
| DriveMoE 同质 PPO | 1 | - | DriveMoE | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_aoljsqfo.pt` | `.logs/val/runs/2MOE.1/eval_pufferrandom_moeppo_vs_idm_gpu3_20260707_185316.log` |
| DriveMoE 同质 PPO | 1 | - | DriveMoE | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_aoljsqfo.pt` | `.logs/val/runs/2MOE.1/eval_pufferrandom_moeppo_vs_moeppo_gpu1_20260707_185315.log` |
| DriveMoE 同质 PPO | 2 | - | DriveMoE | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_jhme2lc6.pt` | `.logs/val/runs/2MOE.2/eval_pufferinter_moeppo_repeat_vs_idm_gpu0_20260708_134804.log` |
| DriveMoE 同质 PPO | 2 | - | DriveMoE | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_jhme2lc6.pt` | `.logs/val/runs/2MOE.2/eval_pufferinter_moeppo_repeat_vs_ppo_gpu1_20260708_134805.log` |
| DriveMoE 同质 PPO | 2 | - | DriveMoE | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_jhme2lc6.pt` | `.logs/val/runs/2MOE.2/eval_pufferrandom_moeppo_repeat_vs_idm_gpu2_20260708_134806.log` |
| DriveMoE 同质 PPO | 2 | - | DriveMoE | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_jhme2lc6.pt` | `.logs/val/runs/2MOE.2/eval_pufferrandom_moeppo_repeat_vs_ppo_gpu3_20260708_134807.log` |
| DriveMoE3 同质 PPO | 1 | - | DriveMoE3 | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_vm8y7nks.pt` | `.logs/val/runs/3MOE.1/eval_pufferinter_moe3ppo_vs_idm_gpu0_20260708_173012.log` |
| DriveMoE3 同质 PPO | 1 | - | DriveMoE3 | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_vm8y7nks.pt` | `.logs/val/runs/3MOE.1/eval_pufferinter_moe3ppo_vs_ppo_gpu1_20260708_173013.log` |
| DriveMoE3 同质 PPO | 1 | - | DriveMoE3 | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_vm8y7nks.pt` | `.logs/val/runs/3MOE.1/eval_pufferrandom_moe3ppo_vs_idm_gpu2_20260708_173014.log` |
| DriveMoE3 同质 PPO | 1 | - | DriveMoE3 | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_vm8y7nks.pt` | `.logs/val/runs/3MOE.1/eval_pufferrandom_moe3ppo_vs_ppo_gpu3_20260708_173015.log` |
| DriveMoE3 同质 PPO | 2 | - | DriveMoE3 | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_8qo9cvm7.pt` | `.logs/val/runs/3MOE.2/eval_pufferinter_moe3ppo_repeat_vs_idm_gpu6_20260708_173409.log` |
| DriveMoE3 同质 PPO | 2 | - | DriveMoE3 | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_8qo9cvm7.pt` | `.logs/val/runs/3MOE.2/eval_pufferinter_moe3ppo_repeat_vs_ppo_gpu7_20260708_173410.log` |
| DriveMoE3 同质 PPO | 2 | - | DriveMoE3 | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_8qo9cvm7.pt` | `.logs/val/runs/3MOE.2/eval_pufferrandom_moe3ppo_repeat_vs_idm_gpu0_20260708_182923.log` |
| DriveMoE3 同质 PPO | 2 | - | DriveMoE3 | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_8qo9cvm7.pt` | `.logs/val/runs/3MOE.2/eval_pufferrandom_moe3ppo_repeat_vs_ppo_gpu2_20260708_182924.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 1 | p0 | Drive | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_x9earvkh/model_policy_0_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.1/p0/eval_mixmoe_p0_pufferinter_Drive_vs_idm_gpu1_20260708_221827.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 1 | p0 | Drive | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_x9earvkh/model_policy_0_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.1/p0/eval_mixmoe_p0_pufferinter_Drive_vs_ppo_gpu2_20260708_221828.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 1 | p0 | Drive | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_x9earvkh/model_policy_0_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.1/p0/eval_mixmoe_p0_pufferrandom_Drive_vs_idm_gpu3_20260708_221829.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 1 | p0 | Drive | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_x9earvkh/model_policy_0_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.1/p0/eval_mixmoe_p0_pufferrandom_Drive_vs_ppo_gpu4_20260708_221830.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 1 | p1 | DriveMoE | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_x9earvkh/model_policy_1_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.1/p1/eval_mixmoe_p1_pufferinter_DriveMoE_vs_idm_gpu5_20260708_221831.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 1 | p1 | DriveMoE | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_x9earvkh/model_policy_1_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.1/p1/eval_mixmoe_p1_pufferinter_DriveMoE_vs_ppo_gpu6_20260708_221832.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 1 | p1 | DriveMoE | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_x9earvkh/model_policy_1_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.1/p1/eval_mixmoe_p1_pufferrandom_DriveMoE_vs_idm_gpu7_20260708_221833.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 1 | p1 | DriveMoE | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_x9earvkh/model_policy_1_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.1/p1/eval_mixmoe_p1_pufferrandom_DriveMoE_vs_ppo_gpu1_20260708_233709.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 1 | p2 | DriveMoE3 | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_x9earvkh/model_policy_2_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.1/p2/eval_mixmoe_p2_pufferinter_DriveMoE3_vs_idm_gpu3_20260708_233710.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 1 | p2 | DriveMoE3 | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_x9earvkh/model_policy_2_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.1/p2/eval_mixmoe_p2_pufferinter_DriveMoE3_vs_ppo_gpu5_20260708_233711.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 1 | p2 | DriveMoE3 | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_x9earvkh/model_policy_2_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.1/p2/eval_mixmoe_p2_pufferrandom_DriveMoE3_vs_idm_gpu7_20260708_233712.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 1 | p2 | DriveMoE3 | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_x9earvkh/model_policy_2_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.1/p2/eval_mixmoe_p2_pufferrandom_DriveMoE3_vs_ppo_gpu0_20260708_233713.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 2 | p0 | Drive | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_7u0evx5k/model_policy_0_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.2/p0/eval_mixmoe2_p0_pufferinter_Drive_vs_idm_gpu2_20260709_003222.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 2 | p0 | Drive | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_7u0evx5k/model_policy_0_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.2/p0/eval_mixmoe2_p0_pufferinter_Drive_vs_ppo_gpu3_20260709_003223.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 2 | p0 | Drive | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_7u0evx5k/model_policy_0_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.2/p0/eval_mixmoe2_p0_pufferrandom_Drive_vs_idm_gpu4_20260709_003224.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 2 | p0 | Drive | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_7u0evx5k/model_policy_0_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.2/p0/eval_mixmoe2_p0_pufferrandom_Drive_vs_ppo_gpu6_20260709_003225.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 2 | p1 | DriveMoE | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_7u0evx5k/model_policy_1_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.2/p1/eval_mixmoe2_p1_pufferinter_DriveMoE_vs_idm_gpu7_20260709_003226.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 2 | p1 | DriveMoE | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_7u0evx5k/model_policy_1_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.2/p1/eval_mixmoe2_p1_pufferinter_DriveMoE_vs_ppo_gpu0_20260709_003556.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 2 | p1 | DriveMoE | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_7u0evx5k/model_policy_1_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.2/p1/eval_mixmoe2_p1_pufferrandom_DriveMoE_vs_idm_gpu1_20260709_003557.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 2 | p1 | DriveMoE | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_7u0evx5k/model_policy_1_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.2/p1/eval_mixmoe2_p1_pufferrandom_DriveMoE_vs_ppo_gpu2_20260709_003558.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 2 | p2 | DriveMoE3 | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_7u0evx5k/model_policy_2_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.2/p2/eval_mixmoe2_p2_pufferinter_DriveMoE3_vs_idm_gpu3_20260709_003559.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 2 | p2 | DriveMoE3 | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_7u0evx5k/model_policy_2_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.2/p2/eval_mixmoe2_p2_pufferinter_DriveMoE3_vs_ppo_gpu4_20260709_003600.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 2 | p2 | DriveMoE3 | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_7u0evx5k/model_policy_2_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.2/p2/eval_mixmoe2_p2_pufferrandom_DriveMoE3_vs_idm_gpu5_20260709_003601.log` |
| Drive/DriveMoE/DriveMoE3 异质 PPO | 2 | p2 | DriveMoE3 | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_7u0evx5k/model_policy_2_puffer_drive_003815.pt` | `.logs/val/runs/MIXMOE.2/p2/eval_mixmoe2_p2_pufferrandom_DriveMoE3_vs_ppo_gpu6_20260709_003602.log` |
