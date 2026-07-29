# Run2 感知层混合智能训练与验证总结

## 1. 实验思路

本次 Perception 实验目标是在当前 `behavior-bench` 平台中非侵入复现 `pufferdrive_re` 的感知层混智核心语义：同一训练环境中同时存在低/中/高三种感知等级的 PPO-controlled agents，并让每个等级策略只用自己的 on-policy 轨迹更新。

本实验没有修改环境、C binding 或默认 `Drive` 策略，而是新增 perception-masked policy class：

| 等级 | 策略 | 感知范围 | 训练含义 |
|---:|---|---|---|
| 0 | `DrivePerceptionLow + Recurrent` | 前方 60 度扇形，50m | 低感知智能体 |
| 1 | `DrivePerceptionMid + Recurrent` | 前方 120 度扇形，50m | 中感知智能体 |
| 2 | `Drive + Recurrent` | 原始全局 observation | 高感知智能体 |

实现方式与 `pufferdrive_re` 的概念对应如下：

| `pufferdrive_re` | 当前 `behavior-bench` |
|---|---|
| `group_id` 分配 low/mid/high | `mix_ppo` 的 `policy_id` / `segment_policy_ids` |
| `DriveMixedPolicy.level_policies` | `mix_ppo` 管理多个独立 policy |
| `MixedRecurrent` 每 level 一个 LSTM | 每个 mixed policy 独立包 `Recurrent` |
| `repeat_policy_samples=[3,3,3]` | batch/total steps 放大 3 倍，采集更多 raw rollout |

## 2. 训练配置

### 2.1 同质训练

每个感知等级单独训练：

```text
batch_size = 524288
total_timesteps = 2000000000
rnn = Recurrent(256,256)
env.num_maps = 10000
```

| 训练组 | 策略 | W&B run | 权重目录 | 训练日志 |
|---|---|---|---|---|
| Homogeneous Low | `DrivePerceptionLow` | `r8jh6epn` | `experiments/puffer_drive_r8jh6epn` | `.logs/train/run2/perception_low_homogeneous_2b_batch1x_run2_20260713_204739.log` |
| Homogeneous Mid | `DrivePerceptionMid` | `877i3fj2` | `experiments/puffer_drive_877i3fj2` | `.logs/train/run2/perception_mid_homogeneous_2b_batch1x_run2_20260713_204751.log` |
| Homogeneous High | `Drive` | `zegl2eqd` | `experiments/puffer_drive_zegl2eqd` | `.logs/train/run2/perception_high_drive_homogeneous_2b_batch1x_run2_20260713_204809.log` |

### 2.2 混合训练

混合训练按 `low:mid:high = 1:1:1`：

```text
batch_size = 1572864
total_timesteps = 6000000000
mix_ppo_policy_names = DrivePerceptionLow,DrivePerceptionMid,Drive
mix_ppo_rnn_names = Recurrent,Recurrent,Recurrent
```

| 训练组 | 方式 | W&B run | 权重目录 | 训练日志 |
|---|---|---|---|---|
| Mixed Single GPU | 单卡 batch3x / 6B | `d2spiva4` | `experiments/puffer_drive_d2spiva4` | `.logs/train/run2/perception_mixed_low_mid_high_6b_batch3x_run2_20260713_204819.log` |
| Mixed DDP3 | 3 卡 DDP，全局 batch/steps 等价 6B | `gob17caq` | `experiments/puffer_drive_gob17caq` | `.logs/train/run2/perception_mixed_low_mid_high_ddp3_global6b_run2_20260714_010819.log` |

## 3. 训练结束指标

### 3.1 同质训练最终指标

| 策略 | Completion | Collision | DNF | Episode Return | Agent Steps |
|---|---:|---:|---:|---:|---:|
| Low `DrivePerceptionLow` | 93.7% | 2.9% | 2.6% | 0.879 | 2.017B |
| Mid `DrivePerceptionMid` | 94.5% | 1.6% | 3.3% | 0.913 | 2.017B |
| High `Drive` | 96.7% | 0.6% | 2.1% | 0.952 | 2.017B |

### 3.2 混合训练最终指标

| 训练组 | Overall Completion | Overall Collision | Overall Return | p0 Low Collision | p1 Mid Collision | p2 High Collision |
|---|---:|---:|---:|---:|---:|---:|
| Mixed Single GPU | 95.0% | 1.7% | 0.916 | 2.5% | 1.7% | 0.8% |
| Mixed DDP3 | 95.5% | 1.6% | 0.922 | 2.3% | 1.7% | 0.8% |

## 4. `pufferinter + IDM` 最终验证结果

所有验证设置：

```text
split = pufferinter
traffic = idm
map_ids = all
```

实际执行时，18 个验证 run 的覆盖情况一致：

```text
请求地图数 = 1000
有效评估地图数 = 589
跳过地图数 = 411
```

跳过原因是对应地图中的 ego 不在 active agents 中。因此，下表所有验证指标的实际有效分母均为 589，而不是 1000。`map_ids = all` 表示请求全量地图，不代表全部地图都成功进入统计。

指标口径：

- `Goal`：589 张有效地图中的目标到达率。
- `Collision`：先计算每张地图内的 step 级碰撞率，再对 589 张地图取平均。
- `At-Fault`：每张地图是否发生过责任碰撞的二值发生率，再对 589 张地图取平均。
- `Offroad`：先计算每张地图内的 step 级 offroad rate，再跨地图取平均。
- `Reward`：每张地图 total reward 的平均值。

### 4.1 同质最终权重验证

结果目录：

```text
/home/fanyuqi/wsc/behavior-bench/.logs/val/run2/results
```

| 策略 | 验证目录 | Goal | Collision | At-Fault | Offroad | Reward |
|---|---|---:|---:|---:|---:|---:|
| Low `DrivePerceptionLow` | `homogeneous_perception_low_2b_batch1x_run2/DrivePerceptionLow_Recurrent` | 70.8% | 12.1% | 8.5% | 0.9% | -0.46 |
| Mid `DrivePerceptionMid` | `homogeneous_perception_mid_2b_batch1x_run2/DrivePerceptionMid_Recurrent` | 75.2% | 7.5% | 3.9% | 1.0% | -0.43 |
| High `Drive` | `homogeneous_perception_high_drive_2b_batch1x_run2/Drive_Recurrent` | 82.9% | 5.8% | 4.2% | 0.7% | -0.70 |

### 4.2 Mixed Single GPU 最终权重验证

| Slot | 策略 | 验证目录 | Goal | Collision | At-Fault | Offroad | Reward |
|---|---|---|---:|---:|---:|---:|---:|
| p0 | Low `DrivePerceptionLow` | `perception_mixed_low_mid_high_6b_batch3x_run2/p0_DrivePerceptionLow_Recurrent` | 71.5% | 13.9% | 9.7% | 1.0% | -0.47 |
| p1 | Mid `DrivePerceptionMid` | `perception_mixed_low_mid_high_6b_batch3x_run2/p1_DrivePerceptionMid_Recurrent` | 76.1% | 9.7% | 7.0% | 1.7% | -0.55 |
| p2 | High `Drive` | `perception_mixed_low_mid_high_6b_batch3x_run2/p2_Drive_Recurrent` | 78.8% | 6.3% | 4.8% | 1.9% | -0.62 |

### 4.3 Mixed DDP3 最终权重验证

| Slot | 策略 | 验证目录 | Goal | Collision | At-Fault | Offroad | Reward |
|---|---|---|---:|---:|---:|---:|---:|
| p0 | Low `DrivePerceptionLow` | `perception_mixed_low_mid_high_ddp3_global6b_run2/p0_DrivePerceptionLow_Recurrent` | 74.7% | 12.6% | 7.8% | 1.2% | -0.56 |
| p1 | Mid `DrivePerceptionMid` | `perception_mixed_low_mid_high_ddp3_global6b_run2/p1_DrivePerceptionMid_Recurrent` | 74.7% | 9.7% | 5.9% | 1.0% | -0.56 |
| p2 | High `Drive` | `perception_mixed_low_mid_high_ddp3_global6b_run2/p2_Drive_Recurrent` | 82.9% | 2.6% | 1.7% | 0.9% | -0.64 |

## 5. 1000 epoch / `001000` 验证结果

### 5.1 同质 1000 checkpoint

结果目录：

```text
/home/fanyuqi/wsc/behavior-bench/.logs/val/run2/results1000
```

| 策略 | Goal | Collision | At-Fault | Offroad | Reward |
|---|---:|---:|---:|---:|---:|
| Low `DrivePerceptionLow` | 69.4% | 17.3% | 12.9% | 1.2% | -0.63 |
| Mid `DrivePerceptionMid` | 72.3% | 10.9% | 7.0% | 2.4% | -0.58 |
| High `Drive` | 78.6% | 8.3% | 5.4% | 2.2% | -0.84 |

### 5.2 Mixed Single GPU 1000 checkpoint

| Slot | 策略 | Goal | Collision | At-Fault | Offroad | Reward |
|---|---|---:|---:|---:|---:|---:|
| p0 | Low `DrivePerceptionLow` | 64.7% | 19.4% | 14.9% | 3.4% | -0.71 |
| p1 | Mid `DrivePerceptionMid` | 72.0% | 12.4% | 7.5% | 3.1% | -0.68 |
| p2 | High `Drive` | 77.3% | 6.3% | 5.3% | 4.2% | -0.65 |

### 5.3 Mixed DDP3 1000 checkpoint

| Slot | 策略 | Goal | Collision | At-Fault | Offroad | Reward |
|---|---|---:|---:|---:|---:|---:|
| p0 | Low `DrivePerceptionLow` | 68.8% | 19.7% | 15.3% | 1.7% | -0.67 |
| p1 | Mid `DrivePerceptionMid` | 71.3% | 15.8% | 9.9% | 1.4% | -0.80 |
| p2 | High `Drive` | 79.0% | 5.3% | 2.7% | 3.9% | -0.69 |

## 6. 最终权重综合对比

本节只比较最终 checkpoint，不包含 1000 epoch 中间权重。核心问题是：同样在 `pufferinter + IDM` 上，混合训练是否比对应等级的同质训练更好。

### 6.1 按感知等级对比

| 感知等级 | 同质训练 Goal / Coll / Fault | Mixed Single Goal / Coll / Fault | Mixed DDP3 Goal / Coll / Fault | 主要观察 |
|---|---|---|---|---|
| Low | 70.8% / 12.1% / 8.5% | 71.5% / 13.9% / 9.7% | 74.7% / 12.6% / 7.8% | DDP3 版本 low 的 goal 更高，fault 略优于同质，但 collision 基本接近 |
| Mid | 75.2% / 7.5% / 3.9% | 76.1% / 9.7% / 7.0% | 74.7% / 9.7% / 5.9% | Mixed Single 的 goal 略高于同质，但两种 mixed 的 collision/fault 均更高，综合上没有稳定优势 |
| High | 82.9% / 5.8% / 4.2% | 78.8% / 6.3% / 4.8% | 82.9% / 2.6% / 1.7% | DDP3 mixed high 明显优于同质 high 的 collision/fault |

### 6.2 按训练方式对比

| 训练方式 | 平均 Goal | 平均 Collision | 平均 At-Fault | 平均 Offroad | 说明 |
|---|---:|---:|---:|---:|---|
| Homogeneous 三等级均值 | 76.3% | 8.4% | 5.5% | 0.8% | 三个等级分别独立训练后再平均 |
| Mixed Single 三等级均值 | 75.4% | 10.0% | 7.1% | 1.5% | 单卡 batch3x 混合训练 |
| Mixed DDP3 三等级均值 | 77.4% | 8.3% | 5.1% | 1.0% | 三卡 DDP，全局预算等价 6B |

以上均值均使用各 `summary.csv` 的原始精度先平均，最后统一保留一位小数，避免对表中已取整值再次求平均。

从均值上看，Mixed Single 不如同质均值；Mixed DDP3 与同质均值接近，并略优于同质的平均 goal 和 at-fault。分等级看，平均 goal 的提升主要来自 Low（70.8% → 74.7%），High 的 goal 与同质相同（均为 82.9%）；平均 at-fault 的改善则主要来自 High（4.2% → 1.7%）。

### 6.3 最重要的结论

本轮最终权重结果显示：

```text
感知等级本身有效：low < mid < high 的性能阶梯清晰。
本次单 seed 结果中，混合训练没有同时提升所有等级。
Mixed DDP3 p2 high 的 collision 从 5.8% 降到 2.6%，fault 从 4.2% 降到 1.7%，是本轮最值得进一步复验的观察。
```

因此，如果论文主目标是“混合智能博弈提升高等级策略对异质交通的鲁棒性”，当前最有价值的候选结果是：

```text
Mixed DDP3 p2 high 在 collision / at-fault 上优于 Homogeneous high
```

如果主目标是“所有智能等级都提升”，当前证据还不充分：Low 主要提升 goal，Mid 仅 Mixed Single 的 goal 略升，而 collision/fault 未同步改善。

## 7. 重要日志与权重

### 7.1 训练日志

| 训练组 | 日志 |
|---|---|
| Homogeneous Low | `.logs/train/run2/perception_low_homogeneous_2b_batch1x_run2_20260713_204739.log` |
| Homogeneous Mid | `.logs/train/run2/perception_mid_homogeneous_2b_batch1x_run2_20260713_204751.log` |
| Homogeneous High | `.logs/train/run2/perception_high_drive_homogeneous_2b_batch1x_run2_20260713_204809.log` |
| Mixed Single GPU | `.logs/train/run2/perception_mixed_low_mid_high_6b_batch3x_run2_20260713_204819.log` |
| Mixed DDP3 | `.logs/train/run2/perception_mixed_low_mid_high_ddp3_global6b_run2_20260714_010819.log` |

### 7.2 权重路径

| 训练组 | 权重 |
|---|---|
| Homogeneous Low | `experiments/puffer_drive_r8jh6epn/model_puffer_drive_003815.pt` |
| Homogeneous Mid | `experiments/puffer_drive_877i3fj2/model_puffer_drive_003815.pt` |
| Homogeneous High | `experiments/puffer_drive_zegl2eqd/model_puffer_drive_003815.pt` |
| Mixed Single p0 Low | `experiments/puffer_drive_d2spiva4/model_policy_0_puffer_drive_003815.pt` |
| Mixed Single p1 Mid | `experiments/puffer_drive_d2spiva4/model_policy_1_puffer_drive_003815.pt` |
| Mixed Single p2 High | `experiments/puffer_drive_d2spiva4/model_policy_2_puffer_drive_003815.pt` |
| Mixed DDP3 p0 Low | `experiments/puffer_drive_gob17caq/model_policy_0_puffer_drive_003815.pt` |
| Mixed DDP3 p1 Mid | `experiments/puffer_drive_gob17caq/model_policy_1_puffer_drive_003815.pt` |
| Mixed DDP3 p2 High | `experiments/puffer_drive_gob17caq/model_policy_2_puffer_drive_003815.pt` |

## 8. 结论

### 8.1 感知等级设计形成了清晰性能阶梯

同质最终验证中，感知越完整，`pufferinter + IDM` 表现越好：

```text
Low:  Goal 70.8%, Collision 12.1%
Mid:  Goal 75.2%, Collision 7.5%
High: Goal 82.9%, Collision 5.8%
```

该单次实验结果与 `DrivePerceptionLow/Mid/Drive` 的感知能力设计预期一致，支持当前 mask 语义确实产生了可观测的性能差异；是否能跨 seed 稳定复现仍需进一步验证。

### 8.2 混合训练没有提升所有等级，但本次 DDP3 High 的碰撞指标更好

Mixed Single 中 high 的 collision 为 6.3%，略差于同质 high 的 5.8%。但 DDP3 mixed 中 high 的 collision 降到 2.6%，at-fault 降到 1.7%，明显好于同质 high。

这表明本次 DDP3 run 的 high policy 终值优于单卡和同质对应项，但单 seed 结果不足以证明 DDP3 更稳定。该差异也可能来自训练随机性或并行采样，需要多 seed repeat 确认。

### 8.3 Low/Mid 从混合中收益有限

低感知策略在混合训练中 collision 仍然较高：

```text
Homogeneous Low: 12.1%
Mixed Single Low: 13.9%
Mixed DDP3 Low: 12.6%
```

中感知策略的 Mixed Single goal 略高于同质（76.1% vs 75.2%），但 collision 和 at-fault 均更高；若只看 collision：

```text
Homogeneous Mid: 7.5%
Mixed Single Mid: 9.7%
Mixed DDP3 Mid: 9.7%
```

因此，本轮可直接确认的是 DDP3 high 的 collision/at-fault 更低，而不能据此证明混合交互主要帮助 high-level policy 的具体机制。该机制解释需要多 seed 和消融实验支持。

### 8.4 1000 checkpoint 与最终 checkpoint 的关系

`001000` checkpoint 的验证表现整体低于最终 checkpoint，尤其 low/mid collision 较高。这说明早期权重不足以代表最终性能；仅凭两个 checkpoint 不能严格判定收敛状态。

### 8.5 实验限制

1. 当前比较基于单 seed，不能估计训练随机性带来的方差。
2. 每个 checkpoint 仅进行一次验证，且 1000 张请求地图中只有 589 张进入统计。
3. 当前结果仅覆盖 `pufferinter + IDM`，尚未验证跨 traffic 或 held-out 设置的泛化。
4. 小幅差异不应直接解释为稳定提升；尤其需要 repeat 验证 Mixed DDP3 p2 high 的结果。

### 8.6 下一步建议

1. 对 Mixed DDP3 重复 seed，确认 high-level collision 2.6% 是否稳定。
2. 排查并修复 411 张地图因 ego 不在 active agents 而被跳过的问题，再进行完整 1000-map 评估。
3. 增加 PDM held-out 验证，检查收益是否只对 IDM 成立。
4. 如果目标是提升 high-level policy，可以尝试调比例：
   ```text
   low:mid:high = 1:1:2
   ```
5. 如果目标是改善 low/mid 本身，需要考虑更强的低/中感知辅助训练，而不仅是混合交互。
