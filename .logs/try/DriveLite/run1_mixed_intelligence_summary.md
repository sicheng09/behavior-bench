# Run1 混合智能策略训练与验证总结

## 1. 实验思路

本次实验的目标是通过混合不同“策略智能等级”的 PPO-controlled agents，增加训练交互的异质性，从而观察高等级策略在未见 IDM 车流上的泛化是否改善。实验不改变 observation 感知输入，所有策略仍然接收相同 ego、partner、road 信息；差异只体现在策略网络和时序记忆能力。

本次对话中最终使用了三类策略等级：

| 等级 | 策略 | 训练/验证名称 | 策略含义 | 参数量 |
|---|---|---|---|---:|
| L3 | `Drive + Recurrent(256,256)` | `Drive_Recurrent` | 完整 Drive 单帧编码 + LSTM 时序 belief，最高等级策略 | 614,236 |
| L2 | `Drive` no LSTM | `Drive_NoLSTM` | 同样单帧编码，但没有时序记忆，只做 reactive decision | 87,900 |
| L1 | `DriveLite` no LSTM | `DriveLite_NoLSTM` | 同样输入，更轻量的 feedforward reactive policy | 43,420 |
| Ablation | `DriveLite + Recurrent(224,224)` | `DriveLite_Recurrent224` | 轻量单帧网络 + LSTM，用于区分“轻量网络”和“无时序记忆”的影响 | 446,620 |

围绕这些策略，实际跑了两组混合训练和三个同质对照：

| 训练组 | 组成 | 训练预算 | 目的 |
|---|---|---:|---|
| Mixed A | p0 `Drive_Recurrent`, p1 `Drive_NoLSTM`, p2 `DriveLite_NoLSTM` | 6B / batch 1,572,864 | 三等级混合智能主实验 |
| Mixed B | p0 `Drive_Recurrent`, p1 `DriveLite_Recurrent224`, p2 `Drive_NoLSTM` | 6B / batch 1,572,864 | 检查低容量策略如果保留 LSTM 是否不同 |
| Homogeneous Drive no LSTM | `Drive_NoLSTM` | 2B / batch 524,288 | Mixed 中 p1/p2 的同质对照 |
| Homogeneous DriveLite no LSTM | `DriveLite_NoLSTM` | 2B / batch 524,288 | Mixed A 中 p2 的同质对照 |
| Homogeneous DriveLite LSTM | `DriveLite_Recurrent224` | 2B / batch 524,288 | Mixed B 中 p1 的同质对照 |

混合训练使用 1:1:1 的 `mix_ppo_policy_mix`，并把总 batch 和总 steps 放大 3 倍，使每个策略等级在每次 update 中约得到同质训练等价的 batch：

```text
homogeneous per-policy batch = 524,288
mixed total batch            = 3 * 524,288 = 1,572,864
homogeneous per-policy steps = 2B
mixed total steps            = 3 * 2B = 6B
```

由于 `env.num_agents=1024` 不能被 3 整除，每个策略的 agent slot 数会有约 0.1% 级别的整数误差，但训练预算设计上是按每个策略与同质训练等价对齐的。

## 2. 训练结果汇总

本节指标取自各训练日志在 W&B 同步完成后的最后一次 Evaluate 窗口；该窗口与 W&B Run Summary 的总体指标一致，但可能不同于同步前 epoch 3815 的即时 dashboard。百分比保留一位小数，Episode Return 保留三位小数。

`2B` / `6B` 表示名义训练预算。由于按完整 batch 推进，实际 `agent_steps` 分别为：

```text
同质训练 = 2,016,935,936 ≈ 2.017B
混合训练 = 6,050,807,808 ≈ 6.051B
```

### 2.1 Mixed A：`Drive_Recurrent + Drive_NoLSTM + DriveLite_NoLSTM`

训练日志：`/home/fanyuqi/wsc/behavior-bench/.logs/train/run1/mix_intelligence_3level_lstm_drive_drivelite_6b_batch3x_run1_20260713_015723.log`

权重目录：`/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_drkkc451`

| 策略 | 权重 | Completion | Collision | Offroad | Episode Return |
|---|---|---:|---:|---:|---:|
| p0 `Drive_Recurrent` | `model_policy_0_puffer_drive_003815.pt` | 96.2% | 0.9% | 0.5% | 0.946 |
| p1 `Drive_NoLSTM` | `model_policy_1_puffer_drive_003815.pt` | 96.6% | 1.0% | 0.6% | 0.947 |
| p2 `DriveLite_NoLSTM` | `model_policy_2_puffer_drive_003815.pt` | 89.3% | 5.0% | 4.1% | 0.786 |
| Overall | `mix_model_puffer_drive_003815.pt` | 94.0% | 2.3% | 1.8% | 0.893 |

### 2.2 Mixed B：`Drive_Recurrent + DriveLite_Recurrent224 + Drive_NoLSTM`

训练日志：`/home/fanyuqi/wsc/behavior-bench/.logs/train/run1/mix_drive_lstm_drivelite_lstm224_drive_nolstm_6b_batch3x_run1_20260713_020607.log`

权重目录：`/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_ejdmrr6z`

| 策略 | 权重 | Completion | Collision | Offroad | Episode Return |
|---|---|---:|---:|---:|---:|
| p0 `Drive_Recurrent` | `model_policy_0_puffer_drive_003815.pt` | 96.4% | 0.9% | 0.5% | 0.946 |
| p1 `DriveLite_Recurrent224` | `model_policy_1_puffer_drive_003815.pt` | 90.6% | 3.9% | 1.8% | 0.830 |
| p2 `Drive_NoLSTM` | `model_policy_2_puffer_drive_003815.pt` | 96.5% | 1.0% | 0.6% | 0.946 |
| Overall | `mix_model_puffer_drive_003815.pt` | 94.5% | 1.9% | 1.0% | 0.908 |

### 2.3 同质训练对照

| 训练组 | 策略 | 日志 | 权重 | Completion | Collision | Offroad | Episode Return |
|---|---|---|---|---:|---:|---:|---:|
| Homogeneous Drive no LSTM | `Drive_NoLSTM` | `.logs/train/run1/homogeneous_drive_nolstm_2b_batch1x_run1_restart_20260713_132915.log` | `experiments/puffer_drive_fi37a551/model_puffer_drive_003815.pt` | 96.9% | 0.8% | 0.8% | 0.950 |
| Homogeneous DriveLite no LSTM | `DriveLite_NoLSTM` | `.logs/train/run1/homogeneous_drivelite_nolstm_2b_batch1x_run1_restart_20260713_132927.log` | `experiments/puffer_drive_gqlcb43j/model_puffer_drive_003815.pt` | 80.9% | 8.6% | 9.0% | 0.624 |
| Homogeneous DriveLite LSTM224 | `DriveLite_Recurrent224` | `.logs/train/run1/homogeneous_drivelite_lstm224_2b_batch1x_run1_20260713_020125.log` | `experiments/puffer_drive_7vmm4a8m/model_puffer_drive_003815.pt` | 88.8% | 5.6% | 2.0% | 0.787 |

## 3. pufferinter + IDM 验证结果

所有验证均为：

```text
split = pufferinter
traffic = idm
map_ids = all
```

实际执行时，每个主验证 run 均请求 1000 张地图，其中 589 张进入统计、411 张因 ego 不在 active agents 中被跳过。因此下表指标的有效分母为 589，而不是 1000。

指标口径：

- `Goal`：589 张有效地图中的目标到达率。
- `Collision`：每张地图内 step 级 collision rate 的跨地图平均。
- `At-Fault`：每张地图是否发生过责任碰撞的二值发生率。
- `Offroad`：每张地图内 step 级 offroad rate 的跨地图平均。
- `Reward`：每张地图 total reward 的平均值。

### 3.1 同质训练验证

| 训练组 | 策略 | 验证日志/目录 | Goal | Collision | At-Fault | Offroad | Reward |
|---|---|---|---:|---:|---:|---:|---:|
| Drive.2 baseline | `Drive_Recurrent` | `.logs/val/run1/results/Drive.2/eval_pufferinter_ppo_vs_idm_gpu2_20260707_115909.log` | 83.9% | 4.4% | 3.1% | 1.0% | -0.66 |
| Homogeneous Drive no LSTM | `Drive_NoLSTM` | `.logs/val/run1/results/homogeneous_drive_nolstm_2b_batch1x_run1/Drive_NoLSTM/` | 78.3% | 9.3% | 7.5% | 1.5% | -0.92 |
| Homogeneous DriveLite no LSTM | `DriveLite_NoLSTM` | `.logs/val/run1/results/homogeneous_drivelite_nolstm_2b_batch1x_run1/DriveLite_NoLSTM/` | 53.1% | 31.8% | 28.4% | 5.9% | -0.94 |
| Homogeneous DriveLite LSTM224 | `DriveLite_Recurrent224` | `.logs/val/run1/results/homogeneous_drivelite_lstm224_2b_batch1x_run1/DriveLite_Recurrent224/` | 56.7% | 28.4% | 25.0% | 3.4% | -0.75 |

`Drive.2` baseline 的数值可由所列 eval log 末行核对（Goal 494/589），但该次运行写出的 `summary.csv` / `per_map.csv` 已不在当前工作区。`Drive.2` 目录内另外三个 `summary.csv` 属于不同 split/traffic 组合，不能替代本表的 `pufferinter + IDM` baseline。

### 3.2 Mixed A 验证

结果目录：`/home/fanyuqi/wsc/behavior-bench/.logs/val/run1/results/mix_intelligence_3level_lstm_drive_drivelite_6b_batch3x_run1`

| Slot | 策略 | 验证目录 | Goal | Collision | At-Fault | Offroad | Reward |
|---|---|---|---:|---:|---:|---:|---:|
| p0 | `Drive_Recurrent` | `p0/` | 80.5% | 6.6% | 5.3% | 1.2% | -0.76 |
| p1 | `Drive_NoLSTM` | `p1/` | 78.6% | 8.5% | 7.6% | 3.1% | -0.93 |
| p2 | `DriveLite_NoLSTM` | `p2/` | 55.4% | 32.6% | 28.5% | 3.1% | -0.90 |

Mixed A 的 `summary.csv` / `per_map.csv` 位于上表 `p0/`、`p1/`、`p2/`；对应的 config/eval log 另存于 `p0_Drive_Recurrent/`、`p1_Drive_NoLSTM/`、`p2_DriveLite_NoLSTM/`。这是产物目录分离，不是重复验证。

### 3.3 Mixed B 验证

结果目录：`/home/fanyuqi/wsc/behavior-bench/.logs/val/run1/results/mix_drive_lstm_drivelite_lstm224_drive_nolstm_6b_batch3x_run1`

| Slot | 策略 | 验证目录 | Goal | Collision | At-Fault | Offroad | Reward |
|---|---|---|---:|---:|---:|---:|---:|
| p0 | `Drive_Recurrent` | `p0_Drive_Recurrent/` | 80.8% | 5.4% | 4.4% | 0.9% | -0.65 |
| p1 | `DriveLite_Recurrent224` | `p1_DriveLite_Recurrent224/` | 55.4% | 24.3% | 18.7% | 3.9% | -0.46 |
| p2 | `Drive_NoLSTM` | `p2_Drive_NoLSTM/` | 81.2% | 6.3% | 4.9% | 1.5% | -0.88 |

## 4. 重要日志与权重索引

### 4.1 训练日志

| 名称 | 日志 |
|---|---|
| Mixed A | `/home/fanyuqi/wsc/behavior-bench/.logs/train/run1/mix_intelligence_3level_lstm_drive_drivelite_6b_batch3x_run1_20260713_015723.log` |
| Mixed B | `/home/fanyuqi/wsc/behavior-bench/.logs/train/run1/mix_drive_lstm_drivelite_lstm224_drive_nolstm_6b_batch3x_run1_20260713_020607.log` |
| Homogeneous Drive no LSTM | `/home/fanyuqi/wsc/behavior-bench/.logs/train/run1/homogeneous_drive_nolstm_2b_batch1x_run1_restart_20260713_132915.log` |
| Homogeneous DriveLite no LSTM | `/home/fanyuqi/wsc/behavior-bench/.logs/train/run1/homogeneous_drivelite_nolstm_2b_batch1x_run1_restart_20260713_132927.log` |
| Homogeneous DriveLite LSTM224 | `/home/fanyuqi/wsc/behavior-bench/.logs/train/run1/homogeneous_drivelite_lstm224_2b_batch1x_run1_20260713_020125.log` |

### 4.2 权重路径

| 名称 | 权重路径 |
|---|---|
| Mixed A p0 `Drive_Recurrent` | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_drkkc451/model_policy_0_puffer_drive_003815.pt` |
| Mixed A p1 `Drive_NoLSTM` | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_drkkc451/model_policy_1_puffer_drive_003815.pt` |
| Mixed A p2 `DriveLite_NoLSTM` | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_drkkc451/model_policy_2_puffer_drive_003815.pt` |
| Mixed B p0 `Drive_Recurrent` | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_ejdmrr6z/model_policy_0_puffer_drive_003815.pt` |
| Mixed B p1 `DriveLite_Recurrent224` | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_ejdmrr6z/model_policy_1_puffer_drive_003815.pt` |
| Mixed B p2 `Drive_NoLSTM` | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_ejdmrr6z/model_policy_2_puffer_drive_003815.pt` |
| Homogeneous Drive no LSTM | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_fi37a551/model_puffer_drive_003815.pt` |
| Homogeneous DriveLite no LSTM | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_gqlcb43j/model_puffer_drive_003815.pt` |
| Homogeneous DriveLite LSTM224 | `/home/fanyuqi/wsc/behavior-bench/experiments/puffer_drive_7vmm4a8m/model_puffer_drive_003815.pt` |

## 5. 结论

### 5.1 Drive 系列与 DriveLite 系列形成明显性能差异

同质训练与混合训练都显示 `Drive` 系列整体强于 `DriveLite` 系列；但 L3 与 L2 在训练内几乎同分，不能把本轮结果解释成均匀的三级性能阶梯。`pufferinter + IDM` 上的单次验证排序为：

```text
Drive_Recurrent > Drive_NoLSTM > DriveLite_Recurrent224 > DriveLite_NoLSTM
```

`DriveLite_NoLSTM` 的 collision 明显更高，同质为 31.8%，Mixed A 中为 32.6%。这表明轻量策略在当前设置下具有可观测的性能和安全性差异，但单 seed 结果不能单独证明差异完全由网络容量导致。

### 5.2 无 LSTM Drive 在训练内表现很好，但 IDM 泛化弱于 Drive+LSTM baseline

同质 `Drive_NoLSTM` 训练内 completion 达到 96.9%，collision 只有 0.8%，但在 `pufferinter + IDM` 上 collision 升到 9.3%，高于 `Drive.2` baseline 的 4.4%。该结果与“时序记忆/behavior belief 可能有助于跨车流泛化”的假设一致。

需要注意，run1 缺少同 pipeline、同预算、同 seed 的 2B Homogeneous `Drive_Recurrent` 对照；`Drive.2` 是独立 baseline，且同时混入网络容量和训练随机性差异，因此这里不是严格的 LSTM 因果消融。

### 5.3 当前混合训练没有超过 Drive+LSTM baseline；Mixed B 单次 p0 更接近 baseline

高等级策略对比：

| 策略来源 | Goal | Collision | At-Fault | Offroad |
|---|---:|---:|---:|---:|
| Drive.2 baseline `Drive_Recurrent` | 83.9% | 4.4% | 3.1% | 1.0% |
| Mixed A p0 `Drive_Recurrent` | 80.5% | 6.6% | 5.3% | 1.2% |
| Mixed B p0 `Drive_Recurrent` | 80.8% | 5.4% | 4.4% | 0.9% |

本次结果中，Mixed B 的高等级 p0 比 Mixed A p0 更接近 baseline，collision 从 6.6% 降到 5.4%，at-fault 从 5.3% 降到 4.4%。但 Mixed A → B 同时改变了 DriveLite 架构和 slot 配方，且每组仅一个 seed，因此不能据此证明 Mixed B 更稳定，也不能把差异单独归因于 LSTM 或“交互难度更平滑”。

### 5.4 混合训练对低等级策略本身不一定全面提升

低等级对比：

| 策略 | 同质验证 Collision | 混合验证 Collision |
|---|---:|---:|
| `Drive_NoLSTM` in Mixed A | 9.3% | 8.5% |
| `Drive_NoLSTM` in Mixed B | 9.3% | 6.3% |
| `DriveLite_NoLSTM` in Mixed A | 31.8% | 32.6% |
| `DriveLite_Recurrent224` in Mixed B | 28.4% | 24.3% |

`Drive_NoLSTM` 在混合训练中部分指标改善：Mixed B 的 goal 从 78.3% 升至 81.2%，collision 从 9.3% 降至 6.3%。Mixed A 的 collision 从 9.3% 降至 8.5%，但 offroad 从 1.5% 升至 3.1%，并非全面改善。`DriveLite_NoLSTM` 没有改善；`DriveLite_Recurrent224` 在 Mixed B 中 collision 从同质 28.4% 降到 24.3%。这些单次结果支持继续研究时序记忆，但不足以隔离其因果贡献。

### 5.5 下一步建议

Mixed B 是值得 repeat 的候选方向：

```text
Drive_Recurrent + DriveLite_Recurrent224 + Drive_NoLSTM
```

本次 Mixed B p0 的 collision/at-fault 低于 Mixed A p0，但 500M 对照中 Mixed A p0 反而更低，尚未形成跨预算一致的 Mixed B 优势。后续如果继续实验，可以优先：

1. 对 Mixed A/B 进行成组多 seed repeat，确认 p0 差异是否稳定。
2. 增加 PDM held-out 验证，检查结论是否只对 IDM 成立。
3. 若目标是提升高等级 p0，考虑降低最低等级比例，例如 `Drive_Recurrent:DriveLite_Recurrent224:Drive_NoLSTM = 2:1:1`。

## 6. 500M Budget 对照实验

这一组实验用于比较“直接以 500M 为总训练日程训练出的最终策略”和“从 2B/6B 长训练中截取中间 checkpoint”之间的差异。因为 learning-rate schedule 由 `total_timesteps` 决定，500M 短训练的最终权重不等价于 2B 长训练中的 500M 中间权重。

训练预算如下：

```text
同质训练:
  batch_size = 524,288
  total_timesteps = 500,000,000
  final checkpoint = 000954

混合训练:
  batch_size = 1,572,864
  total_timesteps = 1,500,000,000
  mix = 1:1:1
  per-policy raw samples ~= 500M
  final checkpoint = 000954
```

验证设置与主实验一致：

```text
split = pufferinter
traffic = idm
map_ids = all
```

结果目录：

```text
/home/fanyuqi/wsc/behavior-bench/.logs/val/run1/results/500M
```

### 6.1 500M 同质训练验证

| 训练组 | 策略 | 验证目录 | Goal | Collision | At-Fault | Offroad | Reward |
|---|---|---|---:|---:|---:|---:|---:|
| Homogeneous Drive LSTM | `Drive_Recurrent` | `homogeneous_drive_lstm_500m_batch1x_run1/Drive_Recurrent` | 81.3% | 5.9% | 3.2% | 1.4% | -0.72 |
| Homogeneous Drive no LSTM | `Drive_NoLSTM` | `homogeneous_drive_nolstm_500m_batch1x_run1/Drive_NoLSTM` | 74.4% | 13.1% | 10.9% | 2.7% | -0.90 |
| Homogeneous DriveLite no LSTM | `DriveLite_NoLSTM` | `homogeneous_drivelite_nolstm_500m_batch1x_run1/DriveLite_NoLSTM` | 48.6% | 29.0% | 26.0% | 4.2% | -0.84 |
| Homogeneous DriveLite LSTM224 | `DriveLite_Recurrent224` | `homogeneous_drivelite_lstm224_500m_batch1x_run1/DriveLite_Recurrent224` | 54.7% | 32.9% | 30.2% | 3.4% | -0.99 |

### 6.2 500M Mixed A 验证

`Mixed A = Drive_Recurrent + Drive_NoLSTM + DriveLite_NoLSTM`

| Slot | 策略 | 验证目录 | Goal | Collision | At-Fault | Offroad | Reward |
|---|---|---|---:|---:|---:|---:|---:|
| p0 | `Drive_Recurrent` | `mix_intelligence_3level_1500m_batch3x_run1_500m/p0_Drive_Recurrent` | 80.8% | 3.7% | 2.9% | 1.5% | -0.61 |
| p1 | `Drive_NoLSTM` | `mix_intelligence_3level_1500m_batch3x_run1_500m/p1_Drive_NoLSTM` | 78.3% | 7.8% | 6.3% | 1.9% | -0.93 |
| p2 | `DriveLite_NoLSTM` | `mix_intelligence_3level_1500m_batch3x_run1_500m/p2_DriveLite_NoLSTM` | 47.4% | 31.9% | 28.4% | 2.2% | -0.83 |

### 6.3 500M Mixed B 验证

`Mixed B = Drive_Recurrent + DriveLite_Recurrent224 + Drive_NoLSTM`

| Slot | 策略 | 验证目录 | Goal | Collision | At-Fault | Offroad | Reward |
|---|---|---|---:|---:|---:|---:|---:|
| p0 | `Drive_Recurrent` | `mix_drive_lstm_drivelite_lstm224_drive_nolstm_1500m_batch3x_run1_500m/p0_Drive_Recurrent` | 80.7% | 4.9% | 3.4% | 1.9% | -0.68 |
| p1 | `DriveLite_Recurrent224` | `mix_drive_lstm_drivelite_lstm224_drive_nolstm_1500m_batch3x_run1_500m/p1_DriveLite_Recurrent224` | 55.4% | 32.4% | 29.4% | 2.2% | -0.90 |
| p2 | `Drive_NoLSTM` | `mix_drive_lstm_drivelite_lstm224_drive_nolstm_1500m_batch3x_run1_500m/p2_Drive_NoLSTM` | 76.9% | 9.7% | 8.0% | 2.9% | -0.94 |

### 6.4 500M partial repeat

run1 还包含两个局部复验。它们只覆盖 Homogeneous `Drive_NoLSTM` 和 Mixed B p2，不是 Mixed A/B 全组 repeat：

| Repeat | 策略 | Goal | Collision | At-Fault | Offroad | Reward |
|---|---|---:|---:|---:|---:|---:|
| Homogeneous repeat | `Drive_NoLSTM` | 72.2% | 14.6% | 11.4% | 3.7% | -0.95 |
| Mixed B p2 repeat | `Drive_NoLSTM` | 77.2% | 9.8% | 8.1% | 2.0% | -0.93 |

首轮与 repeat 中，Mixed B p2 的 collision 分别为 9.7% / 9.8%，方向较一致；同质 collision 为 13.1% / 14.6%。现有 partial repeat 支持 Mixed B p2 相对同质的改善方向，但不能验证 Mixed B p0，也不能替代 Mixed A/B 全组多 seed 对照。

### 6.5 500M 与 2B/6B 主实验对比

#### 6.5.1 `Drive_Recurrent`

| 来源 | Goal | Collision | At-Fault | Offroad |
|---|---:|---:|---:|---:|
| 500M Homogeneous Drive LSTM | 81.3% | 5.9% | 3.2% | 1.4% |
| 2B Drive.2 baseline | 83.9% | 4.4% | 3.1% | 1.0% |
| 500M Mixed A p0 | 80.8% | 3.7% | 2.9% | 1.5% |
| 6B Mixed A p0 | 80.5% | 6.6% | 5.3% | 1.2% |
| 500M Mixed B p0 | 80.7% | 4.9% | 3.4% | 1.9% |
| 6B Mixed B p0 | 80.8% | 5.4% | 4.4% | 0.9% |

`Drive_Recurrent` 在 500M Homogeneous 下 goal 高于两个 500M mixed p0，但 `500M Mixed A p0` 的 collision/fault 更低。长训练下，2B Drive baseline 仍是更均衡的 high-level 基线，6B mixed p0 没有超过它。

#### 6.5.2 `Drive_NoLSTM`

| 来源 | Goal | Collision | At-Fault | Offroad |
|---|---:|---:|---:|---:|
| 500M Homogeneous Drive no LSTM | 74.4% | 13.1% | 10.9% | 2.7% |
| 2B Homogeneous Drive no LSTM | 78.3% | 9.3% | 7.5% | 1.5% |
| 500M Mixed A p1 | 78.3% | 7.8% | 6.3% | 1.9% |
| 6B Mixed A p1 | 78.6% | 8.5% | 7.6% | 3.1% |
| 500M Mixed B p2 | 76.9% | 9.7% | 8.0% | 2.9% |
| 6B Mixed B p2 | 81.2% | 6.3% | 4.9% | 1.5% |

`Drive_NoLSTM` 在本次 500M 同质 run 中低于 2B 同质终值，但由于 learning-rate schedule 不同，不能仅凭这两个终点严格判定“未充分收敛”。500M Mixed A 对 `Drive_NoLSTM` 有明显改善；500M Mixed B p2 的 collision 为 9.7%，高于 6B Mixed B p2 的 6.3%，但低于 500M 同质的 13.1%。

#### 6.5.3 `DriveLite_NoLSTM` / `DriveLite_Recurrent224`

| 来源 | 策略 | Goal | Collision | At-Fault | Offroad |
|---|---|---:|---:|---:|---:|
| 500M Homogeneous DriveLite no LSTM | `DriveLite_NoLSTM` | 48.6% | 29.0% | 26.0% | 4.2% |
| 2B Homogeneous DriveLite no LSTM | `DriveLite_NoLSTM` | 53.1% | 31.8% | 28.4% | 5.9% |
| 500M Mixed A p2 | `DriveLite_NoLSTM` | 47.4% | 31.9% | 28.4% | 2.2% |
| 6B Mixed A p2 | `DriveLite_NoLSTM` | 55.4% | 32.6% | 28.5% | 3.1% |
| 500M Homogeneous DriveLite LSTM224 | `DriveLite_Recurrent224` | 54.7% | 32.9% | 30.2% | 3.4% |
| 2B Homogeneous DriveLite LSTM224 | `DriveLite_Recurrent224` | 56.7% | 28.4% | 25.0% | 3.4% |
| 500M Mixed B p1 | `DriveLite_Recurrent224` | 55.4% | 32.4% | 29.4% | 2.2% |
| 6B Mixed B p1 | `DriveLite_Recurrent224` | 55.4% | 24.3% | 18.7% | 3.9% |

`DriveLite` 系列整体都明显弱于 `Drive` 系列。500M 下，无论是否加 LSTM，collision 都在 29% 以上；2B/6B 后 `DriveLite_Recurrent224` 有明显改善，尤其 6B Mixed B p1 的 collision 降到 24.3%，但仍远高于 `Drive_Recurrent` 和 `Drive_NoLSTM`。

#### 6.5.4 500M 混合训练对低等级策略本身不一定全面提升

| 策略 | 同质验证 Collision | 混合验证 Collision | 同质 Goal | 混合 Goal | 观察 |
|---|---:|---:|---:|---:|---|
| `Drive_NoLSTM` in Mixed A | 13.1% | 7.8% | 74.4% | 78.3% | 明显改善，collision/fault/goal 都优于 500M 同质 |
| `Drive_NoLSTM` in Mixed B | 13.1% | 9.7% | 74.4% | 76.9% | 有改善，但不如 Mixed A，也不属于异常强结果 |
| `DriveLite_NoLSTM` in Mixed A | 29.0% | 31.9% | 48.6% | 47.4% | 没有提升，安全性略差 |
| `DriveLite_Recurrent224` in Mixed B | 32.9% | 32.4% | 54.7% | 55.4% | 基本持平，只有小幅改善 |

这个表更清楚地说明：500M 下混合训练的收益主要集中在 `Drive_NoLSTM`，尤其是 Mixed A 的 p1；`DriveLite` 系列并没有因为混合训练获得同等幅度提升。

#### 6.5.5 分组结论

500M 结果的主要观察是：

1. 本次 `Drive_Recurrent` 短预算结果中，mixed p0 的收益有限，主要表现为 collision/fault 可下降但 goal 不提升；单 run 不能证明稳定性。
2. `Drive_NoLSTM` 在 mixed 中有改善，其中 `500M Mixed A p1` 改善更明显；`500M Mixed B p2` 只是温和改善，不是异常强结果。
3. 本次 `DriveLite` 的 500M 终值仍明显低于 Drive 系列；加 LSTM 未在该预算下带来明确优势。

### 6.6 500M 结论

500M budget 与 2B/6B 主实验不完全等价，因为 learning-rate schedule 以 `total_timesteps` 为准。500M 结果更适合作为“短预算最终策略”的对照，而不是长训练中间 checkpoint 的替代。

当前 500M 结果显示：

```text
本次 Drive_Recurrent 的 Goal/Collision 综合表现仍然最强，但尚无多 seed 稳定性证据。
DriveLite_NoLSTM 在 500M 下仍明显弱。
DriveLite_Recurrent224 的 500M 同质结果没有明显优于 DriveLite_NoLSTM，甚至 collision 更高。
混合训练在部分 slot 上能降低 collision，但不同 slot 的收益不均衡；500M Mixed B p2 并未出现此前误读的 2.6% collision。
```

现有两个 partial repeat 只覆盖 Homogeneous `Drive_NoLSTM` 和 Mixed B p2。后续若要严谨比较 500M budget，仍需补齐 Mixed A/B 全组多 seed repeat，重点确认 `Drive_NoLSTM` 相对同质训练的改善以及 p0 差异是否稳定。
