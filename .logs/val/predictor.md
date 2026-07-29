## 1. 普通 Drive 策略 vs DriveLatentWorldModel

普通 `Drive + Recurrent` 是“观测编码 + LSTM 记忆 + 动作/价值头”：

```text
obs_t
 ├─ ego encoder
 ├─ partner encoder + max-pool
 └─ road encoder + max-pool
        ↓
 concat feature
        ↓
 shared embedding h_t
        ↓
 LSTMWrapper
        ↓
 actor logits / value
        ↓
 action_t
```

`Drive` 本体只负责把当前 observation 编成 hidden，并解码 action/value。时间信息由外层 `LSTMWrapper` 维护：eval 时每步用 `LSTMCell` 更新 `lstm_h/lstm_c`；train 时按 `(batch, time)` 展开，逐步处理 `bptt_horizon` 内的序列。它不显式预测下一状态，只学习“当前该怎么做”和“当前状态价值”。

`DriveLatentWorldModel + LatentWorldModelWrapper` 在 `Drive` 编码器基础上多了 latent transition：

```text
obs_t ──encoder──> z_t ──actor/value──> action_t, value_t
  ↑                  │
  │                  ↓
obs_{t+1} encoder  transition(z_t, action_t) -> z_hat_{t+1}
                     │
                     └─ MSE(z_hat_{t+1}, z_{t+1}) auxiliary loss
```

它把 hidden 解释成 latent state `z_t`。`transition_model` 输入是 `[z_{t-1}, onehot(action_{t-1})]`，输出预测下一 latent。`LatentWorldModelWrapper` 兼容 `LSTMWrapper` 接口，但 `lstm_h` 实际存的是 latent `z`，`lstm_c` 只是同形状零张量。训练时每隔 `imagination_horizon` 步重新用真实 obs 编码一次，其余步可用 transition 在 latent 空间推进，并计算 transition consistency loss。

因此二者核心差别是：

```text
Drive:
  记忆来自 LSTM，不预测世界。

DriveLatentWorldModel:
  记忆来自 latent z，并学习 action-conditioned latent dynamics。
```

不过当前 benchmark 的 `WorldModelPlanner` 仍是 reactive `horizon=1`：每步编码当前 obs 后直接输出动作，没有真正用 `imagine()` 做多步规划。因此它的“世界模型”能力主要体现在训练辅助损失和 latent 表征上，尚未完全转化为 PDM 式前向搜索。
