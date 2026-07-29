# Run3 对抗训练 vs 同质对照

**评测协议**：ego（policy 0）vs IDM · split=`pufferinter` · maps=all

**成功标准**：相对同质，至少不伤 Goal，且 At-fault 不升高。

> 尚未完成 / 缺失：纯 PPO + IDM50%

## 1. 主指标（绝对值）

| 设定 | 说明 | Goal ↑ | Collision ↓ | At-fault ↓ | Offroad ↓ |
| --- | --- | --- | --- | --- | --- |
| **同质 500M（对照）** | 对照 | 81.32% | 5.94% | 3.23% | 1.36% |
| **Adv 500M baseline** | 默认对手奖励 | 79.46% | 4.92% | 4.07% | 2.38% |
| **Adv 500M + IDM50%** | 训练混入 50% IDM | 82.00% | 4.07% | 3.23% | 1.87% |
| **Adv 4B default** | 同默认奖励，加长训 | 79.12% | 5.94% | 5.26% | 1.87% |
| **Adv 500M high_normality** | normality 0.02→0.10 | 76.74% | 5.26% | 4.41% | 1.87% |
| **Adv 500M weak_goal** | goal=0.15 | 77.59% | 6.79% | 6.45% | 2.21% |
| **Adv 500M low_ego_cost** | ego_cost↓ / fault↑ | 77.25% | 8.49% | 7.47% | 2.38% |
| **Adv 500M conservative** | 偏保守对手性格 | 73.68% | 9.85% | 9.34% | 3.40% |

## 2. 相对同质的差值（pp = percentage points）

正值表示该指标数值升高；对 Goal 升高更好，对其余三项升高更差。

| 设定 | Δ Goal | Δ Collision | Δ At-fault | Δ Offroad | 判定 |
| --- | --- | --- | --- | --- | --- |
| Adv 500M baseline | -1.86pp | -1.02pp | +0.84pp | +1.02pp | 更差（goal -1.9pp；fault +0.8pp） |
| Adv 500M + IDM50% | +0.68pp | -1.87pp | +0.00pp | +0.51pp | 更好/持平（相对同质无明显恶化） |
| Adv 4B default | -2.20pp | +0.00pp | +2.03pp | +0.51pp | 更差（goal -2.2pp；fault +2.0pp） |
| Adv 500M high_normality | -4.58pp | -0.68pp | +1.18pp | +0.51pp | 更差（goal -4.6pp；fault +1.2pp） |
| Adv 500M weak_goal | -3.73pp | +0.85pp | +3.22pp | +0.85pp | 更差（goal -3.7pp；fault +3.2pp；coll +0.9pp） |
| Adv 500M low_ego_cost | -4.07pp | +2.55pp | +4.24pp | +1.02pp | 更差（goal -4.1pp；fault +4.2pp；coll +2.6pp） |
| Adv 500M conservative | -7.64pp | +3.91pp | +6.11pp | +2.04pp | 更差（goal -7.6pp；fault +6.1pp；coll +3.9pp） |

## 3. 碰撞形态（碰撞 map 数）

同质以 **侧向** 为主；对抗各组普遍变成 **前方 + 高 at-fault 占比**。

| 设定 | 碰撞 maps | 其中 at-fault | Front | Lateral | 其它 |
| --- | --- | --- | --- | --- | --- |
| 同质 500M | 35 | 19/35 (54%) | 12 | 18 | 5 |
| Adv 500M baseline | 29 | 24/29 (83%) | 17 | 11 | 1 |
| Adv 500M + IDM50% | 24 | 19/24 (79%) | 8 | 15 | 1 |
| Adv 4B default | 35 | 31/35 (89%) | 25 | 8 | 2 |
| Adv 500M high_normality | 31 | 26/31 (84%) | 20 | 7 | 4 |
| Adv 500M weak_goal | 40 | 38/40 (95%) | 31 | 7 | 2 |
| Adv 500M low_ego_cost | 50 | 44/50 (88%) | 34 | 14 | 2 |
| Adv 500M conservative | 58 | 55/58 (95%) | 42 | 14 | 2 |

## 4. 结论（按相对同质的综合表现）

| 排名 | 设定 | 一句话 |
| --- | --- | --- |
| 1 | Adv 500M + IDM50%（训练混入 50% IDM） | 更好/持平：相对同质无明显恶化 |
| 2 | Adv 500M baseline（默认对手奖励） | 更差：goal -1.9pp；fault +0.8pp |
| 3 | Adv 4B default（同默认奖励，加长训） | 更差：goal -2.2pp；fault +2.0pp |
| 4 | Adv 500M high_normality（normality 0.02→0.10） | 更差：goal -4.6pp；fault +1.2pp |
| 5 | Adv 500M weak_goal（goal=0.15） | 更差：goal -3.7pp；fault +3.2pp；coll +0.9pp |
| 6 | Adv 500M low_ego_cost（ego_cost↓ / fault↑） | 更差：goal -4.1pp；fault +4.2pp；coll +2.6pp |
| 7 | Adv 500M conservative（偏保守对手性格） | 更差：goal -7.6pp；fault +6.1pp；coll +3.9pp |

### 要点

1. **没有一组同时打赢同质**（Goal + At-fault）。当前相对最好：`Adv 500M + IDM50%`。
2. **加长到 4B 不能修复**：相对 500M baseline，At-fault 更差，不是「训不够」。
3. **改对手奖励/性格的消融未奏效**；conservative / low_ego_cost 最差。
4. 共同失败模式：碰撞从侧向转向 **前方顶撞**，at-fault 占比升高 → 对手施压轴与 IDM 评测分布不匹配。

## 附录：结果目录

- **同质 500M**：`/home/fanyuqi/wsc/behavior-bench/.logs/val/run1/results/500M/homogeneous_drive_lstm_500m_batch1x_run1/Drive_Recurrent/20260714_150238_c150a5`
- **Adv 500M baseline**：`/home/fanyuqi/wsc/behavior-bench/.logs/val/run3/results/adv_mix_ego_opponent_drive_lstm_ddp2_500m_run3/p0_ego_Drive_Recurrent/20260717_012514_e7de5d`
- **Adv 500M + IDM50%**：`/home/fanyuqi/wsc/behavior-bench/.logs/val/run3/results/adv_mix_ego_opponent_drive_lstm_ddp2_500m_idm50_run3/p0_ego_Drive_Recurrent/20260717_211238_5267cc`
- **Adv 4B default**：`/home/fanyuqi/wsc/behavior-bench/.logs/val/run3/results/adv_mix_ego_opponent_drive_lstm_ddp2_batch2x_4b_run3/p0_ego_Drive_Recurrent/20260717_033332_c57f63`
- **Adv 500M high_normality**：`/home/fanyuqi/wsc/behavior-bench/.logs/val/run3/results/adv_mix_ego_opponent_drive_lstm_ddp2_500m_high_normality_run3/p0_ego_Drive_Recurrent/20260717_041022_00bfa3`
- **Adv 500M weak_goal**：`/home/fanyuqi/wsc/behavior-bench/.logs/val/run3/results/adv_mix_ego_opponent_drive_lstm_ddp2_500m_weak_goal_run3/p0_ego_Drive_Recurrent/20260717_012716_1f30ed`
- **Adv 500M low_ego_cost**：`/home/fanyuqi/wsc/behavior-bench/.logs/val/run3/results/adv_mix_ego_opponent_drive_lstm_ddp2_500m_low_ego_cost_run3/p0_ego_Drive_Recurrent/20260717_041022_0c8d13`
- **Adv 500M conservative**：`/home/fanyuqi/wsc/behavior-bench/.logs/val/run3/results/adv_mix_ego_opponent_drive_lstm_ddp2_500m_conservative_run3/p0_ego_Drive_Recurrent/20260717_042021_2db8ee`
