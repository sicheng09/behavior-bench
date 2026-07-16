# Perception Run2 实验总结优化设计

## 目标

在不改变原始实验数据、日志路径和章节主结构的前提下，修订
`.logs/try/Perception/perception_experiment_summary.md`，使统计口径、有效分母和结论边界明确且一致。

## 修改范围

仅修改 `perception_experiment_summary.md`：

1. 将 Mixed Single 三等级原始值先平均再取整：
   - 平均 Goal：`75.5%` 改为 `75.4%`
   - 平均 At-Fault：`7.2%` 改为 `7.1%`
2. 在验证设置中补充：18 个验证 run 均为 589 张有效地图、411 张跳过地图；所有验证率的有效分母为 589。
3. 说明指标语义：
   - Collision 是各地图 step 级碰撞率的跨地图平均。
   - At-Fault 是各地图是否发生责任碰撞的二值发生率。
4. 修订定性结论：
   - Mixed Single Mid 的 Goal 高于同质 Mid，但 Collision 和 At-Fault 更差。
   - DDP3 平均 Goal 改善主要来自 Low；平均 At-Fault 改善主要来自 High。
   - 将“更稳定”“未收敛”和机制因果解释改为单次实验可支持的审慎措辞。
5. 补充单 seed、单次评估限制和 repeat 建议。

## 保持不变

- §2 训练配置、W&B run、权重和日志路径。
- §3 的 5 组训练终值。
- §4–§5 的 18 组单 run 验证指标。
- 代码复现指南及任何源码、日志、CSV。

## 验收标准

- 文档中的原始训练和验证表值继续与日志、`summary.csv` 一致。
- §6.2 派生均值统一由原始精度计算，最后一步取整。
- 正文不再包含与表格冲突的 Mid/Goal 描述。
- 所有“稳定性、收敛、机制”表述均明确为观察或待验证假设。
- Markdown 表格与章节层级保持可读。
