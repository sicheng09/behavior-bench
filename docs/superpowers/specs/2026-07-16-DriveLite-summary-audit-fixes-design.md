# DriveLite Run1 总结审计修订设计

## 目标

仅修订 `.logs/try/DriveLite/run1_mixed_intelligence_summary.md`，使文档与训练/验证日志一致，并让结论强度符合单 seed、单次验证证据。

## 修改内容

1. 保留已核实的 2B/6B 与 500M 主表数值。
2. 补充验证有效分母：每个主 run 为 589/1000 张地图，411 张被跳过。
3. 说明 Collision、At-Fault、Offroad、Goal 的实际统计口径。
4. 说明 §2 训练指标来自 post-W&B Evaluate；实际步数为 2.017B/6.051B，2B/6B 是名义预算。
5. 修正 Mixed A 验证产物路径：配置/日志目录与 `summary.csv` 的 `p0/p1/p2` 目录分离。
6. 修正 §6.4.4 第 308 行：500M Drive_NoLSTM 改善更明显的是 Mixed A p1。
7. 纳入两个现有 partial repeat：
   - Homogeneous Drive_NoLSTM 500M repeat。
   - Mixed B p2 Drive_NoLSTM 500M repeat。
8. 补充选择性汇报遗漏：
   - Mixed B p2 的 Goal 也优于同质。
   - Mixed A p1 的 Collision 改善但 Offroad 退化。
9. 将“更稳、导致、证明机制、未充分收敛”等表述降级为本次观察或待验证假设。
10. 明确 Drive.2 baseline 仅有可核对日志、CSV 产物缺失，且不是同 pipeline 的严格对照。

## 保持不变

- 不修改训练日志、验证 CSV、代码、权重或代码复现指南。
- 不将不同 split/traffic 的 Drive.2 重跑误并入 pufferinter + IDM baseline。
- 不把 partial repeat 表述成完整 Mixed A/B repeat。

## 验收标准

- 已核实的主表数值不变。
- 第 308 行不再与第 315 行矛盾。
- 文档包含 589/1000、post-W&B、partial repeat 和单 seed 限制。
- 所有机制与稳定性结论均明确证据边界。
- Markdown 表格列数和章节编号正确。
