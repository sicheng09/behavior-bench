# DriveLite Run1 Summary Audit Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 根据训练、验证和结论三路审计结果修订 DriveLite Run1 总结，同时保持已核实的原始表值不变。

**Architecture:** 仅编辑 `.logs/try/DriveLite/run1_mixed_intelligence_summary.md`。先补统计来源与验证分母，再修正路径、repeat 和硬性矛盾，最后统一收紧结论措辞并验证关键文本。

**Tech Stack:** Markdown、Cursor 文件读取与检索。

## Global Constraints

- 不修改日志、CSV、代码、权重或 `run1_code_changes_reference.md`。
- 不修改已核实的主训练和主验证表值。
- 不把不同 split/traffic 的 Drive.2 重跑并入 pufferinter + IDM baseline。
- partial repeat 必须明确为局部复验，不能表述为完整实验 repeat。

---

### Task 1: 修订 DriveLite Run1 总结

**Files:**
- Modify: `.logs/try/DriveLite/run1_mixed_intelligence_summary.md`

**Interfaces:**
- Consumes: `train/run1` 训练日志、`val/run1` 验证产物、审计设计说明。
- Produces: 数值、路径、统计口径和结论边界一致的总结文档。

- [ ] **Step 1: 补充训练统计来源**

在 §2 说明指标取自 post-W&B Evaluate；2B/6B 是名义预算，实际 agent steps 分别约为 2.017B/6.051B。

- [ ] **Step 2: 补充验证分母与指标定义**

在 §3 设置后写明每个主 run 请求 1000 张地图、有效 589、跳过 411；定义 Goal、Collision、At-Fault、Offroad、Reward。

- [ ] **Step 3: 修正验证产物说明**

说明 Drive.2 baseline 仅保留 eval log、CSV/per_map 缺失；Mixed A 的 config/eval.log 与 summary.csv 分别位于命名目录和 p0/p1/p2 目录。

- [ ] **Step 4: 修订 §5 结论**

将“更稳、导致、说明机制”等改为单 seed 观察或假设；说明缺少同 pipeline 的 2B Homogeneous Drive_Recurrent 对照；补充 Mixed B p2 Goal 改善和 Mixed A p1 Offroad 退化。

- [ ] **Step 5: 增加 500M partial repeat**

新增小节列出：

```text
Homo Drive_NoLSTM repeat: Goal 72.2%, Collision 14.6%, At-Fault 11.4%, Offroad 3.7%, Reward -0.95
Mixed B p2 repeat: Goal 77.2%, Collision 9.8%, At-Fault 8.1%, Offroad 2.0%, Reward -0.93
```

明确它们只覆盖一个同质策略和 Mixed B p2。

- [ ] **Step 6: 修正硬性矛盾**

将原第 308 行“尤其是 Mixed B p2”改为“尤其是 Mixed A p1”，并确保后续分组结论一致。

- [ ] **Step 7: 收紧 500M 结论**

把“稳定、未充分收敛、最稳”等改为“本次单 run 终值”；说明 500M 与 2B/6B learning-rate schedule 不同。

- [ ] **Step 8: 验证文档**

确认包含：

```text
589
post-W&B
partial repeat
Mixed A p1
单 seed
```

确认主表数值未变、Markdown 表格列数一致、章节编号连续。

当前工作区已有大量其他未提交改动，本任务不执行 Git 提交。
