# Perception Run2 Summary Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修订 Perception Run2 实验总结中的聚合口径、验证覆盖率和定性结论，同时保持原始训练与验证表值不变。

**Architecture:** 只修改一份 Markdown 总结文件。以训练日志、验证 `summary.csv` 和已完成的三路审计为事实来源，先修订统计说明和派生均值，再收紧结论措辞，最后进行定向文本检查与全文复读。

**Tech Stack:** Markdown、Cursor 文件检索与读取工具。

## Global Constraints

- 仅修改 `.logs/try/Perception/perception_experiment_summary.md`。
- 不修改训练日志、验证 CSV、代码复现指南或源码。
- §3 的训练终值和 §4–§5 的 18 组单 run 验证表值保持不变。
- 派生均值使用原始精度先计算，最后一步保留一位小数。
- 结论必须区分观测结果、稳定性判断和机制假设。

---

### Task 1: 修订实验总结

**Files:**
- Modify: `.logs/try/Perception/perception_experiment_summary.md:76-186`
- Modify: `.logs/try/Perception/perception_experiment_summary.md:214-266`

**Interfaces:**
- Consumes: 5 份训练日志末尾 W&B Run summary、18 份验证 `summary.csv`、有效地图数 `589/1000`。
- Produces: 统计口径一致、结论与表格一致的 Perception Run2 总结文档。

- [ ] **Step 1: 补充验证有效分母和指标语义**

在 §4 的验证设置后说明：

```markdown
有效评估地图数 = 589 / 1000
跳过地图数 = 411
```

并注明 Collision 是地图内 step 级碰撞率的跨地图平均，At-Fault 是地图级二值发生率。

- [ ] **Step 2: 修正 §6.2 的 Mixed Single 聚合值**

将：

```text
平均 Goal 75.5% → 75.4%
平均 At-Fault 7.2% → 7.1%
```

表后补充“均值由 `summary.csv` 原始精度先平均，最后统一取整”。

- [ ] **Step 3: 修订 §6.1 与 §6.2 的归因**

明确：

```text
Mixed Single Mid Goal 76.1% 高于同质 75.2%，但 Collision 和 At-Fault 更差。
DDP3 平均 Goal 的提升主要来自 Low；平均 At-Fault 的改善主要来自 High。
```

- [ ] **Step 4: 收紧 §6.3 与 §8 的结论措辞**

将“更稳定”“未收敛”“主要帮助”等因果或稳定性断言改为：

```text
本次单 seed 结果显示……
001000 checkpoint 的验证表现整体低于最终 checkpoint……
一种可能解释是……仍需多 seed / repeat 验证。
```

- [ ] **Step 5: 补充实验限制**

在结论或建议前增加限制说明：

```text
当前比较基于单 seed、每个 checkpoint 一次 589-map 有效评估；
小幅差异不应直接解释为稳定提升。
```

- [ ] **Step 6: 定向验证关键文本**

检查文档应包含：

```text
589 / 1000
75.4%
7.1%
单 seed
```

检查旧值 `75.5%`、`7.2%` 不再作为 Mixed Single 三等级均值出现；确认 §3–§5 表格未被改变。

- [ ] **Step 7: 全文复读**

确认 Markdown 表格列数一致、章节编号连续、正文不再声称 High 驱动 Goal 均值改善，也不再笼统声称所有 Mixed Mid 指标均未超过同质。

本目录不是 Git 仓库，因此不执行提交步骤。
