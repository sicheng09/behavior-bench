# run_repaet / repeat · 优异配置复训复测

日期：2026-07-23  
对照首轮：`.logs/train/run_repaet/result/`  
几何：与首轮相同（每策略对齐 H0 500M）

## 入选标准

从首轮 vs-IDM / vs-SMART / vs-Expert 中挑：

1. **碰撞率约 2.x%**，或  
2. **相对 H0 降幅很大**（尤其跨协议稳健 / 出现翻转）

| 标签 | 首轮 W&B | 复跑 W&B | GPU | 为何复跑 |
| --- | --- | --- | --- | --- |
| **DriveLite A stratified** | `r0jht7bf` | [`bly4mcn2`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/bly4mcn2) | 0 | IDM Coll **2.72%**（−53%）；SMART **2.89%**（−57%） |
| **Perception stratified** | `yov2516v` | [`q2lm8afu`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/q2lm8afu) | 1 | IDM Coll **2.72%**（−53%）；评测 `policy_2` |
| **MIXMOE stratified** | `o4trtoxc` | [`0eyyzzwx`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/0eyyzzwx) | 3 | SMART Coll **4.24%**（−38%）；shared 反而差 → 验证 strat 翻转 |
| **DriveLite B stratified** | `qgytyxt4` | [`jy3p9j4h`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/jy3p9j4h) | 7 | Expert Coll **14.09%**（−16%，全场最优 Expert） |

## 目录

```text
repeat/
  drivelite_mixA_stratified/   # train log + test_{IDM,Expert,Smart}
  perception_stratified/
  mixmoe_stratified/
  launch_selected_repeats.sh
```

训后评测：`idm` / `expert` / `smart`（多卡并行；结果在各自 `test_*` 下）。

全部 4×3 评测结束后，tmux `wait-repeat-summarize` 会自动写入：

- `repeat/result/repeat_vs_first_summary.md`（H0 / 首轮 / 复跑对照表 + 碰撞归因）
- `repeat/result/metrics_main.csv` / `metrics_attribution.csv`

## 启动

```bash
bash .logs/train/run_repaet/repeat/launch_selected_repeats.sh
```

默认 GPU：DriveLite A→0，Perception→1，MIXMOE→3。
