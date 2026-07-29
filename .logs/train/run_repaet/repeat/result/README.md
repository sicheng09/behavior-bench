# repeat/result

全部 4 组复训（A/B/Perception/MIXMOE stratified）× 3 协议（IDM/Expert/SMART）评测结束后，后台 session `wait-repeat-summarize` 会自动生成：

- `repeat_vs_first_summary.md` — 每个实验一张表：H0 / 首轮 / 复跑 + 碰撞归因对比
- `metrics_main.csv` / `metrics_attribution.csv`

手动重跑：
```bash
python3 .logs/train/run_repaet/repeat/result/build_repeat_summary.py --wait
# 或结果已齐时
python3 .logs/train/run_repaet/repeat/result/build_repeat_summary.py --once
```
