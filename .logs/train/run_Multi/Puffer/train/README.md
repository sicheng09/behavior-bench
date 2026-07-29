# run_Multi / Puffer / train · Hybrid 辅助预训练

启动风格与 `.logs/train/run_repaet/repeat` 对齐：

- 每路一个目录 + `start.sh`
- `tmux` 后台会话（不用 `script`，避免 ANSI 刷屏乱码）
- 日志：`tee` → `${RUN_NAME}_YYYYMMDD_HHMMSS.log`
- 不用 `script`（旧 `terminal.log` 里大量 `\033` 来自全屏重绘）

## 当前三路（辅助策略 500M）

| 目录 | 策略 | GPU | tmux |
| --- | --- | ---: | --- |
| `pretrain_low/` | `HybridDriveLiteLow` · 无 LSTM · 60°/50m | 1 | `multi-pretrain-low-gpu1` |
| `pretrain_mid/` | `HybridDriveMid` · Recurrent · 120°/50m | 2 | `multi-pretrain-mid-gpu2` |
| `pretrain_high/` | `HybridDriveMoE3High` · Recurrent · 全局 | 3 | `multi-pretrain-high-gpu3` |

配方见上级 `../hybrid_training_non侵入式修改说明.md`（入口可不同；本目录以 `start.sh` 为准）。

## 启动 / 重跑

```bash
bash .logs/train/run_Multi/Puffer/train/launch_pretrain.sh
# 或单路
GPU_IDS=1 bash .logs/train/run_Multi/Puffer/train/pretrain_low/start.sh
```

完成后取各目录下最终 `model_puffer_drive_*.pt`（或 `experiments/` 对应 run）作为冻结辅助权重，再开 Original target mix。
