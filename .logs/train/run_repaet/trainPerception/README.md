# run_repaet / trainPerception · 感知混智对齐复训（500M / policy）

日期：2026-07-22  
对照：`.logs/try/Perception/`（旧 Mixed 只放大 batch/steps，**未**放大 minibatch）  
对齐：与 `run_repaet/trainMOE` / `trainLite` 相同

## 配方

```text
mix = low:1,mid:1,high:1
names = DrivePerceptionLow,DrivePerceptionMid,Drive
rnn   = Recurrent,Recurrent,Recurrent
```

预算（对齐 H0 500M/策略）：

| 量 | H0 | 本实验 |
| --- | ---: | ---: |
| `batch_size` | 524288 | **1572864** |
| `total_timesteps` | 500M | **1.5B** |
| shared `minibatch` | 32768 | **98304** |
| stratified | — | **mb32768 × 16 / policy** |

## 启动

```bash
cd "$HOME/wsc/behavior-bench"
SAMPLING=shared     GPU_IDS=0 bash .logs/train/run_repaet/trainPerception/start_perception_mix_1500m.sh
SAMPLING=stratified GPU_IDS=7 bash .logs/train/run_repaet/trainPerception/start_perception_mix_1500m.sh
# 或
bash .logs/train/run_repaet/trainPerception/launch_all.sh
```
