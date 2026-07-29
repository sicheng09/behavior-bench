# run_repaet / trainLite · DriveLite 混智对齐复训（500M / policy）

日期：2026-07-22  
对照：`.logs/try/DriveLite/`（旧 Mixed A/B 只放大了 batch/steps，**未**放大 minibatch）  
对齐：与 `run_repaet/trainMOE` 的 MIXMOE 复训相同（参考 try_IDM optimizer alignment）

## 配方

| 标签 | 组成 | rnn / sizes |
| --- | --- | --- |
| **Mixed A** | `Drive,Drive,DriveLite` | `Recurrent,None,None` |
| **Mixed B** | `Drive,DriveLite,Drive` | `Recurrent,Recurrent,None` + sizes `256,224,0` |

预算（三路 1:1:1 → 对齐 H0 500M/策略）：

| 量 | H0 | 本实验 |
| --- | ---: | ---: |
| `batch_size` | 524288 | **1572864** |
| `total_timesteps` | 500M | **1.5B** |
| shared `minibatch` | 32768 | **98304** |
| stratified per-policy | mb32768 × 16 | **同左 ×3 路** |

## 启动

```bash
cd "$HOME/wsc/behavior-bench"
# VARIANT=A|B  SAMPLING=shared|stratified  GPU_IDS=<n>
VARIANT=A SAMPLING=shared     GPU_IDS=3 bash .logs/train/run_repaet/trainLite/start_drivelite_mix_1500m.sh
VARIANT=A SAMPLING=stratified GPU_IDS=4 bash .logs/train/run_repaet/trainLite/start_drivelite_mix_1500m.sh
VARIANT=B SAMPLING=shared     GPU_IDS=5 bash .logs/train/run_repaet/trainLite/start_drivelite_mix_1500m.sh
VARIANT=B SAMPLING=stratified GPU_IDS=6 bash .logs/train/run_repaet/trainLite/start_drivelite_mix_1500m.sh
```

或一键四路：`bash .logs/train/run_repaet/trainLite/launch_all_A_B.sh`
