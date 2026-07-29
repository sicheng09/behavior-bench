# run_repaet · MIXMOE 优化器对齐复训（500M / policy）

日期：2026-07-22  
对照：`.logs/try/MOE/`（旧 MIXMOE3x 只放大了 batch/steps，**未**放大 minibatch）  
对齐规则：`.logs/train/run4/try_IDM/train/README_optimizer_alignment.md`  
同质预算参考：`.logs/train/run4/problem/`（Drive+LSTM 500M）

## 目标

重跑 `Drive : DriveMoE : DriveMoE3 = 1:1:1` 混合训练，使**每个策略**的：

- 每外层 update 样本量
- 每 `opt.step` 样本量（期望）
- 每 epoch `opt.step` 次数
- 总环境步

均对齐 run4 同质 H0 的 500M 配方。

## 预算对照

| 量 | H0 同质 | 本实验（3 路 1:1:1） |
| --- | ---: | ---: |
| `batch_size` | 524288 | **1572864**（×3） |
| `total_timesteps` | 500M | **1.5B**（×3） |
| `minibatch_size` / `max_minibatch_size` | 32768 | **98304**（×3） |
| 每 epoch opt.step | 16 | **16** |
| 每步每策略样本（期望） | 32768 | **~32768** |
| 每策略总环境步 | ≈500M | **≈500M** |
| 外层 epoch | ≈954 → ckpt `000954` | ≈954 → ckpt `000954` |

旧 MIXMOE3x 问题：`batch×3` 但 `minibatch` 仍 32768 → 每 epoch 48 次更碎的 step、每步每策略 ≈10922 样本。

## 两路对照

| 标签 | 采样 | 几何控制 | 脚本 |
| --- | --- | --- | --- |
| **shared mb98k** | `shared` | 全局 `minibatch=98304`（×3） | `trainMOE/start_mixmoe_mb98k_1500m.sh` |
| **stratified** | `stratified` | 每路 `policy_mb=32768` × `16` steps | `trainMOE/start_mixmoe_stratified_1500m.sh` |

stratified 仍共用一份 rollout 张量，但按 `segment_policy_ids` 建**逻辑独立池**，每路在自己池内采样/更新（见 `run4/try_IDM/train/stratified_mix_training.md`）。

## 启动

```bash
cd "$HOME/wsc/behavior-bench"
# shared（已开 GPU1）
bash .logs/train/run_repaet/trainMOE/start_mixmoe_mb98k_1500m.sh
# stratified（默认 GPU2）
bash .logs/train/run_repaet/trainMOE/start_mixmoe_stratified_1500m.sh
```

| | shared | stratified |
| --- | --- | --- |
| tmux | `mixmoe-mb98k-1500m-gpu1` | `mixmoe-strat-1500m-gpu2` |
| W&B group | `run_repaet-mixmoe` | 同左 |
| 验收字段 | 常规 mix_ppo stats | `policy_i/stratified_*` |

## 训后评测（仿 try_IDM）

各子目录 `test_IDM/` / `test_Expert/` / `test_Smart/`：标准 `Drive+Recurrent` vs IDM / Expert / SMART（`pufferinter` · `MAP_IDS=all`）。

```bash
# 训后自动 wait → IDM（训练期用）
bash .logs/train/run_repaet/launch_all_wait_eval_idm.sh

# 权重已齐时立刻并行 Expert / SMART（复用原 train GPU 映射）
bash .logs/train/run_repaet/launch_all_eval_expert.sh
bash .logs/train/run_repaet/launch_all_eval_smart.sh
```

- MIXMOE / DriveLite：`policy_0`
- Perception：`policy_2`（高感知 Drive 槽）
- SMART 交通权重：`weights/SMART_epoch_030.pt`（可用 `SMART_WEIGHTS` 覆盖）
- 公共脚本：`test_helpers/run_one_policy_eval.sh`（`traffic=idm|expert|smart|ppo`）
