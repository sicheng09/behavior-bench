# Behavior-Bench 中的 SMART

> 目录：`.logs/train/run4/Smart/`  
> 目的：汇总本仓库与 SMART 相关的训练、权重、评测与代码入口（非 RL Drive 训练）。  
> 更新日期：2026-07-21

---

## 1. SMART 是什么？

在本仓库里，**SMART 不是 `puffer train` 训出的驾驶策略**，而是：

1. **轨迹预测模型**（基于 [SMART](https://github.com/rainmaker22/SMART)，arXiv:2405.15677）  
2. 用 **WOMD 专家轨迹做监督学习（预训练）**  
3. 评测时封装成 **planner / traffic controller**：预测未来轨迹 → 比例控制转成 `(accel, steer)`

与 Drive PPO / ConservativeMix 的关系：

| | Drive PPO | SMART |
| --- | --- | --- |
| 训练入口 | `puffer train puffer_drive*` | `python -m pufferlib.prediction.puffer_prediction pretrain` |
| 目标 | RL 最大化回报 | 预测下一 motion token（CE） |
| 评测角色 | `--planner.type ppo` / `--traffic.type ppo` | `--planner.type smart` / `--traffic.type smart` |
| 可选耦合 | `smart_ref_weights_path` 做 SPACeR KL（Drive ini 里，默认空） | — |

官方说明见仓库根 `README.md`「Available Planners / Traffic」与 `pufferlib/prediction/README.md`。

---

## 2. 代码布局

| 路径 | 作用 |
| --- | --- |
| `pufferlib/prediction/README.md` | **训练文档主入口** |
| `pufferlib/prediction/puffer_prediction.py` | CLI：`warmup-cache` / `pretrain` / `finetune` |
| `pufferlib/prediction/smart/` | 模型实现（decoder / attention / tokens） |
| `pufferlib/prediction/smart/prediction_model.py` | `PredictionModel` 封装 |
| `pufferlib/prediction/smart/tokens/cluster_frame_5_2048.pkl` | 运动 token 码本（veh/ped/cyc 各 2048） |
| `pufferlib/prediction/smart/tokens/map_traj_token5.pkl` | 地图 token 码本 |
| `pufferlib/planning/smart.py` | 评测用 `SMARTPlanner` / `SMARTConfig` |
| `pufferlib/planning/registry.py` | `planner/traffic.type == smart` 注册 |
| `pufferlib/config/prediction/smart*.ini` | 训练超参 |
| `pufferlib/config/evaluation.ini` | `[planner.smart]` / `[traffic.smart]` |
| `pufferlib/nuplan_integration/smart_planner.py` | nuPlan 仿真用 SMART planner |
| `weights/SMART_epoch_030.pt` | 本机自带权重（见 §4） |

架构要点（摘自 prediction README）：

- 输入：场景图（agents + tokenized road polylines）  
- 边：时间边、A2A（60 m）、Map→Agent（30 m）  
- 输出：每 agent 每步 2048-class motion token logits  
- 历史 / 未来默认：`num_historical_steps=11`，`num_future_steps=80`，`shift=5`

---

## 3. 如何训练（本仓库流程）

### 3.1 配置

| 文件 | 规模 | 备注 |
| --- | --- | --- |
| `pufferlib/config/prediction/smart.ini` | ~7M（`hidden_dim=128`, agent layers=6） | 默认大模型 |
| `pufferlib/config/prediction/smart_1m.ini` | ~1M（`hidden_dim=64`, layers 更浅） | 常用；本机权重属此档 |
| `smart_1m_slurm.ini` / `smart_slurm.ini` | 集群路径变体 | `data_dir` 指向集群 binaries |
| `smart_overfit.ini` | 过拟合调试 | |

`smart_1m.ini` 关键关键数据路径（需按机器改）：

```ini
data_dir = /data/ag_nr/share/binaries_full
cache_dir = /data/ag_nr/share/smart_cache_vru
checkpoint_dir = experiments/prediction/smart_1M
```

### 3.2 命令

```bash
cd /home/fanyuqi/wsc/behavior-bench
# 1) 预缓存（推荐）
python -m pufferlib.prediction.puffer_prediction warmup-cache \
  --config pufferlib/config/prediction/smart_1m.ini --workers 80

# 2) 预训练（单卡 / 多卡）
python -m pufferlib.prediction.puffer_prediction pretrain \
  --config pufferlib/config/prediction/smart_1m.ini

torchrun --nproc_per_node=8 -m pufferlib.prediction.puffer_prediction pretrain \
  --config pufferlib/config/prediction/smart_1m.ini

# 3) 可选：在 RL rollout transitions 上 finetune
python -m pufferlib.prediction.puffer_prediction finetune \
  --config pufferlib/config/prediction/smart_1m.ini
```

训练超参（`smart_1m.ini` 默认）：`max_epochs=100`，`lr=5e-4`，`batch_size=4`，checkpoint 每 5 epoch 写入 `experiments/prediction/smart_1M/`。

---

## 4. 本机已有权重

| 文件 | 说明 |
| --- | --- |
| `weights/SMART_epoch_030.pt` | 已存在；约 **1.23M** 参数，`epoch=30`，`hidden_dim=64`（1M 档） |

Checkpoint 字段大致包括：`epoch`、`model_state_dict`、`optimizer_state_dict`、`config`、`wandb_run_id` 等。

根 README 里还提到 `weights/smart_epoch_030.pt` / `smart_1M_epoch_029.pt` 等命名；**本机实际文件名为 `weights/SMART_epoch_030.pt`**。评测时请传真实路径。

加载逻辑：`pufferlib/planning/smart.py` → `load_smart_model()` 读 `model_state_dict` + `config`，并加载 motion/map codebook。

---

## 5. 如何在评测里使用

配置段（`pufferlib/config/evaluation.ini`）：

```ini
[planner.smart]
weights_path =
device = cuda
temperature = 1.0
greedy = True
repredict_interval = 5

[traffic.smart]
; 同上
```

示例（与根 README 一致，路径改成本机文件）：

```bash
export DRIVE_BINARIES_DATA_ROOT=/path/to/binaries   # 或 data/eval_splits 等

# SMART 作 ego
python pufferlib/ocean/benchmark/eval.py \
  --planner.type smart \
  --planner.smart.weights-path weights/SMART_epoch_030.pt \
  --map-ids 0-10

# PPO ego vs SMART traffic
python pufferlib/ocean/benchmark/eval.py \
  --planner.type ppo --planner.ppo.weights-path <ppo.pt> \
  --traffic.type smart \
  --traffic.smart.weights-path weights/SMART_epoch_030.pt
```

运行时行为（`SMARTPlanner`）：

1. 缓冲历史状态（默认 11 步）  
2. 每隔 `repredict_interval`（默认 5，对应训练 `shift`）做一次自回归预测  
3. 用 `k_accel` / `k_steer` 比例控制把预测位置变成动作  

单元测试（有权重才跑）：`tests/test_eval_planners.py`（期望路径曾为 `experiments/smart_epoch_030.pt`，与本机 `weights/` 可能不一致，跑测前需对齐）。

---

## 6. 其它相关用途

### 6.1 Drive RL 的 SMART 参考（可选）

`drive.ini` / `drive_conservative_mix.ini` 等：

```ini
; SPACeR KL regularization with SMART reference model
smart_ref_weights_path =
smart_ref_shift = 5
```

默认**空**，不影响当前 try_IDM / try_Multi 的 PPO 训练。填入 SMART 权重后可对 RL 策略加 KL 正则（人类似行为约束）。

### 6.2 nuPlan

`pufferlib/nuplan_integration/smart_planner.py` +  
`pufferlib/nuplan_integration/config/simulation/planner/smart_planner.yaml`  
把同一套预测模型接到 nuPlan 仿真 planner。

---

## 7. 和当前 run4 实验的关系

| 实验目录 | 是否用 SMART |
| --- | --- |
| `run4/problem`（H0 Drive） | 否（PPO vs IDM/PPO） |
| `run4/try_IDM` / `try_Multi` | 否（ConservativeMix PPO） |
| `run4/Smart`（本目录） | **文档汇总**；训练/评测 SMART 时在此记日志与笔记 |

### run4 评测（已对齐 try_IDM / try_Multi）

脚本：`test/run_smart_ego_eval.sh`  
协议：`pufferinter` / `map-ids=all`；vs IDM + vs PPO（PPO 交通默认 H0 r2 `u9ymfcqr`）。

```bash
bash .logs/train/run4/Smart/test/run_smart_ego_eval.sh \
  smart_ego_epoch030 weights/SMART_epoch_030.pt <gpu> idm

bash .logs/train/run4/Smart/test/run_smart_ego_eval.sh \
  smart_ego_epoch030 weights/SMART_epoch_030.pt <gpu> ppo
```

产物：`test/smart_ego_epoch030/SMART/`（`summary.csv`、`collision_snapshots.json`、干净 wrapper log）。

---

## 8. 快速索引

```text
训练文档     pufferlib/prediction/README.md
训练配置     pufferlib/config/prediction/smart_1m.ini
评测配置     pufferlib/config/evaluation.ini  → [planner.smart] / [traffic.smart]
Planner 实现 pufferlib/planning/smart.py
本机权重     weights/SMART_epoch_030.pt
原论文实现   https://github.com/rainmaker22/SMART
```

---

## 9. 一句话

**SMART = 监督训练的运动 token 预测模型；本仓库用它当可插拔的 ego/traffic 控制器，与 Drive PPO 训练管线分离。本机可直接用 `weights/SMART_epoch_030.pt` 做评测，或按 `smart_1m.ini` 在 WOMD binaries 上重新 pretrain。**
