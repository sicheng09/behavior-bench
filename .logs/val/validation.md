# BehaviorBench 验证环境与步骤

本文档整理当前代码库支持的主要验证方式，参考：

- `docs/src/evaluation.md`
- `docs/src/wosac.md`
- `pufferlib/pufferl.py::eval()`
- `pufferlib/ocean/benchmark/`
- `.logs/env/env-install-conda.md`
- `.logs/train/train-puffer-start.md`

## 1. 验证前环境准备

先进入仓库根目录，并激活训练环境：

```bash
cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

export REPO_ROOT="$(pwd)"
export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/resources/drive/binaries"
```

如果当前机器的 `data/` 和 `resources/drive/binaries/` 是软链接，也建议仍然使用上面的仓库相对路径。这样命令在不同设备上只需要改 `cd` 的仓库位置。

确认二进制地图存在：

```bash
ls "$DRIVE_BINARIES_DATA_ROOT"
find "$DRIVE_BINARIES_DATA_ROOT" -maxdepth 2 -type f -name "*.bin" | head
```

常用 split：

```text
$DRIVE_BINARIES_DATA_ROOT/training
$DRIVE_BINARIES_DATA_ROOT/validation
$REPO_ROOT/data/eval_splits/interactive1k
$REPO_ROOT/data/eval_splits/random1k
```

## 2. 验证入口概览

当前有三类常用验证：

```text
1. puffer eval puffer_drive
   - 直接加载 PPO checkpoint 做标准 rollout / WOSAC / human replay。

2. pufferlib/ocean/benchmark/eval.py
   - 更完整的 benchmark 框架，可配置 planner / traffic / split / map_ids。

3. sanity maps
   - 小场景快速 smoke test，适合检查训练环境和策略是否能跑通。
```

## 3. 单策略 checkpoint 的标准验证

普通 PPO checkpoint 路径通常形如：

```text
experiments/puffer_drive_<run_id>/model_puffer_drive_<epoch>.pt
```

最小标准 eval：

```bash
puffer eval puffer_drive \
  --load-model-path experiments/puffer_drive_<run_id>/model_puffer_drive_<epoch>.pt
```

如果只想快速确认模型能加载并 rollout，可以先用较小地图数：

```bash
puffer eval puffer_drive \
  --load-model-path experiments/puffer_drive_<run_id>/model_puffer_drive_<epoch>.pt \
  --eval.split validation \
  --eval.num-maps 5
```

## 4. WOSAC realism 验证

WOSAC 用来评估模拟轨迹和人类轨迹分布的接近程度。代码入口在 `pufferlib/pufferl.py::eval()`，当 `--eval.wosac-realism-eval True` 时会走 `WOSACEvaluator`。

推荐命令：

```bash
puffer eval puffer_drive \
  --load-model-path experiments/puffer_drive_<run_id>/model_puffer_drive_<epoch>.pt \
  --eval.wosac-realism-eval True \
  --eval.human-replay-eval False \
  --eval.split validation \
  --eval.num-maps 20 \
  --eval.backend PufferEnv \
  --eval.wosac-num-rollouts 32 \
  --eval.wosac-init-steps 10 \
  --eval.wosac-control-mode control_wosac \
  --eval.wosac-init-mode create_all_valid \
  --eval.wosac-goal-behavior 2 \
  --eval.wosac-goal-radius 2.0 \
  --eval.wosac-aggregate-results True
```

输出中会包含：

```text
WOSAC_METRICS_START
{... json metrics ...}
WOSAC_METRICS_END
```

常关注指标：

```text
realism_meta_score
ade
min_ade
total_num_agents
```

## 5. Human replay 验证

Human replay 用来测试 policy 作为 ego / SDC 时，与日志中人类交通参与者共存的表现。代码入口同样是 `pufferlib/pufferl.py::eval()`，当 `--eval.human-replay-eval True` 时走 `HumanReplayEvaluator`。

推荐命令：

```bash
puffer eval puffer_drive \
  --load-model-path experiments/puffer_drive_<run_id>/model_puffer_drive_<epoch>.pt \
  --eval.wosac-realism-eval False \
  --eval.human-replay-eval True \
  --eval.split validation \
  --eval.num-maps 20 \
  --eval.backend PufferEnv \
  --eval.human-replay-control-mode control_sdc_only
```

输出中会包含：

```text
HUMAN_REPLAY_METRICS_START
{... json metrics ...}
HUMAN_REPLAY_METRICS_END
```

常关注指标：

```text
collision_rate
offroad_rate
completion_rate
```

## 6. Benchmark eval 框架

`pufferlib/ocean/benchmark/eval.py` 是更完整的 benchmark 入口，适合评估不同 planner / traffic 设置。

### 6.1 统一参数区

```bash
cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

export REPO_ROOT="$(pwd)"

# ===== GPU 与输出 =====
# GPU_ID:
#   指定本次 eval 使用的物理 GPU。通过 CUDA_VISIBLE_DEVICES 生效。
#   例如 GPU_ID="0"、"1"、"5"。
# OUTPUT_ROOT:
#   eval.py 会在该目录下自动创建 <timestamp>_<uuid>/，
#   并保存 config.json、eval.log、per_map.csv 等结果。
export GPU_ID="0"
export OUTPUT_ROOT="$REPO_ROOT/experiments"

# ===== 数据 split =====
# 正式 benchmark 默认使用 Interactive1k 兼容 split：
#   DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/data/eval_splits"
#   EVAL_SPLIT="pufferinter"
# Random1k:
#   EVAL_SPLIT="pufferrandom"
# 普通 validation:
#   DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/resources/drive/binaries"
#   EVAL_SPLIT="validation"
# 注意：当前 Drive 白名单支持 pufferinter/pufferrandom；
# 它们分别作为 interactive1k/random1k 的兼容别名。
export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/data/eval_splits"
export EVAL_SPLIT="pufferinter"

# ===== 地图范围 =====
# MAP_IDS:
#   正式完整 benchmark 用 "all"。
#   快速调试可改为 "0-10"。
#   也支持逗号列表，如 "0,3,8,15"。
export MAP_IDS="all"

# ===== planner / traffic =====
# PLANNER_TYPE:
#   默认 ppo。
#   常见可选：pdm, ppo, smart, idm, hybrid,
#   conditioned_aggr, conditioned_normal, conditioned_caut。
#   具体配置见 pufferlib/config/evaluation.ini 的 [planner] 和 [planner.<type>]。
export PLANNER_TYPE="ppo"

# TRAFFIC_TYPE:
#   默认 idm。
#   常见可选：idm, pdm, ppo, smart, expert,
#   constant_velocity, conditioned_mix, conditioned_aggr,
#   conditioned_normal, conditioned_caut。
#   具体配置见 pufferlib/config/evaluation.ini 的 [traffic] 和 [traffic.<type>]。
export TRAFFIC_TYPE="idm"

# PPO_WEIGHTS_PATH:
#   当 PLANNER_TYPE="ppo" 时需要设置。
#   推荐使用单策略 checkpoint，例如 experiments/puffer_drive_<run_id>.pt
#   或 experiments/puffer_drive_<run_id>/model_puffer_drive_<epoch>.pt。
export PPO_WEIGHTS_PATH="$REPO_ROOT/experiments/puffer_drive_xt5biagd.pt"

# PPO_POLICY_CLASS_NAME / PPO_INPUT_SIZE / PPO_HIDDEN_SIZE:
#   必须与 PPO_WEIGHTS_PATH 对应 checkpoint 的训练架构一致。
#   默认 Drive；如果评估 DriveMoE checkpoint，改成 DriveMoE。
#   当前默认 Drive / DriveMoE / DriveTransformer / DriveGameFormer 等
#   都使用 input_size=64, hidden_size=256。
export PPO_POLICY_CLASS_NAME="Drive"
export PPO_INPUT_SIZE="64"
export PPO_HIDDEN_SIZE="256"
export PPO_REWARD_CONDITIONING="False"

# TRAFFIC_PPO_WEIGHTS_PATH:
#   仅当 TRAFFIC_TYPE="ppo" 时需要设置。
#   注意：PLANNER_TYPE="ppo" 和 TRAFFIC_TYPE="ppo" 使用同一个 PPOPlanner 类，
#   但它们是两个独立实例，权重参数不共享。
#   如果 traffic 也要用 PPO，必须显式传 --traffic.ppo.weights-path；
#   否则 traffic PPO 会随机初始化。
export TRAFFIC_PPO_WEIGHTS_PATH="$REPO_ROOT/experiments/puffer_drive_xt5biagd.pt"
export TRAFFIC_PPO_POLICY_CLASS_NAME="Drive"
export TRAFFIC_PPO_INPUT_SIZE="64"
export TRAFFIC_PPO_HIDDEN_SIZE="256"
export TRAFFIC_PPO_REWARD_CONDITIONING="False"

# ===== 可选可视化 =====
# EVAL_VIZ:
#   True 时保存场景级可视化/动图，速度更慢、占用更多磁盘。
# PLANNER_VIZ:
#   True 时保存 planner 内部可视化，例如 PDM proposals 等。
# 正式大规模 benchmark 默认关闭。
export EVAL_VIZ="False"
export PLANNER_VIZ="False"
```

如果只是测试框架是否能跑通，可以先不用 PPO 权重，跑默认 PDM ego vs IDM traffic：

```bash
python pufferlib/ocean/benchmark/eval.py \
  --eval.split "$EVAL_SPLIT" \
  --map-ids "$MAP_IDS"
```

### 6.2 前台运行 PPO ego vs IDM traffic

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python pufferlib/ocean/benchmark/eval.py \
  --output-dir "$OUTPUT_ROOT" \
  --planner.type "$PLANNER_TYPE" \
  --planner.ppo.weights-path "$PPO_WEIGHTS_PATH" \
  --planner.ppo.device cuda \
  --planner.ppo.policy-class-name "$PPO_POLICY_CLASS_NAME" \
  --planner.ppo.input-size "$PPO_INPUT_SIZE" \
  --planner.ppo.hidden-size "$PPO_HIDDEN_SIZE" \
  --planner.ppo.reward-conditioning "$PPO_REWARD_CONDITIONING" \
  --traffic.type "$TRAFFIC_TYPE" \
  --traffic.ppo.weights-path "$TRAFFIC_PPO_WEIGHTS_PATH" \
  --traffic.ppo.device cuda \
  --traffic.ppo.policy-class-name "$TRAFFIC_PPO_POLICY_CLASS_NAME" \
  --traffic.ppo.input-size "$TRAFFIC_PPO_INPUT_SIZE" \
  --traffic.ppo.hidden-size "$TRAFFIC_PPO_HIDDEN_SIZE" \
  --traffic.ppo.reward-conditioning "$TRAFFIC_PPO_REWARD_CONDITIONING" \
  --eval.split "$EVAL_SPLIT" \
  --eval.viz "$EVAL_VIZ" \
  --eval.planner-viz "$PLANNER_VIZ" \
  --map-ids "$MAP_IDS"
```

`eval.py` 会自动创建输出目录并写日志：

```text
$OUTPUT_ROOT/<timestamp>_<uuid>/
  config.json
  eval.log
  per_map.csv
```

### 6.3 tmux 后台运行

长时间 benchmark 建议用 tmux 后台运行。下面模板会把终端输出同时写到 `.logs/val/runs/`：

```bash
mkdir -p "$REPO_ROOT/.logs/val/runs"

export TMUX_SESSION="eval_${EVAL_SPLIT}_${PLANNER_TYPE}_vs_${TRAFFIC_TYPE}_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S)"
export EVAL_LOG="$REPO_ROOT/.logs/val/runs/${TMUX_SESSION}.log"

tmux new-session -d -s "$TMUX_SESSION" "bash -lc '
set -euo pipefail

cd \"$REPO_ROOT\"
source \"$HOME/miniconda3/etc/profile.d/conda.sh\"
conda activate behavior-bench

# 这里再次 export，是为了让 tmux 内部环境自包含；
# 外层 export 负责生成 tmux 命令，内层 export 方便 attach 后检查配置。
export REPO_ROOT=\"$REPO_ROOT\"
export OUTPUT_ROOT=\"$OUTPUT_ROOT\"
export DRIVE_BINARIES_DATA_ROOT=\"$DRIVE_BINARIES_DATA_ROOT\"
export CUDA_VISIBLE_DEVICES=\"$GPU_ID\"
export GPU_ID=\"$GPU_ID\"
export EVAL_SPLIT=\"$EVAL_SPLIT\"
export MAP_IDS=\"$MAP_IDS\"
export PLANNER_TYPE=\"$PLANNER_TYPE\"
export TRAFFIC_TYPE=\"$TRAFFIC_TYPE\"
export PPO_WEIGHTS_PATH=\"$PPO_WEIGHTS_PATH\"
export PPO_POLICY_CLASS_NAME=\"$PPO_POLICY_CLASS_NAME\"
export PPO_INPUT_SIZE=\"$PPO_INPUT_SIZE\"
export PPO_HIDDEN_SIZE=\"$PPO_HIDDEN_SIZE\"
export PPO_REWARD_CONDITIONING=\"$PPO_REWARD_CONDITIONING\"
export TRAFFIC_PPO_WEIGHTS_PATH=\"$TRAFFIC_PPO_WEIGHTS_PATH\"
export TRAFFIC_PPO_POLICY_CLASS_NAME=\"$TRAFFIC_PPO_POLICY_CLASS_NAME\"
export TRAFFIC_PPO_INPUT_SIZE=\"$TRAFFIC_PPO_INPUT_SIZE\"
export TRAFFIC_PPO_HIDDEN_SIZE=\"$TRAFFIC_PPO_HIDDEN_SIZE\"
export TRAFFIC_PPO_REWARD_CONDITIONING=\"$TRAFFIC_PPO_REWARD_CONDITIONING\"
export EVAL_VIZ=\"$EVAL_VIZ\"
export PLANNER_VIZ=\"$PLANNER_VIZ\"
export EVAL_LOG=\"$EVAL_LOG\"

python pufferlib/ocean/benchmark/eval.py \
  --output-dir \"$OUTPUT_ROOT\" \
  --planner.type \"$PLANNER_TYPE\" \
  --planner.ppo.weights-path \"$PPO_WEIGHTS_PATH\" \
  --planner.ppo.device cuda \
  --planner.ppo.policy-class-name \"$PPO_POLICY_CLASS_NAME\" \
  --planner.ppo.input-size \"$PPO_INPUT_SIZE\" \
  --planner.ppo.hidden-size \"$PPO_HIDDEN_SIZE\" \
  --planner.ppo.reward-conditioning \"$PPO_REWARD_CONDITIONING\" \
  --traffic.type \"$TRAFFIC_TYPE\" \
  --traffic.ppo.weights-path \"$TRAFFIC_PPO_WEIGHTS_PATH\" \
  --traffic.ppo.device cuda \
  --traffic.ppo.policy-class-name \"$TRAFFIC_PPO_POLICY_CLASS_NAME\" \
  --traffic.ppo.input-size \"$TRAFFIC_PPO_INPUT_SIZE\" \
  --traffic.ppo.hidden-size \"$TRAFFIC_PPO_HIDDEN_SIZE\" \
  --traffic.ppo.reward-conditioning \"$TRAFFIC_PPO_REWARD_CONDITIONING\" \
  --eval.split \"$EVAL_SPLIT\" \
  --eval.viz \"$EVAL_VIZ\" \
  --eval.planner-viz \"$PLANNER_VIZ\" \
  --map-ids \"$MAP_IDS\" \
  2>&1 | tee -a \"$EVAL_LOG\"
'"

echo "Started tmux session: $TMUX_SESSION"
echo "Attach with: tmux attach -t $TMUX_SESSION"
echo "Log: $EVAL_LOG"
```

### 6.4 查看、停止和清理

查看 tmux session：

```bash
tmux ls
tmux attach -t "$TMUX_SESSION"
```

实时查看 wrapper 日志：

```bash
tail -f "$EVAL_LOG"
```

查看 `eval.py` 自动生成的实验日志：

```bash
ls -td "$OUTPUT_ROOT"/* | head
tail -f "$(ls -td "$OUTPUT_ROOT"/* | head -1)/eval.log"
```

优雅停止：

```bash
tmux send-keys -t "$TMUX_SESSION" C-c
sleep 5
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
```

查找残留 eval 进程：

```bash
pgrep -af "pufferlib/ocean/benchmark/eval.py" || true
```

必要时终止残留进程：

```bash
pkill -TERM -f "pufferlib/ocean/benchmark/eval.py" || true
sleep 5
pkill -KILL -f "pufferlib/ocean/benchmark/eval.py" || true
```

### 6.5 常见 split 设置

普通 validation：

```bash
export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/resources/drive/binaries"
export EVAL_SPLIT="validation"
export MAP_IDS="0-50"
```

Interactive1k：

```bash
export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/data/eval_splits"
export EVAL_SPLIT="pufferinter"
export MAP_IDS="all"
```

Random1k：

```bash
export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/data/eval_splits"
export EVAL_SPLIT="pufferrandom"
export MAP_IDS="all"
```

## 7. Sanity maps smoke test

Sanity maps 是轻量小场景，适合快速检查环境、渲染和训练入口是否正常。

单个或少量 sanity maps：

```bash
puffer sanity puffer_drive \
  --wandb \
  --wandb-name sanity-demo \
  --sanity-maps forward_goal_in_front s_curve
```

全部 sanity maps：

```bash
puffer sanity puffer_drive \
  --wandb \
  --wandb-name sanity-all
```

短跑 sanity 建议关闭学习率退火：

```bash
--train.anneal-lr False
```

## 8. Multi-PPO / mix_ppo 验证注意事项

当前 `mix_ppo` 训练会保存两类文件：

```text
model_puffer_drive_<epoch>.pt
mix_model_puffer_drive_<epoch>.pt
trainer_state.pt
```

代码位置：`pufferlib/pufferl.py::save_checkpoint()`。

含义：

```text
model_*.pt
  只保存 self.uncompiled_policy，也就是 policy0 / learner 的 state_dict。

mix_model_*.pt
  保存所有 PPO-like policies 的 state_dict 列表、policy_trainable 和 agent_policy_ids。
```

重要限制：当前普通 `puffer eval --load-model-path ...` 的加载路径主要走 `load_policy()`，它直接加载单个 policy 的 state_dict；`load_mixed_policies()` 支持通过 `--train.mix-ppo-policy-paths` 给每个 policy 单独传 checkpoint，但当前代码还没有直接读取 `mix_model_*.pt` 并恢复完整异质 policy 组合的 eval 入口。

因此：

```text
评估单个 learner/policy0：
  可以直接用 model_*.pt。

评估完整 mix_ppo 异质策略组合：
  当前需要额外实现 mix_model_*.pt loader，或先拆出每个 policy 的独立 state_dict，
  再通过 --train.mix-ppo-policy-paths 分别传入。
```

在没有额外 loader 的情况下，不建议误用：

```bash
puffer eval puffer_drive \
  --load-model-path experiments/.../model_puffer_drive_<epoch>.pt \
  --train.mix-ppo True \
  --train.mix-ppo-policy-names Drive,DriveTransformer,DriveGameFormer
```

因为这样只会可靠加载第一个 policy，其他 policy 可能是随机初始化，不能代表完整异质训练结果。

## 9. 推荐验证顺序

单策略 PPO：

```text
1. 确认 DRIVE_BINARIES_DATA_ROOT。
2. 用 validation + num_maps=5 做 puffer eval smoke。
3. 跑 WOSAC validation num_maps=20。
4. 跑 human replay validation num_maps=20。
5. 需要完整 benchmark 时再跑 pufferlib/ocean/benchmark/eval.py。
```

Multi-PPO：

```text
1. 先看训练期间 dashboard / W&B 中 p0/p1/p2 Stats。
2. 单独评估 policy0 时用 model_*.pt。
3. 如果要评估完整异质组合，先补 mix_model loader 或拆分 mix_model。
4. 不要把只含 policy0 的 model_*.pt 当成完整 mix_ppo checkpoint。
```

## 10. 常见问题

### 找不到地图

检查：

```bash
echo "$DRIVE_BINARIES_DATA_ROOT"
ls "$DRIVE_BINARIES_DATA_ROOT"
find "$DRIVE_BINARIES_DATA_ROOT" -maxdepth 2 -type f -name "*.bin" | head
```

### WOSAC 只能用 `PufferEnv`

`pufferlib/pufferl.py::eval()` 中有断言：

```text
WOSAC evaluation only supports PufferEnv backend.
```

所以 WOSAC 命令里建议显式写：

```bash
--eval.backend PufferEnv
```

### 训练内 eval 和离线 eval

训练内 eval 由 `pufferlib/utils.py` 的 subprocess 函数触发，并写入 W&B 的 `eval/...` 指标。正式训练模板中目前保持关闭，推荐训练结束后用本文件中的离线命令单独运行。
