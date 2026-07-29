# Stratified Mix-PPO：共享 Rollout Buffer + 每策略逻辑独立池

日期：2026-07-18  
适用环境：`puffer_drive_conservative_mix`  
目的：混合训练时固定 ego 的有效 minibatch 和 optimizer step 数，消除历史 C-A/C-B 中全局采样后切分造成的更新频率混杂。

## 1. 新旧模式

历史启动命令不变，默认仍为：

```text
mix_ppo_sampling = shared
```

新模式必须显式启用：

```text
mix_ppo_sampling = stratified
```

两种模式都只保存一份 rollout 张量。`stratified` 在训练阶段根据
`segment_policy_ids` 建立逻辑索引池；每个 policy 在自己的池内计算/归一化
priority、固定采样自己的 minibatch，并独立执行 optimizer step。

## 2. 推荐主控制：C-B50 optimizer-matched

与 H0 对齐的 ego 训练几何：

| 项 | H0 | B50 stratified ego |
| --- | ---: | ---: |
| rollout samples / outer epoch | 524288 | 524288 |
| minibatch | 32768 | 32768 |
| optimizer steps / outer epoch | 16 | 16 |
| total ego rollout samples | 500M | 约 500M |
| update_epochs | 1 | 1 |

启动命令：

```bash
cd /home/fanyuqi/wsc/behavior-bench
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

export DRIVE_BINARIES_DATA_ROOT="$PWD/resources/drive/binaries"
export CUDA_VISIBLE_DEVICES=0

puffer train puffer_drive_conservative_mix \
  --config pufferlib/config/ocean/drive_conservative_mix.ini \
  --train.name cons_mix_B50_stratified_optimizer_matched_run4 \
  --train.batch-size 1048576 \
  --train.total-timesteps 1000000000 \
  --train.minibatch-size 32768 \
  --train.max-minibatch-size 32768 \
  --train.bptt-horizon 32 \
  --train.update-epochs 1 \
  --env.num-maps 10000 \
  --env.mix-traffic False \
  --env.ppo-fraction 1.0 \
  --env.idm-fraction 0.0 \
  --env.expert-fraction 0.0 \
  --env.partner-mode reward_shaping \
  --env.partner-target-headway 1.8 \
  --env.partner-w-center 0.05 \
  --env.partner-w-align 0.05 \
  --env.partner-w-steer 0.05 \
  --env.partner-w-gap 0.10 \
  --train.mix-ppo True \
  --train.mix-ppo-policy-mix "ego:0.5,partner:0.5" \
  --train.mix-ppo-policy-names "Drive,Drive" \
  --train.mix-ppo-rnn-names "Recurrent,Recurrent" \
  --train.mix-ppo-policy-paths "," \
  --train.mix-ppo-policy-trainable "True,True" \
  --train.mix-ppo-sampling stratified \
  --train.mix-ppo-policy-minibatch-sizes "32768,32768" \
  --wandb \
  --wandb-project behavior-bench \
  --wandb-group run4-try-idm-stratified
```

`mix_ppo_policy_update_steps` 未指定时自动计算：

```text
steps_i = int(update_epochs × policy_pool_transitions / policy_minibatch_size)
```

B50 中每个池约 524288 transitions，因此两路都自动得到 16 steps。

## 3. C-B33：降低保守伙伴暴露

ego:partner=`2/3:1/3` 时，对齐 ego 500M rollout 预算：

```bash
puffer train puffer_drive_conservative_mix \
  --config pufferlib/config/ocean/drive_conservative_mix.ini \
  --train.name cons_mix_B33_stratified_run4 \
  --train.batch-size 786432 \
  --train.total-timesteps 750000000 \
  --train.minibatch-size 32768 \
  --train.max-minibatch-size 32768 \
  --train.bptt-horizon 32 \
  --train.update-epochs 1 \
  --env.num-maps 10000 \
  --env.mix-traffic False \
  --env.idm-fraction 0.0 \
  --env.partner-mode reward_shaping \
  --env.partner-target-headway 1.8 \
  --train.mix-ppo True \
  --train.mix-ppo-policy-mix "ego:0.6666667,partner:0.3333333" \
  --train.mix-ppo-policy-names "Drive,Drive" \
  --train.mix-ppo-rnn-names "Recurrent,Recurrent" \
  --train.mix-ppo-policy-paths "," \
  --train.mix-ppo-policy-trainable "True,True" \
  --train.mix-ppo-sampling stratified \
  --train.mix-ppo-policy-minibatch-sizes "32768,16384" \
  --train.mix-ppo-policy-update-steps "16,16" \
  --wandb \
  --wandb-project behavior-bench \
  --wandb-group run4-try-idm-stratified
```

1024 agent slots 无法被 2/3 严格整除，因此这里显式锁定两路各 16 steps；ego
每个外层 epoch 固定消费 `16×32768=524288` transitions。partner 使用较小
minibatch，避免其较小 rollout 池限制 ego 的训练几何。

## 4. 多伙伴池模板

示例：ego 50%，四种 partner 各 12.5%。总 batch/steps 与 B50 相同：

```text
mix_ppo_policy_mix = ego:0.5,p1:0.125,p2:0.125,p3:0.125,p4:0.125
mix_ppo_policy_minibatch_sizes = 32768,8192,8192,8192,8192
mix_ppo_policy_update_steps = 16,16,16,16,16
```

若 partner 为 frozen snapshot：

```text
mix_ppo_policy_trainable = True,False,False,False,False
mix_ppo_policy_paths = ,<p1.pt>,<p2.pt>,<p3.pt>,<p4.pt>
mix_ppo_policy_minibatch_sizes = 32768,0,0,0,0
mix_ppo_policy_update_steps = 16,0,0,0,0
```

冻结 partner 只产生交互动作，不执行 loss/backward/optimizer。

## 5. 训练日志验收

新模式记录以下 loss/log 字段（`i` 为 policy id）：

```text
policy_i/stratified_rollout_samples
policy_i/stratified_minibatch_size
policy_i/stratified_optimizer_steps
policy_i/stratified_sampled_transitions
policy_i/stratified_unique_segments
policy_i/stratified_sample_reuse_ratio
policy_i/stratified_sampled_priority_mass_mean
```

B50 的确认条件：

```text
policy_0 rollout_samples     = 524288
policy_0 minibatch_size      = 32768
policy_0 optimizer_steps     = 16
policy_0 sampled_transitions = 524288
```

若这些数字不满足，不进入正式长训评测。

## 6. 非侵入边界与撤回

- 默认 `mix_ppo_sampling=shared`，历史命令和 checkpoint 行为不变。
- `mix_ppo=False` 不创建逻辑池。
- 不修改默认 `puffer_drive`、`drive.ini`、Drive C 动力学、IDM/PDM 或评测路径。
- rollout observations/actions/rewards 不复制；只新增 policy index tensors。
- DDP 模式会在任何 policy forward/backward 前通过 collective 校验各 rank 的
  policy pool 大小、minibatch/update/trainable 配置及最终 optimizer step 数；
  不一致时所有 rank 提前报错，避免部分 rank 进入 backward 后发生 NCCL hang。
- 非 DDP 或 `world_size=1` 时上述校验直接返回，不改变单卡采样和更新逻辑。
- 删除 `pufferlib/policy_sampling.py`、对应测试、`pufferl.py` 中受配置保护的入口和 conservative ini 的三个字段即可撤回。

## 7. 已完成验证

```text
tests/test_policy_sampling.py
tests/test_mix_ppo.py
tests/test_conservative_*.py
```

新增 DDP 保护后的针对性测试结果：`53 passed`。其中真实小训练烟测验证了两个
逻辑池各自获得固定 rollout samples、minibatch size、optimizer steps 和
sampled transitions；另完成双进程 Gloo mismatch smoke，两个 rank 均在训练前
同步退出。

---

## 8. try_IDM 复现长训（A/B stratified，2026-07-19）

相对 mb65k shared 单跑，用 stratified 复现 C-A / C-B（降低“偶然性 + 错误优化几何”混淆）：

```bash
bash .logs/train/run4/try_IDM/train/launch_cons_mix_A_B_stratified.sh
```

| 标签 | tmux | RUN_NAME | W&B | Eval |
| --- | --- | --- | --- | --- |
| C-A | `cons-mix-A-strat-1b-gpu0` | `cons_mix_A_steer0333_stratified_1b_tryIDM` | `e4gq9dom` | GPU 2/3 |
| C-B | `cons-mix-B-strat-1b-gpu1` | `cons_mix_B_shape_stratified_1b_tryIDM` | `u2qysy6b` | GPU 4/5 |

几何：`policy_mb=32768,32768`，`update_steps=16,16`，`batch=1048576`，`steps=1B`。  
训完自动评测 → `.logs/train/run4/try_IDM/test/cons_mix_*_stratified_1b_tryIDM/`。  
长训前验收：日志中 `policy_0/stratified_optimizer_steps=16` 且 `rollout_samples=524288`。

---

## 9. 与旧训练逻辑对比 + 旧启动命令（追加）

> 本节只作对照说明，不改变默认 `shared` 行为。  
> 共同前提（50:50、`batch_size=1048576`）：先填满**一份**共享 rollout；  
> 段数 \(=1048576/32=32768\)，ego/partner 各 **16384 段 = 524288 transitions**（精确对半）。

### 9.1 三种实现怎么从整池走到 `opt.step`

```text
共享 rollout 填满 1,048,576
        │
        ├─ 旧 try（shared, mb=32768）── 未对齐 H0 优化几何 ─────────────┐
        │  循环次数 = 1048576/32768 = 32                                  │
        │  每步：全局抽 32768 → 切开 → ego 期望 ~16384 → 各策略更新         │
        │  ego：32 steps × ~16384 样本（更碎、更频）                       │
        │
        ├─ try_IDM mb65k（shared, mb=65536）── 预算对齐（期望）───────────┐
        │  循环次数 = 1048576/65536 = 16（精确）                           │
        │  每步：全局抽 65536 → 切开 → ego 期望 32768 → 各策略更新          │
        │  ego：16 steps × 期望 32768（步内数量会抖）                       │
        │
        └─ stratified（本文件）── 几何锁死 ──────────────────────────────┐
           循环次数 = 配置 16（或 pool/mb 自动推出）                         │
           每步：只从该策略池抽 32768 → 仅该策略更新                          │
           ego：16 steps × 固定 32768（不依赖切开期望）                       │
```

| 量 | 旧 try shared mb32768 | try_IDM shared mb65k | stratified（现） |
| --- | ---: | ---: | ---: |
| `mix_ppo_sampling` | shared | shared | **stratified** |
| 全局 `minibatch` | 32768 | **65536** | 32768（几乎不控几何） |
| 每 epoch 全局循环 / ego steps | **32** | **16** | **16**（per-policy） |
| 每步 ego 样本量 | 期望 ~16384 | 期望 ~32768 | **固定 32768** |
| 单次抽（一步内） | 无放回 | 无放回 | 无放回 |
| 跨 16/32 步 | 有放回（每步重抽） | 同左 | 同左 |
| 与 H0 优化几何 | 不对齐 | 期望对齐 | **配置对齐（更严）** |

等价性：mb65k 与 stratified 在「16 steps × 每步 32768 ego」的**训练预算（期望/配置）**上对齐；  
**不是**同一采样分布（前者全局抽再切开，后者条件在策略池上抽）。stratified 更严、可日志验收。

### 9.2 旧启动命令（shared）

**A · 旧 try（mb=32768，优化几何未对齐）** — 脚本：`try/run/start_cons_mix_A_batch2x_1b.sh`

```bash
cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench
export DRIVE_BINARIES_DATA_ROOT="$PWD/resources/drive/binaries"
export CUDA_VISIBLE_DEVICES=0

puffer train puffer_drive_conservative_mix \
  --config pufferlib/config/ocean/drive_conservative_mix.ini \
  --train.name cons_mix_A_steer0333_batch2x_1b_run4 \
  --train.batch-size 1048576 \
  --train.total-timesteps 1000000000 \
  --train.minibatch-size 32768 \
  --train.max-minibatch-size 32768 \
  --train.bptt-horizon 32 \
  --train.update-epochs 1 \
  --env.num-maps 10000 \
  --env.mix-traffic False \
  --env.partner-mode action_constraint \
  --env.partner-max-abs-steer 0.333 \
  --train.mix-ppo True \
  --train.mix-ppo-policy-mix "ego:0.5,partner:0.5" \
  --train.mix-ppo-policy-names "Drive,DriveSteerConstrained" \
  --train.mix-ppo-rnn-names "Recurrent,Recurrent" \
  --train.mix-ppo-policy-paths "," \
  --train.mix-ppo-policy-trainable "True,True" \
  --wandb --wandb-project behavior-bench --wandb-group run4-try-cons-mix
```

**B · 旧 try（mb=32768）** — 脚本：`try/run/start_cons_mix_B_batch2x_1b.sh`  
与上相同，但：

```text
--env.partner-mode reward_shaping
--env.partner-target-headway 1.8
--train.mix-ppo-policy-names "Drive,Drive"
--train.name cons_mix_B_shape_batch2x_1b_run4
```

**A/B · try_IDM mb65k（shared，期望对齐 H0）** — 脚本：`try_IDM/train/start_cons_mix_{A,B}_mb65k_1b.sh`  
相对旧 try **仅**把全局 minibatch 提到 65536（采样仍为 shared）：

```bash
# 与旧 try 相同骨架，关键差异：
--train.minibatch-size 65536 \
--train.max-minibatch-size 65536 \
# 无 --train.mix-ppo-sampling（默认 shared）
# A: partner_mode=action_constraint + Drive,DriveSteerConstrained
# B: partner_mode=reward_shaping + Drive,Drive
# 一键：bash .logs/train/run4/try_IDM/train/launch_cons_mix_A_B_mb65k.sh
```

**现 · stratified（配置对齐 H0）** — 见本文 §2 / §8；相对 mb65k 的关键增量：

```text
--train.minibatch-size 32768
--train.max-minibatch-size 32768
--train.mix-ppo-sampling stratified
--train.mix-ppo-policy-minibatch-sizes "32768,32768"
--train.mix-ppo-policy-update-steps "16,16"
```

---

## 10. 自动复训判定（追加，2026-07-19）

stratified C-B 训完+评测完后，由 `auto_decide_retrain_strat_B.sh`（tmux `auto-decide-strat-B`）对照 mb65k B 自动决定是否再训 1 次：

- 门槛翻转（PASS→FAIL）
- |Coll−2.55|≥1.5pp 或 |Fault−1.87|≥1.5pp
- vs-PPO Coll >1.68%

决策写入 `.logs/train/run4/try_IDM/result/stratified_B_retrain_decision.md`；若触发则启动 `cons_mix_B_shape_stratified_1b_tryIDM_r2`。

---

## 11. Stratified DDP 一致性保护（2026-07-19）

### 11.1 背景与旧 shared DDP 行为

旧 `shared mix_ppo` 已有真实 DDP2 长训完成记录。旧训练循环还会在每个
minibatch 中通过 `all_reduce` 检查某个 policy 是否只在部分 rank 出现；如果
出现这种分叉，会在 backward 前报错，避免 DDP 等待不存在的梯度。

stratified 模式改为每个 rank 从共享 rollout storage 本地建立逻辑 policy pool，
并按 policy 独立执行更新。如果不同 rank 得到不同的 pool 大小、训练配置或
optimizer step 数，可能导致各 rank 对 DDP policy 执行不同次数的
forward/backward，最终表现为 NCCL hang。该风险不涉及轨迹被按 timestep 拆分
或 policy ID 错位，而是 DDP 调用次数的一致性问题。

### 11.2 新增保护

实现位置：`pufferlib/pufferl.py::_assert_stratified_ddp_values_match`。

stratified 训练会在任何 policy forward/backward 之前依次校验：

1. 每个 policy 的 rollout pool segment 数；
2. 每个 policy 的 minibatch size；
3. 配置的 policy update steps（自动计算用 `-1` 表示）；
4. 每个 policy 的 trainable 状态；
5. `resolve_policy_update_steps` 得到的最终 optimizer step 数。

校验使用 distributed `all_gather` 收集所有 rank 的整数向量。只要任一 rank
不同，每个 rank 都会在进入训练循环前抛出 `pufferlib.APIUsageError`，错误信息
包含各 rank 的实际值。例如：

```text
stratified mix_ppo DDP requires identical per-policy rollout pool sizes
on every rank; got [[16384, 16384], [16384, 8192]]
```

这样可以把原来可能出现的无提示 NCCL 等待转化为可定位、可复现的配置错误。

### 11.3 单卡与历史功能边界

- `torch.distributed` 未初始化时立即返回；
- distributed 已初始化但 `world_size=1` 时立即返回；
- 单卡不执行 `all_gather`，不增加训练通信；
- 不修改 pool 构建、优先级采样、advantage、loss、optimizer 或更新顺序；
- 不修改默认 `mix_ppo_sampling=shared` 分支；
- 不修改 checkpoint 格式和已有启动参数；
- 修改代码不会影响已经启动的 Python 训练进程；当前 A/B 本身也是两个独立的
  单卡任务，不属于 DDP。

因此本保护不会改变现有单卡 stratified/shared 训练的优化几何和训练结果逻辑。

### 11.4 DDP batch 语义保持不变

配置中的 `mix_ppo_policy_minibatch_sizes` 仍表示**每个 rank**的 minibatch。
例如两卡 DDP 配置：

```text
mix_ppo_policy_minibatch_sizes = 32768,32768
mix_ppo_policy_update_steps    = 16,16
```

则每个 policy 每次同步更新的全局有效 minibatch 为 `32768 × 2 = 65536`，更新
次数仍为 16。该行为与仓库原有 DDP 语义一致，本次保护没有自动缩放 batch。
如果目标是匹配单卡 `32768` 的全局优化几何，两卡时应显式设置每 rank 为
`16384,16384`。

### 11.5 验证结果

- `tests/test_policy_sampling.py`、`tests/test_mix_ppo.py` 和全部 conservative
  针对性测试：`53 passed`；
- 单卡 no-op 测试确认 distributed 未初始化时不会调用 `all_gather`；
- mismatch 单元测试确认 rank 值不同时抛出明确的 `APIUsageError`；
- 真实双进程 CPU/Gloo mismatch smoke：两个 rank 均在训练前同步捕获错误并
  正常退出，没有 deadlock；
- `git diff --check` 通过；
- 当前 A/B 单卡长训日志未出现 Traceback、NCCL、RuntimeError 或
  APIUsageError。
