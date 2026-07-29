# ConservativeMix（run4/try）· Agent 交接备忘

> 写给下一个 agent：读完本文 + 文内链接即可接续，无需重扫全对话。  
> 日期：2026-07-18  
> 仓库根：`/home/fanyuqi/wsc/behavior-bench`  
> 本目录：`.logs/train/run4/try/`

---

## 1. 我们要解决什么问题？

**现象（同质自博弈 H0）**：`Drive+Recurrent` 在全 PPO 同伴上训完后，pufferinter **vs IDM** 碰撞明显升高（约 5–6%），**vs 同权重 PPO** 却接近 0。碰撞以 **ACTIVE_LATERAL** 为主、**ACTIVE_FRONT** 为辅。

**根因判断**：训练分布（会横向配合的 PPO 同伴）与评测分布（车道跟随、纵向保距 \(T=1.5\)、横向钝的 IDM）偏移。

**目标**：用**可训的保守伙伴**覆盖「少大转角 / 保距 / 不主动让道」等 IDM 类特征，**训练不接入评测 IDM**；相对 H0 降低 vs-IDM Coll 或 At-fault（相对 −≥15%），且 vs-PPO Coll 不恶化超过 +1 pp。

**问题证据与 H0 数字**：

- `.logs/train/run4/problem/homogeneous_drive_lstm_500m_results.md`
- H0 评测目录现多为：`.logs/train/run4/problem/test_IDM/`（原 `test/` 可能已改名）

**成功门槛**（相对 H0 run4_r2 `u9ymfcqr`）：见  
`.logs/train/run4/try/launch_commands.md` §Success gate，以及结果汇总 §1.3。

| 指标 | H0 r2 | 目标 |
| --- | ---: | --- |
| vs IDM Coll | 5.77% | ≤ 4.90%（−15% rel） |
| vs IDM At-fault | 4.58% | ≤ 3.89%（−15% rel） |
| vs PPO Coll | 0.68% | ≤ 1.68%（+1 pp） |

通过：IDM 上 Coll **或** Fault 达标，**且** PPO Coll 不超上限。

---

## 2. 方案是什么？（ConservativeMix）

**方案 3**：统一 env `puffer_drive_conservative_mix`，`mix_ppo` 双策略 50:50：

| | policy 0 = ego | policy 1 = partner |
|---|---|---|
| **C-A** | 普通 `Drive`+原奖励 | `DriveSteerConstrained`：采样前 mask `|steer|>0.333` |
| **C-B** | 普通 `Drive`+原奖励 | 普通 `Drive` + **partner 专用**奖励塑形 |

**B 的 shaping（仅 partner）**：

\[
r'=\mathrm{clip}(r_\text{base}+w_c r_\text{center}+w_a r_\text{align}-w_s|\mathrm{steer}|+w_g\cdot g,\,-1,1)
\]

- \(r_\text{center}/r_\text{align}\)：**复用** C 侧 raw `l_center` / `l_align`（未新增 C 奖励通道）
- \(g\)：同场景前车 bumper gap → 时距相对 \(T_\text{train}=1.8\)（评测 IDM 为 1.5，刻意不完全抄）
- 默认权重：`w_c=w_a=w_s=0.05`，`w_g=0.10`

**不是完整跟车模型**：只软引导时距，不输出 IDM 加速度。

设计 / 计划全文：

- `.logs/train/run4/try/2026-07-18-conservative-mixed-training-design.zh.md`
- `.logs/train/run4/try/2026-07-18-conservative-mixed-training-plan.md`

---

## 3. 非侵入实现（如何做到「可撤回」）

**原则**：不改默认 `puffer_drive` / `drive.ini` / C 动力学热路径；新增可删包。

| 允许 | 禁止 |
|---|---|
| 新包 `pufferlib/conservative/` | 改 `drive.h` 默认奖励/IDM |
| 新 ini `drive_conservative_mix.ini` | 改 `drive.ini` 默认值 |
| `environment.py` 追加一项懒加载注册 | 让 `mix_ppo=False` 的 Drive 走 conservative |
| 新测试 `tests/test_conservative_*.py` | 训练接评测 IDM |

**代码位置**：

```text
pufferlib/conservative/
  action_constraint.py   # 合法 steer mask、logits -inf
  policy.py              # DriveSteerConstrained
  reward.py              # PartnerShapingEvaluator + gap
  env.py                 # ConservativeMixDrive（继承 Drive）
  config.py              # partner_mode / 权重 / mixed_scene_rate 告警
  REVERT.md              # 撤回清单
pufferlib/config/ocean/drive_conservative_mix.ini
pufferlib/ocean/environment.py  # "drive_conservative_mix" → ConservativeMixDrive
```

**撤回**：`pufferlib/conservative/REVERT.md`（副本：`conservative-mix-revert.md`）。

**角色分配**：复用现有 `mix_ppo` + `assign_policy_ids`（全局 deficit 交错，非整场景同质）。  
审计：`assignment/mixed_scene_rate`（W&B；阈值 0.8 以下才 warn）。B 实测 ≈0.97。

---

## 4. 做了哪些实验？

### 4.1 训练配方（A/B、r1/r2 相同超参）

对齐同质 H0 的 **ego** 预算（50:50 → 总 batch/steps ×2）：

| 项 | 值 |
|---|---|
| env | `puffer_drive_conservative_mix` |
| batch_size | 1048576 |
| total_timesteps | 1e9 |
| updates | ≈953 → ckpt epoch **000954** |
| minibatch / bptt / update_epochs | 32768 / 32 / 1 |
| num_maps | 10000 |
| mix_traffic | False |
| mix_ppo | ego:0.5, partner:0.5，两策略均可训 |

**关于 seed**：ini 写 `seed=42`，但 `pufferl` 里 `torch/np/random.manual_seed` **被注释**；r1/r2 是同配置再跑，**不是**显式换 seed。同质 H0 r1/r2 同理。

### 4.2 四次训练 run

| 标签 | W&B | GPU | 训练日志 |
|---|---|---:|---|
| C-A r1 | `6qdti7k0` | 0 | `run/cons_mix_A_steer0333_batch2x_1b_run4_20260718_040248.log` |
| C-B r1 | `l8z87oaa` | 1 | `run/cons_mix_B_shape_batch2x_1b_run4_20260718_040810.log` |
| C-A r2 | `t1uqsvzw` | 0 | `run/cons_mix_A_steer0333_batch2x_1b_run4_r2_20260718_112001.log` |
| C-B r2 | `i1mvzmos` | 1 | `run/cons_mix_B_shape_batch2x_1b_run4_r2_20260718_112001.log` |

评测用 **ego = `model_policy_0_*_000954.pt`**（不是 policy 1）。

### 4.3 评测协议

与 H0 / `validation.md` 一致：

- split=`pufferinter`，`map_ids=all`（有效约 589）
- vs **IDM** 与 vs **同权重 PPO**（ego=traffic=同一 policy-0 ckpt）
- 入口：`python pufferlib/ocean/benchmark/eval.py`
- 结果根：`.logs/train/run4/try/test/<run_dir>/Drive_Recurrent/`

评测脚本：

- `test/run_one_policy_eval.sh`
- `test/wait_and_eval_cons_mix.sh`
- `test/resolve_final_ego_checkpoint.sh`
- `test/README.md`

---

## 5. 指标与结论（摘要）

**完整表 + 碰撞分解 + r1/r2 对比**：  
→ **`.logs/train/run4/try/result/conservative_mix_A_B_results.md`**（必读）  
机器表：`result/metrics_summary.csv`

### 5.1 主数字（vs IDM / vs PPO）

| Run | IDM Coll | IDM Fault | PPO Coll | Gate |
|---|---:|---:|---:|---|
| H0 r2 | 5.77% | 4.58% | 0.68% | baseline |
| C-A r1 | 5.09% | 4.58% | 0.00% | IDM FAIL |
| C-A r2 | 4.58% | 3.74% | 0.85% | PASS |
| C-B r1 | 4.24% | 3.23% | 0.17% | PASS |
| C-B r2 | 4.41% | 3.23% | 0.68% | PASS |

均值：C-B Coll≈4.33%、Fault=3.23%（双跑均过）；C-A 有方差（仅 r2 过 IDM）。

### 5.2 机制解读（简）

- **A**：Lateral↓ 但 Front 易↑ → At-fault 不稳；硬约束横向，纵向贴挤可能补偿。  
- **B**：Front 稳定压到 ~5；gap+lane shaping 更像「保守交通」。  
- **推荐主配方：C-B**；报结果时给双跑范围。

### 5.3 读指标优先级

1. Collision + At-fault（主）  
2. Goal（能力）  
3. Offroad（防用出路换低撞）  
4. vs PPO Coll（泛化门）

---

## 6. 训练 / 评测如何启动

### 6.1 一键复跑（推荐）

```bash
cd "$HOME/wsc/behavior-bench"
bash .logs/train/run4/try/run/launch_cons_mix_A_B_r2.sh
```

会：启 A/B 训练 tmux → 抓 W&B id → 挂 `wait_and_eval`（训完自动 vs IDM + vs PPO）。

单路 r1 脚本（历史）：

- `run/start_cons_mix_A_batch2x_1b.sh`
- `run/start_cons_mix_B_batch2x_1b.sh`
- 说明：`run/launch_cons_mix_A_batch2x_1b.md`、`launch_commands.md`

### 6.2 训练命令骨架（与脚本一致）

**A：**

```bash
puffer train puffer_drive_conservative_mix \
  --config pufferlib/config/ocean/drive_conservative_mix.ini \
  --train.batch-size 1048576 --train.total-timesteps 1000000000 \
  --train.minibatch-size 32768 --train.bptt-horizon 32 --train.update-epochs 1 \
  --env.num-maps 10000 --env.mix-traffic False \
  --env.partner-mode action_constraint --env.partner-max-abs-steer 0.333 \
  --train.mix-ppo True \
  --train.mix-ppo-policy-mix "ego:0.5,partner:0.5" \
  --train.mix-ppo-policy-names "Drive,DriveSteerConstrained" \
  --train.mix-ppo-rnn-names "Recurrent,Recurrent" \
  --train.mix-ppo-policy-trainable "True,True" \
  --wandb --wandb-project behavior-bench --wandb-group run4-try-cons-mix
```

**B：** 同上，但：

```text
--env.partner-mode reward_shaping
--env.partner-target-headway 1.8
--train.mix-ppo-policy-names "Drive,Drive"
```

环境：`conda activate behavior-bench`；  
`DRIVE_BINARIES_DATA_ROOT=$REPO/resources/drive/binaries`（训练）；  
评测用 `$REPO/data/eval_splits` + split `pufferinter`。

### 6.3 手动评测示例

```bash
MAP_IDS=all bash .logs/train/run4/try/test/run_one_policy_eval.sh \
  <train_dir_name> Drive_Recurrent <weights_path> \
  Drive Recurrent 256 256 <gpu> idm   # 或 ppo
```

vs PPO 时脚本会对 traffic 传同一 `weights_path`。

---

## 7. 目录地图

```text
.logs/train/run4/try/
  AGENT_HANDOFF.md                          ← 本文件
  2026-07-18-conservative-mixed-training-design.zh.md
  2026-07-18-conservative-mixed-training-plan.md
  launch_commands.md
  conservative-mix-revert.md
  run/          # 训练脚本 + 日志
  test/         # 评测脚本 + 各 run 的 eval 产物
  result/       # 汇总 md + metrics_summary.csv
```

相关外部：

- 评测说明：`.logs/val/validation.md`
- H0 问题汇总：`.logs/train/run4/problem/homogeneous_drive_lstm_500m_results.md`
- 单元测试：`tests/test_conservative_*.py`（约 31 passed）

---

## 8. 已知坑 / 实现细节（下一个 agent 易踩）

1. **评测权重**：vs PPO 必须 ego=traffic=**policy 0**；已在各 `config.json` 核对 r1/r2。  
2. **ckpt 命名**：mix_ppo 用 `model_policy_0_puffer_drive_conservative_mix_XXXXXX.pt`，目录 `experiments/puffer_drive_conservative_mix_<wandb_id>/`。  
3. **mixed_scene_rate**：终端 dashboard **不打印**，看 W&B `environment/conservative/assignment/...`；B 才稳定打 shaping 指标。  
4. **wait_and_eval**：曾出现「tmux has-session 立刻判结束」的假完成日志，但 eval 进程/日志仍在跑；以 `summary.csv` + 日志末行 `589` maps 为准。  
5. **勿改**默认 `puffer_drive` 路径做实验；新实验继续挂在 `conservative` 包或新 env 名上。  
6. Canvas（可选）：`~/.cursor/projects/home-fanyuqi-wsc-behavior-bench/canvases/cons-mix-run4-try-eval.canvas.tsx`（r1 分析视图，未含完整 r2）。

---

## 9. 建议的下一步（未做）

按优先级：

1. **C-AB**：`partner_mode=both`（steer mask + reward shaping），同预算训+测双跑。  
2. **B 消融**：`partner_target_headway`、`partner_w_gap` 等。  
3. 若要真·多 seed：打开 `pufferl` 的 `torch.manual_seed` 并显式扫 `--train.seed`。  
4. 论文/报告：以 `result/conservative_mix_A_B_results.md` 为表源，强调 C-B 稳健、C-A 方差。

---

## 10. 一句话状态

**ConservativeMix 已落地且可撤回；C-A/C-B 各完成 r1+r2 训练与 pufferinter（IDM/PPO）评测。C-B 双跑过门槛，推荐为主配方；指标与路径见 `result/conservative_mix_A_B_results.md`。**
