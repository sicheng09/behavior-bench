# Run4 下一阶段：面向未见交通背景的行为多样化训练

日期：2026-07-18  
输入证据：`../try/AGENT_HANDOFF.md`、`../try/result/conservative_mix_A_B_results.md`、各评测目录中的 `per_map.csv` / `collision_snapshots.json`  
目标：进一步降低 pufferinter vs IDM 的 Collision / At-fault，同时避免把失败转移为 Offroad、Goal 下降或同权重 PPO 退化；训练中不接入 IDM/PDM 控制器，并将 PDM 保留为最终盲测背景。

---

## 1. 当前证据与需要修正的结论

### 1.1 双跑均值

| 方法 | IDM Coll ↓ | IDM Fault ↓ | IDM Offroad ↓ | IDM Goal ↑ | PPO Coll ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0（r1/r2 均值） | 5.60% | 4.41% | **1.02%** | 81.49% | 0.60% |
| C-A（r1/r2 均值） | 4.84% | 4.16% | 1.45% | 81.58% | 0.43% |
| C-B（r1/r2 均值） | **4.33%** | **3.23%** | 1.53% | **82.17%** | 0.43% |

C-B 相对 H0 双跑均值约降低 Collision 22.7%、At-fault 26.8%，Goal 和 vs-PPO 没有明显牺牲，是现阶段唯一值得作为主配方延伸的方案。

但不能只写“C-B 已解决问题”：

1. C-B 的 IDM Offroad 从 H0 均值 1.02% 上升到 1.53%（约多 3 张图/589 maps）。绝对值不大、当前无显著性，但方向连续两跑偏高，应视作明确风险。
2. 按 map 配对的双侧 exact McNemar 检验，H0 与 C-B 四种 seed 交叉比较中，Collision / At-fault 的 `p` 约 0.16–0.34，尚未到 0.05。当前结果是“稳定的效果量信号”，不是充分统计确认。
3. C-B 稳定把 Front 压到 5 张，但剩余失败仍以 Lateral 为主；说明继续增强 gap shaping 的边际收益可能有限。
4. C-A 的硬 steer mask 在两跑中都改变了失败类型谱，并增加 Front 占比。因此 C-AB 不是下一轮最高优先级，硬约束还可能强化“刚性障碍物”分布和 Offroad 风险。

### 1.2 机制假设

当前最符合数据的解释是：

```text
同质 H0
  -> 伙伴过度协同，IDM 下 Front + Lateral 增多

C-B 50:50
  -> 保守伙伴暴露改善纵向节奏，Front 稳定下降
  -> 但单一保守风格占比偏高，ego 可能学会更激进地绕过“刚性车流”
  -> Offroad 有上移风险，剩余碰撞集中为 Lateral
```

因此下一步不是把保守性继续加重，而是降低单一行为模式的支配程度，并逐步引入多个可学习/已学习伙伴风格。

---

## 2. 预注册的多目标选择规则

所有候选使用相同 pufferinter map 集、确定性评测、相同 episode/config。禁止只按 IDM Collision 选模型。

### 2.1 主目标

相对 H0 双跑均值：

- IDM Collision 相对下降至少 15%；
- IDM At-fault 相对下降至少 15%。

这次将原来的“Collision 或 Fault”改成“二者同时”，避免指标选择性汇报。

### 2.2 非劣约束

以百分点为单位，候选的双 seed 均值及 worst-seed 均需检查：

| 指标 | 建议容忍上限 |
| --- | --- |
| IDM Offroad | 不高于 H0 均值 +0.34 pp（约 2 maps/589） |
| IDM Goal | 不低于 H0 均值 -1.0 pp |
| vs PPO Collision | 不高于 H0 均值 +0.5 pp |
| vs PPO Goal | 不低于 H0 均值 -1.0 pp |

增加一个防止失败迁移的复合指标：

```text
SafetyViolation(map) = Collision(map) OR Offroad(map)
```

主报告必须给出 SafetyViolation、Collision、At-fault、Offroad 的 map-level paired 差异；若 Collision 降低但 SafetyViolation 不降，不晋级。

### 2.3 统计报告

- 比例：报告双 seed 的 mean、range、worst seed；不只报最好一次。
- 同一 map 集上的二值指标：paired bootstrap 95% CI + exact McNemar。
- 训练 seed 是推断单位；最终候选至少 3 个真实可控 seed。两次未显式固定初始化的复跑不能冒充严格 seed 实验。
- 1 map = 0.17 pp。所有百分点变化同时报告“净变化 map 数”，避免小样本比例看起来过度精确。

---

## 3. 实验阶段与优先级

## Stage 0：不训练，先做行为压力评测

目的：确认 C-B 改善的是一段行为分布，而不是只对默认 `T=1.5, v0=15` 有效。

候选 checkpoint：H0 r1/r2、C-B r1/r2。开发背景：

| Dev 背景 | 配置意图 |
| --- | --- |
| IDM-default | `T=1.5, v0=15`，保持历史主指标 |
| IDM-short | `T=1.0, v0=20`，更短时距/更高目标速度 |
| IDM-long | `T=2.0, v0=12`，更谨慎、制动更早 |
| constant-velocity | 不跟随 ego 协同，测试刚性纵向行为 |
| same-weight PPO | 保持同质能力守门 |

每个背景均使用 all 589 effective maps，输出 per-map 表和碰撞类型。先比较 H0/C-B 的 worst-background SafetyViolation，而不是立即启动新训练。

注意：`traffic.type=pdm` 不进入本阶段。PDM 是最终 held-out 背景；在候选选择完成前不查看 PDM 结果。

## Stage 1：低成本主实验——C-B33

### H1：降低单一保守伙伴暴露可保留 IDM 收益并回收 Offroad

把 ego:partner 从 0.5:0.5 改为 **2/3:1/3**，其他 C-B shaping 不变：

```text
partner_mode = reward_shaping
mix_ppo_policy_mix = ego:0.6666667,partner:0.3333333
partner_target_headway = 1.8
w_center/w_align/w_steer/w_gap = 0.05/0.05/0.05/0.10
```

对齐 ego 500M 预算：

```text
batch_size      = 786432       # 1.5 × 524288；ego/update 仍约 524288
total_timesteps = 750000000    # 1.5 × 500M；ego 总步数仍约 500M
minibatch_size  = 49152        # 1.5 × 32768；每个混合 minibatch 的 ego 样本仍约 32768
max_minibatch_size = 49152     # 与 minibatch 相同，避免进入当前未归一化的梯度累积分支
bptt_horizon    = 32
update_epochs   = 1
```

这里必须同时缩放 `batch_size`、`total_timesteps` 和 `minibatch_size`。当前
mix-PPO 是先从全局 buffer 采样 minibatch、再按 policy id 分开算 loss；如果只扩大
batch 而保持 minibatch=32768，ego 每个 optimizer step 只有约 21845 个样本，且每个
外层 epoch 的 optimizer step 从 16 增为 24。上面的 49152 才同时对齐 ego
minibatch、外层 epoch 数和 optimizer step 数。

同理，历史 B50 的完全优化等价配置应为 `batch_size=1048576`、
`minibatch_size=max_minibatch_size=65536`、`total_timesteps=1B`。已有 B50 使用 32768，虽然对齐了
ego 总样本和外层 epoch，却把 optimizer step 数从 16 增到 32；因此下一阶段应先补
一个 `B50-optimizer-matched` 作为消除该混杂因素的控制实验。

为什么优先：无需改训练核心或 conservative 实现；计算量比 C-B50 少 25%；直接检验当前最重要的“保守暴露过量导致失败迁移”假设。

预测：

- IDM Front 仍显著低于 H0，但可能略高于 B50 的固定 5 张；
- IDM Offroad 接近 H0/C-A；
- Lateral 和 PPO 指标优于或不劣于 B50；
- 若 Coll/Fault 改善完全消失，则说明 B50 的收益依赖高暴露，而非可泛化的响应策略。

主比较：H0、B50、B33；至少两个真实固定 seed 做筛选，晋级后补第三 seed。

## Stage 2：机制消融——只做两个，不进行大网格

若 B33 未满足全部约束，才运行以下消融；先用一个固定 seed 筛选，再给优胜者补 seed。

| ID | partner mix | `(w_center,w_align,w_steer,w_gap)` | 假设 |
| --- | ---: | --- | --- |
| B33-gap | 1/3 | `(0,0,0,0.10)` | 纵向 gap 是 Front 改善主因，去掉横向刚性可降低 Offroad |
| B33-soft | 1/3 | `(0.025,0.025,0.025,0.10)` | 保留软横向引导但降低“车道墙”效应，兼顾 Lateral/Offroad |

不优先运行：

- `w_gap > 0.10` 或更大 headway：现有 Front 已稳定为 5，继续加码缺乏证据；
- `C-A-tight`：已知有 Front 风险；
- `C-AB`：只有当 B33/B33-soft 的 Lateral 明显反弹、而 Offroad 已通过约束时，才作为诊断实验，且先用较软 steer 阈值，不从 0.333 硬组合开始。

## Stage 3：主方法升级——DiversePartnerPool

若 Stage 1/2 证明“多样性优于单一保守风格”，再实现策略池，而不是继续调一个 shaping 点。

建议训练构成：

```text
policy 0: ego，trainable，50%
policy 1: 普通 PPO partner snapshot A，frozen，12.5%
policy 2: 普通 PPO partner snapshot B，frozen，12.5%
policy 3: conservative partner snapshot A，frozen，12.5%
policy 4: conservative partner snapshot B，frozen，12.5%
```

候选伙伴全部来自 PPO 训练权重，不含 IDM/PDM。普通伙伴覆盖协同行为，保守伙伴覆盖不主动让道/保距行为；冻结池避免 ego 与单一同步学习伙伴再次形成私有 convention。

实现边界：

- 新增独立 `pufferlib/conservative_population/`（或 conservative 下的独立 env 类）和新 ini；
- 默认 `puffer_drive`、`drive.ini`、`pufferl.py`、C 动力学、评测路径保持不变；
- role 语义只在新 env 内定义为 `policy_id==0 -> ego`、`policy_id>0 -> partner`；
- `mix_ppo_policy_paths` / `mix_ppo_policy_trainable` 复用现有能力；不在通用 mix_ppo 热路径增加特殊分支；
- 新单测覆盖：多 policy role 分配、ego 奖励不变、frozen partner 不更新、旧 env 非侵入。

该阶段的确认性比较只需要：最佳 B 单策略方案 vs DiversePartnerPool；不要同时引入 curriculum、对抗奖励或新网络结构。

---

## 4. 真实 seed 与复现方案

当前 `pufferl.PuffeRL.__init__` 中 Python/NumPy/Torch 的 `manual_seed` 被注释，而且 policy 在该位置之前已构造，因此 ini 中 `seed=42` 不能控制网络初始化。下一轮必须修正实验方法，但不应改默认训练热路径。

推荐新增**实验专用 launcher**，在调用现有 `load_env/load_policy/train` 之前执行：

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

要求：

- launcher 仅放在 `.logs/train/run4/next/run/` 或独立实验工具中；
- 不取消/修改 `pufferl.py` 中现有注释，不改变旧命令语义；
- 将 seed、git commit、完整解析后 config 写入每个 run 目录；
- 首先做同 seed 的两次 1–2 update 烟测，核对初始参数 hash 和首批日志一致，再启动长训。

建议确认 seed：`101, 202`；最终获胜者补 `303`。不要把当前 r1/r2 标签改称 seed 1/2。

---

## 5. 晋级与停止规则

按以下顺序执行，任何阶段不满足条件就停止扩展该分支：

1. **Stage 0**：若 C-B 在 IDM-short/long 中的 worst-background SafetyViolation 不优于 H0，则先修正行为机制，不训练 B33。
2. **B33 seed 101**：若 IDM Coll 与 Fault 均未比 H0 降 10%，停止 B33，转 B33-gap/B33-soft。
3. **B33 seed 202**：仅当双 seed 均满足主目标，且 Offroad/PPO/Goal 满足非劣约束时晋级。
4. **第三 seed**：只给一个最优单策略候选和（若实现）一个 DiversePartnerPool 候选补 seed 303。
5. **最终盲测**：锁定唯一 checkpoint selection rule 后，一次性评测 `traffic.type=pdm`；不因 PDM 结果返回调参。若要再迭代，必须声明开启新的研究阶段，并使用新的 held-out traffic/split。

最终模型采用约束式、词典序选择：

```text
先通过所有非劣约束
  -> 最小化 worst-seed × worst-dev-background SafetyViolation
  -> 最小化 At-fault
  -> 最大化 Goal
```

这样不会用 Reward 抵消安全退化，也不会以平均值掩盖一个坏 seed/坏背景。

---

## 6. 最终 PDM 盲测

仓库已支持 `MultiAgentPDMPlanner` 作为 traffic controller。锁定候选后使用现有评测入口：

```bash
python pufferlib/ocean/benchmark/eval.py \
  --planner.type ppo \
  --planner.ppo.weights-path <LOCKED_EGO_WEIGHT> \
  --planner.ppo.policy-class-name Drive \
  --planner.ppo.rnn-name Recurrent \
  --planner.ppo.rnn-input-size 256 \
  --planner.ppo.rnn-hidden-size 256 \
  --traffic.type pdm \
  --eval.split pufferinter \
  --map-ids all \
  --output-dir <LOCKED_OUTPUT_DIR>
```

同时评测 H0、B50 和锁定的新候选，但在全部任务完成前不读取中间结果。PDM 计算量显著高于 IDM（每个交通 agent 内含 proposal batch），先用 5 maps 做纯运行烟测，正式盲测仍用 all maps。

PDM 表必须报告与 IDM 完全相同的 Goal / Collision / At-fault / Offroad / SafetyViolation 和碰撞类型，不能在看到结果后换主指标。

---

## 7. 推荐执行顺序（最小充分矩阵）

| 顺序 | 实验 | 新增长训数 | 决策用途 |
| ---: | --- | ---: | --- |
| 0 | H0/B50 行为压力评测 | 0 | 判断收益是否跨 IDM 参数 |
| 1 | B50-optimizer-matched seed 101 + B33 seed 101 | 2 | 消除 optimizer 混杂并筛选暴露比例 |
| 2 | B33-gap、B33-soft seed 101 | 最多 2 | 仅在 B33 不过约束时定位机制 |
| 3 | 最佳消融补 seed 202 | 最多 1 | 确认 |
| 4 | DiversePartnerPool 101/202 | 2 | 验证伙伴多样性/去共适应主张 |
| 5 | 两个 finalist 补 303 | 2 | 稳健性 |
| 6 | 唯一规则锁权重后 PDM all | 0 训练 | 真正 OOD 盲测 |

最理想路径只需先增加 2 次长训；不要一开始做 headway × gap weight × steer threshold 的笛卡尔网格。

---

## 8. 研究依据

- 交通参与者行为参数随机化被用于提升自动驾驶 RL 向未见交通流的迁移，支持从单一伙伴点转向行为分布覆盖：Lin, Xie, Liu, *Autonomous vehicle decision and control through reinforcement learning with traffic flow randomization* (2024), https://arxiv.org/abs/2403.02882
- 标准 self-play 容易形成任意且单一的 convention，跨伙伴泛化受限；多样 convention / mixed play 是直接相关的机制证据：Sarkar, Shih, Sadigh, *Diverse Conventions for Human-AI Collaboration* (NeurIPS 2023), https://openreview.net/forum?id=MljeRycu9s
- 仅有“多样”并不充分，伙伴还需形成有意义的专门行为；这支持保留普通与保守两个可解释行为簇，而不是无目标噪声：Charakorn et al., *Diversity Is Not All You Need: Training A Robust Cooperative Agent Needs Specialist Partners* (NeurIPS 2024), https://openreview.net/forum?id=15460JjocO

这些工作用于支持实验方向，不替代本仓库中的确认性消融与盲测。
