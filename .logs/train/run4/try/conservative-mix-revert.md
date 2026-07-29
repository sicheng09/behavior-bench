# ConservativeMix：可退回性质与退回方式

本功能以**增量包**方式接入，合并进主分支后仍可整包撤回，且默认训练路径不受影响。

设计依据：`.logs/train/run4/try/2026-07-18-conservative-mixed-training-design.zh.md` §4.2–4.4。

---

## 1. 可退回性质（保证什么）

| 保证 | 含义 |
| --- | --- |
| **默认路径零行为变化** | 未启动 `puffer_drive_conservative_mix` 时，不加载本包逻辑；`puffer train puffer_drive` 与合并前语义一致 |
| **启动方式不变** | 旧命令、旧 `drive.ini`、旧评测入口无需改参数即可复现历史实验 |
| **整包可删** | 撤回 = 删除本功能新增文件 + 去掉一行 env 注册；无需改写 C 动力学或默认奖励 |
| **不污染复现** | 本功能不修改 `drive.h` / 默认 `Drive` 奖励 / `mix_traffic` 默认值 / 评测 at-fault 定义 |

合并进 `main-behavior-bench` **不等于不可逆**。合并只是把增量纳入主线；退回后旧实验仍可按原命令复现。

---

## 2. 本功能触及的文件（增量清单）

**新增**

- `pufferlib/conservative/`（整目录，含本文件）
- `pufferlib/config/ocean/drive_conservative_mix.ini`  
  （若 `config` → `pufferlib/config` 为符号链接，则 `config/ocean/` 下为同一文件）
- `tests/test_conservative_*.py`

**修改（仅追加）**

- `pufferlib/ocean/environment.py`：`MAKE_FUNCTIONS` 中增加一项  
  `"drive_conservative_mix": lazy_import("pufferlib.conservative.env", "ConservativeMixDrive")`

**明确未改（撤回时也不应去改）**

- `pufferlib/config/ocean/drive.ini` 默认值
- `pufferlib/pufferl.py`、`pufferlib/policy_mix.py`
- `pufferlib/ocean/drive/drive.py` / `drive.h` / `binding.c`
- `pufferlib/ocean/torch.py` 源码（伙伴策略仅在新 env 构造时 runtime 注册）
- 评测 `collision_classifier` / PDM 语义

---

## 3. 推荐退回方式（业务回退，保留 git 历史）

在仓库根目录执行：

```bash
# 1) 删除新包与测试
rm -rf pufferlib/conservative
rm -f tests/test_conservative_*.py

# 2) 删除新 ini（config 若为 symlink，删 pufferlib 侧即可）
rm -f pufferlib/config/ocean/drive_conservative_mix.ini
rm -f config/ocean/drive_conservative_mix.ini

# 3) 编辑 pufferlib/ocean/environment.py
#    删除 MAKE_FUNCTIONS 中的 "drive_conservative_mix" 条目（含 lazy_import 那几行）

# 4) 验收：旧路径仍可用
python -c "from pufferlib.pufferl import load_config; print(load_config('puffer_drive')['env_name'])"
# 期望：puffer_drive

# 可选：回归既有测试
python -m pytest tests/test_mix_ppo.py tests/test_adversarial_mix_env.py -q
```

完成后提交一次即可，例如：`revert: remove ConservativeMix pack`。

此时 `puffer train puffer_drive_conservative_mix` 将不可用；`puffer train puffer_drive` 恢复为「从未引入本功能」的行为面。

---

## 4. 用 git 反做合并（历史回退）

适用于希望提交历史上也去掉本次引入的场景。

**本次合入记录（fast-forward，无独立 merge commit）：**

| 项 | 值 |
| --- | --- |
| 合入前主线 SHA | `5428d1793cc8df50a689bc6407cdec97953696d0` |
| 合入后 tip（含本说明） | 见 `git log -1 --oneline`；首次合入 tip 为 `521129a5`，随后 REVERT 补记为 `41043ccc` |
| 合入方式 | `git merge feature/conservative-mix` → **fast-forward** |
| 功能提交区间 | `5428d179..41043ccc`（含 REVERT 文档） |

因是 fast-forward，**没有** `git revert -m 1 <merge_commit>` 可用的双亲合并提交；历史回退用下面两种方式之一。

### 4.1 合入后尚未 push（或可改写远程）

```bash
git reset --hard 5428d1793cc8df50a689bc6407cdec97953696d0
```

危险：会丢掉该 SHA 之后的本地提交（含本功能全部 commits）。仅在确定没有需保留的后续提交时使用。

### 4.2 合入后已 push（推荐：逆向提交，不改写历史）

```bash
# 反做整个功能区间（会生成一批或一个反向提交，视 git 版本/策略而定）
git revert --no-commit 5428d179..41043ccc
git commit -m "revert: remove ConservativeMix (5428d179..41043ccc)"
```

若区间上又叠了无关提交，优先改用 §3 业务回退，避免误伤。
### 4.3 仅停用、不删代码

不删文件，只停止使用新 env 即可：继续用 `puffer train puffer_drive`，不要传 `puffer_drive_conservative_mix`。代码留在树里但默认路径不执行。

---

## 5. 退回验收清单

- [ ] `pufferlib/conservative/` 不存在（若走 §3）
- [ ] `MAKE_FUNCTIONS` 无 `drive_conservative_mix`
- [ ] `load_config('puffer_drive')` 成功且 `env_name == puffer_drive`
- [ ] `drive.ini` 中无 `partner_mode` / `DriveSteerConstrained`
- [ ] 需要时：既有 `test_mix_ppo` / adversarial 测试通过

---

## 6. 相关文档

| 文档 | 路径 |
| --- | --- |
| 设计（含非侵入硬约束） | `.logs/train/run4/try/2026-07-18-conservative-mixed-training-design.zh.md` |
| 实施计划 | `.logs/train/run4/try/2026-07-18-conservative-mixed-training-plan.md` |
| 启动命令 | `.logs/train/run4/try/launch_commands.md` |
| 本说明（随代码） | `pufferlib/conservative/REVERT.md` |
| 本说明（实验目录副本） | `.logs/train/run4/try/conservative-mix-revert.md` |
