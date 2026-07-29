# DriveConditioned 2B · creward profiles vs IDM

同一 checkpoint（`homogeneous_drive_conditioned_2b_batch1x_run4` / W&B `dfkvxczz`），三种评测时 creward 风格：

| 目录 | planner.type | 风格 |
| --- | --- | --- |
| `conditioned_aggr_vs_idm/` | `conditioned_aggr` | 激进 |
| `conditioned_normal_vs_idm/` | `conditioned_normal` | 普通 |
| `conditioned_caut_vs_idm/` | `conditioned_caut` | 保守 |

协议：`pufferinter` · `map-ids=all` · `traffic.type=idm`（与 `tset/500M` / H0 vs IDM 一致）。  
creward 数值取自 `pufferlib/config/evaluation.ini` 默认段。

自动流程（等 2B 训练结束）：

```bash
bash .logs/train/run4/DriveConditioned/tset/2B/wait_and_eval_creward_vs_idm.sh
```
