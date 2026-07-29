# trainMOE / test_Smart

Drive+Recurrent（`policy_0`）vs **SMART** traffic，协议同 `test_IDM` / `test_Expert`：

- split=`pufferinter` · `MAP_IDS=all`
- traffic=`smart` · 权重默认 `weights/SMART_epoch_030.pt`
- 结果：`<train_dir>/Drive_Recurrent/`

```bash
bash .logs/train/run_repaet/trainMOE/test_Smart/launch_eval.sh
```
