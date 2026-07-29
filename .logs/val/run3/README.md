# Run3 validation (adversarial mix)

协议对齐 `.logs/val/validation.md` / run1·run2：

- planner：PPO ego（`model_policy_0_*`）
- traffic：**IDM**
- split：**pufferinter**
- maps：`all`
- 输出：`results/<train_dir>/<policy_dir>/<timestamp_uuid>/`

## 500M baseline 训练结束后自动评测

等待 tmux `adv-mix-drive-lstm-ddp2-500m-gpu34` 结束，加载
`experiments/puffer_drive_adversarial_wpj0xhn8/model_policy_0_*.pt`，在 GPU 3 上跑 ego vs IDM：

```bash
bash /home/fanyuqi/wsc/behavior-bench/.logs/val/run3/wait_and_eval_ddp2_500m.sh
```

## 500M weak-goal 消融结束后自动评测

等待 tmux `adv-mix-drive-lstm-ddp2-500m-weak-goal-gpu56` 结束，加载
`experiments/puffer_drive_adversarial_jbx6brjj/model_policy_0_*.pt`，在 GPU 5 上跑 ego vs IDM：

```bash
bash /home/fanyuqi/wsc/behavior-bench/.logs/val/run3/wait_and_eval_ddp2_500m_weak_goal.sh
```

结果：`results/adv_mix_ego_opponent_drive_lstm_ddp2_500m_weak_goal_run3/p0_ego_Drive_Recurrent/`

单次手动评测：

```bash
MAP_IDS=all bash /home/fanyuqi/wsc/behavior-bench/.logs/val/run3/run_one_policy_eval.sh \
  adv_mix_ego_opponent_drive_lstm_ddp2_500m_run3 \
  p0_ego_Drive_Recurrent \
  /path/to/model_policy_0_puffer_drive_adversarial_XXXXXX.pt \
  Drive Recurrent 256 256 3
```
