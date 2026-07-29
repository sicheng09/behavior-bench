# ConservativeMix Phase A · 对齐同质 500M 的启动说明

## 预算对齐（硬约束）

同质 run4（`homogeneous_drive_lstm_500m_batch1x`）：

```text
policy              = Drive + Recurrent
batch_size          = 524288
total_timesteps     = 500000000
updates             ≈ 500M / 524288 ≈ 953
minibatch_size      = 32768
bptt_horizon        = 32
update_epochs       = 1
num_maps            = 10000
mix_traffic         = False
```

本 run：`ego:partner = 0.5:0.5` 两策略混智。按 `train-muliti-ppo.md` / DriveLite 惯例，**总 batch 与总 steps 按策略数放大**，使 **ego（目标策略）** 每次 update 与总 agent-steps 对齐同质：

```text
N_policies                  = 2
train.batch_size            = 2 × 524288 = 1048576   # ego ≈ 524288 / update
train.total_timesteps       = 2 × 500000000 = 1000000000
updates                     ≈ 1B / 1048576 ≈ 953     # 与同质同频
ego per-update samples      ≈ 524288
ego total agent-steps       ≈ 500M
partner                     = DriveSteerConstrained（横向约束，非评测 IDM）
```

其余超参与同质 / `drive.ini` 一致：`minibatch_size=32768`、`bptt_horizon=32`、`update_epochs=1`、`learning_rate=0.003`、`num_maps=10000`、训练内 WOSAC/human-replay 关闭。

## 启动

```bash
bash /home/fanyuqi/wsc/behavior-bench/.logs/train/run4/try/run/start_cons_mix_A_batch2x_1b.sh
```

日志目录：`.logs/train/run4/try/run/`  
退回说明：`pufferlib/conservative/REVERT.md`
