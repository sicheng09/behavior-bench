# H0 PPO ego vs PDM traffic

- Protocol: same as `test_IDM` (`pufferinter`, `map-ids=all`)
- Ego: Drive + Recurrent 256
- Traffic: `pdm` (evaluation.ini defaults)
- Output: `problem/test_PDM/<run>/Drive_Recurrent/`

| Run | W&B | Weights | GPU | tmux |
| --- | --- | --- | --- | --- |
| `homogeneous_drive_lstm_500m_batch1x_run1` | fdfw3v5e | `experiments/puffer_drive_fdfw3v5e/model_puffer_drive_000954.pt` | 2 | `h0-vs-pdm-run1-gpu2` |
| `…_run4_r1` | 1ytcyfuk | `experiments/puffer_drive_1ytcyfuk/model_puffer_drive_000954.pt` | 3 | `h0-vs-pdm-r1-gpu3` |
| `…_run4_r2` | u9ymfcqr | `experiments/puffer_drive_u9ymfcqr/model_puffer_drive_000954.pt` | 7 | `h0-vs-pdm-r2-gpu7` |
