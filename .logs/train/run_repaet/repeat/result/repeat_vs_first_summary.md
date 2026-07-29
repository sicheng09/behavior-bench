# run_repaet / repeat · 复训复测指标总结

> 生成时间：2026-07-24 05:09:35   
> 协议：`pufferinter` · `MAP_IDS=all`（有效图约 589）· ego=`Drive+Recurrent`  
> 基线：H0 run4_r2（[`u9ymfcqr`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/u9ymfcqr)）  
> 对照：首轮 stratified（`.logs/train/run_repaet/result/`）vs 本轮 `repeat/`  

**加粗规则（主表 second 列）**：相对 H0，Goal/Reward 更高加粗；Collision / At-fault / Offroad 更低加粗。

## 0. Run 索引

| Experiment | 首轮 W&B | 复跑 W&B | 复跑目录 |
| --- | --- | --- | --- |
| DriveLite A strat | [`r0jht7bf`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/r0jht7bf) | [`bly4mcn2`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/bly4mcn2) | `repeat/drivelite_mixA_stratified/` |
| DriveLite B strat | [`qgytyxt4`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/qgytyxt4) | [`jy3p9j4h`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/jy3p9j4h) | `repeat/drivelite_mixB_stratified/` |
| Perception strat | [`yov2516v`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/yov2516v) | [`q2lm8afu`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/q2lm8afu) | `repeat/perception_stratified/` |
| MIXMOE strat | [`o4trtoxc`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/o4trtoxc) | [`0eyyzzwx`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/0eyyzzwx) | `repeat/mixmoe_stratified/` |

---

## 1. vs IDM · 主指标（H0 / 首轮 / 复跑）

### DriveLite A strat

首轮 [`r0jht7bf`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/r0jht7bf) · 复跑 [`bly4mcn2`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/bly4mcn2)

| Metric | H0 baseline | 首轮 (1st) | 复跑 (2nd) | Δ 2nd−1st | Δ 2nd vs H0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Goal ↑ | 81.66% | 82.68% | **83.36%** | +0.68pp | +2.1% |
| Collision ↓ | 5.77% | 2.72% | **2.89%** | +0.17pp | -49.9% |
| At-fault ↓ | 4.58% | 1.70% | **1.87%** | +0.17pp | -59.2% |
| Offroad ↓ | 0.85% | 1.02% | 1.19% | +0.17pp | +40.0% |
| Reward ↑ | -0.6755 | -0.5017 | **-0.5744** | -0.0727 | +0.1011 |

**可复现性速览**：Coll 2nd−1st = +0.17pp； Fault 2nd−1st = +0.17pp。

### DriveLite B strat

首轮 [`qgytyxt4`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/qgytyxt4) · 复跑 [`jy3p9j4h`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/jy3p9j4h)

| Metric | H0 baseline | 首轮 (1st) | 复跑 (2nd) | Δ 2nd−1st | Δ 2nd vs H0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Goal ↑ | 81.66% | 81.15% | 81.15% | +0.00pp | -0.6% |
| Collision ↓ | 5.77% | 5.09% | **4.41%** | -0.68pp | -23.6% |
| At-fault ↓ | 4.58% | 4.07% | **3.74%** | -0.33pp | -18.3% |
| Offroad ↓ | 0.85% | 1.70% | 1.53% | -0.17pp | +80.0% |
| Reward ↑ | -0.6755 | -0.6929 | **-0.6558** | +0.0371 | +0.0197 |

**可复现性速览**：Coll 2nd−1st = -0.68pp； Fault 2nd−1st = -0.33pp。

### Perception strat

首轮 [`yov2516v`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/yov2516v) · 复跑 [`q2lm8afu`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/q2lm8afu)

| Metric | H0 baseline | 首轮 (1st) | 复跑 (2nd) | Δ 2nd−1st | Δ 2nd vs H0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Goal ↑ | 81.66% | 81.15% | 80.81% | -0.34pp | -1.0% |
| Collision ↓ | 5.77% | 2.72% | **3.23%** | +0.51pp | -44.0% |
| At-fault ↓ | 4.58% | 2.38% | **2.55%** | +0.17pp | -44.3% |
| Offroad ↓ | 0.85% | 1.53% | 0.85% | -0.68pp | +0.0% |
| Reward ↑ | -0.6755 | -0.5420 | **-0.5651** | -0.0231 | +0.1104 |

**可复现性速览**：Coll 2nd−1st = +0.51pp； Fault 2nd−1st = +0.17pp。

### MIXMOE strat

首轮 [`o4trtoxc`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/o4trtoxc) · 复跑 [`0eyyzzwx`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/0eyyzzwx)

| Metric | H0 baseline | 首轮 (1st) | 复跑 (2nd) | Δ 2nd−1st | Δ 2nd vs H0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Goal ↑ | 81.66% | 80.98% | **83.19%** | +2.21pp | +1.9% |
| Collision ↓ | 5.77% | 5.94% | **4.92%** | -1.02pp | -14.7% |
| At-fault ↓ | 4.58% | 4.92% | **4.41%** | -0.51pp | -3.7% |
| Offroad ↓ | 0.85% | 1.19% | **0.51%** | -0.68pp | -40.0% |
| Reward ↑ | -0.6755 | -0.7215 | **-0.6107** | +0.1108 | +0.0648 |

**可复现性速览**：Coll 2nd−1st = -1.02pp； Fault 2nd−1st = -0.51pp。

### vs IDM · 四实验总览（Collision）

| Experiment | H0 Coll | 1st Coll | 2nd Coll | Δ2−1 | 2nd vs H0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DriveLite A strat | 5.77% | 2.72% | 2.89% | +0.17pp | -49.9% |
| DriveLite B strat | 5.77% | 5.09% | 4.41% | -0.68pp | -23.6% |
| Perception strat | 5.77% | 2.72% | 3.23% | +0.51pp | -44.0% |
| MIXMOE strat | 5.77% | 5.94% | 4.92% | -1.02pp | -14.7% |

---

## 2. vs Expert · 主指标（H0 / 首轮 / 复跑）

### DriveLite A strat

首轮 [`r0jht7bf`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/r0jht7bf) · 复跑 [`bly4mcn2`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/bly4mcn2)

| Metric | H0 baseline | 首轮 (1st) | 复跑 (2nd) | Δ 2nd−1st | Δ 2nd vs H0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Goal ↑ | 71.31% | 73.85% | **74.02%** | +0.17pp | +3.8% |
| Collision ↓ | 16.81% | 14.43% | **13.41%** | -1.02pp | -20.2% |
| At-fault ↓ | 15.28% | 13.41% | **12.56%** | -0.85pp | -17.8% |
| Offroad ↓ | 0.68% | 1.02% | 1.02% | +0.00pp | +50.0% |
| Reward ↑ | -0.8165 | -0.7010 | **-0.7730** | -0.0720 | +0.0435 |

**可复现性速览**：Coll 2nd−1st = -1.02pp； Fault 2nd−1st = -0.85pp。

### DriveLite B strat

首轮 [`qgytyxt4`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/qgytyxt4) · 复跑 [`jy3p9j4h`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/jy3p9j4h)

| Metric | H0 baseline | 首轮 (1st) | 复跑 (2nd) | Δ 2nd−1st | Δ 2nd vs H0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Goal ↑ | 71.31% | 72.50% | **74.02%** | +1.52pp | +3.8% |
| Collision ↓ | 16.81% | 14.09% | **13.75%** | -0.34pp | -18.2% |
| At-fault ↓ | 15.28% | 12.22% | **12.73%** | +0.51pp | -16.7% |
| Offroad ↓ | 0.68% | 1.87% | 1.02% | -0.85pp | +50.0% |
| Reward ↑ | -0.8165 | -0.7937 | **-0.7775** | +0.0162 | +0.0390 |

**可复现性速览**：Coll 2nd−1st = -0.34pp； Fault 2nd−1st = +0.51pp。

### Perception strat

首轮 [`yov2516v`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/yov2516v) · 复跑 [`q2lm8afu`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/q2lm8afu)

| Metric | H0 baseline | 首轮 (1st) | 复跑 (2nd) | Δ 2nd−1st | Δ 2nd vs H0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Goal ↑ | 71.31% | 71.65% | 71.14% | -0.51pp | -0.2% |
| Collision ↓ | 16.81% | 15.62% | **16.30%** | +0.68pp | -3.0% |
| At-fault ↓ | 15.28% | 13.58% | **14.09%** | +0.51pp | -7.8% |
| Offroad ↓ | 0.68% | 0.68% | 0.68% | +0.00pp | +0.0% |
| Reward ↑ | -0.8165 | -0.7071 | **-0.8092** | -0.1021 | +0.0073 |

**可复现性速览**：Coll 2nd−1st = +0.68pp； Fault 2nd−1st = +0.51pp。

### MIXMOE strat

首轮 [`o4trtoxc`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/o4trtoxc) · 复跑 [`0eyyzzwx`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/0eyyzzwx)

| Metric | H0 baseline | 首轮 (1st) | 复跑 (2nd) | Δ 2nd−1st | Δ 2nd vs H0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Goal ↑ | 71.31% | 71.99% | 70.97% | -1.02pp | -0.5% |
| Collision ↓ | 16.81% | 16.81% | **15.11%** | -1.70pp | -10.1% |
| At-fault ↓ | 15.28% | 14.77% | **13.58%** | -1.19pp | -11.1% |
| Offroad ↓ | 0.68% | 1.36% | 0.68% | -0.68pp | +0.0% |
| Reward ↑ | -0.8165 | -0.8985 | **-0.8090** | +0.0895 | +0.0075 |

**可复现性速览**：Coll 2nd−1st = -1.70pp； Fault 2nd−1st = -1.19pp。

### vs Expert · 四实验总览（Collision）

| Experiment | H0 Coll | 1st Coll | 2nd Coll | Δ2−1 | 2nd vs H0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DriveLite A strat | 16.81% | 14.43% | 13.41% | -1.02pp | -20.2% |
| DriveLite B strat | 16.81% | 14.09% | 13.75% | -0.34pp | -18.2% |
| Perception strat | 16.81% | 15.62% | 16.30% | +0.68pp | -3.0% |
| MIXMOE strat | 16.81% | 16.81% | 15.11% | -1.70pp | -10.1% |

---

## 3. vs Smart · 主指标（H0 / 首轮 / 复跑）

### DriveLite A strat

首轮 [`r0jht7bf`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/r0jht7bf) · 复跑 [`bly4mcn2`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/bly4mcn2)

| Metric | H0 baseline | 首轮 (1st) | 复跑 (2nd) | Δ 2nd−1st | Δ 2nd vs H0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Goal ↑ | 80.48% | 84.21% | **83.53%** | -0.68pp | +3.8% |
| Collision ↓ | 6.79% | 2.89% | **3.40%** | +0.51pp | -49.9% |
| At-fault ↓ | 4.92% | 2.21% | **2.72%** | +0.51pp | -44.7% |
| Offroad ↓ | 0.85% | 0.68% | **0.68%** | +0.00pp | -20.0% |
| Reward ↑ | -0.7411 | -0.5965 | **-0.6855** | -0.0890 | +0.0556 |

**可复现性速览**：Coll 2nd−1st = +0.51pp； Fault 2nd−1st = +0.51pp。

### DriveLite B strat

首轮 [`qgytyxt4`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/qgytyxt4) · 复跑 [`jy3p9j4h`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/jy3p9j4h)

| Metric | H0 baseline | 首轮 (1st) | 复跑 (2nd) | Δ 2nd−1st | Δ 2nd vs H0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Goal ↑ | 80.48% | 80.48% | **82.17%** | +1.69pp | +2.1% |
| Collision ↓ | 6.79% | 5.09% | **3.40%** | -1.69pp | -49.9% |
| At-fault ↓ | 4.92% | 4.58% | **2.38%** | -2.20pp | -51.6% |
| Offroad ↓ | 0.85% | 2.55% | 1.87% | -0.68pp | +120.0% |
| Reward ↑ | -0.7411 | -0.7598 | -0.7420 | +0.0178 | -0.0009 |

**可复现性速览**：Coll 2nd−1st = -1.69pp； Fault 2nd−1st = -2.20pp。

### Perception strat

首轮 [`yov2516v`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/yov2516v) · 复跑 [`q2lm8afu`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/q2lm8afu)

| Metric | H0 baseline | 首轮 (1st) | 复跑 (2nd) | Δ 2nd−1st | Δ 2nd vs H0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Goal ↑ | 80.48% | 81.32% | **81.32%** | +0.00pp | +1.0% |
| Collision ↓ | 6.79% | 4.07% | **4.41%** | +0.34pp | -35.1% |
| At-fault ↓ | 4.92% | 2.55% | **3.23%** | +0.68pp | -34.3% |
| Offroad ↓ | 0.85% | 1.19% | 1.53% | +0.34pp | +80.0% |
| Reward ↑ | -0.7411 | -0.6369 | **-0.7000** | -0.0631 | +0.0411 |

**可复现性速览**：Coll 2nd−1st = +0.34pp； Fault 2nd−1st = +0.68pp。

### MIXMOE strat

首轮 [`o4trtoxc`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/o4trtoxc) · 复跑 [`0eyyzzwx`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/0eyyzzwx)

| Metric | H0 baseline | 首轮 (1st) | 复跑 (2nd) | Δ 2nd−1st | Δ 2nd vs H0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Goal ↑ | 80.48% | 82.51% | **81.83%** | -0.68pp | +1.7% |
| Collision ↓ | 6.79% | 4.24% | **5.43%** | +1.19pp | -20.0% |
| At-fault ↓ | 4.92% | 3.57% | **4.41%** | +0.84pp | -10.4% |
| Offroad ↓ | 0.85% | 1.70% | 0.85% | -0.85pp | +0.0% |
| Reward ↑ | -0.7411 | -0.7569 | **-0.7239** | +0.0330 | +0.0172 |

**可复现性速览**：Coll 2nd−1st = +1.19pp； Fault 2nd−1st = +0.84pp。

### vs Smart · 四实验总览（Collision）

| Experiment | H0 Coll | 1st Coll | 2nd Coll | Δ2−1 | 2nd vs H0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DriveLite A strat | 6.79% | 2.89% | 3.40% | +0.51pp | -49.9% |
| DriveLite B strat | 6.79% | 5.09% | 3.40% | -1.69pp | -49.9% |
| Perception strat | 6.79% | 4.07% | 4.41% | +0.34pp | -35.1% |
| MIXMOE strat | 6.79% | 4.24% | 5.43% | +1.19pp | -20.0% |

---

## 5. 碰撞归因对比（maps：lateral / front / rear / stopped）

计数来自 `collision_snapshots.json` 的 `collision_type`：

- lateral ← `ACTIVE_LATERAL_COLLISION`
- front ← `ACTIVE_FRONT_COLLISION`
- rear ← `ACTIVE_REAR_COLLISION`
- stopped ← `STOPPED_*_COLLISION`

### vs IDM

#### DriveLite A strat

| Phase | Coll maps | Lateral | Front | Rear | Stopped |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 | 34 | 25 | 7 | 1 | 1 |
| 1st | 16 | 4 | 7 | 3 | 2 |
| 2nd | 17 | 10 | 4 | 1 | 2 |
| Δ2−1 | +1 | +6 | -3 | -2 | +0 |

#### DriveLite B strat

| Phase | Coll maps | Lateral | Front | Rear | Stopped |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 | 34 | 25 | 7 | 1 | 1 |
| 1st | 30 | 10 | 17 | 0 | 3 |
| 2nd | 26 | 10 | 12 | 1 | 3 |
| Δ2−1 | -4 | +0 | -5 | +1 | +0 |

#### Perception strat

| Phase | Coll maps | Lateral | Front | Rear | Stopped |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 | 34 | 25 | 7 | 1 | 1 |
| 1st | 16 | 8 | 5 | 0 | 3 |
| 2nd | 19 | 12 | 4 | 1 | 2 |
| Δ2−1 | +3 | +4 | -1 | +1 | -1 |

#### MIXMOE strat

| Phase | Coll maps | Lateral | Front | Rear | Stopped |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 | 34 | 25 | 7 | 1 | 1 |
| 1st | 35 | 22 | 11 | 0 | 2 |
| 2nd | 29 | 17 | 10 | 0 | 2 |
| Δ2−1 | -6 | -5 | -1 | +0 | +0 |

**vs IDM 宽表**

| Experiment | Phase | Maps | Lat | Front | Rear | Stop |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H0 | baseline | 34 | 25 | 7 | 1 | 1 |
| DriveLite A strat | 1st | 16 | 4 | 7 | 3 | 2 |
| DriveLite A strat | 2nd | 17 | 10 | 4 | 1 | 2 |
| DriveLite B strat | 1st | 30 | 10 | 17 | 0 | 3 |
| DriveLite B strat | 2nd | 26 | 10 | 12 | 1 | 3 |
| Perception strat | 1st | 16 | 8 | 5 | 0 | 3 |
| Perception strat | 2nd | 19 | 12 | 4 | 1 | 2 |
| MIXMOE strat | 1st | 35 | 22 | 11 | 0 | 2 |
| MIXMOE strat | 2nd | 29 | 17 | 10 | 0 | 2 |

---

### vs Expert

#### DriveLite A strat

| Phase | Coll maps | Lateral | Front | Rear | Stopped |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 | 99 | 33 | 27 | 9 | 30 |
| 1st | 85 | 19 | 23 | 4 | 39 |
| 2nd | 79 | 11 | 33 | 5 | 30 |
| Δ2−1 | -6 | -8 | +10 | +1 | -9 |

#### DriveLite B strat

| Phase | Coll maps | Lateral | Front | Rear | Stopped |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 | 99 | 33 | 27 | 9 | 30 |
| 1st | 83 | 19 | 28 | 6 | 30 |
| 2nd | 81 | 23 | 26 | 3 | 29 |
| Δ2−1 | -2 | +4 | -2 | -3 | -1 |

#### Perception strat

| Phase | Coll maps | Lateral | Front | Rear | Stopped |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 | 99 | 33 | 27 | 9 | 30 |
| 1st | 92 | 18 | 28 | 11 | 35 |
| 2nd | 96 | 27 | 23 | 9 | 37 |
| Δ2−1 | +4 | +9 | -5 | -2 | +2 |

#### MIXMOE strat

| Phase | Coll maps | Lateral | Front | Rear | Stopped |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 | 99 | 33 | 27 | 9 | 30 |
| 1st | 99 | 30 | 29 | 6 | 34 |
| 2nd | 89 | 26 | 28 | 6 | 29 |
| Δ2−1 | -10 | -4 | -1 | +0 | -5 |

**vs Expert 宽表**

| Experiment | Phase | Maps | Lat | Front | Rear | Stop |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H0 | baseline | 99 | 33 | 27 | 9 | 30 |
| DriveLite A strat | 1st | 85 | 19 | 23 | 4 | 39 |
| DriveLite A strat | 2nd | 79 | 11 | 33 | 5 | 30 |
| DriveLite B strat | 1st | 83 | 19 | 28 | 6 | 30 |
| DriveLite B strat | 2nd | 81 | 23 | 26 | 3 | 29 |
| Perception strat | 1st | 92 | 18 | 28 | 11 | 35 |
| Perception strat | 2nd | 96 | 27 | 23 | 9 | 37 |
| MIXMOE strat | 1st | 99 | 30 | 29 | 6 | 34 |
| MIXMOE strat | 2nd | 89 | 26 | 28 | 6 | 29 |

---

### vs Smart

#### DriveLite A strat

| Phase | Coll maps | Lateral | Front | Rear | Stopped |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 | 40 | 23 | 9 | 7 | 1 |
| 1st | 17 | 11 | 2 | 2 | 2 |
| 2nd | 20 | 6 | 9 | 4 | 1 |
| Δ2−1 | +3 | -5 | +7 | +2 | -1 |

#### DriveLite B strat

| Phase | Coll maps | Lateral | Front | Rear | Stopped |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 | 40 | 23 | 9 | 7 | 1 |
| 1st | 30 | 10 | 13 | 3 | 4 |
| 2nd | 20 | 7 | 9 | 4 | 0 |
| Δ2−1 | -10 | -3 | -4 | +1 | -4 |

#### Perception strat

| Phase | Coll maps | Lateral | Front | Rear | Stopped |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 | 40 | 23 | 9 | 7 | 1 |
| 1st | 24 | 11 | 5 | 7 | 1 |
| 2nd | 26 | 17 | 2 | 6 | 1 |
| Δ2−1 | +2 | +6 | -3 | -1 | +0 |

#### MIXMOE strat

| Phase | Coll maps | Lateral | Front | Rear | Stopped |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 | 40 | 23 | 9 | 7 | 1 |
| 1st | 25 | 17 | 6 | 1 | 1 |
| 2nd | 32 | 17 | 10 | 4 | 1 |
| Δ2−1 | +7 | +0 | +4 | +3 | +0 |

**vs Smart 宽表**

| Experiment | Phase | Maps | Lat | Front | Rear | Stop |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H0 | baseline | 40 | 23 | 9 | 7 | 1 |
| DriveLite A strat | 1st | 17 | 11 | 2 | 2 | 2 |
| DriveLite A strat | 2nd | 20 | 6 | 9 | 4 | 1 |
| DriveLite B strat | 1st | 30 | 10 | 13 | 3 | 4 |
| DriveLite B strat | 2nd | 20 | 7 | 9 | 4 | 0 |
| Perception strat | 1st | 24 | 11 | 5 | 7 | 1 |
| Perception strat | 2nd | 26 | 17 | 2 | 6 | 1 |
| MIXMOE strat | 1st | 25 | 17 | 6 | 1 | 1 |
| MIXMOE strat | 2nd | 32 | 17 | 10 | 4 | 1 |

---

## 6. 结论要点（自动草稿）

- **vs IDM**
  - DriveLite A strat: Coll 2.72% → 2.89%（Δ +0.17pp；vs H0 -49.9%）
  - DriveLite B strat: Coll 5.09% → 4.41%（Δ -0.68pp；vs H0 -23.6%）
  - Perception strat: Coll 2.72% → 3.23%（Δ +0.51pp；vs H0 -44.0%）
  - MIXMOE strat: Coll 5.94% → 4.92%（Δ -1.02pp；vs H0 -14.7%）

- **vs Expert**
  - DriveLite A strat: Coll 14.43% → 13.41%（Δ -1.02pp；vs H0 -20.2%）
  - DriveLite B strat: Coll 14.09% → 13.75%（Δ -0.34pp；vs H0 -18.2%）
  - Perception strat: Coll 15.62% → 16.30%（Δ +0.68pp；vs H0 -3.0%）
  - MIXMOE strat: Coll 16.81% → 15.11%（Δ -1.70pp；vs H0 -10.1%）

- **vs Smart**
  - DriveLite A strat: Coll 2.89% → 3.40%（Δ +0.51pp；vs H0 -49.9%）
  - DriveLite B strat: Coll 5.09% → 3.40%（Δ -1.69pp；vs H0 -49.9%）
  - Perception strat: Coll 4.07% → 4.41%（Δ +0.34pp；vs H0 -35.1%）
  - MIXMOE strat: Coll 4.24% → 5.43%（Δ +1.19pp；vs H0 -20.0%）

---

机器可读表：同目录 `metrics_main.csv` / `metrics_attribution.csv`。

