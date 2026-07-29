# H0 vs PDM relaunch (serial, anti-OOM)

- Date: 2026-07-22
- Prior failure: **kernel OOM killer** (not user Ctrl-C, not app `sys.exit`)
  - dmesg killed pids `3462511` / `3462520` / `3462654` matching the three parallel `eval.py`
  - anon-rss ≈ 309 GiB / 492 GiB / **977 GiB** before kill; host ~1 TiB, **no swap**
- Cause: `MultiAgentPDMPlanner` = each traffic agent owns a `PDMPlanner` with **15** `DriveBatch` env copies → memory scales with traffic count (logs show up to ~127 agents)
- Fix: `run_h0_vs_pdm_serial.sh` — **one run at a time**, refuse start if `MemAvailable < 80 GiB`, RSS monitor every 60s
- Protocol unchanged: `pufferinter` / `map-ids=all` / evaluation.ini PDM defaults
- GPU: 2 (CPU-heavy; GPU barely used)
- tmux: `h0-vs-pdm-serial-gpu2`
