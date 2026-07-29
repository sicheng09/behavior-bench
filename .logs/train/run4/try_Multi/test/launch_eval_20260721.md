# try_Multi eval relaunch (clean logs)

- previous partial evals deleted
- weights: experiments/puffer_drive_conservative_mix_y62wqvky/model_policy_0_..._000954.pt
- protocol: pufferinter / map-ids=all; vs IDM + vs PPO
- log sanitization: TERM=dumb + strip_ansi on wrapper tee
- tmux IDM: try-multi-eval-idm-gpu2 (GPU 2)
- tmux PPO: try-multi-eval-ppo-gpu3 (GPU 3)
2026-07-21T19:17:08+08:00
