# try_Multi · freeze plvd8yph policy_0 as partner

- date: 2026-07-21T20:28:04+08:00
- frozen: weights/plvd8yph_policy_0_000954.pt (C-B mb65k ego, Coll 2.55%)
- contrast: previous run froze policy_1 → vs IDM Coll 5.09%
- tmux train: try-multi-ego-scratch-frozenB0-gpu1 (GPU 1)
- tmux wait+eval: wait-eval-try-multi-frozenB0 (eval GPU 2/4)
- sampling: shared, mb65k, trainable True,False
