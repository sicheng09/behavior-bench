#!/usr/bin/env python3
"""Run puffer CLI using the adversarial worktree checkout in-process.

The conda env editable install points at the main repo. This wrapper remaps the
editable finder to the worktree for this process only, so
`puffer_drive_adversarial` is available without changing the global install.

---------------------------------------------------------------------------
Rollback (undo local adversarial / related commits on main-behavior-bench)
---------------------------------------------------------------------------
These commits were local-only (not pushed). Choose one:

1) Drop ALL local commits ahead of origin (back to remote main-behavior-bench):

    cd "$HOME/wsc/behavior-bench"
    git checkout main-behavior-bench
    git reset --hard origin/main-behavior-bench

2) Keep earlier local baseline commits, drop only adversarial series
   (reset to chore: ignore local worktrees):

    cd "$HOME/wsc/behavior-bench"
    git checkout main-behavior-bench
    git reset --hard 89574944

Optional: stop the run3 training session first:

    tmux send-keys -t adv-mix-drive-lstm-batch2x-4b-gpu0 C-c
    # or: tmux kill-session -t adv-mix-drive-lstm-batch2x-4b-gpu0

Optional: remove the feature worktree after rollback:

    cd "$HOME/wsc/behavior-bench"
    git worktree remove .worktrees/adversarial-mixed-training
    git branch -D feature/adversarial-mixed-training

WARNING: ``git reset --hard`` discards uncommitted changes on that branch.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path

MAIN = Path("/home/fanyuqi/wsc/behavior-bench").resolve()
WT = Path(
    "/home/fanyuqi/wsc/behavior-bench/.worktrees/adversarial-mixed-training"
).resolve()


def _remap(path: str) -> str:
    p = Path(path)
    try:
        rel = p.relative_to(MAIN)
    except ValueError:
        return path
    candidate = WT / rel
    return str(candidate if candidate.exists() else p)


def _patch_editable_finder() -> None:
    import __editable___pufferlib_3_0_0_finder as ed

    ed.MAPPING = {key: _remap(value) for key, value in ed.MAPPING.items()}
    ed.NAMESPACES = {
        key: [_remap(value) for value in values]
        for key, values in ed.NAMESPACES.items()
    }
    ed.MAPPING["pufferlib"] = str(WT / "pufferlib")
    ed.MAPPING["config"] = str(WT / "pufferlib" / "config")
    ed.MAPPING["pufferlib.config"] = str(WT / "pufferlib" / "config")
    ed.MAPPING["pufferlib.config.ocean"] = str(
        WT / "pufferlib" / "config" / "ocean"
    )
    ed.NAMESPACES["config"] = [str(WT / "pufferlib" / "config")]
    ed.NAMESPACES["pufferlib.config"] = [str(WT / "pufferlib" / "config")]
    ed.NAMESPACES["pufferlib.config.ocean"] = [
        str(WT / "pufferlib" / "config" / "ocean")
    ]
    ed.NAMESPACES["config.ocean"] = [str(WT / "pufferlib" / "config" / "ocean")]


def main() -> None:
    _patch_editable_finder()
    from pufferlib.pufferl import main as puffer_main

    puffer_main()


if __name__ == "__main__":
    main()
