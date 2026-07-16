#!/usr/bin/env python3
"""Compare step SPS: Drive vs AdversarialMixDrive (same env settings)."""

import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ["DRIVE_BINARIES_DATA_ROOT"] = str(
    ROOT / "resources" / "drive" / "binaries"
)
os.environ["PUFFER_DISABLE_VIDEO"] = "1"

from pufferlib.adversarial.env import AdversarialMixDrive
from pufferlib.ocean.drive.drive import Drive

NUM_AGENTS = 64
NUM_MAPS = 1
WARM_STEPS = 1000
MEAS_STEPS = 10000
ATN_CACHE = 16

COMMON = dict(
    num_agents=NUM_AGENTS,
    num_maps=NUM_MAPS,
    action_type="discrete",
    init_mode="create_all_valid",
    control_mode="control_agents",
    episode_length=91,
    resample_frequency=100000,
    report_interval=10**9,
    split="training",
)


def make_actions(env):
    spaces = env.single_action_space
    return np.stack(
        [
            np.random.randint(0, space.n, (ATN_CACHE, env.num_agents))
            for space in spaces
        ],
        axis=-1,
    )


def bench(name, factory):
    env = factory()
    env.reset()
    actions = make_actions(env)
    n = env.num_agents
    for i in range(WARM_STEPS):
        env.step(actions[i % ATN_CACHE])
    t0 = time.perf_counter()
    for i in range(MEAS_STEPS):
        env.step(actions[i % ATN_CACHE])
    dt = time.perf_counter() - t0
    sps = n * MEAS_STEPS / dt
    env.close()
    print(f"{name}: agents={n} meas_steps={MEAS_STEPS} wall={dt:.3f}s sps={sps:.1f}")
    return sps


def main():
    policy_ids = ([0, 1] * ((NUM_AGENTS + 1) // 2))[:NUM_AGENTS]
    cfg = str(ROOT / "pufferlib/config/adversarial/opponent_mix.yaml")

    def make_baseline():
        return Drive(
            **COMMON,
            policy_log_ids=policy_ids,
            policy_log_count=2,
        )

    def make_adv():
        return AdversarialMixDrive(
            adversarial_config_path=cfg,
            **COMMON,
            policy_log_ids=policy_ids,
            policy_log_count=2,
        )

    b1 = bench("baseline_1", make_baseline)
    a1 = bench("adversarial_1", make_adv)
    b2 = bench("baseline_2", make_baseline)
    a2 = bench("adversarial_2", make_adv)

    baseline = (b1 + b2) / 2
    adversarial = (a1 + a2) / 2
    overhead = 1.0 - adversarial / baseline
    print("---")
    print(f"baseline_sps={baseline:.1f}")
    print(f"adversarial_sps={adversarial:.1f}")
    print(f"overhead={overhead:.4f} ({overhead * 100:.2f}%)")
    if overhead > 0.10:
        raise SystemExit("FAIL: overhead > 10%")
    print("PASS")


if __name__ == "__main__":
    main()
