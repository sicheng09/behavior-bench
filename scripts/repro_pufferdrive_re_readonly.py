#!/usr/bin/env python3
"""Read-only reproduction harness for /home/fanyuqi/pufferdrive_re.

This script imports the target repository but writes every artifact under the
current behavior-bench workspace. It supports the legacy homogeneous/mixed
training paths plus a homogeneous-only initialization ablation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


TARGET_REPO = Path("/home/fanyuqi/pufferdrive_re").resolve()
WORKSPACE = Path("/home/fanyuqi/wsc/behavior-bench").resolve()
DEFAULT_OUTPUT_ROOT = WORKSPACE / ".logs/repro/pufferdrive_re_500m"
DEFAULT_TRAINING_DIR = Path("/data2/puffer/data/GPUDrive_medium/binaries/training")
DEFAULT_VALIDATION_DIR = Path("/data2/puffer/data/GPUDrive_medium/binaries/validation")


def _assert_target_imports() -> None:
    import pufferlib

    imported = Path(pufferlib.__file__).resolve()
    if TARGET_REPO not in imported.parents:
        raise RuntimeError(
            f"Expected pufferlib from {TARGET_REPO}, imported {imported}. "
            "Launch with PYTHONPATH=/home/fanyuqi/pufferdrive_re."
        )


_assert_target_imports()

from pufferlib.pufferl import (  # noqa: E402
    PuffeRL,
    load_config,
    load_env,
    load_policy,
    repeat_factors_for_level_distribution,
)
import pufferlib.models as puffer_models  # noqa: E402
import pufferlib.ocean.torch as ocean_torch  # noqa: E402
import pufferlib.pytorch as puffer_pytorch  # noqa: E402

LEGACY_RECURRENT = ocean_torch.Recurrent


class SafeLSTMWrapper(puffer_models.LSTMWrapper):
    """LSTMWrapper that preserves the already initialized base policy."""

    def __init__(self, env, policy, input_size=128, hidden_size=128):
        nn.Module.__init__(self)
        self.obs_shape = env.single_observation_space.shape
        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_continuous = self.policy.is_continuous

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.cell.weight_ih = self.lstm.weight_ih_l0
        self.cell.weight_hh = self.lstm.weight_hh_l0
        self.cell.bias_ih = self.lstm.bias_ih_l0
        self.cell.bias_hh = self.lstm.bias_hh_l0


def apply_variant_patch(variant: str) -> None:
    ocean_torch.Recurrent = SafeLSTMWrapper if variant == "h-fixed-init" else LEGACY_RECURRENT


def set_process_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def env_name_for_variant(variant: str) -> str:
    if variant in ("h-legacy", "h-fixed-init"):
        return "puffer_drive_light_clean_high"
    if variant == "m-legacy":
        return "puffer_drive_mixed_light_clean_enhanced"
    raise ValueError(f"Unknown variant: {variant}")


def load_clean_config(env_name: str) -> dict[str, Any]:
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        return load_config(env_name)
    finally:
        sys.argv = original_argv


def distributed_context() -> tuple[int, int, bool]:
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, False
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", world_size=world_size)
    return local_rank, world_size, True


def per_rank_total_steps(total_steps: int, world_size: int) -> int:
    return (int(total_steps) + int(world_size) - 1) // int(world_size)


def tensor_list(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def unwrap_policy(policy):
    if isinstance(policy, torch.nn.parallel.DistributedDataParallel):
        return policy.module
    return policy


def actor_modules(policy) -> list[tuple[str, nn.Linear]]:
    policy = unwrap_policy(policy)
    modules: list[tuple[str, nn.Linear]] = []
    if hasattr(policy, "policy"):
        base = policy.policy
    else:
        base = policy

    if hasattr(base, "level_policies"):
        for index, level_policy in enumerate(base.level_policies):
            modules.append((f"level_{index}.actor", level_policy.actor))
    elif hasattr(base, "actor"):
        modules.append(("actor", base.actor))
    return modules


def parameter_diagnostics(policy) -> dict[str, Any]:
    result: dict[str, Any] = {"actors": {}}
    for name, module in actor_modules(policy):
        weight = module.weight.detach().float()
        result["actors"][name] = {
            "weight_norm": float(weight.norm().cpu()),
            "weight_std": float(weight.std().cpu()),
            "weight_abs_max": float(weight.abs().max().cpu()),
            "bias_norm": float(module.bias.detach().float().norm().cpu()),
        }

    base = unwrap_policy(policy)
    result["model_parameter_count"] = sum(p.numel() for p in base.parameters())
    result["wrapper_class"] = type(base).__name__
    result["base_policy_class"] = type(getattr(base, "policy", base)).__name__
    return result


def rollout_diagnostics(trainer: PuffeRL, policy, sample_size: int = 512) -> dict[str, Any]:
    stats = trainer.evaluate()
    observations = trainer.observations[:sample_size, 0].to(trainer.config["device"])
    base = unwrap_policy(policy)
    state: dict[str, Any] = {}
    if trainer.config["use_rnn"]:
        hidden_size = int(base.hidden_size)
        state["lstm_h"] = torch.zeros(observations.shape[0], hidden_size, device=observations.device)
        state["lstm_c"] = torch.zeros(observations.shape[0], hidden_size, device=observations.device)

    with torch.no_grad():
        logits, _ = base.forward_eval(observations, state)
        if isinstance(logits, torch.distributions.Distribution):
            raise RuntimeError("Expected discrete logits")
        flat_logits = torch.cat(logits, dim=-1)
        _, _, entropy = puffer_pytorch.sample_logits(logits)

    metric_summary = {}
    for key, values in stats.items():
        try:
            metric_summary[key] = float(np.nanmean(values))
        except Exception:
            continue
    return {
        "sample_size": int(observations.shape[0]),
        "logits_mean": float(flat_logits.float().mean().cpu()),
        "logits_std": float(flat_logits.float().std().cpu()),
        "entropy_mean": float(entropy.float().mean().cpu()),
        "rollout_metrics": metric_summary,
    }


def configure_args(
    variant: str,
    map_dir: Path,
    num_maps: int,
    data_dir: Path,
    total_steps: int,
    seed: int,
    device: str | int,
    world_size: int,
    checkpoint_interval: int,
) -> tuple[str, dict[str, Any]]:
    env_name = env_name_for_variant(variant)
    args = load_clean_config(env_name)
    args["env"]["map_dir"] = str(map_dir)
    args["env"]["num_maps"] = int(num_maps)
    args["eval"]["map_dir"] = str(map_dir)
    args["eval"]["render_self_play_eval"] = False
    args["eval"]["render_human_replay_eval"] = False
    args["train"]["seed"] = int(seed)
    args["train"]["data_dir"] = str(data_dir)
    args["train"]["checkpoint_interval"] = int(checkpoint_interval)
    args["train"]["total_timesteps"] = per_rank_total_steps(total_steps, world_size)
    args["train"]["device"] = device

    if args["policy_name"] == "DriveMixedPolicy":
        distribution = list(args["policy"]["level_distribution"])
        args["train"]["repeat_policy_samples"] = True
        args["train"]["repeat_policy_levels"] = list(range(len(distribution)))
        args["train"]["repeat_policy_factors"] = repeat_factors_for_level_distribution(distribution)
    return env_name, args


def build_policy_and_trainer(env_name: str, args: dict[str, Any]):
    vecenv = load_env(env_name, args)
    policy = load_policy(args, vecenv, env_name)
    train_config = dict(**args["train"], env=env_name, eval=args["eval"])
    trainer = PuffeRL(train_config, vecenv, policy, logger=None, full_args=args)
    return vecenv, policy, trainer, train_config


def command_diagnose(parsed) -> None:
    output_root = Path(parsed.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in parsed.variants:
        set_process_seeds(parsed.seed)
        apply_variant_patch(variant)
        data_dir = output_root / "diagnostics" / variant
        env_name, args = configure_args(
            variant=variant,
            map_dir=Path(parsed.map_dir),
            num_maps=parsed.num_maps,
            data_dir=data_dir,
            total_steps=parsed.diagnostic_steps,
            seed=parsed.seed,
            device=parsed.device,
            world_size=1,
            checkpoint_interval=999999,
        )
        vecenv, policy, trainer, _ = build_policy_and_trainer(env_name, args)
        try:
            row = {
                "variant": variant,
                "seed": parsed.seed,
                "target_repo": str(TARGET_REPO),
                "parameter_diagnostics": parameter_diagnostics(policy),
                "rollout_diagnostics": rollout_diagnostics(trainer, policy),
            }
            rows.append(row)
        finally:
            if hasattr(trainer, "utilization"):
                trainer.utilization.stop()
            vecenv.close()

    output_path = output_root / "initialization_diagnostics.json"
    output_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(rows, sort_keys=True))


def command_train(parsed) -> None:
    local_rank, world_size, is_distributed = distributed_context()
    is_rank0 = local_rank == 0
    started = time.time()
    set_process_seeds(parsed.seed)
    apply_variant_patch(parsed.variant)

    output_root = Path(parsed.output_root).resolve()
    run_dir = output_root / "runs" / parsed.variant
    data_dir = run_dir / "checkpoints"
    summary_path = run_dir / "summary.json"
    diagnostics_path = run_dir / "initialization.json"
    train_log_path = run_dir / "train_metrics.jsonl"

    env_name, args = configure_args(
        variant=parsed.variant,
        map_dir=Path(parsed.map_dir),
        num_maps=parsed.num_maps,
        data_dir=data_dir,
        total_steps=parsed.total_env_steps,
        seed=parsed.seed,
        device=local_rank if is_distributed else parsed.device,
        world_size=world_size,
        checkpoint_interval=parsed.checkpoint_interval,
    )
    vecenv = load_env(env_name, args)
    policy = load_policy(args, vecenv, env_name)
    init_diagnostics = parameter_diagnostics(policy)

    if is_distributed:
        policy = policy.to(local_rank)
        ddp = torch.nn.parallel.DistributedDataParallel(
            policy,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=hasattr(policy, "parameters_for_level"),
        )
        if hasattr(policy, "hidden_size"):
            ddp.hidden_size = policy.hidden_size
        ddp.forward_eval = policy.forward_eval
        policy = ddp.to(local_rank)

    train_config = dict(**args["train"], env=env_name, eval=args["eval"])
    trainer = PuffeRL(train_config, vecenv, policy, logger=None, full_args=args)
    if is_rank0:
        run_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(
            json.dumps(
                {
                    "variant": parsed.variant,
                    "seed": parsed.seed,
                    "explicit_process_seeding": True,
                    "diagnostics": init_diagnostics,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    try:
        while not trainer.should_stop_training():
            if str(train_config["device"]) == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            trainer.evaluate()
            if str(train_config["device"]) == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            trainer.train()
            if is_rank0:
                epoch_metrics = {
                    "epoch": int(trainer.epoch),
                    "local_agent_steps": int(trainer.global_step),
                    "aggregate_agent_steps": int(trainer.global_step * world_size),
                    "learning_rate": float(trainer.optimizer.param_groups[0]["lr"]),
                    **{
                        f"environment/{key}": float(value)
                        for key, value in trainer.last_stats.items()
                        if np.isscalar(value)
                    },
                    **{
                        f"losses/{key}": float(value)
                        for key, value in trainer.losses.items()
                        if np.isscalar(value)
                    },
                }
                train_log_path.parent.mkdir(parents=True, exist_ok=True)
                with train_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(epoch_metrics, sort_keys=True) + "\n")

        if is_distributed:
            torch.distributed.barrier()
        final_checkpoint = trainer.save_checkpoint()
        if is_rank0:
            summary = {
                "variant": parsed.variant,
                "env_name": env_name,
                "elapsed_seconds": time.time() - started,
                "final_checkpoint": final_checkpoint,
                "world_size": world_size,
                "trainer_global_step": trainer.global_step,
                "aggregate_global_step": trainer.global_step * world_size,
                "target_aggregate_env_steps": parsed.total_env_steps,
                "local_total_env_steps": args["train"]["total_timesteps"],
                "map_dir": str(parsed.map_dir),
                "num_maps": parsed.num_maps,
                "seed": parsed.seed,
                "explicit_process_seeding": True,
                "policy": args["policy"],
                "train": {
                    key: args["train"].get(key)
                    for key in (
                        "batch_size",
                        "minibatch_size",
                        "rollout_horizon",
                        "learning_rate",
                        "update_epochs",
                        "repeat_policy_samples",
                        "repeat_policy_levels",
                        "repeat_policy_factors",
                    )
                    if key in args["train"]
                },
                "raw_level_agent_steps": tensor_list(getattr(trainer, "raw_level_agent_steps", None)),
                "repeat_level_train_steps": tensor_list(getattr(trainer, "repeat_level_train_steps", None)),
            }
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        if hasattr(trainer, "utilization"):
            trainer.utilization.stop()
        vecenv.close()
        if is_distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


METRIC_KEYS = [
    "n",
    "score",
    "completion_rate",
    "episode_return",
    "episode_length",
    "dnf_rate",
    "offroad_rate",
    "collision_rate",
    "lane_alignment_rate",
    "offroad_per_agent",
    "collisions_per_agent",
    "goals_reached_this_episode",
    "speed_at_goal",
    "policy_eval_reward_mean",
]


def infer_num_maps(path: Path) -> int:
    return len(list(path.glob("map_*.bin")))


def mean_stats(stats) -> dict[str, float]:
    result = {}
    for key, values in stats.items():
        try:
            result[key] = float(np.nanmean(values))
        except Exception:
            continue
    return result


def evaluate_checkpoint(
    variant: str,
    checkpoint: Path,
    map_dir: Path,
    rollouts: int,
    device: str,
    output_root: Path,
    forced_level: int | None,
) -> dict[str, Any]:
    apply_variant_patch(variant)
    env_name = env_name_for_variant(variant)
    args = load_clean_config(env_name)
    args["load_model_path"] = str(checkpoint)
    args["train"].update(
        {
            "device": device,
            "compile": False,
            "optimizer": "adam",
            "total_timesteps": 1,
            "checkpoint_interval": 999999,
            "data_dir": str(output_root / "_eval_tmp"),
        }
    )
    args["vec"].update({"num_workers": 16, "num_envs": 16, "batch_size": 4, "seed": 123})
    args["env"].update(
        {
            "map_dir": str(map_dir),
            "num_maps": infer_num_maps(map_dir),
            "init_mode": "create_all_valid",
            "control_mode": "control_agents",
            "num_agents": 1024,
        }
    )
    args["eval"].update(
        {
            "map_dir": str(map_dir),
            "eval_interval": 999999,
            "self_play_eval": False,
            "human_replay_eval": False,
            "wosac_realism_eval": False,
            "render_self_play_eval": False,
            "render_human_replay_eval": False,
        }
    )
    args["wandb"] = False
    args["neptune"] = False
    if forced_level is not None:
        distribution = [0.0] * int(args["policy"].get("num_groups", 3))
        distribution[int(forced_level)] = 1.0
        args["policy"]["level_distribution"] = distribution

    vecenv = load_env(env_name, args)
    policy = load_policy(args, vecenv, env_name)
    trainer = PuffeRL(dict(**args["train"], env=env_name, eval=args["eval"]), vecenv, policy, logger=None, full_args=args)
    try:
        for _ in range(rollouts):
            trainer.evaluate()
        return {
            "variant": variant,
            "checkpoint": str(checkpoint),
            "map_dir": str(map_dir),
            "rollouts": rollouts,
            "forced_level": forced_level,
            "metrics": mean_stats(trainer.stats),
        }
    finally:
        if hasattr(trainer, "utilization"):
            trainer.utilization.stop()
        vecenv.close()


def command_eval(parsed) -> None:
    output_root = Path(parsed.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    map_dirs = [
        ("training", Path(parsed.training_dir)),
        ("validation", Path(parsed.validation_dir)),
    ]
    rows = []
    forced_levels = [None]
    if parsed.variant == "m-legacy":
        forced_levels.extend([0, 1, 2])

    for split, map_dir in map_dirs:
        for forced_level in forced_levels:
            row = evaluate_checkpoint(
                variant=parsed.variant,
                checkpoint=Path(parsed.checkpoint).resolve(),
                map_dir=map_dir,
                rollouts=parsed.rollouts,
                device=parsed.device,
                output_root=output_root,
                forced_level=forced_level,
            )
            row["split"] = split
            rows.append(row)

    stem = f"{parsed.variant}_train_validation_{parsed.rollouts}rollouts"
    json_path = output_root / f"{stem}.json"
    csv_path = output_root / f"{stem}.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "variant", "checkpoint", "map_dir", "forced_level", "rollouts", *METRIC_KEYS],
        )
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in writer.fieldnames}
            for key in METRIC_KEYS:
                flat[key] = row["metrics"].get(key)
            writer.writerow(flat)
    print(json.dumps(rows, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)

    diagnose = subparsers.add_parser("diagnose")
    diagnose.add_argument(
        "--variants",
        nargs="+",
        choices=["h-legacy", "h-fixed-init", "m-legacy"],
        default=["h-legacy", "h-fixed-init", "m-legacy"],
    )
    diagnose.add_argument("--map-dir", default=str(DEFAULT_TRAINING_DIR))
    diagnose.add_argument("--num-maps", type=int, default=10000)
    diagnose.add_argument("--diagnostic-steps", type=int, default=524288)
    diagnose.add_argument("--seed", type=int, default=42)
    diagnose.add_argument("--device", default="cuda")
    diagnose.set_defaults(func=command_diagnose)

    train = subparsers.add_parser("train")
    train.add_argument("--variant", required=True, choices=["h-legacy", "h-fixed-init", "m-legacy"])
    train.add_argument("--map-dir", default=str(DEFAULT_TRAINING_DIR))
    train.add_argument("--num-maps", type=int, default=10000)
    train.add_argument("--total-env-steps", type=int, default=500_000_000)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cuda")
    train.add_argument("--checkpoint-interval", type=int, default=999999)
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("--variant", required=True, choices=["h-legacy", "h-fixed-init", "m-legacy"])
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--training-dir", default=str(DEFAULT_TRAINING_DIR))
    evaluate.add_argument("--validation-dir", default=str(DEFAULT_VALIDATION_DIR))
    evaluate.add_argument("--rollouts", type=int, default=32)
    evaluate.add_argument("--device", default="cuda")
    evaluate.set_defaults(func=command_eval)
    return parser


def main() -> None:
    parsed = build_parser().parse_args()
    parsed.func(parsed)


if __name__ == "__main__":
    main()
