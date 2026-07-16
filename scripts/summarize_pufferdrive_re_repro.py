#!/usr/bin/env python3
"""Build normalized training-curve artifacts from reproduction outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def load_jsonl_curve(path: Path, variant: str) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        rows.append(
            {
                "variant": variant,
                "epoch": item["epoch"],
                "aggregate_agent_steps": item["aggregate_agent_steps"],
                "completion_rate": item.get("environment/completion_rate"),
                "score": item.get("environment/score"),
                "collision_rate": item.get("environment/collision_rate"),
                "offroad_rate": item.get("environment/offroad_rate"),
                "dnf_rate": item.get("environment/dnf_rate"),
            }
        )
    return rows


def metric(block: str, name: str) -> float | None:
    match = re.search(rf"{re.escape(name)}\s+(-?\d+\.\d+)", block)
    return float(match.group(1)) if match else None


def parse_steps(value: str) -> int:
    suffix = value[-1]
    multiplier = {"K": 1_000, "M": 1_000_000}.get(suffix, 1)
    number = value[:-1] if suffix in ("K", "M") else value
    return int(round(float(number) * multiplier))


def load_dashboard_curve(path: Path, variant: str) -> list[dict]:
    text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    rows_by_epoch = {}
    for block in text.split("╭"):
        epoch_match = re.search(r"Epoch\s+(\d+)", block)
        steps_match = re.search(r"Steps\s+([0-9.]+[MK]?)", block)
        if not epoch_match or not steps_match or "completion_rate" not in block:
            continue
        epoch = int(epoch_match.group(1))
        rows_by_epoch[epoch] = {
            "variant": variant,
            "epoch": epoch,
            "aggregate_agent_steps": parse_steps(steps_match.group(1)),
            "completion_rate": metric(block, "completion_rate"),
            "score": metric(block, "score"),
            "collision_rate": metric(block, "collision_rate"),
            "offroad_rate": metric(block, "offroad_rate"),
            "dnf_rate": metric(block, "dnf_rate"),
        }
    return [rows_by_epoch[epoch] for epoch in sorted(rows_by_epoch)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/home/fanyuqi/wsc/behavior-bench/.logs/repro/pufferdrive_re_500m",
    )
    parsed = parser.parse_args()
    root = Path(parsed.root)

    rows = load_dashboard_curve(root / "logs/train_h-legacy.log", "h-legacy")
    rows += load_jsonl_curve(root / "runs/h-fixed-init/train_metrics.jsonl", "h-fixed-init")
    rows += load_jsonl_curve(root / "runs/m-legacy/train_metrics.jsonl", "m-legacy")

    output_json = root / "training_curves.json"
    output_csv = root / "training_curves.csv"
    output_json.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "epoch",
                "aggregate_agent_steps",
                "completion_rate",
                "score",
                "collision_rate",
                "offroad_rate",
                "dnf_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "json": str(output_json), "csv": str(output_csv)}))


if __name__ == "__main__":
    main()
