#!/usr/bin/env python3
"""Build repeat vs first-round vs H0 summary markdown + CSVs.

Waits until all 4 stratified repeats finish idm/expert/smart evals, then writes:
  repeat/result/repeat_vs_first_summary.md
  repeat/result/metrics_*.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO = Path("/home/fanyuqi/wsc/behavior-bench")
REPEAT = REPO / ".logs/train/run_repaet/repeat"
OUT_DIR = REPEAT / "result"
FIRST_ROOT = REPO / ".logs/train/run_repaet"
H0_ROOT = REPO / ".logs/train/run4/problem"

TRAFFICS = ("IDM", "Expert", "Smart")
TRAFFIC_DIR = {"IDM": "test_IDM", "Expert": "test_Expert", "Smart": "test_Smart"}

# collision_type → attribution bucket (matches first-round metrics_summary.csv)
TYPE_BUCKET = {
    "ACTIVE_LATERAL_COLLISION": "lateral",
    "ACTIVE_FRONT_COLLISION": "front",
    "ACTIVE_REAR_COLLISION": "rear",
    "STOPPED_TRACK_COLLISION": "stopped",
    "STOPPED_EGO_COLLISION": "stopped",
}


@dataclass
class RunSpec:
    label: str
    first_wandb: str
    first_family_dir: str  # under FIRST_ROOT
    first_train_name: str
    repeat_subdir: str
    repeat_train_name: str
    repeat_wandb_file: Path


RUNS = [
    RunSpec(
        "DriveLite A strat",
        "r0jht7bf",
        "trainLite",
        "drivelite_mixA_stratified_1500m_run_repaet",
        "drivelite_mixA_stratified",
        "drivelite_mixA_stratified_1500m_run_repaet_repeat",
        REPEAT / "drivelite_mixA_stratified" / "wandb_run_id.txt",
    ),
    RunSpec(
        "DriveLite B strat",
        "qgytyxt4",
        "trainLite",
        "drivelite_mixB_stratified_1500m_run_repaet",
        "drivelite_mixB_stratified",
        "drivelite_mixB_stratified_1500m_run_repaet_repeat",
        REPEAT / "drivelite_mixB_stratified" / "wandb_run_id.txt",
    ),
    RunSpec(
        "Perception strat",
        "yov2516v",
        "trainPerception",
        "perception_mix_low_mid_high_stratified_1500m_run_repaet",
        "perception_stratified",
        "perception_mix_low_mid_high_stratified_1500m_run_repaet_repeat",
        REPEAT / "perception_stratified" / "wandb_run_id.txt",
    ),
    RunSpec(
        "MIXMOE strat",
        "o4trtoxc",
        "trainMOE",
        "mix_moe_drive_moe_moe3_stratified_1500m_run_repaet",
        "mixmoe_stratified",
        "mix_moe_drive_moe_moe3_stratified_1500m_run_repaet_repeat",
        REPEAT / "mixmoe_stratified" / "wandb_run_id.txt",
    ),
]

H0 = {
    "label": "H0 run4_r2",
    "wandb": "u9ymfcqr",
    "train_name": "homogeneous_drive_lstm_500m_batch1x_run4_r2",
}


def _map_count(run_dir: Path) -> int:
    pmc = run_dir / "per_map.csv"
    if not pmc.is_file():
        return 0
    try:
        with pmc.open() as f:
            # header + rows
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


def _traffic_type(run_dir: Path) -> str:
    cfg = run_dir / "config.json"
    if not cfg.is_file():
        return ""
    try:
        return str(json.loads(cfg.read_text()).get("traffic", {}).get("type", "")).lower()
    except (OSError, json.JSONDecodeError, AttributeError):
        return ""


def newest_run_dir(root: Path, expect_traffic: Optional[str] = None) -> Optional[Path]:
    """Pick best finished eval rundir.

    Prefer matching traffic.type (idm/expert/smart), then ~589-map full runs, then newest.
    """
    if not root.is_dir():
        return None
    cands = [p for p in root.iterdir() if p.is_dir() and (p / "summary.csv").is_file()]
    if not cands:
        return None
    want = (expect_traffic or "").lower()
    scored = []
    for p in cands:
        n = _map_count(p)
        tt = _traffic_type(p)
        traffic_ok = 1 if (want and tt == want) else (0 if want else 1)
        full = 1 if 500 <= n <= 700 else 0
        scored.append((traffic_ok, full, n, p.stat().st_mtime, p))
    scored.sort(reverse=True)
    return scored[0][-1]


def read_summary(summary_csv: Path) -> dict:
    out = {}
    with summary_csv.open() as f:
        for row in csv.DictReader(f):
            m = row["metric"]
            try:
                out[m] = float(row["mean"]) if row["mean"] not in ("", "nan") else float("nan")
            except ValueError:
                out[m] = row["mean"]
    return out


def attr_from_snapshots(snap_path: Path) -> dict:
    buckets = Counter({"lateral": 0, "front": 0, "rear": 0, "stopped": 0})
    if not snap_path.is_file():
        return {**buckets, "collision_maps": 0, "raw_types": {}}
    data = json.loads(snap_path.read_text())
    raw = Counter()
    maps = set()
    for item in data:
        maps.add(item.get("map_id"))
        t = item.get("collision_type", "UNKNOWN")
        raw[t] += 1
        buckets[TYPE_BUCKET.get(t, "stopped" if "STOPPED" in t else "front")] += 1
        # unknown non-stopped → count as front bucket fallback already; keep raw
    return {
        "lateral": buckets["lateral"],
        "front": buckets["front"],
        "rear": buckets["rear"],
        "stopped": buckets["stopped"],
        "collision_maps": len(maps),
        "raw_types": dict(raw),
    }


def load_eval(base: Path, train_name: str, expect_traffic: Optional[str] = None) -> Optional[dict]:
    # Infer traffic from parent folder name test_IDM / test_Expert / test_Smart if needed
    if expect_traffic is None:
        for part in base.parts:
            if part.startswith("test_"):
                expect_traffic = part.replace("test_", "").lower()
                break
    run_dir = newest_run_dir(base / train_name / "Drive_Recurrent", expect_traffic=expect_traffic)
    if run_dir is None:
        return None
    summary = read_summary(run_dir / "summary.csv")
    attr = attr_from_snapshots(run_dir / "collision_snapshots.json")
    return {
        "run_dir": str(run_dir),
        "goal_pct": 100.0 * summary.get("goal_reached_rate", float("nan")),
        "collision_pct": 100.0 * summary.get("collision_rate", float("nan")),
        "at_fault_pct": 100.0 * summary.get("at_fault_collision_rate", float("nan")),
        "offroad_pct": 100.0 * summary.get("offroad_rate", float("nan")),
        "reward": summary.get("total_reward", float("nan")),
        **{k: attr[k] for k in ("lateral", "front", "rear", "stopped", "collision_maps")},
        "raw_types": attr["raw_types"],
    }


def pct(x: float, digits: int = 2) -> str:
    if x != x:  # nan
        return "—"
    return f"{x:.{digits}f}%"


def num(x: float, digits: int = 4) -> str:
    if x != x:
        return "—"
    return f"{x:.{digits}f}"


def rel_delta(new: float, base: float) -> str:
    if new != new or base != base or base == 0:
        return "—"
    return f"{(new - base) / base * 100:+.1f}%"


def abs_delta(new: float, old: float, unit: str = "pp") -> str:
    if new != new or old != old:
        return "—"
    d = new - old
    sign = "+" if d >= 0 else ""
    if unit == "pp":
        return f"{sign}{d:.2f}pp"
    return f"{sign}{d:.4f}"


def read_wandb(path: Path, fallback: str = "—") -> str:
    if path.is_file():
        return path.read_text().strip() or fallback
    # try discover from train log
    parent = path.parent
    logs = sorted(parent.glob("*_run_repaet_repeat_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for log in logs:
        text = log.read_text(errors="ignore")
        m = re.search(r"wandb\.ai/[^/]+/[^/]+/runs/([a-z0-9]+)", text)
        if m:
            return m.group(1)
        m = re.search(r"setting up run ([a-z0-9]+)", text)
        if m:
            return m.group(1)
    return fallback


def all_repeat_ready() -> tuple[bool, list[str]]:
    missing = []
    for spec in RUNS:
        for traffic in TRAFFICS:
            base = REPEAT / spec.repeat_subdir / TRAFFIC_DIR[traffic]
            m = load_eval(base, spec.repeat_train_name)
            if m is None:
                missing.append(f"{spec.label}/{traffic}")
    return (len(missing) == 0, missing)


def gather() -> dict:
    """Return nested dict [label][traffic][phase] -> metrics."""
    data = {}
    for traffic in TRAFFICS:
        h0 = load_eval(H0_ROOT / TRAFFIC_DIR[traffic], H0["train_name"])
        if h0 is None:
            raise FileNotFoundError(f"missing H0 {traffic}")
        data.setdefault(H0["label"], {})[traffic] = {
            "H0": h0,
            "wandb": H0["wandb"],
        }

    for spec in RUNS:
        wandb2 = read_wandb(spec.repeat_wandb_file)
        for traffic in TRAFFICS:
            first = load_eval(
                FIRST_ROOT / spec.first_family_dir / TRAFFIC_DIR[traffic],
                spec.first_train_name,
            )
            second = load_eval(
                REPEAT / spec.repeat_subdir / TRAFFIC_DIR[traffic],
                spec.repeat_train_name,
            )
            if first is None:
                raise FileNotFoundError(f"missing first {spec.label}/{traffic}")
            if second is None:
                raise FileNotFoundError(f"missing repeat {spec.label}/{traffic}")
            data.setdefault(spec.label, {})[traffic] = {
                "first": first,
                "second": second,
                "first_wandb": spec.first_wandb,
                "second_wandb": wandb2,
            }
    return data


def bold_better(val_str: str, better: bool) -> str:
    return f"**{val_str}**" if better else val_str


def metric_row_triple(h0, first, second, key, lower_better: bool) -> tuple[str, str, str]:
    hv, fv, sv = h0[key], first[key], second[key]

    def mark(v, ref_h0, ref_first=None):
        s = pct(v) if key.endswith("_pct") else (num(v, 4) if key == "reward" else str(int(v) if v == v else "—"))
        # highlight second if beats H0 on the metric direction
        if key == "reward":
            good_h0 = v > ref_h0
        elif key.endswith("_pct") or key in ("collision_maps",):
            good_h0 = v < ref_h0 if lower_better else v > ref_h0
        else:
            good_h0 = False
        return bold_better(s, good_h0 and (v == sv))

    if key.endswith("_pct"):
        return pct(hv), pct(fv), pct(sv)
    if key == "reward":
        return num(hv, 4), num(fv, 4), num(sv, 4)
    return str(int(hv)), str(int(fv)), str(int(sv))


def write_md(data: dict, path: Path) -> None:
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S %z")
    lines += [
        "# run_repaet / repeat · 复训复测指标总结",
        "",
        f"> 生成时间：{now}  ",
        f"> 协议：`pufferinter` · `MAP_IDS=all`（有效图约 589）· ego=`Drive+Recurrent`  ",
        f"> 基线：H0 run4_r2（[`{H0['wandb']}`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/{H0['wandb']})）  ",
        "> 对照：首轮 stratified（`.logs/train/run_repaet/result/`）vs 本轮 `repeat/`  ",
        "",
        "**加粗规则（主表 second 列）**：相对 H0，Goal/Reward 更高加粗；Collision / At-fault / Offroad 更低加粗。",
        "",
        "## 0. Run 索引",
        "",
        "| Experiment | 首轮 W&B | 复跑 W&B | 复跑目录 |",
        "| --- | --- | --- | --- |",
    ]
    for spec in RUNS:
        w2 = data[spec.label]["IDM"]["second_wandb"]
        lines.append(
            f"| {spec.label} | [`{spec.first_wandb}`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/{spec.first_wandb}) "
            f"| [`{w2}`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/{w2}) "
            f"| `repeat/{spec.repeat_subdir}/` |"
        )
    lines += ["", "---", ""]

    for traffic in TRAFFICS:
        lines += [
            f"## {TRAFFICS.index(traffic)+1}. vs {traffic} · 主指标（H0 / 首轮 / 复跑）",
            "",
        ]
        for spec in RUNS:
            h0 = data[H0["label"]][traffic]["H0"]
            blk = data[spec.label][traffic]
            first, second = blk["first"], blk["second"]
            lines += [
                f"### {spec.label}",
                "",
                f"首轮 [`{blk['first_wandb']}`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/{blk['first_wandb']}) · "
                f"复跑 [`{blk['second_wandb']}`](https://wandb.ai/sichengwang-beihang-university/behavior-bench/runs/{blk['second_wandb']})",
                "",
                "| Metric | H0 baseline | 首轮 (1st) | 复跑 (2nd) | Δ 2nd−1st | Δ 2nd vs H0 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
            rows = [
                ("Goal ↑", "goal_pct", False),
                ("Collision ↓", "collision_pct", True),
                ("At-fault ↓", "at_fault_pct", True),
                ("Offroad ↓", "offroad_pct", True),
                ("Reward ↑", "reward", False),
            ]
            for name, key, lower in rows:
                hv, fv, sv = h0[key], first[key], second[key]
                if key == "reward":
                    hs, fs, ss = num(hv, 4), num(fv, 4), num(sv, 4)
                    d12 = abs_delta(sv, fv, "raw")
                    good = sv > hv
                else:
                    hs, fs, ss = pct(hv), pct(fv), pct(sv)
                    d12 = abs_delta(sv, fv, "pp")
                    good = (sv < hv) if lower else (sv > hv)
                ss_fmt = bold_better(ss, good)
                d_h0 = rel_delta(sv, hv) if key != "reward" else abs_delta(sv, hv, "raw")
                lines.append(f"| {name} | {hs} | {fs} | {ss_fmt} | {d12} | {d_h0} |")

            lines += [
                "",
                f"**可复现性速览**：Coll 2nd−1st = {abs_delta(second['collision_pct'], first['collision_pct'])}；"
                f" Fault 2nd−1st = {abs_delta(second['at_fault_pct'], first['at_fault_pct'])}。",
                "",
            ]

        # overview table for this traffic
        lines += [
            f"### vs {traffic} · 四实验总览（Collision）",
            "",
            "| Experiment | H0 Coll | 1st Coll | 2nd Coll | Δ2−1 | 2nd vs H0 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        h0c = data[H0["label"]][traffic]["H0"]["collision_pct"]
        for spec in RUNS:
            f = data[spec.label][traffic]["first"]["collision_pct"]
            s = data[spec.label][traffic]["second"]["collision_pct"]
            lines.append(
                f"| {spec.label} | {pct(h0c)} | {pct(f)} | {pct(s)} | {abs_delta(s, f)} | {rel_delta(s, h0c)} |"
            )
        lines += ["", "---", ""]

    # Attribution section
    lines += [
        "## 5. 碰撞归因对比（maps：lateral / front / rear / stopped）",
        "",
        "计数来自 `collision_snapshots.json` 的 `collision_type`：",
        "",
        "- lateral ← `ACTIVE_LATERAL_COLLISION`",
        "- front ← `ACTIVE_FRONT_COLLISION`",
        "- rear ← `ACTIVE_REAR_COLLISION`",
        "- stopped ← `STOPPED_*_COLLISION`",
        "",
    ]
    for traffic in TRAFFICS:
        lines += [
            f"### vs {traffic}",
            "",
        ]
        for spec in RUNS:
            h0 = data[H0["label"]][traffic]["H0"]
            first = data[spec.label][traffic]["first"]
            second = data[spec.label][traffic]["second"]
            lines += [
                f"#### {spec.label}",
                "",
                "| Phase | Coll maps | Lateral | Front | Rear | Stopped |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                f"| H0 | {h0['collision_maps']} | {h0['lateral']} | {h0['front']} | {h0['rear']} | {h0['stopped']} |",
                f"| 1st | {first['collision_maps']} | {first['lateral']} | {first['front']} | {first['rear']} | {first['stopped']} |",
                f"| 2nd | {second['collision_maps']} | {second['lateral']} | {second['front']} | {second['rear']} | {second['stopped']} |",
                f"| Δ2−1 | {second['collision_maps']-first['collision_maps']:+d} | "
                f"{second['lateral']-first['lateral']:+d} | {second['front']-first['front']:+d} | "
                f"{second['rear']-first['rear']:+d} | {second['stopped']-first['stopped']:+d} |",
                "",
            ]
        # wide table
        lines += [
            f"**vs {traffic} 宽表**",
            "",
            "| Experiment | Phase | Maps | Lat | Front | Rear | Stop |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        h0 = data[H0["label"]][traffic]["H0"]
        lines.append(
            f"| H0 | baseline | {h0['collision_maps']} | {h0['lateral']} | {h0['front']} | {h0['rear']} | {h0['stopped']} |"
        )
        for spec in RUNS:
            for phase, key in (("1st", "first"), ("2nd", "second")):
                m = data[spec.label][traffic][key]
                lines.append(
                    f"| {spec.label} | {phase} | {m['collision_maps']} | {m['lateral']} | "
                    f"{m['front']} | {m['rear']} | {m['stopped']} |"
                )
        lines += ["", "---", ""]

    lines += [
        "## 6. 结论要点（自动草稿）",
        "",
    ]
    for traffic in TRAFFICS:
        lines.append(f"- **vs {traffic}**")
        for spec in RUNS:
            f = data[spec.label][traffic]["first"]["collision_pct"]
            s = data[spec.label][traffic]["second"]["collision_pct"]
            h = data[H0["label"]][traffic]["H0"]["collision_pct"]
            lines.append(
                f"  - {spec.label}: Coll {pct(f)} → {pct(s)}（Δ {abs_delta(s, f)}；vs H0 {rel_delta(s, h)}）"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "机器可读表：同目录 `metrics_main.csv` / `metrics_attribution.csv`。",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csvs(data: dict) -> None:
    main_path = OUT_DIR / "metrics_main.csv"
    attr_path = OUT_DIR / "metrics_attribution.csv"
    with main_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "experiment",
                "traffic",
                "phase",
                "wandb",
                "goal_pct",
                "collision_pct",
                "at_fault_pct",
                "offroad_pct",
                "reward",
            ]
        )
        for traffic in TRAFFICS:
            h0 = data[H0["label"]][traffic]["H0"]
            w.writerow(
                [
                    H0["label"],
                    traffic,
                    "H0",
                    H0["wandb"],
                    f"{h0['goal_pct']:.4f}",
                    f"{h0['collision_pct']:.4f}",
                    f"{h0['at_fault_pct']:.4f}",
                    f"{h0['offroad_pct']:.4f}",
                    f"{h0['reward']:.6f}",
                ]
            )
            for spec in RUNS:
                blk = data[spec.label][traffic]
                for phase, key, wb in (
                    ("1st", "first", blk["first_wandb"]),
                    ("2nd", "second", blk["second_wandb"]),
                ):
                    m = blk[key]
                    w.writerow(
                        [
                            spec.label,
                            traffic,
                            phase,
                            wb,
                            f"{m['goal_pct']:.4f}",
                            f"{m['collision_pct']:.4f}",
                            f"{m['at_fault_pct']:.4f}",
                            f"{m['offroad_pct']:.4f}",
                            f"{m['reward']:.6f}",
                        ]
                    )

    with attr_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "experiment",
                "traffic",
                "phase",
                "wandb",
                "collision_maps",
                "lateral",
                "front",
                "rear",
                "stopped",
            ]
        )
        for traffic in TRAFFICS:
            h0 = data[H0["label"]][traffic]["H0"]
            w.writerow(
                [
                    H0["label"],
                    traffic,
                    "H0",
                    H0["wandb"],
                    h0["collision_maps"],
                    h0["lateral"],
                    h0["front"],
                    h0["rear"],
                    h0["stopped"],
                ]
            )
            for spec in RUNS:
                blk = data[spec.label][traffic]
                for phase, key, wb in (
                    ("1st", "first", blk["first_wandb"]),
                    ("2nd", "second", blk["second_wandb"]),
                ):
                    m = blk[key]
                    w.writerow(
                        [
                            spec.label,
                            traffic,
                            phase,
                            wb,
                            m["collision_maps"],
                            m["lateral"],
                            m["front"],
                            m["rear"],
                            m["stopped"],
                        ]
                    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", action="store_true", help="poll until all repeat evals ready")
    ap.add_argument("--poll-seconds", type=int, default=120)
    ap.add_argument("--once", action="store_true", help="build now or fail if incomplete")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status_log = OUT_DIR / "build_summary_watch.log"

    def log(msg: str) -> None:
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        with status_log.open("a") as f:
            f.write(line + "\n")

    if args.wait:
        log("Waiting for 4×3 repeat evals to finish...")
        while True:
            ready, missing = all_repeat_ready()
            if ready:
                log("All repeat evals ready.")
                break
            log(f"Missing ({len(missing)}): {', '.join(missing[:8])}{'...' if len(missing)>8 else ''}")
            time.sleep(args.poll_seconds)
    else:
        ready, missing = all_repeat_ready()
        if not ready:
            log(f"Incomplete: {missing}")
            if args.once:
                return 1
            log("Use --wait to poll, or --once to fail hard.")
            return 1

    data = gather()
    md = OUT_DIR / "repeat_vs_first_summary.md"
    write_md(data, md)
    write_csvs(data)
    log(f"Wrote {md}")
    log(f"Wrote {OUT_DIR / 'metrics_main.csv'}")
    log(f"Wrote {OUT_DIR / 'metrics_attribution.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
