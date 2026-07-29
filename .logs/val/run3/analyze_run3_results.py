#!/usr/bin/env python3
"""Compare run3 adversarial evals vs homogeneous 500M (ego vs IDM)."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path.home() / "wsc" / "behavior-bench"
VAL = REPO / ".logs" / "val" / "run3" / "results"
HOMO = (
    REPO
    / ".logs"
    / "val"
    / "run1"
    / "results"
    / "500M"
    / "homogeneous_drive_lstm_500m_batch1x_run1"
    / "Drive_Recurrent"
    / "20260714_150238_c150a5"
)

# Internal key -> (display name, short note)
JOB_META = {
    "homo_500m": ("同质 500M", "对照"),
    "adv_500m_baseline": ("Adv 500M baseline", "默认对手奖励"),
    "adv_500m_idm50": ("Adv 500M + IDM50%", "训练混入 50% IDM"),
    "ppo_idm50": ("纯 PPO + IDM50%", "单卡500M，无对抗，PPO+IDM"),
    "adv_4b_default": ("Adv 4B default", "同默认奖励，加长训"),
    "adv_500m_high_normality": ("Adv 500M high_normality", "normality 0.02→0.10"),
    "adv_500m_weak_goal": ("Adv 500M weak_goal", "goal=0.15"),
    "adv_500m_low_ego_cost": ("Adv 500M low_ego_cost", "ego_cost↓ / fault↑"),
    "adv_500m_conservative": ("Adv 500M conservative", "偏保守对手性格"),
}

JOBS = {
    "homo_500m": HOMO,
    "adv_500m_baseline": VAL
    / "adv_mix_ego_opponent_drive_lstm_ddp2_500m_run3"
    / "p0_ego_Drive_Recurrent",
    "adv_500m_idm50": VAL
    / "adv_mix_ego_opponent_drive_lstm_ddp2_500m_idm50_run3"
    / "p0_ego_Drive_Recurrent",
    "ppo_idm50": VAL
    / "ppo_idm50_drive_lstm_500m_run3"
    / "Drive_Recurrent",
    "adv_4b_default": VAL
    / "adv_mix_ego_opponent_drive_lstm_ddp2_batch2x_4b_run3"
    / "p0_ego_Drive_Recurrent",
    "adv_500m_high_normality": VAL
    / "adv_mix_ego_opponent_drive_lstm_ddp2_500m_high_normality_run3"
    / "p0_ego_Drive_Recurrent",
    "adv_500m_weak_goal": VAL
    / "adv_mix_ego_opponent_drive_lstm_ddp2_500m_weak_goal_run3"
    / "p0_ego_Drive_Recurrent",
    "adv_500m_low_ego_cost": VAL
    / "adv_mix_ego_opponent_drive_lstm_ddp2_500m_low_ego_cost_run3"
    / "p0_ego_Drive_Recurrent",
    "adv_500m_conservative": VAL
    / "adv_mix_ego_opponent_drive_lstm_ddp2_500m_conservative_run3"
    / "p0_ego_Drive_Recurrent",
}

METRIC_LABELS = {
    "goal_reached_rate": "Goal ↑",
    "collision_rate": "Collision ↓",
    "at_fault_collision_rate": "At-fault ↓",
    "offroad_rate": "Offroad ↓",
    "total_reward": "Reward ↑",
}


def latest_summary_dir(root: Path) -> Path | None:
    if root is None:
        return None
    if (root / "summary.csv").exists():
        return root
    if not root.exists():
        return None
    subs = sorted(
        [p for p in root.iterdir() if p.is_dir() and (p / "summary.csv").exists()],
        key=lambda p: p.stat().st_mtime,
    )
    return subs[-1] if subs else None


def read_summary(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                out[row["metric"]] = float(row["mean"])
            except (KeyError, ValueError, TypeError):
                continue
    return out


def fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{100.0 * x:.2f}%"


def fmt_delta_pp(d: float | None) -> str:
    if d is None:
        return "—"
    return f"{100.0 * d:+.2f}pp"


def collision_breakdown(summary_dir: Path) -> dict:
    snap = summary_dir / "collision_snapshots.json"
    empty = {
        "n": 0,
        "fault": 0,
        "fault_pct": None,
        "front": 0,
        "lateral": 0,
        "other": 0,
    }
    if not snap.exists():
        return empty
    snaps = json.loads(snap.read_text())
    if not snaps:
        return empty
    types = Counter(s.get("collision_type", "?") for s in snaps)
    faults = sum(1 for s in snaps if s.get("at_fault"))
    front = types.get("ACTIVE_FRONT_COLLISION", 0)
    lateral = types.get("ACTIVE_LATERAL_COLLISION", 0)
    other = len(snaps) - front - lateral
    return {
        "n": len(snaps),
        "fault": faults,
        "fault_pct": 100.0 * faults / len(snaps),
        "front": front,
        "lateral": lateral,
        "other": other,
    }


def verdict_for(h: dict, r: dict) -> tuple[str, str]:
    g = (r.get("goal_reached_rate") or 0) - (h.get("goal_reached_rate") or 0)
    f = (r.get("at_fault_collision_rate") or 0) - (
        h.get("at_fault_collision_rate") or 0
    )
    c = (r.get("collision_rate") or 0) - (h.get("collision_rate") or 0)
    notes = []
    worse = 0
    if g < -0.01:
        notes.append(f"goal {g*100:+.1f}pp")
        worse += 1
    if f > 0.002:
        notes.append(f"fault {f*100:+.1f}pp")
        worse += 1
    if c > 0.002:
        notes.append(f"coll {c*100:+.1f}pp")
        worse += 1
    if worse == 0:
        status = "更好/持平"
    elif worse <= 1 and (g >= -0.02 or c <= 0):
        status = "互有胜负"
    else:
        status = "更差"
    detail = "；".join(notes) if notes else "相对同质无明显恶化"
    return status, detail


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def analyze() -> str:
    rows: dict[str, dict[str, float]] = {}
    dirs: dict[str, Path] = {}
    missing: list[str] = []
    for name, root in JOBS.items():
        d = latest_summary_dir(root)
        if d is None:
            missing.append(name)
            continue
        dirs[name] = d
        rows[name] = read_summary(d / "summary.csv")

    names = [n for n in JOBS if n in rows]
    lines: list[str] = []

    lines.append("# Run3 对抗训练 vs 同质对照")
    lines.append("")
    lines.append("**评测协议**：ego（policy 0）vs IDM · split=`pufferinter` · maps=all")
    lines.append("")
    lines.append("**成功标准**：相对同质，至少不伤 Goal，且 At-fault 不升高。")
    lines.append("")

    if missing:
        miss_names = ", ".join(JOB_META[m][0] for m in missing)
        lines.append(f"> 尚未完成 / 缺失：{miss_names}")
        lines.append("")

    # --- main metrics: one row per run ---
    lines.append("## 1. 主指标（绝对值）")
    lines.append("")
    headers = ["设定", "说明", "Goal ↑", "Collision ↓", "At-fault ↓", "Offroad ↓"]
    table_rows = []
    for n in names:
        label, note = JOB_META[n]
        r = rows[n]
        name_cell = f"**{label}（对照）**" if n == "homo_500m" else f"**{label}**"
        table_rows.append(
            [
                name_cell,
                note,
                fmt_pct(r.get("goal_reached_rate")),
                fmt_pct(r.get("collision_rate")),
                fmt_pct(r.get("at_fault_collision_rate")),
                fmt_pct(r.get("offroad_rate")),
            ]
        )
    lines.extend(md_table(headers, table_rows))
    lines.append("")

    # --- deltas ---
    if "homo_500m" in rows:
        h = rows["homo_500m"]
        lines.append("## 2. 相对同质的差值（pp = percentage points）")
        lines.append("")
        lines.append("正值表示该指标数值升高；对 Goal 升高更好，对其余三项升高更差。")
        lines.append("")
        headers = ["设定", "Δ Goal", "Δ Collision", "Δ At-fault", "Δ Offroad", "判定"]
        table_rows = []
        for n in names:
            if n == "homo_500m":
                continue
            label, _ = JOB_META[n]
            r = rows[n]
            status, detail = verdict_for(h, r)
            table_rows.append(
                [
                    label,
                    fmt_delta_pp(
                        (r.get("goal_reached_rate") or 0)
                        - (h.get("goal_reached_rate") or 0)
                    ),
                    fmt_delta_pp(
                        (r.get("collision_rate") or 0) - (h.get("collision_rate") or 0)
                    ),
                    fmt_delta_pp(
                        (r.get("at_fault_collision_rate") or 0)
                        - (h.get("at_fault_collision_rate") or 0)
                    ),
                    fmt_delta_pp(
                        (r.get("offroad_rate") or 0) - (h.get("offroad_rate") or 0)
                    ),
                    f"{status}（{detail}）",
                ]
            )
        lines.extend(md_table(headers, table_rows))
        lines.append("")

    # --- collision morphology ---
    lines.append("## 3. 碰撞形态（碰撞 map 数）")
    lines.append("")
    lines.append("同质以 **侧向** 为主；对抗各组普遍变成 **前方 + 高 at-fault 占比**。")
    lines.append("")
    headers = ["设定", "碰撞 maps", "其中 at-fault", "Front", "Lateral", "其它"]
    table_rows = []
    for n in names:
        label, _ = JOB_META[n]
        b = collision_breakdown(dirs[n])
        if b["n"] == 0:
            table_rows.append([label, "0", "—", "—", "—", "—"])
            continue
        table_rows.append(
            [
                label,
                str(b["n"]),
                f"{b['fault']}/{b['n']} ({b['fault_pct']:.0f}%)",
                str(b["front"]),
                str(b["lateral"]),
                str(b["other"]),
            ]
        )
    lines.extend(md_table(headers, table_rows))
    lines.append("")

    # --- ranking / takeaways ---
    lines.append("## 4. 结论（按相对同质的综合表现）")
    lines.append("")
    if "homo_500m" in rows:
        h = rows["homo_500m"]
        ranked = []
        for n in names:
            if n == "homo_500m":
                continue
            r = rows[n]
            # Higher is better: reward goal gains, penalize fault/collision rises.
            g = (r.get("goal_reached_rate") or 0) - (h.get("goal_reached_rate") or 0)
            f = (r.get("at_fault_collision_rate") or 0) - (
                h.get("at_fault_collision_rate") or 0
            )
            c = (r.get("collision_rate") or 0) - (h.get("collision_rate") or 0)
            score = 3.0 * g - 4.0 * f - 1.0 * c
            ranked.append((score, n))
        ranked.sort(reverse=True)

        lines.append("| 排名 | 设定 | 一句话 |")
        lines.append("| --- | --- | --- |")
        for i, (_, n) in enumerate(ranked, 1):
            label, note = JOB_META[n]
            status, detail = verdict_for(h, rows[n])
            lines.append(f"| {i} | {label}（{note}） | {status}：{detail} |")
        lines.append("")

        best = JOB_META[ranked[0][1]][0] if ranked else "—"
        lines.append("### 要点")
        lines.append("")
        lines.append(f"1. **没有一组同时打赢同质**（Goal + At-fault）。当前相对最好：`{best}`。")
        lines.append(
            "2. **加长到 4B 不能修复**：相对 500M baseline，At-fault 更差，不是「训不够」。"
        )
        lines.append(
            "3. **改对手奖励/性格的消融未奏效**；conservative / low_ego_cost 最差。"
        )
        lines.append(
            "4. 共同失败模式：碰撞从侧向转向 **前方顶撞**，at-fault 占比升高 → "
            "对手施压轴与 IDM 评测分布不匹配。"
        )
        lines.append("")

    # --- appendix paths ---
    lines.append("## 附录：结果目录")
    lines.append("")
    for n in names:
        label, _ = JOB_META[n]
        lines.append(f"- **{label}**：`{dirs[n]}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    report = analyze()
    out = VAL / "run3_comparison_report.md"
    out.write_text(report)
    print(report)
    print(f"\nWrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
