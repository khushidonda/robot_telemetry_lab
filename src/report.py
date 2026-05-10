"""
report.py
=========
Writes `outputs/summary_report.txt` — a short, ops-readable brief aligned with
fleet telemetry review (not certified safety analysis).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import pandas as pd

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parser import load_telemetry  # noqa: E402
from robot_truth import add_velocity_consistency, divergence_segments  # noqa: E402
from safety_categories import run_all_detectors  # noqa: E402
from sensor_integrity import compute_timing_stats  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs"


def _fmt_mmss(t: float) -> str:
    m = int(t // 60)
    s = int(round(t - 60 * m))
    return f"{m:02d}:{s:02d}"


def _dur_line(t0: float, t1: float) -> str:
    sec = max(0.0, float(t1 - t0))
    if sec < 120:
        return f"{sec:.1f}s"
    return f"{sec/60:.1f} min ({sec:.0f}s)"


def build_report(df: pd.DataFrame) -> str:
    df2 = add_velocity_consistency(df)
    events = run_all_detectors(df2)
    divs = divergence_segments(df2)

    t0 = float(df2["t"].min())
    t1 = float(df2["t"].max())
    counts = Counter(e.category for e in events)

    src = str(df2.attrs.get("source", "unknown"))
    imu_fb = str(df2.attrs.get("imu_fallback", "")).lower() == "true"
    pose_fb = str(df2.attrs.get("pose_fallback", "")).lower() == "true"
    trunc = df2.attrs.get("truncated_odom_rows") or df2.attrs.get("truncated_rows")

    lines: list[str] = []

    lines.append("FLEET TELEMETRY BRIEF (demo)")
    lines.append("")
    lines.append(f"Data: {src}")
    if imu_fb:
        lines.append("Note: IMU-only bag — motion is an integration proxy, not wheel odometry.")
        lines.append(
            "Heuristic thresholds auto-relax on IMU-proxy paths so flags stay interpretable "
            "(triage, not incident certification)."
        )
    elif pose_fb:
        lines.append("Note: PoseStamped trajectory — speeds are derived from pose deltas.")
    if trunc:
        lines.append(f"Note: loaded first {trunc} messages (cap for fast iteration).")
    lines.append(f"Clip: {_dur_line(t0, t1)}")
    lines.append("")

    # JD / resume alignment: triage flags from telemetry (explicitly not certification)
    lines.append("Review flags (heuristics):")
    if not counts:
        lines.append("- None above threshold in this window.")
    else:
        for k in ["sudden_stop", "sharp_turn", "stall", "velocity_spike"]:
            n = int(counts.get(k, 0))
            if n:
                lines.append(f"- {k.replace('_', ' ')}: {n}")
    lines.append(f"- Total flagged intervals: {len(events)}")
    lines.append("")

    lines.append("Top events:")
    if not events:
        lines.append("- None")
    else:
        for e in events[:5]:
            lines.append(
                f"- {_fmt_mmss(e.t_start)}–{_fmt_mmss(e.t_end)} | {e.category} | {e.description}"
            )
        if len(events) > 5:
            lines.append(f"- … +{len(events) - 5} more (full list in dashboard)")
    lines.append("")

    if imu_fb:
        ts = compute_timing_stats(df2)
        lines.append("Sensor timing / table integrity (IMU-only; no second motion source):")
        lines.append(
            f"- Median Δt: {ts['median_dt_ms']:.2f} ms (~{ts['median_hz']:.2f} Hz), "
            f"p99 Δt: {ts['p99_dt_ms']:.1f} ms, max Δt: {ts['max_dt_ms']:.1f} ms"
        )
        lines.append(
            f"- Spacing spikes above {ts['gap_thresh_ms']:.1f} ms: {int(ts['n_gaps'])} "
            "(coarse proxy for gaps / jitter in the timestamp column)"
        )
        lines.append(
            "- Classic robot-truth (independent A vs B) is not applicable without wheel odometry; "
            "the dashboard swaps the speed overlay for this integrity view."
        )
    else:
        lines.append("Robot-truth (self-consistency):")
        if len(divs):
            top = divs.sort_values("max_div", ascending=False).head(2)
            for _, r in top.iterrows():
                lines.append(
                    f"- Mismatch {_fmt_mmss(float(r.t_start))}–{_fmt_mmss(float(r.t_end))} "
                    f"(peak Δ ≈ {float(r.max_div):.2f})"
                )
        else:
            lines.append("- No sustained mismatch between reported speed and motion-derived speed.")
    lines.append("")

    lines.append(
        "Next at scale: version metric definitions with eng/validation; validate flags against "
        "mission logs; expand from single-robot clip to fleet rollups (uptime, dock success, stops/hr)."
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_telemetry()
    text = build_report(df)
    (OUT_DIR / "summary_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
