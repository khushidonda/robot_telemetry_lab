"""
analyzer.py
===========
Single entry that runs the **offline pipeline**: load → robot-truth → detectors →
`outputs/` artifacts (PNGs, CSV, text report, benchmark JSON).

Matches the lab file plan where `analyzer.py` is the “main pipeline + benchmark” driver.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark_events import run_synthetic_benchmark  # noqa: E402
from parser import load_telemetry  # noqa: E402
from report import build_report  # noqa: E402
from robot_truth import add_velocity_consistency, divergence_segments  # noqa: E402
from safety_categories import SafetyEvent, run_all_detectors  # noqa: E402
from visualizer import export_pipeline_pngs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs"


def _events_to_rows(events: list[SafetyEvent]) -> list[dict]:
    return [
        {
            "t_start": e.t_start,
            "t_end": e.t_end,
            "category": e.category,
            "severity": e.severity,
            "description": e.description,
        }
        for e in events
    ]


def _speed_labels(imu_fb: bool) -> tuple[str, str | None]:
    if imu_fb:
        return (
            "Proxy speed (IMU-integrated)",
            "Self-consistency: proxy speed vs motion-derived speed (orange = sustained mismatch)",
        )
    return (
        "Reported speed (message / table)",
        "Robot-truth divergence: reported vs motion-derived speed (orange = sustained mismatch)",
    )


def run_pipeline() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df0 = load_telemetry()
    df = add_velocity_consistency(df0)
    events = run_all_detectors(df)
    divs = divergence_segments(df)

    imu_fb = str(df0.attrs.get("imu_fallback", "")).lower() == "true"
    speed_label, rt_title = _speed_labels(imu_fb)

    pd.DataFrame(_events_to_rows(events)).to_csv(OUT_DIR / "safety_events.csv", index=False)
    (OUT_DIR / "summary_report.txt").write_text(build_report(df0), encoding="utf-8")

    bench = run_synthetic_benchmark()
    (OUT_DIR / "benchmark.txt").write_text(bench["text"], encoding="utf-8")
    (OUT_DIR / "benchmark.json").write_text(json.dumps(bench["json"], indent=2), encoding="utf-8")

    export_pipeline_pngs(
        df,
        events,
        divs,
        OUT_DIR,
        speed_primary_label=speed_label,
        robot_truth_title=rt_title,
        benchmark_rows=int(bench["json"]["rows"]),
        benchmark_slow_s=float(bench["json"]["slow_seconds"]),
        benchmark_fast_s=float(bench["json"]["fast_seconds"]),
        imu_fallback=imu_fb,
    )

    print("Wrote outputs:")
    chart_truth_name = "chart_sensor_integrity.png" if imu_fb else "chart_robot_truth.png"
    for name in [
        "chart_motion_profile.png",
        "chart_safety_events.png",
        chart_truth_name,
        "chart_events_by_category.png",
        "chart_benchmark.png",
        "safety_events.csv",
        "summary_report.txt",
        "benchmark.txt",
        "benchmark.json",
    ]:
        print(f"  - outputs/{name}")


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
