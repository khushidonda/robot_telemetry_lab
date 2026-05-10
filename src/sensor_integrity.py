"""
sensor_integrity.py
===================
Timing / spacing checks on a single telemetry table clock (no second motion source).

Used for IMU-only bags where a classic "robot-truth vs odom" overlay is not meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_timing_stats(df: pd.DataFrame) -> dict[str, float]:
    """Inter-sample Δt statistics on the `t` column (seconds)."""
    t = df["t"].to_numpy(dtype=float)
    if t.size < 2:
        return {
            "median_dt_ms": 0.0,
            "p99_dt_ms": 0.0,
            "max_dt_ms": 0.0,
            "median_hz": 0.0,
            "n_gaps": 0.0,
            "gap_thresh_ms": 0.0,
        }

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return {
            "median_dt_ms": 0.0,
            "p99_dt_ms": 0.0,
            "max_dt_ms": 0.0,
            "median_hz": 0.0,
            "n_gaps": 0.0,
            "gap_thresh_ms": 0.0,
        }

    med = float(np.median(dt))
    p99 = float(np.percentile(dt, 99))
    mx = float(np.max(dt))
    gap_thresh = max(med * 8.0, 5e-3)  # 8× median spacing, floor 5 ms
    n_gaps = float(np.sum(dt > gap_thresh))
    med_hz = float(1.0 / med) if med > 0 else 0.0

    return {
        "median_dt_ms": med * 1000.0,
        "p99_dt_ms": p99 * 1000.0,
        "max_dt_ms": mx * 1000.0,
        "median_hz": med_hz,
        "n_gaps": n_gaps,
        "gap_thresh_ms": gap_thresh * 1000.0,
    }
