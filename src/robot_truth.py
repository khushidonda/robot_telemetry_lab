"""
robot_truth.py
==============
WHAT THIS FILE IS:
  A small "robot truth" style check for a demo.

CORE IDEA (very important):
  The robot (or simulator) REPORTS a speed in messages (speed_r).
  We can ALSO estimate speed from how position (x, y) changes over time.

  If those disagree a lot, something may be off (timing, slip, weird odometry,
  message mismatch, etc.). This is a REVIEW SIGNAL, not automatic diagnosis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_velocity_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    RETURNS:
      A NEW dataframe with extra columns:
        speed_d   : derived speed from (x,y) deltas
        div_speed : |speed_r - speed_d|
        div_flag  : True where mismatch is "big enough" to highlight
    """
    out = df.copy()

    t = out["t"].to_numpy(dtype=float)
    x = out["x"].to_numpy(dtype=float)
    y = out["y"].to_numpy(dtype=float)
    imu_fb = str(out.attrs.get("imu_fallback", "")).lower() == "true"

    # Time deltas; avoid divide-by-zero
    dt = np.concatenate([[np.nan], np.diff(t)])
    dt = np.where(dt <= 1e-9, np.nan, dt)

    if imu_fb and t.size >= 3:
        # IMU proxy path: forward `diff(x)/dt` matches integrated `vx` almost exactly by construction,
        # which makes the robot-truth panel look "fake aligned." Use central `np.gradient` on (x,y,t)
        # so the comparison is a real consistency check (still the same clip, not two sensors).
        vx_d = np.gradient(x, t, edge_order=2)
        vy_d = np.gradient(y, t, edge_order=2)
        vx_d = np.nan_to_num(vx_d, nan=0.0, posinf=0.0, neginf=0.0)
        vy_d = np.nan_to_num(vy_d, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        # Position deltas (standard odometry / pose tables)
        dx = np.concatenate([[0.0], np.diff(x)])
        dy = np.concatenate([[0.0], np.diff(y)])

        # Derived velocity components
        vx_d = dx / dt
        vy_d = dy / dt
        vx_d = np.nan_to_num(vx_d, nan=0.0, posinf=0.0, neginf=0.0)
        vy_d = np.nan_to_num(vy_d, nan=0.0, posinf=0.0, neginf=0.0)

    speed_d = np.hypot(vx_d, vy_d)
    speed_r = out["speed_r"].to_numpy(dtype=float)

    div = np.abs(speed_r - speed_d)

    # Threshold scales with motion so we don't flag noise when basically stopped
    if imu_fb:
        thresh = np.maximum(0.06, 0.22 * np.maximum(speed_r, 0.04))
    else:
        thresh = np.maximum(0.12, 0.35 * np.maximum(speed_r, 0.05))
    flag = (div > thresh) & (speed_r > 0.05)

    out["speed_d"] = speed_d
    out["div_speed"] = div
    out["div_flag"] = flag
    # Preserve load_telemetry metadata so downstream heuristics (IMU vs odom) stay honest.
    if getattr(df, "attrs", None):
        out.attrs.update(dict(df.attrs))
    return out


def divergence_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turns div_flag True-runs into a small table for charts:
      t_start, t_end, max_div
    """
    if "div_flag" not in df.columns:
        df = add_velocity_consistency(df)

    t = df["t"].to_numpy(float)
    f = df["div_flag"].to_numpy(bool)
    d = df["div_speed"].to_numpy(float)

    idx = np.flatnonzero(f)
    if idx.size == 0:
        return pd.DataFrame(columns=["t_start", "t_end", "max_div"])

    rows = []
    a = int(idx[0])
    b = int(idx[0])
    for k in idx[1:]:
        k = int(k)
        if k == b + 1:
            b = k
        else:
            rows.append((float(t[a]), float(t[b]), float(np.max(d[a : b + 1]))))
            a = b = k
    rows.append((float(t[a]), float(t[b]), float(np.max(d[a : b + 1]))))
    return pd.DataFrame(rows, columns=["t_start", "t_end", "max_div"])
