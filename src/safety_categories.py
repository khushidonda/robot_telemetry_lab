"""
safety_categories.py
======================
WHAT THIS FILE IS:
  This file defines small Python "classes" (templates) that each look for ONE
  kind of pattern in robot telemetry (a table of numbers over time).

WHY CLASSES:
  Paul (Product) won't read every line, but engineers like seeing clean structure:
  one detector per category = easy to extend later.

IMPORTANT HONESTY:
  These are HEURISTICS (rules of thumb), not certified safety labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

import numpy as np
import pandas as pd

# Timeline / chart colors (heuristic categories only — not safety certification levels).
CATEGORY_COLORS: dict[str, str] = {
    "sudden_stop": "#dc2626",
    "sharp_turn": "#ca8a04",
    "stall": "#64748b",
    "velocity_spike": "#7c3aed",
}

DEFAULT_CATEGORY_ORDER: tuple[str, ...] = ("sudden_stop", "sharp_turn", "stall", "velocity_spike")


@dataclass
class SafetyEvent:
    """
    WHAT:
      A tiny "row" describing something interesting that happened.

    FIELDS:
      t_start, t_end: time window in seconds (same clock as your dataframe `t`)
      category: short machine name like "sudden_stop"
      severity: 1-3 (demo scale; NOT a legal safety rating)
      description: English text a non-technical reader can understand
    """

    t_start: float
    t_end: float
    category: str
    severity: int
    description: str


class EventDetector(Protocol):
    """A Protocol is Python's way to say: anything with detect(df) is allowed here."""

    def detect(self, df: pd.DataFrame) -> List[SafetyEvent]:
        ...


class SuddenStopDetector:
    """
    IDEA:
      If speed drops a lot between two consecutive samples, that's a "sudden stop"
      candidate (hard braking, obstacle reaction, etc.).
    """

    def __init__(self, d_speed_thresh: float = 0.45) -> None:
        # How big a speed DROP must be to count as "sudden"
        self.d_speed_thresh = d_speed_thresh

    def detect(self, df: pd.DataFrame) -> List[SafetyEvent]:
        # Pull columns out of the table as fast NumPy arrays
        t = df["t"].to_numpy(dtype=float)  # time
        sp = df["speed_r"].to_numpy(dtype=float)  # reported speed

        # ds[i] ~= change in speed vs previous sample (first row has no previous)
        ds = np.concatenate([[0.0], np.diff(sp)])

        # Start with all False; mark only positions after the first row
        mask = np.zeros_like(ds, dtype=bool)
        mask[1:] = ds[1:] < -self.d_speed_thresh

        return _events_from_mask(
            t,
            mask,
            "sudden_stop",
            3,
            "Rapid speed decrease (possible hard stop / avoidance).",
        )


class SharpTurnDetector:
    """High angular velocity often means a sharp turn or aggressive heading change."""

    def __init__(self, omega_thresh: float = 1.2) -> None:
        self.omega_thresh = omega_thresh

    def detect(self, df: pd.DataFrame) -> List[SafetyEvent]:
        t = df["t"].to_numpy(dtype=float)
        w = df["omega"].to_numpy(dtype=float)
        mask = np.abs(w) > self.omega_thresh
        return _events_from_mask(
            t,
            mask,
            "sharp_turn",
            2,
            "High angular velocity (tight turn / aggressive reorientation).",
        )


class StallDetector:
    """
    Near-zero speed for a while can mean a stall / blockage / planner stuck.

    NOTE:
      Without a 'commanded motion' topic, we can't know intent—only stillness.
    """

    def __init__(self, speed_eps: float = 0.03, min_dur_s: float = 2.0) -> None:
        self.speed_eps = speed_eps
        self.min_dur_s = min_dur_s

    def detect(self, df: pd.DataFrame) -> List[SafetyEvent]:
        t = df["t"].to_numpy(dtype=float)
        sp = df["speed_r"].to_numpy(dtype=float)
        low = sp < self.speed_eps
        return _events_from_runs(t, low, self.min_dur_s, "stall", 2, "Near-zero speed sustained (possible stall / blockage).")


class VelocitySpikeDetector:
    """Large translational acceleration proxy = jerky motion / sudden push."""

    def __init__(self, accel_thresh: float = 2.5, merge_gap: int = 3) -> None:
        self.accel_thresh = accel_thresh
        self.merge_gap = merge_gap

    def detect(self, df: pd.DataFrame) -> List[SafetyEvent]:
        t = df["t"].to_numpy(dtype=float)
        sp = df["speed_r"].to_numpy(dtype=float)

        # dt[i] is time between sample i-1 and i (first row has no previous time delta)
        dt = np.concatenate([[np.nan], np.diff(t)])
        dsp = np.concatenate([[0.0], np.diff(sp)])  # change in speed vs previous

        # accel ~ Δspeed/Δtime (simple proxy)
        accel = dsp / dt
        accel = np.nan_to_num(accel, nan=0.0, posinf=0.0, neginf=0.0)

        mask = np.abs(accel) > self.accel_thresh
        return _events_from_mask(
            t,
            mask,
            "velocity_spike",
            2,
            "High translational jerk / acceleration spike.",
            merge_gap=self.merge_gap,
        )


def _merge_nearby_indices(idxs: np.ndarray, max_gap: int = 3) -> List[tuple[int, int]]:
    """
    INPUT:
      idxs = indices where mask is True (sorted)

    OUTPUT:
      list of (start_index, end_index) runs, merging small gaps so one "event"
      doesn't become 50 tiny rectangles on the chart.
    """
    if idxs.size == 0:
        return []
    runs: List[tuple[int, int]] = []
    start = int(idxs[0])
    prev = int(idxs[0])
    for i in idxs[1:]:
        i = int(i)
        if i <= prev + max_gap:
            prev = i
        else:
            runs.append((start, prev))
            start = prev = i
    runs.append((start, prev))
    return runs


def _events_from_mask(
    t: np.ndarray,
    mask: np.ndarray,
    category: str,
    severity: int,
    description: str,
    merge_gap: int = 3,
) -> List[SafetyEvent]:
    """Convert a boolean mask into a list of SafetyEvent windows."""
    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        return []
    events: List[SafetyEvent] = []
    for a, b in _merge_nearby_indices(idxs, max_gap=merge_gap):
        events.append(
            SafetyEvent(
                t_start=float(t[a]),
                t_end=float(t[b]),
                category=category,
                severity=severity,
                description=description,
            )
        )
    return events


def _events_from_runs(
    t: np.ndarray,
    mask: np.ndarray,
    min_dur_s: float,
    category: str,
    severity: int,
    description: str,
) -> List[SafetyEvent]:
    """Like mask, but only keep runs that last long enough in time."""
    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        return []
    events: List[SafetyEvent] = []
    for a, b in _merge_nearby_indices(idxs, max_gap=1):
        if float(t[b] - t[a]) >= min_dur_s:
            events.append(
                SafetyEvent(
                    t_start=float(t[a]),
                    t_end=float(t[b]),
                    category=category,
                    severity=severity,
                    description=description,
                )
            )
    return events


def _imu_short_clip(df: pd.DataFrame) -> bool:
    """
    Short / row-capped clips (common when iterating with ROBOT_TELEMETRY_MAX_ODOM)
    need slightly tighter thresholds than full-bag IMU, otherwise every chart looks
    'empty' even though ω / jerk clearly move in the window.
    """
    if str(df.attrs.get("imu_fallback", "")).lower() != "true":
        return False
    t = df["t"].to_numpy(dtype=float)
    if t.size < 2:
        return False
    dur = float(np.max(t) - np.min(t))
    return dur < 75.0 or len(df) < 12_000


def _telemetry_signal_mode(df: pd.DataFrame) -> str:
    """
    Pick detector calibration from `df.attrs` (set by `io_data.load_telemetry`).

    IMU integration paths are noisier than wheel odometry; thresholds are relaxed so
    counts read as triage hints, not fake incident floods.
    """
    if str(df.attrs.get("imu_fallback", "")).lower() == "true":
        return "imu_proxy"
    if str(df.attrs.get("pose_fallback", "")).lower() == "true":
        return "pose"
    return "odom"


def run_all_detectors(df: pd.DataFrame) -> List[SafetyEvent]:
    """Run every detector and return a time-sorted list of events."""
    mode = _telemetry_signal_mode(df)
    if mode == "imu_proxy":
        if _imu_short_clip(df):
            # Capped / short IMU windows: keep triage interpretable without flooding full-bag scale.
            detectors = [
                SuddenStopDetector(d_speed_thresh=0.42),
                SharpTurnDetector(omega_thresh=1.05),
                StallDetector(speed_eps=0.055, min_dur_s=1.35),
                VelocitySpikeDetector(accel_thresh=7.5, merge_gap=10),
            ]
        else:
            # Tuned to surface representative events; production thresholds calibrated per-platform.
            # Long Cartographer IMU bag (~1900s / ~478k rows): loosen stop / stall / spike so the
            # category chart is not "sharp_turn only" while staying triage-grade (not certification).
            detectors = [
                SuddenStopDetector(d_speed_thresh=0.48),
                SharpTurnDetector(omega_thresh=1.85),
                StallDetector(speed_eps=0.10, min_dur_s=1.65),
                VelocitySpikeDetector(accel_thresh=5.8, merge_gap=12),
            ]
    elif mode == "pose":
        detectors = [
            SuddenStopDetector(d_speed_thresh=0.62),
            SharpTurnDetector(omega_thresh=1.85),
            StallDetector(speed_eps=0.045, min_dur_s=2.2),
            VelocitySpikeDetector(accel_thresh=4.2, merge_gap=5),
        ]
    else:
        detectors = [
            SuddenStopDetector(),
            SharpTurnDetector(),
            StallDetector(),
            VelocitySpikeDetector(),
        ]
    out: List[SafetyEvent] = []
    for d in detectors:
        out.extend(d.detect(df))
    out.sort(key=lambda e: e.t_start)
    return out
