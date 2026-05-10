"""
detectors_slow.py
=================
WHAT THIS FILE IS:
  Slow versions of detection logic using plain Python loops.

WHY IT EXISTS:
  For learning + for benchmark_events.py:
  loops are easy to read, but slow on big arrays.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from safety_categories import SafetyEvent


def sudden_stops_slow(df: pd.DataFrame, d_speed_thresh: float = 0.45) -> List[SafetyEvent]:
    events: List[SafetyEvent] = []
    t = df["t"].to_numpy(float)
    sp = df["speed_r"].to_numpy(float)
    for i in range(1, len(sp)):
        ds = sp[i] - sp[i - 1]
        if ds < -d_speed_thresh:
            events.append(
                SafetyEvent(
                    t_start=float(t[i - 1]),
                    t_end=float(t[i]),
                    category="sudden_stop",
                    severity=3,
                    description="Rapid speed decrease (loop-detected).",
                )
            )
    return events


def sharp_turns_slow(df: pd.DataFrame, omega_thresh: float = 1.2) -> List[SafetyEvent]:
    events: List[SafetyEvent] = []
    t = df["t"].to_numpy(float)
    w = df["omega"].to_numpy(float)
    for i in range(len(w)):
        if abs(w[i]) > omega_thresh:
            events.append(
                SafetyEvent(
                    t_start=float(t[i]),
                    t_end=float(t[i]),
                    category="sharp_turn",
                    severity=2,
                    description="High angular velocity (loop-detected).",
                )
            )
    return events


def stalls_slow(df: pd.DataFrame, speed_eps: float = 0.03, min_dur_s: float = 2.0) -> List[SafetyEvent]:
    events: List[SafetyEvent] = []
    t = df["t"].to_numpy(float)
    sp = df["speed_r"].to_numpy(float)

    run_start = None
    for i in range(len(sp)):
        if sp[i] < speed_eps:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                if t[i - 1] - t[run_start] >= min_dur_s:
                    events.append(
                        SafetyEvent(
                            t_start=float(t[run_start]),
                            t_end=float(t[i - 1]),
                            category="stall",
                            severity=2,
                            description="Near-zero speed sustained (loop-detected).",
                        )
                    )
                run_start = None
    return events


def velocity_spikes_slow(df: pd.DataFrame, accel_thresh: float = 2.5) -> List[SafetyEvent]:
    events: List[SafetyEvent] = []
    t = df["t"].to_numpy(float)
    sp = df["speed_r"].to_numpy(float)
    for i in range(1, len(sp)):
        dt = t[i] - t[i - 1]
        if dt <= 1e-9:
            continue
        accel = (sp[i] - sp[i - 1]) / dt
        if abs(accel) > accel_thresh:
            events.append(
                SafetyEvent(
                    t_start=float(t[i - 1]),
                    t_end=float(t[i]),
                    category="velocity_spike",
                    severity=2,
                    description="High acceleration (loop-detected).",
                )
            )
    return events


def all_events_slow(df: pd.DataFrame) -> List[SafetyEvent]:
    out: List[SafetyEvent] = []
    out.extend(sudden_stops_slow(df))
    out.extend(sharp_turns_slow(df))
    out.extend(stalls_slow(df))
    out.extend(velocity_spikes_slow(df))
    out.sort(key=lambda e: e.t_start)
    return out
