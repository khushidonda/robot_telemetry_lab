"""
app.py (run with Streamlit)
===========================
Browser dashboard layout follows the lab wireframe:
  [1] KPIs → [2] Motion profile → [3] Safety timeline → [4]/[6] split → [5] Robot-truth → [7] Summary

RUN:
  cd ~/Desktop/robot_telemetry_lab
  source .venv/bin/activate
  streamlit run app.py
"""

from __future__ import annotations

import html as html_mod
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import io_data  # noqa: E402
import parser as parser_mod  # noqa: E402
import robot_truth  # noqa: E402
import safety_categories  # noqa: E402
import sensor_integrity as sensor_integrity_mod  # noqa: E402
import visualizer  # noqa: E402

if os.environ.get("ROBOT_TELEMETRY_FORCE_RELOAD", "1") == "1":  # noqa: E402
    importlib.reload(io_data)
    importlib.reload(parser_mod)
    importlib.reload(robot_truth)
    importlib.reload(safety_categories)
    importlib.reload(sensor_integrity_mod)
    importlib.reload(visualizer)

from parser import load_telemetry  # noqa: E402
from robot_truth import add_velocity_consistency, divergence_segments  # noqa: E402
from safety_categories import DEFAULT_CATEGORY_ORDER, run_all_detectors  # noqa: E402
from sensor_integrity import compute_timing_stats  # noqa: E402
from visualizer import (  # noqa: E402
    estimate_sample_rate_hz,
    figure_benchmark_speedup,
    figure_events_by_category,
    figure_motion_profile,
    figure_robot_truth,
    figure_safety_timeline,
    figure_sensor_integrity,
)

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    pass


def _telemetry_disk_signature(repo_root: Path) -> tuple[tuple[str, int, int], ...]:
    """
    Cheap fingerprint of files under data/ so Streamlit cache invalidates when bags/CSV change.

    Mirrors the usual layout: top-level files plus ROS2 dirs with metadata.yaml + *.db3.
    """
    data_dir = repo_root / "data"
    bits: list[tuple[str, int, int]] = []
    if not data_dir.is_dir():
        return (("__no_data_dir__", 0, 0),)
    for p in sorted(data_dir.iterdir()):
        try:
            if p.is_file():
                st = p.stat()
                bits.append((p.name, int(st.st_mtime_ns), int(st.st_size)))
            elif p.is_dir():
                meta = p / "metadata.yaml"
                if meta.is_file():
                    st = meta.stat()
                    bits.append((f"{p.name}/metadata.yaml", int(st.st_mtime_ns), int(st.st_size)))
                    for db in sorted(p.glob("*.db3")):
                        stb = db.stat()
                        bits.append((f"{p.name}/{db.name}", int(stb.st_mtime_ns), int(stb.st_size)))
        except OSError:
            continue
    return tuple(bits) if bits else (("__empty_data__", 0, 0),)


@st.cache_data(show_spinner="Loading mission…")
def _cached_load_and_detect(
    disk_sig: tuple[tuple[str, int, int], ...],
    max_odom_env: str,
    imu_hp_env: str,
) -> dict[str, object]:
    """
    Full bag read + derived columns + detectors. Cached so moving the mission-window slider
    does not re-read ~500k rows from disk on every Streamlit rerun.
    """
    df0 = load_telemetry()
    df = add_velocity_consistency(df0)
    events = run_all_detectors(df)
    divs = divergence_segments(df)
    imu_fb = str(df0.attrs.get("imu_fallback", "")).lower() == "true"
    if imu_fb:
        _tim = compute_timing_stats(df)
        speed_compare_note = (
            f"**Sensor table timing:** {int(_tim['n_gaps'])} spacing spikes above {_tim['gap_thresh_ms']:.1f} ms "
            f"(median Δt {_tim['median_dt_ms']:.2f} ms). **Not** independent-sensor robot-truth — that needs wheel "
            "odometry or another motion source recorded alongside IMU."
        )
    else:
        speed_compare_note = (
            "Compares **reported speed** vs **motion-derived speed** from position deltas (same clip, two estimators)."
        )
    return {
        "df0": df0,
        "df": df,
        "events": events,
        "divs": divs,
        "speed_compare_note": speed_compare_note,
    }


def _flag_traffic(pct: float) -> str:
    return f"{pct:.1f}%"


def _format_trip_duration(clip_s: float) -> str:
    total = max(0, int(round(float(clip_s))))
    m, s = divmod(total, 60)
    if m == 0:
        return f"{s} second{'s' if s != 1 else ''}"
    if s == 0:
        return f"{m} minute{'s' if m != 1 else ''}"
    return f"{m} minute{'s' if m != 1 else ''} {s} second{'s' if s != 1 else ''}"


def _distance_plain(path_len: float) -> str:
    if path_len > 1000.0:
        km = path_len / 1000.0
        blocks = max(1, int(round(path_len / 80.0)))
        return f"{km:.2f} kilometers (about {blocks} city blocks)"
    return f"{max(0.0, path_len):.0f} meters"


def _pace_plain(median_speed: float) -> str:
    if median_speed < 0.3:
        ref = "slower than a casual walk"
    elif median_speed < 0.6:
        ref = "about walking pace"
    elif median_speed < 1.0:
        ref = "a brisk walk"
    else:
        ref = "faster than a brisk walk"
    return f"Typically {median_speed:.2f} meters per second — {ref}."


def _dominant_category_key(counts: dict[str, int]) -> str | None:
    best_k: str | None = None
    best_n = -1
    for k in DEFAULT_CATEGORY_ORDER:
        n = int(counts.get(k, 0))
        if n > best_n:
            best_n = n
            best_k = k
    return best_k if best_k is not None and best_n > 0 else None


def _unusual_line(total_events: int, counts: dict[str, int]) -> str:
    if total_events == 0:
        return "Nothing flagged. The robot moved smoothly the whole trip."
    dom = _dominant_category_key(counts)
    labels = {
        "velocity_spike": "sudden speed-ups (jerky motion)",
        "sharp_turn": "tight turns or aggressive reorientation",
        "sudden_stop": "sudden stops",
        "stall": "moments of being stuck in place",
    }
    stories = {
        "velocity_spike": "sped up suddenly while navigating around obstacles or people",
        "sharp_turn": "made tight turns or changed heading quickly while navigating around obstacles or people",
        "sudden_stop": "slowed or stopped abruptly, for example to avoid something in its path",
        "stall": "had stretches where it barely moved, as if waiting or blocked",
    }
    if dom is None:
        return (
            f"{total_events} moments were flagged for a closer look. "
            "See the category chart below for the mix of reasons — worth a quick supervisor scan."
        )
    lab = labels.get(dom, "unusual motion")
    story = stories.get(dom, "showed motion worth a quick human review")
    return (
        f"{total_events} moments were flagged for a closer look — mostly {lab}. "
        f"This usually means the robot {story}."
    )


def _data_clean_line(gap_spike_count: int) -> str:
    if gap_spike_count == 0:
        return "✅ Yes — no gaps in the sensor data. We can trust what we see."
    if gap_spike_count == 1:
        return (
            "⚠️ 1 brief sensor gap was detected — worth a closer look before drawing conclusions."
        )
    return (
        f"⚠️ {gap_spike_count} brief sensor gaps were detected — worth a closer look "
        "before drawing conclusions."
    )


def _bottom_line_verdict(clip_flag_pct: float, gap_spike_count: int) -> str:
    if clip_flag_pct < 1.0 and gap_spike_count == 0:
        return "Looks like a normal, healthy trip. No action needed."
    if 1.0 <= clip_flag_pct < 5.0 and gap_spike_count == 0:
        return "Mostly healthy with a few moments worth reviewing."
    if clip_flag_pct >= 5.0 or gap_spike_count > 0:
        return "Several moments need review. Recommend an engineer look at the flagged windows below."
    return "Looks like a normal, healthy trip. No action needed."


def _ops_briefing_card_html(
    *,
    clip_s: float,
    path_len: float,
    median_speed: float,
    total_events: int,
    counts: dict[str, int],
    gap_spike_count: int,
    clip_flag_pct: float,
) -> str:
    dur = html_mod.escape(_format_trip_duration(clip_s))
    dist = html_mod.escape(_distance_plain(path_len))
    pace = html_mod.escape(_pace_plain(median_speed))
    unusual = html_mod.escape(_unusual_line(total_events, counts))
    clean = html_mod.escape(_data_clean_line(gap_spike_count))
    bottom = html_mod.escape(_bottom_line_verdict(clip_flag_pct, gap_spike_count))
    foot = html_mod.escape(
        "Automated triage — designed to give operations teams a quick read on mission health. "
        "The technical detail below is for engineers who want to investigate flagged moments."
    )
    row = '<p style="margin:0 0 14px 0;line-height:1.55;font-size:1.06rem;color:#e8eaed;">{}<br/><span style="font-weight:400;">{}</span></p>'
    title = html_mod.escape("What this robot did (in plain English)")
    parts = [
        f'<div style="background-color:#1a1d23;border:1px solid #2d3139;border-radius:10px;padding:18px 20px;margin:0 0 16px 0;">',
        f'<p style="margin:0 0 16px 0;font-size:1.12rem;font-weight:700;color:#f8fafc;">📋 {title}</p>',
        row.format("<strong>How long was the trip?</strong>", dur),
        row.format("<strong>How far did the robot go?</strong>", dist),
        row.format("<strong>How fast was it usually moving?</strong>", pace),
        row.format("<strong>Did anything unusual happen?</strong>", unusual),
        row.format("<strong>Was the data clean?</strong>", clean),
        row.format("<strong>Bottom line</strong>", bottom),
        f'<p style="margin:12px 0 0 0;font-size:0.92rem;line-height:1.45;color:#9aa0a6;font-style:italic;">{foot}</p>',
        "</div>",
    ]
    return "".join(parts)


st.set_page_config(page_title="Robot Telemetry Lab", layout="wide")

st.title("Robot Telemetry Lab")

_disk_sig = _telemetry_disk_signature(ROOT)
_max_odom_key = os.environ.get("ROBOT_TELEMETRY_MAX_ODOM", "__unset__")
_imu_hp_key = os.environ.get("ROBOT_TELEMETRY_IMU_HP_S", "__unset__")
_bundle = _cached_load_and_detect(_disk_sig, _max_odom_key, _imu_hp_key)
df0 = _bundle["df0"]  # type: ignore[assignment]
df = _bundle["df"]  # type: ignore[assignment]
events = _bundle["events"]  # type: ignore[assignment]
divs = _bundle["divs"]  # type: ignore[assignment]
speed_compare_note = str(_bundle["speed_compare_note"])

source = str(df0.attrs.get("source", "unknown"))
imu_fb = str(df0.attrs.get("imu_fallback", "")).lower() == "true"
pose_fb = str(df0.attrs.get("pose_fallback", "")).lower() == "true"
trunc = df0.attrs.get("truncated_odom_rows") or df0.attrs.get("truncated_rows")

if imu_fb:
    signal_mode = "IMU integration proxy"
elif pose_fb:
    signal_mode = "Pose-derived motion"
else:
    signal_mode = "Odometry (or CSV) motion table"

t = df["t"].to_numpy(dtype=float)
t0 = float(t.min()) if t.size else 0.0
t1 = float(t.max()) if t.size else 0.0
clip_s = max(0.0, t1 - t0)
n_samples = int(len(df))
sample_hz = estimate_sample_rate_hz(t)

if imu_fb:
    speed_primary_label = "Proxy speed (IMU-integrated)"
else:
    speed_primary_label = "Reported speed (message / table)"
rt_title = "Robot-truth: reported vs motion-derived speed (orange = mismatch)"

# --- Sidebar: mission window (entire dashboard follows this range)
with st.sidebar:
    st.header("Mission window")
    if (not np.isfinite(t0)) or (not np.isfinite(t1)) or (t1 <= t0 + 1e-6):
        w_lo, w_hi = float(t0), float(t1)
        st.caption("No zoomable range for this clip.")
    else:
        _span = float(t1 - t0)
        _step = max(_span / 2000.0, 0.01)
        w_lo, w_hi = st.slider(
            "Time range (s from clip start)",
            min_value=float(t0),
            max_value=float(t1),
            value=(float(t0), float(t1)),
            step=float(_step),
        )
    st.caption("KPIs, briefing, charts, summary, and table all use this window. Telemetry load stays cached.")

_mask_plot = (df["t"] >= w_lo) & (df["t"] <= w_hi)
df_win = df.loc[_mask_plot]
# Matplotlib needs at least one row; if the window has no samples, fall back to full clip for figures only.
df_fig = df_win if not df_win.empty else df
_win_empty = df_win.empty and len(df) > 0

events_plot = [e for e in events if float(e.t_end) >= w_lo and float(e.t_start) <= w_hi]

if len(divs):
    divs_plot = divs[(divs["t_start"] <= w_hi) & (divs["t_end"] >= w_lo)].copy()
else:
    divs_plot = divs

clip_win = max(0.0, float(w_hi - w_lo))
t_w = df_win["t"].to_numpy(dtype=float) if len(df_win) else np.array([], dtype=float)
n_samples_win = int(len(df_win))
sample_hz_win = estimate_sample_rate_hz(t_w) if t_w.size >= 2 else 0.0

xv_w = df_win["x"].to_numpy(dtype=float) if len(df_win) else np.array([], dtype=float)
yv_w = df_win["y"].to_numpy(dtype=float) if len(df_win) else np.array([], dtype=float)
if xv_w.size >= 2:
    path_len = float(np.sum(np.hypot(np.diff(xv_w), np.diff(yv_w))))
else:
    path_len = 0.0

flagged_s = 0.0
for e in events_plot:
    a = max(float(e.t_start), float(w_lo))
    b = min(float(e.t_end), float(w_hi))
    flagged_s += max(0.0, b - a)
clip_flag_pct = float(min(100.0, 100.0 * flagged_s / max(clip_win, 1e-9))) if clip_win > 0 else 0.0

counts: dict[str, int] = {k: 0 for k in DEFAULT_CATEGORY_ORDER}
for e in events_plot:
    counts[e.category] = counts.get(e.category, 0) + 1

_spr_w = df_win["speed_r"].to_numpy(dtype=float) if len(df_win) else np.array([], dtype=float)
median_speed = float(np.median(_spr_w)) if _spr_w.size else 0.0
_gap_tim = compute_timing_stats(df_win)
gap_spike_count = int(_gap_tim["n_gaps"])

_display_ds = source.split(":")[-1] if ":" in source else source
_sig_line = "IMU proxy path" if imu_fb else signal_mode
_cap_note = f" · Row cap {trunc}" if trunc else ""
st.warning(
    f"Heuristic triage (not certification). Dataset `{_display_ds}`. Signal: {_sig_line}.{_cap_note}"
)
if _win_empty:
    st.caption("No samples fall in this time window — KPIs reflect the window; plots temporarily show the full clip so charts stay valid.")

st.markdown(
    _ops_briefing_card_html(
        clip_s=clip_win,
        path_len=path_len,
        median_speed=median_speed,
        total_events=len(events_plot),
        counts=counts,
        gap_spike_count=gap_spike_count,
        clip_flag_pct=clip_flag_pct,
    ),
    unsafe_allow_html=True,
)

st.markdown("### Mission overview")
st.caption("What it shows: duration, sample rate, and event count for the **selected time window** (sidebar).")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Duration", f"{clip_win:.2f} s")
k2.metric("Samples", f"{n_samples_win:,}")
k3.metric("Sample rate", f"{sample_hz_win:.2f} Hz")
k4.metric("Events", str(len(events_plot)))

st.caption(
    "What it shows: **path usage** (chord length), **how often** events occur, and **share of time** under any flag "
    "**within the selected window**."
)

triage_per_min = float(len(events_plot)) / max(clip_win / 60.0, 1e-9)

h1, h2, h3 = st.columns(3)
h1.metric("Path length (Σ‖Δr‖)", f"{path_len:.2f} m")
h2.metric("Triage density", f"{triage_per_min:.2f} evt/min")
h3.metric("Time under any flag", f"{_flag_traffic(clip_flag_pct)} of clip")

if imu_fb:
    st.caption(
        "What it shows: **IMU-derived** motion summaries (proxy speed and angular activity) for the **selected window**."
    )
    spr = df_win["speed_r"].to_numpy(dtype=float) if len(df_win) else np.array([], dtype=float)
    omg = np.abs(df_win["omega"].to_numpy(dtype=float)) if len(df_win) else np.array([], dtype=float)
    med_sp = float(np.median(spr)) if spr.size else 0.0
    p95_w = float(np.percentile(omg, 95)) if omg.size else 0.0
    j1, j2 = st.columns(2)
    j1.metric("Median proxy speed", f"{med_sp:.3f} m/s")
    j2.metric("P95 |ω|", f"{p95_w:.3f} rad/s")

st.divider()

st.markdown("### Motion profile")
st.caption(
    "What it shows: **smoothed** linear-accel proxy and angular velocity vs time (5 s rolling mean ±1σ). "
    "Detectors use full-rate data; long traces are decimated for plotting only."
)
fig_m = figure_motion_profile(df_fig)
st.pyplot(fig_m, clear_figure=True)

st.markdown("### Safety-style event timeline")
st.caption(
    "What it shows: speed with **heuristic** event bands by category. Labels **1, 10, 20, …** only; "
    "the event table lists events that overlap this window."
)
fig_t = figure_safety_timeline(df_fig, events_plot, speed_primary_label)
st.pyplot(fig_t, clear_figure=True)

st.markdown("### Category mix & performance")
left, right = st.columns((1, 1))

with left:
    st.caption("What it shows: **how many** merged windows fired per detector category in this **window**.")
    fig_c = figure_events_by_category(counts)
    st.pyplot(fig_c, clear_figure=True)

bench_path = ROOT / "outputs" / "benchmark.json"
bench: dict | None = None
if bench_path.exists():
    try:
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
    except Exception:
        bench = None

with right:
    st.caption("What it shows: **loop vs vectorized** detector runtime on a synthetic table (`outputs/benchmark.json`).")
    if bench:
        r = int(bench.get("rows", 0))
        ss = float(bench.get("slow_seconds", 0.0))
        fs = float(bench.get("fast_seconds", 0.0))
        sp = float(bench.get("speedup_pct_vs_slow", 0.0))
        m1, m2, m3 = st.columns(3)
        m1.metric("Bench rows", f"{r:,}")
        m2.metric("Slow (loops)", f"{ss:.3f}s")
        m3.metric("Fast (vector)", f"{fs:.4f}s")
        st.metric("Speedup vs slow", f"{sp:.1f}%")
        fig_b = figure_benchmark_speedup(r, ss, fs)
        st.pyplot(fig_b, clear_figure=True)
    else:
        st.info("Run `python src/analyzer.py` once to create `outputs/benchmark.json`.")

st.divider()

if imu_fb:
    st.markdown("### Sensor timing & data integrity")
    st.caption(
        "What it shows: **IMU sample spacing** and implied rate (gaps / jitter). "
        "For IMU-only bags this replaces a dual-speed “robot truth” chart, which would be circular."
    )
    fig_si = figure_sensor_integrity(df_fig)
    st.pyplot(fig_si, clear_figure=True)
else:
    st.markdown("### Robot-truth divergence")
    st.caption(
        "What it shows: **reported speed** vs **speed from position deltas**; shaded spans are sustained mismatch."
    )
    fig_r = figure_robot_truth(df_fig, divs_plot, speed_primary_label, title=rt_title)
    st.pyplot(fig_r, clear_figure=True)

st.divider()

st.markdown("### Summary")
st.caption("What it shows: a **plain-language** recap of detector output for the **selected window**.")
if counts and sum(counts.values()) > 0:
    breakdown = ", ".join(
        f"{k.replace('_', ' ')} ×{int(counts.get(k, 0))}" for k in DEFAULT_CATEGORY_ORDER
    )
    st.markdown(
        "In this window, the tool marked **candidate review windows** "
        f"({int(sum(counts.values()))} merged windows). **By category:** {breakdown}. "
        "Treat these as **triage**, not automatic fault labels."
    )
else:
    st.markdown("In this window, **no heuristic windows** overlap the selected range.")

if imu_fb:
    if len(df_win):
        _tim_note = (
            f"**Sensor table timing:** {int(_gap_tim['n_gaps'])} spacing spikes above {_gap_tim['gap_thresh_ms']:.1f} ms "
            f"(median Δt {_gap_tim['median_dt_ms']:.2f} ms). **Not** independent-sensor robot-truth — that needs wheel "
            "odometry or another motion source recorded alongside IMU."
        )
    else:
        _tim_note = "**Sensor table timing:** no samples in this window."
    st.markdown(f"No dual-speed overlay on IMU-only clips. {_tim_note}")
else:
    st.markdown(f"Mismatch segments in this window: **{len(divs_plot)}**. {speed_compare_note}")

st.divider()

st.markdown("### Event table")
st.caption(
    "What it shows: **first 50** events (by time) that overlap this window — category, review level, short description."
)
if not events_plot:
    st.info("No events overlap this time window.")
else:
    ev_ranked = sorted(events_plot, key=lambda e: (e.t_start, e.t_end))
    rows = [
        {
            "evt_id": i,
            "start_s": round(e.t_start, 2),
            "end_s": round(e.t_end, 2),
            "category": e.category,
            "Review level": e.severity,
            "description": e.description,
        }
        for i, e in enumerate(ev_ranked[:50], start=1)
    ]
    try:
        st.dataframe(pd.DataFrame(rows), width="stretch")
    except TypeError:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

