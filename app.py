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


def _flag_traffic(pct: float) -> str:
    if pct < 10.0:
        return f"🟢 {pct:.1f}%"
    if pct < 30.0:
        return f"🟡 {pct:.1f}%"
    return f"🔴 {pct:.1f}%"


st.set_page_config(page_title="Robot Telemetry Lab", layout="wide")

st.title("🤖 Robot Telemetry Lab")
st.caption("Rosbag-derived safety & robot-truth analysis (demo; not certification).")

df0 = load_telemetry()
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

df = add_velocity_consistency(df0)
events = run_all_detectors(df)
divs = divergence_segments(df)

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

_display_ds = source.split(":")[-1] if ":" in source else source
_sig_line = "IMU integration (no wheel `/odom`)" if imu_fb else signal_mode
_compact = (
    f"⚠️ **Heuristic triage only** · 📊 **Dataset:** `{_display_ds}` · 🧮 **Signal:** {_sig_line}"
    + (f" · **Load cap:** `{trunc}` rows" if trunc else "")
)
st.info(_compact)

with st.expander("Methodology details", expanded=False):
    st.markdown(
        """
**Scope:** Heuristic triage only — not certified safety, not validation sign-off, not fleet incident certification.  
Flags are **review aids** so a human can skim where the math got interesting.

**Offline artifacts:** run `python src/analyzer.py` → `outputs/` PNGs, `safety_events.csv`, `benchmark.json`, `summary_report.txt`.

**IMU bags (Cartographer-style):** no in-range wheel **`/odom`** for this public clip — motion is an **IMU integration proxy** (bias removal + moving-mean high-pass on linear accel before integration). Tune the high-pass with `ROBOT_TELEMETRY_IMU_HP_S`.  
**Detector calibration:** full-bag IMU uses a relaxed set so counts stay interpretable; short / row-capped previews auto-tighten slightly so bands are still visible. Thresholds are **tuned to surface representative events from this dataset**; production would be **calibrated per platform** with validation owners.

**Motion profile chart:** shows a **5 s rolling mean ±1σ** for display only; **detectors always run on the full-rate table.**
"""
    )

st.markdown("### Mission overview")
st.caption("Wireframe [1] — KPI strip + fleet-style rollups (still heuristic triage, not certification).")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Duration", f"{clip_s:.2f} s")
k2.metric("Samples", f"{n_samples:,}")
k3.metric("Sample rate", f"{sample_hz:.2f} Hz")
k4.metric("Events", str(len(events)))

# JD / fleet-telemetry headline metrics (honest on any signal path)
xv = df["x"].to_numpy(dtype=float)
yv = df["y"].to_numpy(dtype=float)
if xv.size >= 2:
    path_len = float(np.sum(np.hypot(np.diff(xv), np.diff(yv))))
else:
    path_len = 0.0
triage_per_min = float(len(events)) / max(clip_s / 60.0, 1e-9)
flagged_s = 0.0
for e in events:
    flagged_s += max(0.0, float(e.t_end - e.t_start))
clip_flag_pct = float(min(100.0, 100.0 * flagged_s / max(clip_s, 1e-9))) if clip_s > 0 else 0.0

h1, h2, h3 = st.columns(3)
h1.metric("Path length (Σ‖Δr‖)", f"{path_len:.2f} m")
h2.metric("Triage density", f"{triage_per_min:.2f} evt/min")
h3.metric("Time under any flag", f"{_flag_traffic(clip_flag_pct)} of clip")

st.caption(
    "Path length sums chord steps in **meters** (assumes SI accel integration for this demo bag). "
    "Triage density uses merged heuristic windows; overlap across categories can double-count time in the % row."
)

# IMU-only signal-quality row (defensible motion summaries — not internal hyperparameters)
if imu_fb:
    spr = df["speed_r"].to_numpy(dtype=float)
    omg = np.abs(df["omega"].to_numpy(dtype=float))
    med_sp = float(np.median(spr)) if spr.size else 0.0
    p95_w = float(np.percentile(omg, 95)) if omg.size else 0.0
    j1, j2 = st.columns(2)
    j1.metric("Median proxy speed", f"{med_sp:.3f} m/s")
    j2.metric("P95 |ω| (turniness)", f"{p95_w:.3f} rad/s")

st.divider()

st.markdown("### Motion profile")
st.caption(
    "[2] **5 s rolling mean ±1σ** (display only; detectors use full-rate data). "
    "Long clips are **uniformly decimated to ≤10k vertices** for Matplotlib/Agg stability."
)
fig_m = figure_motion_profile(df)
st.pyplot(fig_m, clear_figure=True)
if imu_fb:
    _hp_disp = str(df0.attrs.get("imu_accel_hp_s", "")).strip()
    if _hp_disp:
        st.caption(
            f"**Analysis setting:** IMU linear-accel high-pass window ≈ **{_hp_disp} s** "
            "(`ROBOT_TELEMETRY_IMU_HP_S`). Used only when building the integration proxy, not for plotting smoothing."
        )

st.markdown("### Safety-style event timeline")
st.caption("[3] Colored bands = heuristic category; numbered chips match the event table.")
fig_t = figure_safety_timeline(df, events, speed_primary_label)
st.pyplot(fig_t, clear_figure=True)

st.markdown("### Category mix & performance story")
st.caption("[4] Event histogram · [6] NumPy / vectorization benchmark (synthetic 120k-row table).")
left, right = st.columns((1, 1))

counts: dict[str, int] = {k: 0 for k in DEFAULT_CATEGORY_ORDER}
for e in events:
    counts[e.category] = counts.get(e.category, 0) + 1

with left:
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
        st.caption("Synthetic table benchmark from `outputs/benchmark.json` (run `python src/analyzer.py`).")
    else:
        st.info("No `outputs/benchmark.json` yet. Run **`python src/analyzer.py`** once to generate it.")

st.divider()

if imu_fb:
    st.markdown("### Sensor timing & data integrity")
    st.caption(
        "[5] IMU-only bags have a single motion source — a dual speed trace is tautological. "
        "This panel is the honest substitute: timestamp spacing + rate jitter."
    )
    st.markdown(
        "⭐ **Interview line:** “No `/odom` in this public bag, so I show **data integrity** "
        "(gaps / jitter) instead of pretending I have cross-sensor robot truth.”"
    )
    fig_si = figure_sensor_integrity(df)
    st.pyplot(fig_si, clear_figure=True)
    st.caption(
        "Sample-rate jitter: **y-axis clipped at 500 Hz**; sub-millisecond sample pairs can create numerical spikes in `1/Δt`."
    )
else:
    st.markdown("### Robot-truth divergence")
    st.caption("[5] Reported vs motion-derived speed when a motion table + twist/pose exists.")
    st.markdown("⭐ Primary consistency chart for stakeholder walkthroughs.")
    fig_r = figure_robot_truth(df, divs, speed_primary_label, title=rt_title)
    st.pyplot(fig_r, clear_figure=True)

st.divider()

st.markdown("### Plain-English summary")
st.caption("[7] Ops-readable recap.")
if counts and sum(counts.values()) > 0:
    parts = [f"{k.replace('_', ' ')} ×{v}" for k, v in sorted(counts.items()) if v]
    st.markdown(
        "During this clip, the tool marked **candidate review windows**: **"
        + ", ".join(parts)
        + "**. Treat these as **triage**, not automatic fault labels."
    )
else:
    st.markdown("During this clip, **no heuristic windows** cleared the (signal-aware) thresholds.")

if imu_fb:
    _div_line = f"**Robot-truth speed overlay:** not shown (single-source clip). {speed_compare_note}"
else:
    _div_line = f"**Robot-truth mismatch windows:** {len(divs)}. {speed_compare_note}"

st.markdown(
    _div_line
    + "  \n**Text brief:** `outputs/summary_report.txt` (from `python src/report.py` or `python src/analyzer.py`)."
)

st.divider()

st.subheader("Event table (first 50)")
if not events:
    st.info("No rows — thresholds did not fire on this clip (expected for some clean segments).")
else:
    ev_ranked = sorted(events, key=lambda e: (e.t_start, e.t_end))
    rows = [
        {
            "evt_id": i,
            "start_s": round(e.t_start, 2),
            "end_s": round(e.t_end, 2),
            "category": e.category,
            "severity": e.severity,
            "description": e.description,
        }
        for i, e in enumerate(ev_ranked[:50], start=1)
    ]
    try:
        st.dataframe(pd.DataFrame(rows), width="stretch")
    except TypeError:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

with st.sidebar:
    st.header("How to run")
    st.code(
        f"cd {ROOT}\n"
        "source .venv/bin/activate\n"
        "# Fast preview (12s-style cap):\n"
        "# export ROBOT_TELEMETRY_MAX_ODOM=3000\n"
        "# Monday showcase (full Cartographer IMU stream — slower first load):\n"
        "# unset ROBOT_TELEMETRY_MAX_ODOM   # uses default ~500k rows (~30 min @170 Hz)\n"
        "# export ROBOT_TELEMETRY_MAX_ODOM=0   # effectively unlimited\n"
        "python src/analyzer.py\n"
        "streamlit run app.py\n",
        language="bash",
    )
    st.markdown("**Outputs:** `outputs/*` (charts, `safety_events.csv`, reports, benchmark JSON).")
    st.markdown("**Modules:** `src/parser.py`, `src/analyzer.py`, `src/visualizer.py`")
