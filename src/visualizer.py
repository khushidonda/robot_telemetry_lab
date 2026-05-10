"""
visualizer.py
=============
Matplotlib figures for the dashboard and static `outputs/chart_*.png` exports.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from safety_categories import CATEGORY_COLORS, DEFAULT_CATEGORY_ORDER, SafetyEvent
from sensor_integrity import compute_timing_stats


def _style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    # Long Cartographer clips (~500k points) can exceed Agg's default path chunk budget.
    mpl.rcParams["agg.path.chunksize"] = 100_000
    mpl.rcParams["path.simplify_threshold"] = 0.2


def _decimate_uniform(max_points: int, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Uniform stride subsample for Matplotlib display (analysis stays full-rate elsewhere)."""
    if not arrays:
        return ()
    n = int(arrays[0].size)
    if n <= max_points:
        return arrays
    step = int(np.ceil(n / float(max_points)))
    sl = slice(None, None, step)
    return tuple(np.asarray(a, dtype=float)[sl] for a in arrays)


def estimate_sample_rate_hz(t: np.ndarray) -> float:
    """Median sample rate from timestamps (Hz)."""
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return 0.0
    dt = np.diff(t)
    dt = dt[dt > 1e-9]
    if dt.size == 0:
        return 0.0
    return float(1.0 / np.median(dt))


def accel_proxy_speed(t: np.ndarray, speed_r: np.ndarray) -> np.ndarray:
    """Simple Δspeed/Δt aligned to each row (first sample padded with 0)."""
    t = np.asarray(t, dtype=float)
    sp = np.asarray(speed_r, dtype=float)
    accel = np.zeros_like(sp, dtype=float)
    if sp.size < 2:
        return accel
    dt = np.diff(t)
    dsp = np.diff(sp)
    safe = dt > 1e-12
    a = np.zeros_like(dsp, dtype=float)
    a[safe] = dsp[safe] / dt[safe]
    accel[1:] = a
    return accel


def _rolling_mean_std(y: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
    """Centered rolling mean and std with a fixed odd window (samples)."""
    win = max(3, int(win))
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=np.float64) / float(win)
    m = np.convolve(y, k, mode="same")
    m2 = np.convolve(y * y, k, mode="same")
    var = np.clip(m2 - m * m, 0.0, None)
    s = np.sqrt(var)
    return m, s


def figure_motion_profile(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (14.0, 4.2),
    *,
    display_window_s: float = 5.0,
) -> plt.Figure:
    """
    [Wireframe §2] Acceleration proxy + angular velocity (shared time axis).

    Plots a **5 s rolling mean ±1σ band** for readability on long high-rate clips.
    Heuristic detectors still consume the full-resolution table elsewhere.
    """
    _style()
    fig, ax1 = plt.subplots(figsize=figsize)
    t = df["t"].to_numpy(dtype=float)
    accel = accel_proxy_speed(t, df["speed_r"].to_numpy(dtype=float))
    omega = df["omega"].to_numpy(dtype=float)

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 1e-12)]
    med_dt = float(np.median(dt)) if dt.size else 0.01
    n_win = max(7, int(round(display_window_s / max(med_dt, 1e-9))))

    am, asd = _rolling_mean_std(accel, n_win)
    wm, wsd = _rolling_mean_std(omega, n_win)

    # Rolling stats still have one value per row; decimate vertices for Agg/Matplotlib limits.
    t_p, am_p, asd_p, wm_p, wsd_p = _decimate_uniform(10_000, t, am, asd, wm, wsd)

    ax1.fill_between(t_p, am_p - asd_p, am_p + asd_p, color="#0f766e", alpha=0.22, linewidth=0.0, label="Accel ±1σ (display)")
    ax1.plot(
        t_p,
        am_p,
        color="#0f766e",
        linewidth=1.45,
        label=f"Accel proxy — {display_window_s:.0f}s rolling mean (display)",
    )
    ax1.set_xlabel("Time (s from clip start)")
    ax1.set_ylabel("Accel proxy", color="#0f766e")
    ax1.tick_params(axis="y", labelcolor="#0f766e")

    ax2 = ax1.twinx()
    ax2.fill_between(t_p, wm_p - wsd_p, wm_p + wsd_p, color="#7c3aed", alpha=0.16, linewidth=0.0, label="ω ±1σ (display)")
    ax2.plot(
        t_p,
        wm_p,
        color="#7c3aed",
        linewidth=1.15,
        alpha=0.95,
        label=f"ω — {display_window_s:.0f}s rolling mean (display)",
    )
    ax2.set_ylabel("Angular velocity ω (rad/s)", color="#7c3aed")
    ax2.tick_params(axis="y", labelcolor="#7c3aed")

    ax1.set_title("Motion profile: rolling mean ±1σ (display only; detectors use full-rate table)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def figure_safety_timeline(
    df: pd.DataFrame,
    events: list[SafetyEvent],
    speed_label: str,
    figsize: tuple[float, float] = (14.0, 3.8),
) -> plt.Figure:
    """[Wireframe §3] Speed trace with colored bands by heuristic category."""
    _style()
    fig, ax = plt.subplots(figsize=figsize)
    t = df["t"].to_numpy(dtype=float)
    sp = df["speed_r"].to_numpy(dtype=float)
    # Draw spans behind the speed line so bands stay visible at alpha < 1.
    for e in events:
        c = CATEGORY_COLORS.get(e.category, "#94a3b8")
        ax.axvspan(e.t_start, e.t_end, color=c, alpha=0.38, zorder=1)
    t_d, sp_d = _decimate_uniform(12_000, t, sp)
    ax.plot(t_d, sp_d, color="#0f172a", linewidth=1.6, label=speed_label, zorder=3)

    ev_sorted = sorted(events, key=lambda e: e.t_start)
    if sp.size:
        ymax = float(np.nanmax(sp))
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = 1.0
    else:
        ymax = 1.0
    ymax *= 1.08
    for i, e in enumerate(ev_sorted, start=1):
        xm = 0.5 * (float(e.t_start) + float(e.t_end))
        ax.text(
            xm,
            ymax * 0.93,
            str(i),
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="#f8fafc",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="#0f172a",
                edgecolor="white",
                linewidth=0.35,
                alpha=0.9,
            ),
            zorder=6,
        )

    # Legend for categories (dedupe)
    seen: set[str] = set()
    for cat in DEFAULT_CATEGORY_ORDER:
        if cat in seen:
            continue
        seen.add(cat)
        ax.plot([], [], color=CATEGORY_COLORS.get(cat, "#94a3b8"), linewidth=6, alpha=0.45, label=cat.replace("_", " "))
    if not events:
        _imu = str(df.attrs.get("imu_fallback", "")).lower() == "true"
        _empty_msg = "No heuristic windows in this clip (thresholds not crossed)."
        if _imu:
            _empty_msg += (
                "\nShort / row-capped IMU clips use a tighter detector calibration so "
                "bands can appear when motion warrants it (full bags keep the relaxed set)."
            )
        ax.text(
            0.5,
            0.92,
            _empty_msg,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="#334155",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fef9c3", edgecolor="#eab308", alpha=0.9),
            zorder=5,
        )
    ax.set_title("Safety-style event timeline (colored bands = heuristic category)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (arb.)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25, zorder=0)
    fig.tight_layout()
    return fig


def figure_events_by_category(counts: dict[str, int], figsize: tuple[float, float] = (6.0, 4.2)) -> plt.Figure:
    """[Wireframe §4] Bar chart of event counts by category."""
    _style()
    labels = [k.replace("_", " ") for k in DEFAULT_CATEGORY_ORDER]
    vals = [int(counts.get(k, 0)) for k in DEFAULT_CATEGORY_ORDER]
    colors = [CATEGORY_COLORS.get(k, "#94a3b8") for k in DEFAULT_CATEGORY_ORDER]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title("Events by category")
    ax.set_ylabel("Count (merged windows)")
    ax.tick_params(axis="x", rotation=20)
    top = max(vals) if vals else 0
    ax.set_ylim(0.0, float(max(4, int(top * 1.25) + 1)))
    if top == 0:
        ax.text(
            0.5,
            0.55,
            "Zero merged windows\nin this clip",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#475569",
        )
    fig.tight_layout()
    return fig


def figure_benchmark_speedup(
    rows: int,
    slow_s: float,
    fast_s: float,
    figsize: tuple[float, float] = (6.0, 4.2),
) -> plt.Figure:
    """[Wireframe §6] Simple runtime comparison (same synthetic scale as benchmark_events)."""
    _style()
    fig, ax = plt.subplots(figsize=figsize)
    names = ["Slow (loops)", "Fast (vectorized)"]
    times = [max(slow_s, 1e-9), max(fast_s, 1e-9)]
    ax.bar(names, times, color=["#94a3b8", "#16a34a"])
    ax.set_ylabel("Seconds (log scale)")
    ttl = "NumPy / vectorization story"
    if rows:
        ttl += f" (n={rows:,})"
    ax.set_title(ttl)
    ax.set_yscale("log")
    lo = min(times) * 0.65
    hi = max(times) * 1.35
    ax.set_ylim(lo, hi)
    ax.grid(True, axis="y", which="both", alpha=0.25)
    # Annotate exact timings so the fast bar is readable even when tiny.
    for i, (name, sec) in enumerate(zip(names, [slow_s, fast_s])):
        ax.annotate(
            f"{sec:.4f}s",
            xy=(i, max(sec, 1e-9)),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#0f172a",
        )
    fig.tight_layout()
    return fig


def figure_robot_truth(
    df: pd.DataFrame,
    divs: pd.DataFrame,
    speed_primary_label: str,
    figsize: tuple[float, float] = (14.0, 5.0),
    title: str | None = None,
) -> plt.Figure:
    """[Wireframe §5] Reported/proxy vs motion-derived speed + mismatch shading."""
    _style()
    fig, ax = plt.subplots(figsize=figsize)
    t = df["t"].to_numpy(dtype=float)
    sr = df["speed_r"].to_numpy(dtype=float)
    sd = df["speed_d"].to_numpy(dtype=float)
    t_p, sd_p, sr_p = _decimate_uniform(12_000, t, sd, sr)
    # When curves coincide, a dashed overlay still reads as "two traces" in screenshots.
    ax.plot(t_p, sd_p, label="Motion-derived speed (Δposition/Δt)", color="#a855f7", linewidth=2.4, alpha=0.55, zorder=2)
    ax.plot(
        t_p,
        sr_p,
        label=speed_primary_label,
        color="#1d4ed8",
        linewidth=1.8,
        linestyle="--",
        zorder=3,
    )
    if len(divs):
        for _, r in divs.iterrows():
            ax.axvspan(float(r.t_start), float(r.t_end), color="#f97316", alpha=0.22)
    ax.set_xlabel("Time (s from clip start)")
    ax.set_ylabel("Speed (arb.)")
    ax.set_title(title or "Robot-truth divergence (orange = sustained mismatch)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def figure_sensor_integrity(df: pd.DataFrame, figsize: tuple[float, float] = (14.0, 5.5)) -> plt.Figure:
    """
    IMU-only / single-table integrity: timestamp spacing + instantaneous sample rate.

    This replaces a tautological 'two speed traces' chart when there is no second motion source.
    """
    _style()
    stats = compute_timing_stats(df)
    t = df["t"].to_numpy(dtype=float)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    if t.size < 2:
        ax0.text(0.5, 0.5, "Not enough samples for timing diagnostics", ha="center", va="center", transform=ax0.transAxes)
        fig.tight_layout()
        return fig

    dt = np.diff(t)
    t_r = t[1:]
    gap_s = stats["gap_thresh_ms"] / 1000.0
    is_gap = dt > gap_s
    inst_hz = 1.0 / np.maximum(dt, 1e-12)

    tr_d, dtm_d = _decimate_uniform(12_000, t_r, dt * 1000.0)
    tr_h, hz_d = _decimate_uniform(12_000, t_r, inst_hz)

    ax0.plot(tr_d, dtm_d, color="#334155", lw=0.85, label="Δt (sample spacing)")
    ax0.axhline(stats["gap_thresh_ms"], color="#dc2626", ls="--", lw=1.05, label=f"Gap threshold ({stats['gap_thresh_ms']:.1f} ms)")
    if np.any(is_gap):
        ax0.scatter(t_r[is_gap], dt[is_gap] * 1000.0, color="#dc2626", s=26, zorder=5, label="Spacing spike / gap")
    ax0.set_ylabel("Δt (ms)")
    ax0.set_title(
        "Sensor timing & data integrity "
        f"(median Δt={stats['median_dt_ms']:.2f} ms, p99={stats['p99_dt_ms']:.1f} ms, "
        f"spikes>{stats['gap_thresh_ms']:.1f} ms: {int(stats['n_gaps'])})"
    )
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.25)

    ax1.plot(tr_h, hz_d, color="#0f766e", lw=0.9)
    ax1.set_ylabel("Instantaneous rate (Hz)")
    ax1.set_xlabel("Time (s from clip start)")
    ax1.set_ylim(0.0, 500.0)
    ax1.set_title(
        f"Sample-rate jitter (median ≈ {stats['median_hz']:.2f} Hz)\n"
        "Y-axis clipped at 500 Hz; sub-millisecond sample pairs produce numerical artifacts."
    )
    ax1.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def save_fig(fig: plt.Figure, path: Path, dpi: int = 140) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def export_pipeline_pngs(
    df: pd.DataFrame,
    events: list[SafetyEvent],
    divs: pd.DataFrame,
    out_dir: Path,
    *,
    speed_primary_label: str,
    robot_truth_title: str | None,
    benchmark_rows: int,
    benchmark_slow_s: float,
    benchmark_fast_s: float,
    imu_fallback: bool = False,
) -> None:
    """Write chart assets matching the lab file plan."""
    out_dir.mkdir(parents=True, exist_ok=True)
    save_fig(figure_motion_profile(df), out_dir / "chart_motion_profile.png")
    save_fig(figure_safety_timeline(df, events, speed_primary_label), out_dir / "chart_safety_events.png")
    if imu_fallback:
        save_fig(figure_sensor_integrity(df), out_dir / "chart_sensor_integrity.png")
    else:
        save_fig(figure_robot_truth(df, divs, speed_primary_label, title=robot_truth_title), out_dir / "chart_robot_truth.png")
    # Extra chart from wireframe (events breakdown); useful for decks.
    counts = {k: 0 for k in DEFAULT_CATEGORY_ORDER}
    for e in events:
        counts[e.category] = counts.get(e.category, 0) + 1
    save_fig(figure_events_by_category(counts), out_dir / "chart_events_by_category.png")
    save_fig(figure_benchmark_speedup(benchmark_rows, benchmark_slow_s, benchmark_fast_s), out_dir / "chart_benchmark.png")
