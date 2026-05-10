"""
benchmark_events.py
=======================
Times "slow loop detectors" vs "fast vectorized detectors" on a larger synthetic table.

OUTPUT:
  Writes `outputs/benchmark.txt` (+ `benchmark.json` when run via `analyzer.py`).

HOW TO RUN:
  cd ~/Desktop/robot_telemetry_lab
  source .venv/bin/activate
  python src/benchmark_events.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detectors_slow import all_events_slow  # noqa: E402
from safety_categories import run_all_detectors  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs"


def _big_synthetic(n: int = 120_000) -> pd.DataFrame:
    """Big table so loop vs vector timing differences are visible."""
    t = np.linspace(0.0, 600.0, n)
    rng = np.random.default_rng(0)
    x = np.cumsum(0.01 * rng.standard_normal(n))
    y = np.cumsum(0.01 * rng.standard_normal(n))
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    sp = np.hypot(vx, vy)
    omega = np.gradient(np.arctan2(vy, vx + 1e-9), t)
    return pd.DataFrame({"t": t, "x": x, "y": y, "vx_r": vx, "vy_r": vy, "omega": omega, "speed_r": sp})


def run_synthetic_benchmark(n: int = 120_000) -> dict:
    """
    Runs the benchmark on a synthetic table and returns text + structured stats.

    Used by `analyzer.py` (writes JSON for the dashboard) and can be reused elsewhere.
    """
    df = _big_synthetic(n)

    start = time.perf_counter()
    slow_events = all_events_slow(df)
    slow_time = time.perf_counter() - start

    start = time.perf_counter()
    fast_events = run_all_detectors(df)
    fast_time = time.perf_counter() - start

    speedup_pct = 0.0 if slow_time <= 0 else (1.0 - (fast_time / slow_time)) * 100.0

    payload = {
        "rows": int(len(df)),
        "slow_seconds": float(slow_time),
        "fast_seconds": float(fast_time),
        "speedup_pct_vs_slow": float(speedup_pct),
        "slow_events": int(len(slow_events)),
        "fast_events": int(len(fast_events)),
    }

    text = "\n".join(
        [
            "Benchmark: loop detectors vs vectorized detectors (demo)",
            f"rows: {payload['rows']:,}",
            f"slow (Python loops): {payload['slow_seconds']:.4f}s | events: {payload['slow_events']}",
            f"fast (vectorized):   {payload['fast_seconds']:.4f}s | events: {payload['fast_events']}",
            f"speedup (% faster than slow): {payload['speedup_pct_vs_slow']:.1f}%",
            "",
            "Notes:",
            "- Event counts can differ slightly because grouping rules differ.",
            "- This is about engineering approach, not certified safety performance.",
        ]
    ) + "\n"

    return {"text": text, "json": payload}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = run_synthetic_benchmark()
    print(result["text"])
    (OUT_DIR / "benchmark.txt").write_text(result["text"], encoding="utf-8")
    (OUT_DIR / "benchmark.json").write_text(json.dumps(result["json"], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
