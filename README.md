# Robot telemetry lab (Desktop project)

This folder should live at:

`~/Desktop/robot_telemetry_lab/`

You already installed Python **3.11.x** + packages into `.venv/`.

---

## Step-by-step: run everything (Monday demo order)

### 1) Open Terminal and “go to the project”

```bash
cd ~/Desktop/robot_telemetry_lab
```

### 2) Activate the virtual environment

```bash
source .venv/bin/activate
```

You should see `(.venv)` in your prompt.

### 3) One-shot pipeline (matches the lab file plan: PNGs + CSV + reports)

```bash
python src/analyzer.py
```

This creates under `outputs/`:

- `chart_motion_profile.png`, `chart_safety_events.png`
- `chart_robot_truth.png` **or** `chart_sensor_integrity.png` (IMU-only bags skip the tautological dual-speed chart)
- `chart_events_by_category.png`, `chart_benchmark.png`
- `safety_events.csv`, `summary_report.txt`, `benchmark.txt`, `benchmark.json`

### 4) (Optional) Benchmark or report alone

```bash
python src/benchmark_events.py
python src/report.py
```

### 5) Launch the dashboard in your browser

```bash
streamlit run app.py
```

The dashboard reads **`outputs/benchmark.json`** for the NumPy speedup panel (generate it with step 3).

Streamlit prints a URL like `http://localhost:8501`.

---

## What each folder/file is for (beginner map)

- `app.py` — Streamlit dashboard (wireframe layout: KPIs → motion → timeline → charts → robot-truth → summary)
- `src/parser.py` — **ingest** API (`load_telemetry` → `io_data`)
- `src/io_data.py` — rosbags-backed loaders (CSV / ROS2 dir / MCAP / `.bag` / synthetic)
- `src/analyzer.py` — **offline pipeline**: detectors + `outputs/*.png` + CSV + reports + benchmark JSON
- `src/visualizer.py` — Matplotlib figures + PNG export helpers
- `src/safety_categories.py` — detector classes + fast NumPy logic + category colors
- `src/detectors_slow.py` — slow loop versions for benchmarking only
- `src/benchmark_events.py` — writes `outputs/benchmark.txt` (+ `benchmark.json` when you run `analyzer.py` or `benchmark_events.py`)
- `src/report.py` — writes `outputs/summary_report.txt`
- `src/robot_truth.py` — reported vs motion-derived speed consistency (non-IMU bags)
- `src/sensor_integrity.py` — timestamp spacing / jitter stats (IMU-only integrity story)
- `docs/robot_truth_v0_1.md` — short definitions doc (good to read out loud)

---

## Optional: use real public data (Cartographer ROS 1 `.bag`)

See **`docs/DATA.md`**.

Fast path (downloads ~470 MB):

```bash
cd ~/Desktop/robot_telemetry_lab
bash scripts/download_cartographer_bag.sh
python src/analyzer.py
streamlit run app.py
```

## Optional: ROS 2 bag directory (`metadata.yaml` + `.db3` / `.mcap`)

Put the **whole bag folder** under `~/Desktop/robot_telemetry_lab/data/<name>/` (see `docs/DATA.md`).

## Optional: IMU proxy tuning (Cartographer-style bags)

If the IMU integration path looks too flat or too noisy, adjust the accel high-pass window (seconds of moving-mean removed before integration):

```bash
export ROBOT_TELEMETRY_IMU_HP_S=3.5
streamlit run app.py
```

**`ROBOT_TELEMETRY_MAX_ODOM`:** defaults to **500,000** rows (~30+ minutes @ ~170 Hz). Use `3000` for instant previews, or `0` / `full` for effectively unlimited (slow first load).

---

## GitHub (push when it works locally)

```bash
cd ~/Desktop/robot_telemetry_lab
git init
git add .
git commit -m "Add robot telemetry demo dashboard"
```

Create a **new empty repo** on GitHub (no README), then:

```bash
git remote add origin https://github.com/<you>/<repo>.git
git branch -M main
git push -u origin main
```
