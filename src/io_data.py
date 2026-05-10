"""
io_data.py
==========
WHAT THIS FILE IS:
  "How we get a table of numbers to analyze."

ORDER OF TRYING THINGS:
  1) If data/telemetry.csv exists -> read it
  2) Else if data/<any_folder>/metadata.yaml exists -> treat that folder as a ROS 2 bag
  3) Else if data/*.mcap exists -> read as MCAP (common for modern ROS 2 recordings)
  4) Else if data/*.bag exists -> read as ROS 1 bag (ex: Google Cartographer public dataset)
  5) Else if data/*.db3 exists -> try reading it as a bag path (works for some layouts)
  6) Else -> generate a built-in synthetic mission (always works offline)

IMPORTANT (real rosbags):
  A "ROS 2 bag" is usually a DIRECTORY like:
    my_run/
      metadata.yaml
      my_run_0.db3

  If you only download the .db3 without metadata.yaml, tools may fail — download the
  whole folder from wherever you got the dataset.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

# REPO_ROOT is the folder that contains data/, src/, app.py
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

def max_bag_messages() -> int:
    """
    Cap per-topic message reads so dev laptops stay responsive.

    Defaults high enough for ~30+ minute Cartographer IMU streams (~170 Hz → ~300k rows).
    Preview / dev: `ROBOT_TELEMETRY_MAX_ODOM=3000`
    Full read: `ROBOT_TELEMETRY_MAX_ODOM=0` (or `full` / `max`)
    """
    raw = os.environ.get("ROBOT_TELEMETRY_MAX_ODOM", "500000").strip().lower()
    if raw in ("0", "full", "max", "none", "unlimited"):
        return 1_000_000_000
    try:
        return max(1, int(raw))
    except ValueError:
        return 500_000


def _synthetic_mission(n: int = 2400, seed: int = 7) -> pd.DataFrame:
    """
    Creates fake-but-plausible telemetry with a few injected "events"
    so the charts are never empty on day 1.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 120.0, n)

    x = np.cumsum(0.04 * np.sin(0.07 * t) + 0.01 * rng.standard_normal(n))
    y = np.cumsum(0.03 * np.cos(0.05 * t) + 0.01 * rng.standard_normal(n))

    vx_d = np.gradient(x, t)
    vy_d = np.gradient(y, t)
    vx_r = vx_d + 0.005 * rng.standard_normal(n)
    vy_r = vy_d + 0.005 * rng.standard_normal(n)
    omega = np.gradient(np.arctan2(vy_r, vx_r), t)
    speed_r = np.hypot(vx_r, vy_r)

    def win(t0, t1):
        return (t >= t0) & (t <= t1)

    omega[win(17.5, 19.0)] += 2.2  # sharp turn
    m = win(43.8, 45.0)
    speed_r[m] *= np.linspace(1.0, 0.05, int(np.sum(m)))  # sudden stop
    speed_r[win(70.0, 74.0)] = 0.01  # stall
    speed_r[win(94.7, 95.3)] += 1.1  # jerk spike
    speed_r[win(29.5, 31.0)] += 0.55  # short mismatch vs motion (robot truth demo)

    df = pd.DataFrame({"t": t, "x": x, "y": y, "vx_r": vx_r, "vy_r": vy_r, "omega": omega, "speed_r": speed_r})
    df.attrs["source"] = "synthetic_mission"
    return df


def _find_ros2_bag_dirs() -> list[Path]:
    """Return candidate bag directories under data/ (folders containing metadata.yaml)."""
    out: list[Path] = []
    if not DATA_DIR.exists():
        return out
    for p in sorted(DATA_DIR.iterdir()):
        if p.is_dir() and (p / "metadata.yaml").exists():
            out.append(p)
    return out


def _typestores_for_path(bagpath: Path):
    """
    ROS 1 bags (*.bag) need ROS1 message definitions.
    ROS 2 bags / mcap / db3 need ROS2 message definitions.
    """
    from rosbags.typesys import Stores, get_typestore  # type: ignore

    stores: list = []
    if bagpath.suffix.lower() == ".bag":
        # Cartographer public bags are often older; try older ROS1 stores first.
        for name in ("ROS1_KINETIC", "ROS1_MELODIC", "ROS1_NOETIC"):
            st = getattr(Stores, name, None)
            if st is not None:
                stores.append(st)
    else:
        for name in ("ROS2_JAZZY", "ROS2_IRON", "ROS2_HUMBLE", "ROS2_FOXY"):
            st = getattr(Stores, name, None)
            if st is not None:
                stores.append(st)

    for st in stores:
        try:
            yield get_typestore(st)  # type: ignore[arg-type]
        except Exception:
            continue

    # Last-resort default (ROS 2 only — do not force ROS2 defs onto ROS1 *.bag)
    if bagpath.suffix.lower() != ".bag":
        try:
            yield get_typestore(Stores.ROS2_HUMBLE)  # type: ignore[arg-type]
        except Exception:
            return


def _pick_nav_odometry_connections(conns) -> list:
    """
    STRICT selection for `nav_msgs/Odometry`.

    IMPORTANT:
      Never fall back to "first N connections". That was incorrectly selecting
      huge laser scan topics (anything alphabetically early), which looks like
      the script is "stuck" forever.
    """
    out: list = []
    for c in conns:
        mt = str(getattr(c, "msgtype", "")).lower()
        # ROS1: nav_msgs/Odometry   ROS2: nav_msgs/msg/Odometry
        if "nav_msgs" in mt and "odometry" in mt:
            out.append(c)

    if not out:
        return []

    exact = [c for c in out if getattr(c, "topic", "") == "/odom"]
    return exact or out


def _pick_imu_connections(conns) -> list:
    """
    Pick `sensor_msgs/Imu` topics.

    Some public Cartographer bags only contain laser + IMU (no Odometry / PoseStamped).
    For a demo dashboard, we can build a *proxy* trajectory by integrating linear acceleration
    (after a crude bias removal). This is NOT ground-truth odometry.
    """
    out: list = []
    for c in conns:
        mt = str(getattr(c, "msgtype", "")).lower()
        mt_compact = mt.replace("/", "").replace("_", "")
        if "sensor_msgs" in mt and "imu" in mt_compact:
            out.append(c)

    if not out:
        return []

    exact = [c for c in out if getattr(c, "topic", "") == "/imu"]
    return exact or out


def _pick_pose_stamped_connections(conns) -> list:
    """Pick `geometry_msgs/PoseStamped` topics as a Cartographer-friendly fallback."""
    cands = []
    for c in conns:
        mt = str(getattr(c, "msgtype", "")).lower()
        mt_compact = mt.replace("/", "").replace("_", "")
        if "posestamped" in mt_compact:
            cands.append(c)

    if not cands:
        return []

    preferred_topics = {"/pose", "/tracked_pose", "/robot_pose", "/base_link_pose"}
    pref = [c for c in cands if getattr(c, "topic", "") in preferred_topics]
    return pref or cands


def _stamp_to_sec(stamp) -> float:
    """Support both ROS1 (secs/nsecs) and ROS2 (sec/nanosec) stamp objects."""
    if stamp is None:
        return 0.0
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
        return float(stamp.secs) + float(stamp.nsecs) * 1e-9
    return 0.0


def _yaw_from_quat(q) -> float:
    """Yaw (rad) from geometry_msgs/Quaternion (x,y,z,w)."""
    x = float(q.x)
    y = float(q.y)
    z = float(q.z)
    w = float(q.w)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _try_read_any_bag(bagpath: Path) -> pd.DataFrame | None:
    """
    Read Odometry-shaped messages using `rosbags` (ROS1 *.bag and ROS2 bags).

    We extract fields compatible with `nav_msgs/Odometry`:
      pose.pose.position + twist.twist.linear/angular
    """
    try:
        from rosbags.highlevel import AnyReader  # type: ignore
    except Exception:
        return None

    rows: list[tuple[float, float, float, float, float, float, float]] = []
    label = bagpath.name
    last_err: str | None = None
    msg_cap = max_bag_messages()

    print(
        f"\n[io_data] Opening: {bagpath}\n[io_data] Message row cap: {msg_cap:,} "
        f"(set ROBOT_TELEMETRY_MAX_ODOM; use 0 or `full` for effectively unlimited)\n",
        flush=True,
    )

    # Progress cadence: for small caps, print more often so it never looks "hung"
    prog_step = max(250, min(20_000, max(1, msg_cap // 5)))

    for typestore in _typestores_for_path(bagpath):
        rows: list[tuple[float, float, float, float, float, float, float]] = []
        try:
            with AnyReader([bagpath], default_typestore=typestore) as reader:
                conns = list(reader.connections)
                odom_conns = _pick_nav_odometry_connections(conns)

                if odom_conns:
                    topics = sorted({c.topic for c in odom_conns})
                    print(f"[io_data] Using nav_msgs/Odometry topic(s): {topics}\n", flush=True)

                    n_ok = 0
                    for connection, ts, rawdata in reader.messages(connections=odom_conns):
                        msg = reader.deserialize(rawdata, connection.msgtype)

                        try:
                            tsec = float(ts) * 1e-9
                        except Exception:
                            tsec = float(ts)

                        try:
                            px = float(msg.pose.pose.position.x)
                            py = float(msg.pose.pose.position.y)
                            vx = float(msg.twist.twist.linear.x)
                            vy = float(msg.twist.twist.linear.y)
                            wz = float(msg.twist.twist.angular.z)
                            sp = float((vx * vx + vy * vy) ** 0.5)
                            rows.append((tsec, px, py, vx, vy, wz, sp))
                            n_ok += 1
                            if n_ok % prog_step == 0:
                                print(f"…loaded {n_ok:,} / {msg_cap:,} odometry rows", flush=True)
                            if n_ok >= msg_cap:
                                break
                        except Exception:
                            continue

                else:
                    # Many Cartographer demo bags do NOT publish `/odom` at all.
                    # Fallback A: PoseStamped trajectory (derive translational speed + yaw rate).
                    # Fallback B: IMU-only bags (integrate linear acceleration after crude bias removal).
                    pose_conns = _pick_pose_stamped_connections(conns)
                    if pose_conns:
                        topics = sorted({c.topic for c in pose_conns})
                        print(
                            "[io_data] No nav_msgs/Odometry found.\n"
                            f"[io_data] Fallback: geometry_msgs/PoseStamped topic(s): {topics}\n",
                            flush=True,
                        )

                        raw_pose: list[tuple[float, float, float, float]] = []
                        n_ok = 0
                        for connection, ts, rawdata in reader.messages(connections=pose_conns):
                            msg = reader.deserialize(rawdata, connection.msgtype)
                            try:
                                tsec = _stamp_to_sec(getattr(msg, "header", None).stamp)
                                if tsec == 0.0:
                                    tsec = float(ts) * 1e-9
                            except Exception:
                                tsec = float(ts) * 1e-9

                            try:
                                px = float(msg.pose.position.x)
                                py = float(msg.pose.position.y)
                                yaw = _yaw_from_quat(msg.pose.orientation)
                                raw_pose.append((tsec, px, py, yaw))
                                n_ok += 1
                                if n_ok % prog_step == 0:
                                    print(f"…loaded {n_ok:,} / {msg_cap:,} pose rows", flush=True)
                                if n_ok >= msg_cap:
                                    break
                            except Exception:
                                continue

                        if len(raw_pose) >= 5:
                            pdf = pd.DataFrame(raw_pose, columns=["t", "x", "y", "yaw"]).sort_values("t")
                            t = pdf["t"].to_numpy(dtype=float)
                            x = pdf["x"].to_numpy(dtype=float)
                            y = pdf["y"].to_numpy(dtype=float)
                            yaw = np.unwrap(pdf["yaw"].to_numpy(dtype=float))

                            vx = np.gradient(x, t)
                            vy = np.gradient(y, t)
                            omega = np.gradient(yaw, t)
                            speed_r = np.hypot(vx, vy)

                            pdf2 = pd.DataFrame(
                                {
                                    "t": t,
                                    "x": x,
                                    "y": y,
                                    "vx_r": vx,
                                    "vy_r": vy,
                                    "omega": omega,
                                    "speed_r": speed_r,
                                }
                            )
                            # Normalize time so charts read "seconds from clip start" (not Unix epoch).
                            pdf2["t"] = pdf2["t"] - float(pdf2["t"].min())
                            pdf2.attrs["source"] = f"rosbag:{label}"
                            pdf2.attrs["pose_fallback"] = "true"
                            if n_ok >= msg_cap:
                                pdf2.attrs["truncated_odom_rows"] = str(msg_cap)
                            return pdf2

                    imu_conns = _pick_imu_connections(conns)
                    if imu_conns:
                        topics = sorted({c.topic for c in imu_conns})
                        print(
                            "[io_data] No nav_msgs/Odometry + no PoseStamped found.\n"
                            f"[io_data] Fallback: sensor_msgs/Imu topic(s): {topics}\n"
                            "[io_data] NOTE: IMU path is a *demo proxy* (accel integration), not true odometry.\n",
                            flush=True,
                        )

                        raw_imu: list[tuple[float, float, float, float, float]] = []
                        n_ok = 0
                        for connection, ts, rawdata in reader.messages(connections=imu_conns):
                            msg = reader.deserialize(rawdata, connection.msgtype)
                            try:
                                tsec = _stamp_to_sec(getattr(msg, "header", None).stamp)
                                if tsec == 0.0:
                                    tsec = float(ts) * 1e-9
                            except Exception:
                                tsec = float(ts) * 1e-9

                            try:
                                ax = float(msg.linear_acceleration.x)
                                ay = float(msg.linear_acceleration.y)
                                az = float(msg.linear_acceleration.z)
                                wz = float(msg.angular_velocity.z)
                                raw_imu.append((tsec, ax, ay, az, wz))
                                n_ok += 1
                                if n_ok % prog_step == 0:
                                    print(f"…loaded {n_ok:,} / {msg_cap:,} IMU rows", flush=True)
                                if n_ok >= msg_cap:
                                    break
                            except Exception:
                                continue

                        if len(raw_imu) >= 5:
                            imu_df = pd.DataFrame(
                                raw_imu, columns=["t", "ax", "ay", "az", "omega"]
                            ).sort_values("t")
                            t = imu_df["t"].to_numpy(dtype=float)
                            ax = imu_df["ax"].to_numpy(dtype=float)
                            ay = imu_df["ay"].to_numpy(dtype=float)
                            wz = imu_df["omega"].to_numpy(dtype=float)

                            # Crude bias removal using the first chunk (sensor mostly stationary at startup).
                            # Bias removal: use a longer startup slice when available (Cartographer bags
                            # often start near stationary; a tiny window leaves residual bias that integrates
                            # into a fake linear "speed ramp" in plots).
                            n_bias = min(max(200, len(ax) // 20), len(ax))
                            ax0 = ax - float(np.median(ax[:n_bias]))
                            ay0 = ay - float(np.median(ay[:n_bias]))

                            dt = np.diff(t, prepend=t[0])
                            dt = np.where(dt <= 1e-9, np.nan, dt)
                            med_dt = float(np.nanmedian(dt))
                            if not np.isfinite(med_dt) or med_dt <= 0:
                                med_dt = 0.01
                            dt = np.nan_to_num(dt, nan=med_dt)

                            # High-pass accelerations (~few Hz and below) so residual DC/bias does not
                            # integrate into a misleading straight-line "velocity" over minutes.
                            _imu_hp_s = float(os.environ.get("ROBOT_TELEMETRY_IMU_HP_S", "2.5"))
                            win = int(round(_imu_hp_s / med_dt))
                            # Cap at array length; `convolve(..., mode="same")` needs w <= len(a).
                            win = max(5, min(len(ax), win))
                            print(
                                f"[io_data] IMU accel high-pass: ~{_imu_hp_s:.1f}s window "
                                f"({win} samples, median Δt={med_dt:.4f}s)\n",
                                flush=True,
                            )

                            def _moving_mean_same(a: np.ndarray, w: int) -> np.ndarray:
                                w = max(3, min(int(w), len(a)))
                                k = np.ones(w, dtype=np.float64) / float(w)
                                return np.convolve(a, k, mode="same")

                            def _hp_moving_mean(a: np.ndarray, w: int) -> np.ndarray:
                                return a - _moving_mean_same(a, w)

                            ax_hp = _hp_moving_mean(ax0, win)
                            ay_hp = _hp_moving_mean(ay0, win)

                            vx = np.cumsum(ax_hp * dt)
                            vy = np.cumsum(ay_hp * dt)
                            x = np.cumsum(vx * dt)
                            y = np.cumsum(vy * dt)
                            speed_r = np.hypot(vx, vy)

                            pdf_imu = pd.DataFrame(
                                {
                                    "t": t,
                                    "x": x,
                                    "y": y,
                                    "vx_r": vx,
                                    "vy_r": vy,
                                    "omega": wz,
                                    "speed_r": speed_r,
                                }
                            )
                            # Normalize time so charts read "seconds from clip start" (not Unix epoch).
                            pdf_imu["t"] = pdf_imu["t"] - float(pdf_imu["t"].min())
                            pdf_imu.attrs["source"] = f"rosbag:{label}"
                            pdf_imu.attrs["imu_fallback"] = "true"
                            pdf_imu.attrs["imu_accel_hp_s"] = f"{_imu_hp_s:.2f}"
                            if n_ok >= msg_cap:
                                pdf_imu.attrs["truncated_odom_rows"] = str(msg_cap)
                            return pdf_imu

                    preview = sorted({(c.topic, str(c.msgtype)) for c in conns})[:25]
                    print(
                        "[io_data] No nav_msgs/Odometry + no PoseStamped + no Imu found.\n"
                        "[io_data] First topics (topic, msgtype) preview:\n"
                        + "\n".join([f"  - {a} :: {b}" for a, b in preview])
                        + "\n",
                        flush=True,
                    )
                    continue

        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            rows = []
            continue

        if len(rows) >= 5:
            break
        rows = []

    if len(rows) < 5:
        if last_err:
            print("\n[io_data] FAILED to read odometry from bag. Last error:\n" + last_err + "\n", flush=True)
        return None

    df = pd.DataFrame(rows, columns=["t", "x", "y", "vx_r", "vy_r", "omega", "speed_r"])
    df = df.sort_values("t").reset_index(drop=True)
    df["t"] = df["t"] - df["t"].min()
    df.attrs["source"] = f"rosbag:{label}"
    _cap = max_bag_messages()
    if len(rows) >= _cap:
        df.attrs["truncated_odom_rows"] = str(_cap)
    return df


def load_telemetry() -> pd.DataFrame:
    """Main function the app calls."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_DIR / "telemetry.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.attrs["source"] = "telemetry.csv"
        return df

    # Real ROS 2 bags: prefer full bag directories under data/
    for bagdir in _find_ros2_bag_dirs():
        df = _try_read_any_bag(bagdir)
        if df is not None:
            return df

    # MCAP files (single-file recordings)
    for mcap in sorted(DATA_DIR.glob("*.mcap")):
        df = _try_read_any_bag(mcap)
        if df is not None:
            return df

    # ROS 1 bag files (*.bag) — e.g. Google Cartographer public backpack dataset
    for bag in sorted(DATA_DIR.glob("*.bag")):
        df = _try_read_any_bag(bag)
        if df is not None:
            return df

    # Sometimes people drop only *.db3 into data/ — try the file, then its parent folder
    for db3 in sorted(DATA_DIR.glob("*.db3")):
        df = _try_read_any_bag(db3)
        if df is not None:
            return df
        parent = db3.parent
        if (parent / "metadata.yaml").exists():
            df = _try_read_any_bag(parent)
            if df is not None:
                return df

    print(
        "\n[io_data] No readable bag/CSV found under data/ — using synthetic_mission fallback.\n",
        flush=True,
    )
    return _synthetic_mission()


def main() -> None:
    df = load_telemetry()
    print("source:", df.attrs.get("source"))
    print(df.head())


if __name__ == "__main__":
    main()
    sys.exit(0)
