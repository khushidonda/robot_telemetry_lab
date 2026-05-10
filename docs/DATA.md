# Where to get **real** bag data (ROS 1 `.bag` or ROS 2 bags)

Your dashboard code tries to extract **`nav_msgs/Odometry`-shaped** messages
(topics whose messages have `pose.pose.position` and `twist.twist.*`).

We parse bags using **`rosbags`** (works for many **ROS 1 `*.bag`** files and **ROS 2** recordings).

---

## Option Cartographer (recommended public “real” dataset — ROS 1)

This is the plan you confirmed with Cartographer’s public backpack dataset:

- **Format:** ROS 1 `.bag` (not ROS 2 `metadata.yaml + .db3`)
- **Topic we prefer:** `/odom`
- **Download (~470 MB):**

```bash
cd ~/Desktop/robot_telemetry_lab
bash scripts/download_cartographer_bag.sh
```

Or manually:

```bash
mkdir -p ~/Desktop/robot_telemetry_lab/data
curl -L -o ~/Desktop/robot_telemetry_lab/data/cartographer_paper_deutsches_museum.bag \
  https://storage.googleapis.com/cartographer-public-data/bags/backpack_2d/cartographer_paper_deutsches_museum.bag
```

Then verify:

```bash
cd ~/Desktop/robot_telemetry_lab
source .venv/bin/activate
python src/io_data.py
```

You want `source:` to show `rosbag:cartographer_paper_deutsches_museum.bag` (not `synthetic_mission`).

### Cartographer `cartographer_paper_deutsches_museum.bag` note

This public bag often **does not include** `nav_msgs/Odometry` on `/odom`.

The loader now:

1. Tries **strict** `nav_msgs/Odometry` topics only (never “random first topics”).
2. If none exist, falls back to **`geometry_msgs/PoseStamped`** (common for SLAM demos) and **derives**
   translational speed + yaw rate for the dashboard.
3. If that still doesn’t exist (your Cartographer bag is **IMU + laser only**), it falls back to **`sensor_msgs/Imu`**
   (prefers `/imu`) and builds a **demo-only proxy path** by integrating linear acceleration after a crude bias removal.

**Honesty for interviews:** the IMU path is **not** “true robot odometry,” it’s a visualization proxy so the dashboard can run on this dataset.

### Why `python src/io_data.py` can feel slow

That Cartographer bag is large. Reading **every** `/odom` message can take minutes.

By default we **cap** how many odometry rows we load (fast demo). You’ll see progress lines like:

`…loaded 20,000 / ~80,000 odometry rows`

To load more (slower, more complete):

```bash
export ROBOT_TELEMETRY_MAX_ODOM=250000
python src/io_data.py
```

**Interview honesty:** this bag is **ROS 1**; your pipeline is “read bag → extract `/odom` → KPIs/charts,” which is the same *shape* as ROS 2 fleet workflows even if the file format differs.

---

## Option A (best story): record your own 30–60 seconds

This is the most “real” data because it’s truly your recording.

Official tutorial (ROS 2):

- [Recording and playing back data (Humble)](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html)

Typical flow (on any machine with ROS 2 installed):

1. Run a robot/sim that publishes **`/odom`** (or record whatever odometry topic your sim uses).
2. Record:

```bash
ros2 bag record /odom
```

3. Stop recording. ROS creates a **folder** like `rosbag2_YYYY_MM_DD-HH_MM_SS/`
   containing `metadata.yaml` + `.db3` (or `.mcap` depending on settings).

4. Copy that **entire folder** into:

`~/Desktop/robot_telemetry_lab/data/<some_short_name>/`

Example:

`~/Desktop/robot_telemetry_lab/data/my_first_bag/metadata.yaml`

---

## Option B: download a published bag (often a folder or zip)

Many datasets ship as:

- a **folder** with `metadata.yaml`, or
- a **zip** you unzip into `data/…`

### Example dataset pages (may require login / may be large)

- IEEE DataPort hosts ROS 2 recordings in some robotics uploads (often MCAP). Example listing page:
  - `https://ieee-dataport.org/documents/rosbag220240719110137-dataset`

**Rule of thumb:** pick something **small** first (seconds to a few minutes), not a multi‑GB autonomy dataset.

---

## Option C: MCAP files (still “real robot recording”)

If you download a **`.mcap`** recording that includes odometry topics, place it here:

`~/Desktop/robot_telemetry_lab/data/<name>.mcap`

Our loader tries `.mcap` after bag directories.

---

## How to place files in *this* project

Preferred layout:

```text
~/Desktop/robot_telemetry_lab/data/
  my_public_bag/
    metadata.yaml
    my_public_bag_0.db3
```

Then restart:

```bash
streamlit run app.py
```

The UI should show `source: rosbag:my_public_bag` (or similar).

---

## If it falls back to “synthetic_mission”

That means none of these worked yet:

- no `telemetry.csv`
- no `data/*/metadata.yaml`
- no readable `*.mcap` with odometry-shaped messages
- no readable `*.db3`

Next debugging step (quick):

```bash
cd ~/Desktop/robot_telemetry_lab
source .venv/bin/activate
python src/io_data.py
```

If it prints `synthetic_mission`, your bag isn’t being picked up or doesn’t contain odometry messages we can parse.
