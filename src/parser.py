"""
parser.py
=========
Public entry for **loading a motion telemetry table** from disk (CSV, ROS bags, MCAP, …).

Implementation lives in `io_data.py` (rosbags-backed paths, caps, fallbacks). This module
exists so the repo layout matches the lab file plan (`parser` = ingest).
"""

from __future__ import annotations

from io_data import load_telemetry

__all__ = ["load_telemetry"]
