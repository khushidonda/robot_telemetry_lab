#!/usr/bin/env bash
# Downloads the public Google Cartographer 2D backpack ROS 1 bag (~470 MB).
# Parser in this repo: rosbags (NOT bagpy).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/data"

DEST="$ROOT/data/cartographer_paper_deutsches_museum.bag"
URL="https://storage.googleapis.com/cartographer-public-data/bags/backpack_2d/cartographer_paper_deutsches_museum.bag"

echo "Downloading to: $DEST"
curl -L --fail --retry 3 --retry-delay 2 -o "$DEST" "$URL"
echo "Done. Run: cd \"$ROOT\" && source .venv/bin/activate && python src/io_data.py"
