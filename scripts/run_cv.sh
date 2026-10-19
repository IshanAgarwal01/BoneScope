#!/usr/bin/env bash
# Train all 5 outer folds for one config, then aggregate.
# Usage: bash scripts/run_cv.sh configs/convnext_tiny.yaml [extra --set overrides...]
#   e.g. bash scripts/run_cv.sh configs/convnext_tiny.yaml data.root=/content/data/raw
# Set RUNS_DIR if you changed output.dir (the aggregator reads from it).
# Folds that already have metrics.json are skipped, so a disconnected Colab can resume.
set -euo pipefail
CONFIG="$1"; shift || true
FOLDS="${FOLDS:-0 1 2 3 4}"
for k in $FOLDS; do
  echo "=== $(basename "$CONFIG") fold $k ==="
  python -m mds.train --config "$CONFIG" --fold "$k" --skip-done --set "$@"
done
python scripts/aggregate_results.py --runs "${RUNS_DIR:-runs}"
