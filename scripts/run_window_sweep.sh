#!/usr/bin/env bash
# Single-window locust run for the Phase 2 Opt 3 batching curve.
#
# The container must already be running at <target-url> with
# BATCHING_WINDOW_MS=<window-ms> set in its environment. This script just
# drives a closed-loop locust run at fixed concurrency and labels the CSV
# by window — the caller restarts the container between invocations.
#
# Usage:
#   scripts/run_window_sweep.sh <target-url> <window-ms-label>
#
# e.g. (window curve at saturation concurrency):
#   LOCUST_USERS=10 scripts/run_window_sweep.sh http://ec2-host:8000 0
#   LOCUST_USERS=10 scripts/run_window_sweep.sh http://ec2-host:8000 2
#   LOCUST_USERS=10 scripts/run_window_sweep.sh http://ec2-host:8000 5
#   LOCUST_USERS=10 scripts/run_window_sweep.sh http://ec2-host:8000 10
#   LOCUST_USERS=10 scripts/run_window_sweep.sh http://ec2-host:8000 20
#   uv run python scripts/summarize_window_sweep.py
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <target-url> <window-ms-label>" >&2
  exit 2
fi

TARGET="$1"
WINDOW="$2"
USERS="${LOCUST_USERS:-10}"
RUNTIME="${LOCUST_RUNTIME:-60s}"

cd "$(dirname "$0")/.."
mkdir -p docs/locust/batching

# Reachability + sanity: the target must report the expected batching window
# so a botched container restart doesn't silently corrupt the curve.
HEALTH=$(python3 -c "
import json, urllib.request
with urllib.request.urlopen('${TARGET}/health', timeout=5) as r:
    print(json.dumps(json.loads(r.read())))
") || { echo "FATAL: ${TARGET}/health unreachable" >&2; exit 3; }
echo "health: ${HEALTH}"

echo "=== window=${WINDOW}ms users=${USERS} runtime=${RUNTIME} ==="
uv run locust \
  -f locustfile.py \
  --headless \
  -u "${USERS}" \
  -r "${USERS}" \
  --run-time "${RUNTIME}" \
  -H "${TARGET}" \
  --only-summary \
  --csv "docs/locust/batching/w${WINDOW}"

echo "saved docs/locust/batching/w${WINDOW}_stats.csv"
