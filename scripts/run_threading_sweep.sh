#!/usr/bin/env bash
# Single-config locust run for the Phase 2 Opt 4 threading-tune curve.
#
# Drives a 60 s closed-loop run at fixed concurrency against the target
# and labels the CSV by the (intra_op, window) tuple. The caller is
# responsible for restarting the container with the right env vars
# between invocations:
#   docker run -e ONNX_INTRA_OP_THREADS=<N> -e BATCHING_WINDOW_MS=<M> ...
#
# Usage:
#   scripts/run_threading_sweep.sh <target-url> <label>
#
# e.g.
#   LOCUST_USERS=10 scripts/run_threading_sweep.sh http://ec2-host:8000 intra2_w0
#   LOCUST_USERS=10 scripts/run_threading_sweep.sh http://ec2-host:8000 intra1_w0
#   LOCUST_USERS=10 scripts/run_threading_sweep.sh http://ec2-host:8000 intra1_w5
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <target-url> <label>" >&2
  exit 2
fi

TARGET="$1"
LABEL="$2"
USERS="${LOCUST_USERS:-10}"
RUNTIME="${LOCUST_RUNTIME:-60s}"

cd "$(dirname "$0")/.."
mkdir -p docs/locust/threading

HEALTH=$(python3 -c "
import json, urllib.request
with urllib.request.urlopen('${TARGET}/health', timeout=5) as r:
    print(json.dumps(json.loads(r.read())))
") || { echo "FATAL: ${TARGET}/health unreachable" >&2; exit 3; }
echo "health: ${HEALTH}"

echo "=== ${LABEL} users=${USERS} runtime=${RUNTIME} ==="
uv run locust \
  -f locustfile.py \
  --headless \
  -u "${USERS}" \
  -r "${USERS}" \
  --run-time "${RUNTIME}" \
  -H "${TARGET}" \
  --only-summary \
  --csv "docs/locust/threading/${LABEL}"

echo "saved docs/locust/threading/${LABEL}_stats.csv"
