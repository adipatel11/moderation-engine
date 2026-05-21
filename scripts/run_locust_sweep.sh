#!/usr/bin/env bash
# Locked Phase 1 concurrency sweep: 1, 5, 10, 25, 50, 100 users × 60 s each.
# Writes per-level CSVs into docs/locust/ then emits a summary markdown table.
#
# Usage:
#   scripts/run_locust_sweep.sh http://<ec2-public-dns>:8000
#
# Honors LOCUST_RUNTIME (default 60s) and LOCUST_USERS (default the standard sweep).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <target-url>  e.g. http://ec2-1-2-3-4.compute-1.amazonaws.com:8000" >&2
  exit 2
fi

TARGET="$1"
RUNTIME="${LOCUST_RUNTIME:-60s}"
USERS="${LOCUST_USERS:-1 5 10 25 50 100}"

cd "$(dirname "$0")/.."
mkdir -p docs/locust

# Quick reachability check before burning a 6-level sweep.
# Use python (not curl) so the same script runs under restrictive harnesses.
if ! python3 -c "
import json, sys, urllib.request
with urllib.request.urlopen('${TARGET}/health', timeout=5) as r:
    body = json.loads(r.read())
sys.exit(0 if body.get('status') == 'ok' and body.get('model_loaded') else 1)
" >/dev/null 2>&1; then
  echo "FATAL: ${TARGET}/health did not respond OK" >&2
  exit 3
fi

for u in $USERS; do
  echo "=== sweep: users=${u} runtime=${RUNTIME} ==="
  uv run locust \
    -f locustfile.py \
    --headless \
    -u "${u}" \
    -r "${u}" \
    --run-time "${RUNTIME}" \
    -H "${TARGET}" \
    --only-summary \
    --csv "docs/locust/u${u}"
done

echo "=== summarizing ==="
uv run python scripts/summarize_locust.py
