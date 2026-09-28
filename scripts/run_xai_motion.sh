#!/usr/bin/env bash
set -euo pipefail
out=outputs/xai-motion-20260928
for attempt in $(seq 1 180); do
    if test -f "$out/server-after.json"; then break; fi
    sleep 5
done
test -f "$out/server-after.json"
restore() { docker compose start vlm; }
trap restore EXIT
docker compose stop vlm
docker compose run --rm --no-deps -e TORCHINDUCTOR_CACHE_DIR=/tmp/torch-xai-cache -e HF_HUB_DISABLE_PROGRESS_BARS=1 dev \
    python scripts/benchmark_xai_motion.py official --output "$out" > "$out/official-run.log" 2>&1
