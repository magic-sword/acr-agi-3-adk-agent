#!/usr/bin/env bash
set -euo pipefail
out=outputs/observe-compare-20260928
python3 scripts/benchmark_observe_compare.py run --output "$out" --backend llama > "$out/llama-run.log" 2>&1
restore() { docker compose start vlm; }
trap restore EXIT
docker compose stop vlm
docker compose run --rm --no-deps -e TORCHINDUCTOR_CACHE_DIR=/tmp/torch-observe-cache -e HF_HUB_DISABLE_PROGRESS_BARS=1 dev \
 python scripts/benchmark_observe_compare.py run --output "$out" --backend official > "$out/official-run.log" 2>&1
