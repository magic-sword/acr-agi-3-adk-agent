#!/usr/bin/env bash
set -euo pipefail
out=outputs/explanation-xai-20260928
restore() { docker compose start vlm; }
trap restore EXIT
docker compose stop vlm
docker compose run --rm --no-deps -e TORCHINDUCTOR_CACHE_DIR=/tmp/torch-explanation-cache -e HF_HUB_DISABLE_PROGRESS_BARS=1 dev \
 python scripts/benchmark_explanation_xai.py run --output "$out" > "$out/run.log" 2>&1
