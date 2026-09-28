#!/usr/bin/env bash
set -euo pipefail
out=outputs/relation-reference-probe-20260928
python3 scripts/probe_relation_fields.py run --output "$out" --backend llama > "$out/llama-run.log" 2>&1
restore() { docker compose start vlm; }
trap restore EXIT
docker compose stop vlm
docker compose run --rm --no-deps -e TORCHINDUCTOR_CACHE_DIR=/tmp/torch-relation-cache -e HF_HUB_DISABLE_PROGRESS_BARS=1 dev \
 python scripts/probe_relation_fields.py run --output "$out" --backend official > "$out/official-run.log" 2>&1
