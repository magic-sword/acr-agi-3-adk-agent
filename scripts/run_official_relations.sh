#!/usr/bin/env bash
# Temporary GPU ownership only; always restore the original configured VLM service.
set -euo pipefail
out=outputs/official-relations-20260928
for attempt in $(seq 1 180); do
    if test -f "$out/server-after.json"; then break; fi
    sleep 5
done
test -f "$out/server-after.json"
python3 - <<'PY'
import json
from pathlib import Path
rows=[json.loads(s) for s in Path('outputs/official-relations-20260928/llama-measurements.jsonl').read_text().splitlines()]
assert len(rows)==68 and sum(not r['warmup'] for r in rows)==66
PY
restore() {
    docker stop arc-qwen-frame-audit >/dev/null 2>&1 || true
    docker compose start vlm
}
trap restore EXIT
docker compose stop vlm
if ! test -f "$out/native-frame-audit-response.json"; then
docker compose run --rm --no-deps --name arc-qwen-frame-audit -p 127.0.0.1:8081:8080 vlm \
    --model /models/Qwen3VL-4B-Instruct-Q4_K_M.gguf \
    --mmproj /models/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf \
    --alias qwen3-vl-4b-instruct --jinja --host 0.0.0.0 --port 8080 \
    --n-gpu-layers 99 --ctx-size 16384 --parallel 1 -lv 5 \
    > "$out/native-frame-audit.log" 2>&1 &
audit_pid=$!
python3 - <<'PY'
import json,time
from pathlib import Path
from scripts.benchmark_attention_selection import http
out=Path('outputs/official-relations-20260928')
for _ in range(60):
    try:
        http('http://127.0.0.1:8081/health',timeout=2)
        break
    except Exception: time.sleep(1)
else: raise RuntimeError('Diagnostic server did not start')
j=next(j for j in json.loads((out/'jobs.json').read_text()) if j['arm']=='llama_base')
r=http('http://127.0.0.1:8081/v1/chat/completions',j['payload'],120)
(out/'native-frame-audit-response.json').write_text(json.dumps(r,indent=2)+'\n')
PY
docker stop arc-qwen-frame-audit
wait "$audit_pid"
fi
docker compose run --rm --no-deps -v "$PWD:$PWD:ro" -e TORCHINDUCTOR_CACHE_DIR=/tmp/torch-audit-cache dev \
    python scripts/benchmark_official_relations.py run --output "$out" --backend official \
    > "$out/official-run.log" 2>&1
