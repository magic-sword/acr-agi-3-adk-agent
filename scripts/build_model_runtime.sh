#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_dir="$root_dir/vendor/llama.cpp"
bundle_dir="$root_dir/.cache/model-cache/qwen3-vl-4b"
llama_commit=bddf8263c31c3dce3212263b00ebd2d98c1a752b
mkdir -p "$bundle_dir" "$root_dir/vendor"
if [ ! -d "$src_dir/.git" ]; then
    git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git "$src_dir"
fi
git -C "$src_dir" fetch --depth 1 origin "$llama_commit"
git -C "$src_dir" checkout --detach "$llama_commit"
docker build -f "$root_dir/scripts/Dockerfile.model-runtime" -t arc-agi3-llama-runtime "$src_dir"
container_id="$(docker create arc-agi3-llama-runtime)"
trap 'docker rm -f "$container_id" >/dev/null' EXIT
docker cp "$container_id:/src/build/bin/." "$bundle_dir/"
docker cp "$container_id:/src/LICENSE" "$bundle_dir/llama-LICENSE"
test -x "$bundle_dir/llama-server"
test -e "$bundle_dir/libllama.so" || { echo 'llama runtime libraries missing' >&2; exit 1; }
echo "$llama_commit" > "$bundle_dir/llama-commit.txt"
echo "Runtime bundle ready: $bundle_dir"
