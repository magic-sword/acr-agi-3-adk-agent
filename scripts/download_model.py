"""Download the official Qwen GGUF and vision projector outside Git."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".cache/model-cache/qwen3-vl-4b"
REPO = "Qwen/Qwen3-VL-4B-Instruct-GGUF"
FILES = (
    "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
    "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf",
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = OUT / name
        if dest.exists() and dest.stat().st_size > 100_000_000:
            with dest.open("rb") as file:
                if file.read(4) == b"GGUF":
                    print(f"Already downloaded: {name}")
                    continue
        url = f"https://huggingface.co/{REPO}/resolve/main/{name}"
        part = dest.with_suffix(dest.suffix + ".part")
        print(f"Downloading {url} -> {dest}", flush=True)
        with urlopen(url, timeout=120) as response, part.open("wb") as file:
            while chunk := response.read(8 * 1024 * 1024):
                file.write(chunk)
        with part.open("rb") as file:
            if file.read(4) != b"GGUF" or part.stat().st_size < 100_000_000:
                raise RuntimeError(f"Incomplete GGUF: {part}")
        part.replace(dest)
    metadata = {
        "title": "ARC AGI 3 Qwen3 VL 4B offline bundle",
        "id": "magicsword001/arc-agi-3-qwen3-vl-4b",
        "licenses": [{"name": "Apache 2.0"}],
    }
    (OUT / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (OUT / "MODEL-SOURCE.txt").write_text(
        f"Source: https://huggingface.co/{REPO}\n"
        "Qwen3-VL-4B-Instruct GGUF model license: Apache-2.0\n"
        "Inference executable: llama.cpp (MIT; see llama-LICENSE)\n"
    )
    print("Bundle ready; build its runtime with: make model-runtime")


if __name__ == "__main__":
    main()
