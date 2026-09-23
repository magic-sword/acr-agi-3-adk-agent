"""Start the attached Qwen3-VL GGUF model inside the offline Kaggle kernel."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


MODEL = "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
PROJECTOR = "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"


def find_bundle() -> Path:
    base = Path("/kaggle/input")
    bundles = sorted(base.glob("**/" + MODEL))
    for file in bundles:
        directory = file.parent
        if (directory / PROJECTOR).is_file() and (directory / "llama-server").is_file():
            return directory
    raise FileNotFoundError(
        "Attach Kaggle dataset magicsword001/arc-agi-3-qwen3-vl-4b "
        "with both GGUF files and llama-server"
    )


def start(bundle: Path) -> subprocess.Popen:
    # Kaggle input is read-only; execute a writable copy (with execute permission).
    runtime = Path("/kaggle/working/llama-runtime")
    runtime.mkdir(parents=True, exist_ok=True)
    for source in bundle.iterdir():
        if source.name == "llama-server" or source.name.startswith("lib") and ".so" in source.name:
            shutil.copy2(source, runtime / source.name, follow_symlinks=True)
    (runtime / "llama-server").chmod(0o755)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(runtime) + ":" + env.get("LD_LIBRARY_PATH", "")
    command = [
        str(runtime / "llama-server"),
        "--model", str(bundle / MODEL),
        "--mmproj", str(bundle / PROJECTOR),
        "--alias", "qwen3-vl-4b-instruct",
        "--host", "127.0.0.1", "--port", "8080",
        "--n-gpu-layers", "99", "--ctx-size", "4096", "--parallel", "1",
    ]
    with Path("/kaggle/working/llama-server.log").open("w") as log:
        return subprocess.Popen(command, env=env, stdout=log,
                                stderr=subprocess.STDOUT)
