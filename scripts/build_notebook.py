"""Package the local ARC agent into a Kaggle competition notebook."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "submission.ipynb"
METADATA = ROOT / "notebooks" / "kernel-metadata.json"
SOURCES = {
    "agent/__init__.py": ROOT / "agent" / "__init__.py",
    "agent/adk_policy.py": ROOT / "agent" / "adk_policy.py",
    "agent/local_vlm.py": ROOT / "agent" / "local_vlm.py",
    "agent/model_runtime.py": ROOT / "agent" / "model_runtime.py",
    "scripts/wait_model.py": ROOT / "scripts" / "wait_model.py",
    "agents/templates/my_agent.py": ROOT / "agent" / "my_agent.py",
}
COMP = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3"


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def build() -> dict:
    metadata = json.loads(METADATA.read_text())
    if metadata.get("enable_internet") is not False:
        raise ValueError("Set enable_internet=false: the baseline is designed to run offline")
    if not metadata.get("competition_sources") or "arc-prize-2026-arc-agi-3" not in metadata["competition_sources"]:
        raise ValueError("Add arc-prize-2026-arc-agi-3 to competition_sources")
    if not metadata.get("enable_gpu"):
        raise ValueError("Local Qwen3-VL inference requires enable_gpu=true")
    if "magicsword001/arc-agi-3-qwen3-vl-4b" not in metadata.get("dataset_sources", []):
        raise ValueError("Attach the Qwen3-VL offline bundle via dataset_sources")
    write_files = "from pathlib import Path\n"
    for rel, src in SOURCES.items():
        if not src.is_file():
            raise FileNotFoundError(src)
        target = Path("/tmp/arc_adk") / rel
        write_files += f"p = Path({str(target)!r})\np.parent.mkdir(parents=True, exist_ok=True)\n_ = p.write_text({src.read_text()!r})\n"

    install = dedent(f'''\
        from pathlib import Path
        import subprocess
        import sys

        wheel_dir = Path("{COMP}/arc_agi_3_wheels")
        if wheel_dir.is_dir():
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--no-index",
                "--find-links", str(wheel_dir), "arc-agi", "python-dotenv",
            ], check=True)
        else:
            # The development image already includes these packages.
            import arc_agi
            import dotenv
            print("Local ARC dependencies: OK")
    ''')

    start_model = dedent(f'''\
        import sys
        from pathlib import Path
        sys.path.insert(0, "/tmp/arc_adk")
        from agent.model_runtime import find_bundle, start

        model_process = None
        if Path("{COMP}").is_dir():
            bundle = find_bundle()
            model_process = start(bundle)
            try:
                from scripts.wait_model import wait
                wait("http://127.0.0.1:8080")
            except Exception:
                model_process.terminate()
                raise RuntimeError(
                    "Offline Qwen3-VL failed to start; inspect /kaggle/working/llama-server.log "
                    "and verify the Kaggle GPU and runtime libraries"
                )
        else:
            print("Local notebook: use make model-up; make eval-model GAME=ls20 STEPS=50")
    ''')
    run = dedent(f'''\
        import os
        if os.environ.get("KAGGLE_IS_COMPETITION_RERUN"):
            import shutil
            import subprocess
            import time
            import urllib.request
            from pathlib import Path

            # Wait for the ARC gateway before starting the framework.
            for attempt in range(120):
                try:
                    with urllib.request.urlopen("http://gateway:8001/api/games", timeout=5) as response:
                        if response.status == 200:
                            break
                except Exception:
                    time.sleep(5)
            else:
                raise RuntimeError("ARC gateway did not become available")

            root = Path("/kaggle/working/ARC-AGI-3-Agents")
            shutil.copytree("{COMP}/ARC-AGI-3-Agents", root, dirs_exist_ok=True)
            shutil.copytree("/tmp/arc_adk/agent", root / "agent", dirs_exist_ok=True)
            shutil.copyfile("/tmp/arc_adk/agents/templates/my_agent.py", root / "agents/templates/my_agent.py")
            (root / "agents/__init__.py").write_text(
                'from typing import Type\\n'
                'from dotenv import load_dotenv\\n'
                'from .agent import Agent, Playback\\n'
                'from .swarm import Swarm\\n'
                'from .templates.random_agent import Random\\n'
                'from .templates.my_agent import MyAgent\\n'
                'load_dotenv()\\n'
                'AVAILABLE_AGENTS: dict[str, Type[Agent]] = {{"random": Random, "myagent": MyAgent}}\\n'
            )
            (root / ".env").write_text(
                "SCHEME=http\\nHOST=gateway\\nPORT=8001\\nARC_API_KEY=test-key-123\\n"
                "ARC_BASE_URL=http://gateway:8001/\\nOPERATION_MODE=online\\n"
                "ENVIRONMENTS_DIR=\\nRECORDINGS_DIR=/kaggle/working/server_recording\\n"
            )
            run_env = os.environ.copy()
            run_env["ADK_MODEL"] = "local/qwen3-vl-4b-instruct"
            run_env["VLM_API_BASE"] = "http://127.0.0.1:8080/v1"
            run_env["MPLBACKEND"] = "agg"
            try:
                subprocess.run(["python", "main.py", "--agent", "myagent"], cwd=root, env=run_env, check=True)
            finally:
                if model_process is not None:
                    model_process.terminate()
        else:
            print("Local play: make eval GAME=ls20 STEPS=50")
    ''')
    dummy = dedent(f'''\
        import os
        from pathlib import Path
        if Path("{COMP}").is_dir() and not os.environ.get("KAGGLE_IS_COMPETITION_RERUN"):
            import pandas as pd
            # During Save & Run the gateway is absent. The real output is
            # produced by the gateway during the competition rerun.
            pd.DataFrame([["1_0", "1", True, 1]],
                columns=["row_id", "game_id", "end_of_game", "score"]
            ).to_parquet("/kaggle/working/submission.parquet", index=False)
    ''')
    gpu = bool(metadata.get("enable_gpu"))
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4" if gpu else "none", "isGpuEnabled": gpu,
                       "isInternetEnabled": False, "language": "python", "sourceType": "notebook"},
        },
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": "# ARC-AGI-3 ADK baseline\nGenerated by scripts/build_notebook.py."},
            code(install),
            code("import google.adk\nprint('Google ADK import: OK')"),
            code(write_files), code(start_model), code(run), code(dummy),
        ],
    }


def main() -> None:
    result = build()
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(result['cells'])} cells)")


if __name__ == "__main__":
    main()
