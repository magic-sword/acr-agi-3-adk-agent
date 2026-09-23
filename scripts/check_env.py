from __future__ import annotations

import importlib
import platform
import sys


def check(module: str, label: str | None = None) -> None:
    label = label or module
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "installed")
        print(f"[OK] {label}: {version}")
    except Exception as exc:
        print(f"[NG] {label}: {exc}")
        raise


print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")

check("jupyterlab", "JupyterLab")
check("google.adk", "Google ADK")
check("torch", "PyTorch")
check("transformers", "Transformers")
check("kaggle", "Kaggle CLI Python package")

import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
