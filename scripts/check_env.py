from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys


def check_import(module: str, label: str | None = None) -> None:
    label = label or module
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "installed")
        print(f"[OK] {label}: {version}")
    except Exception as exc:
        print(f"[NG] {label}: {exc}")
        raise


def check_distribution(distribution: str, label: str | None = None) -> None:
    """Check package installation without importing it.

    The Kaggle package authenticates during import in some releases, so importing
    it would make a local environment check depend on API credentials.
    """
    label = label or distribution
    try:
        version = importlib.metadata.version(distribution)
        print(f"[OK] {label}: {version}")
    except Exception as exc:
        print(f"[NG] {label}: {exc}")
        raise


print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")

check_import("jupyterlab", "JupyterLab")
check_import("google.adk", "Google ADK")
from google.adk import Workflow
assert importlib.metadata.version("google-adk") == "2.0.0", "Rebuild the pinned ADK image"
check_import("arc_agi", "ARC-AGI game engine")
check_import("torch", "PyTorch")
check_import("transformers", "Transformers")
check_distribution("kaggle", "Kaggle CLI Python package")

import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
