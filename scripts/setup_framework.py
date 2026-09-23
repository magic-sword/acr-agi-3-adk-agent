"""Make the vendored ARC framework importable without its optional LLM templates."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
init = ROOT / "vendor" / "ARC-AGI-3-Agents" / "agents" / "__init__.py"
if not init.exists():
    raise SystemExit(f"Missing {init}; run make setup to clone the framework")
init.write_text('''from typing import Type
from dotenv import load_dotenv
from .agent import Agent, Playback
from .swarm import Swarm
from .templates.random_agent import Random
load_dotenv()
AVAILABLE_AGENTS: dict[str, Type[Agent]] = {"random": Random}
''')
print(f"Prepared {init}")
