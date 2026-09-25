"""State-scoped ADK SkillToolsets with on-demand instructions and resources."""

from pathlib import Path
import json

from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset

class ReasoningSkillToolset(SkillToolset):
    """ADK loaders with inline discovery and only supported knowledge tools.

    These packages contain reasoning guidance, not executable scripts or dynamic
    tools. Host FunctionTools are registered independently of skill activation.
    Public hooks keep native loading/results/state behavior without modifying ADK.
    """

    def __init__(self, *, skills):
        for skill in skills:
            if skill.resources.scripts or skill.frontmatter.metadata.get("adk_additional_tools"):
                raise ValueError("Reasoning skills cannot declare scripts or dynamic tools")
        super().__init__(skills=skills)
        self.catalog = tuple({"name": s.name, "description": s.description} for s in skills)

    async def get_tools(self, readonly_context=None):
        return [tool for tool in await super().get_tools(readonly_context)
                if tool.name in {"load_skill", "load_skill_resource"}]

    async def process_llm_request(self, *, tool_context, llm_request):
        # Replace the stock system instruction, which advertises script execution
        # and mandatory discovery. Only L1 goes into the initial request.
        llm_request.append_instructions([
            "Optional reasoning skills (names and applicability only): "
            + json.dumps(self.catalog, separators=(",", ":")),
            "Use a skill only when its specialist method helps resolve the current task. "
            "You may complete the state without loading any skill. "
            "To use one, read its instructions with load_skill(skill_name) first and follow them. "
            "Use load_skill_resource only for a referenced example needed for this decision. "
            "Reuse content already loaded within this reasoning call; do not read the whole catalog. "
            "Skills guide reasoning; the controller owns submission and the host owns "
            "validation, budgets, state transitions and game execution.",
        ])


def skill_instructions(name: str) -> str:
    root = Path(__file__).resolve().parents[1] / "skills"
    return load_skill_from_dir(root / name).instructions


def skill_toolset() -> ReasoningSkillToolset:
    root = Path(__file__).resolve().parents[1] / "skills"
    return ReasoningSkillToolset(skills=[load_skill_from_dir(root / name)
        for name in ("notebook",)])
