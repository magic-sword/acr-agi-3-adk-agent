"""State-scoped ADK SkillToolsets with on-demand instructions and resources."""

from pathlib import Path
import json

from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset

SKILL_NAMES = {
    "S01": "visual-observation",
    "S02": "occlusion-memory",
    "S03": "prediction-verification",
    "S04": "hypothesis-maintenance",
    "S05": "goal-inference",
    "S06": "cause-diagnosis",
    "S07": "discriminating-experiment",
    "S08": "backward-planning",
    "S09": "plan-repair",
    "S10": "temporal-reasoning",
    "S11": "effect-revaluation",
    "S12": "action-grounding",
    "S14": "procedure-reuse",
}

STATE_SKILLS = {
    "DECIDE": ("S01", "S03", "S05", "S06", "S07", "S08"),
    "OBSERVE": (),
    "UPDATE": (), "ACT": (), "COMMIT": (), "CONSOLIDATE": (), "RECOVER": (),
}


def selected_skills(state: str) -> tuple[str, ...]:
    selected = STATE_SKILLS.get(state, ())
    if not selected:
        raise ValueError(f"{state} is host-only and must not create a model skill toolset")
    return tuple(dict.fromkeys(selected))


class ReasoningSkillToolset(SkillToolset):
    """ADK loaders with inline discovery and only supported knowledge tools.

    Registered bundled scripts run through typed host tools, never arbitrary execution.
    Tools are registered independently of skill activation.
    Public hooks keep native loading/results/state behavior without modifying ADK.
    """

    def __init__(self, *, skills, script_sources=None):
        for skill in skills:
            expected = (script_sources or {}).get(skill.name, {})
            if (any(name not in expected or script.src != expected[name]
                    for name, script in skill.resources.scripts.items())
                    or skill.frontmatter.metadata.get("adk_additional_tools")):
                raise ValueError("Unconnected scripts or dynamic tools are not allowed")
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
            "Skills guide reasoning; the state instruction owns submission and the host owns "
            "validation, budgets, state transitions and game execution.",
        ])


def bound_script_sources():
    from .causal import SCRIPT_BINDINGS
    root = Path(__file__).resolve().parents[1] / "skills"
    return {name: {Path(path).name: (root / name / path).read_text()} for name, path in SCRIPT_BINDINGS.items()}


def skill_toolset(state: str) -> ReasoningSkillToolset:
    root = Path(__file__).resolve().parents[1] / "skills"
    return ReasoningSkillToolset(skills=[load_skill_from_dir(root / SKILL_NAMES[s]) for s in selected_skills(state)],
                                 script_sources=bound_script_sources())
