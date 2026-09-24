"""State-scoped ADK SkillToolsets with on-demand instructions and resources."""

from pathlib import Path

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
    "S13": "decision-commit",
    "S14": "procedure-reuse",
    "S15": "budget-recovery",
}

STATE_SKILLS = {
    # OBSERVE also proposes hypotheses and a goal in the current Perception contract.
    "OBSERVE": ("S01", "S02", "S03", "S04", "S05"),
    "PROBE": ("S07", "S10", "S12", "S15"),
    "PLAN": ("S08", "S09", "S10", "S11", "S12", "S14"),
    # Reinterpretation supplements the destination's experiment/planning skills.
    "REVISE": ("S04", "S05", "S06", "S09", "S11"),
    "VERIFY": (), "UPDATE": (), "ACT": (), "COMMIT": (),
    "CONSOLIDATE": (), "RECOVER": (),
}


def selected_skills(state: str, proposal_state: str | None = None) -> tuple[str, ...]:
    selected = STATE_SKILLS[state]
    if state == "REVISE":
        if proposal_state not in ("PROBE", "PLAN"):
            raise ValueError("REVISE requires its PROBE or PLAN destination")
        selected += STATE_SKILLS[proposal_state]
    elif proposal_state is not None:
        raise ValueError("Only REVISE accepts a proposal destination")
    if not selected:
        raise ValueError(f"{state} is host-only and must not create a model skill toolset")
    return tuple(dict.fromkeys(selected))


def instruction(state: str, schema: dict) -> str:
    import json
    return (
        "You solve an unknown visual game. Return ONE JSON object, no prose. "
        "Use only supplied observations, memory and legal actions. Never invent an observed result. "
        "Evidence IDs must refer to supplied observations. Facts are visible measurements, hypotheses are interpretations. "
        "Keep output concise and omit optional empty fields. Keep the existing goal text unchanged unless evidence requires a different goal. Empty goal means retain it. Use skill tools before the final JSON answer.\n"
        + "\nDiscover skills with list_skills, then load the relevant skill with load_skill before reasoning. "
        "Load references with load_skill_resource only when needed. Do not repeatedly load unchanged resources. "
        "Only the final answer must be JSON; intermediate function calls are allowed."
        + "\nRequired JSON shape examples (replace IDs with input IDs; infer your own facts/actions): "
        + ('{"observation_id":"COPY_INPUT_ID","memory_revision":0,"facts":[],"unknowns":["control mapping"],"goal":"","hypotheses":[]}'
           if state == "OBSERVE" else
           '{"observation_id":"COPY_INPUT_ID","memory_revision":0,"purpose":"probe","action":{"action":"ACTION1","reason":"test control"},"experiment":{"question":"Does this control visibly change the board?","alternatives":{"changes":[{"kind":"frame_changed","value":true}],"unchanged":[{"kind":"frame_changed","value":false}]},"discriminator":"compare consecutive real frames","risk":"unknown side effects"}}')
        + "\nDo not return empty observation analysis: report visible relations and specific important unknowns. "
        "experiment.alternatives is an OBJECT whose named values are ARRAYS of predicate objects, never a flat array or single predicate. "
        + "\nPredicates: cell uses x,y and integer color ID; fact uses key,value of a currently visible fact; "
        "state uses the exact game state string; levels_min uses an integer; frame_changed uses boolean. "
        "Plan dependencies must be acyclic, node status must be todo. Every plan action needs completion and "
        "observable effects. A probe needs alternatives with different observable predictions. "
        "delay_steps counts external actions, not model calls. Hypothesis scope is level or game. "
        "Echo observation_id and memory_revision exactly.\nJSON schema:\n"
        + json.dumps(schema, separators=(",", ":"))
    )


def skill_toolset(state: str, proposal_state: str | None = None) -> SkillToolset:
    root = Path(__file__).resolve().parents[1] / "skills"
    return SkillToolset(skills=[load_skill_from_dir(root / SKILL_NAMES[s])
                               for s in selected_skills(state, proposal_state)])
