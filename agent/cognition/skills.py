"""Host-selected skill instructions; no model tool calling is required."""

SKILLS = {
    "S01": "Extract visible objects and relations, keeping color IDs and coordinates. Never assume the meaning of an ACTION number.",
    "S02": "Distinguish currently visible evidence from remembered hidden locations. Absence from the image is not proof of disappearance.",
    "S03": "Compare predicted effects and invariants to real observations. Separate delayed, unobservable and contradicted effects.",
    "S04": "Maintain competing goal/control/dynamics/mode hypotheses, their scope, evidence and counterexamples. Support is not certainty.",
    "S05": "Treat success conditions as hypotheses. Visual completion is not WIN; consider missing or simultaneous conditions.",
    "S06": "Diagnose input errors, identity, control mapping, delay and mode before changing rules. Revise only affected assumptions.",
    "S07": "Design a bounded experiment whose observable outcomes distinguish at least two plausible hypotheses. State cost and irreversible risk.",
    "S08": "Plan backward from the goal. Include prerequisites, staging space, preserved conditions, linked effects and noncommuting operations.",
    "S09": "Repair affected plan dependencies and preserve still-valid completed subgoals. Do not assume undo exists.",
    "S10": "Represent delays and synchronization with step windows. Do not invent WAIT; ACTION effects and time progression are unknown until observed.",
    "S11": "Keep causal effect separate from usefulness: an effect harmful earlier can now achieve the goal.",
    "S12": "Ground one action against current legal controls, coordinates and observed preconditions. No arbitrary fallback or RESET.",
    "S13": "Commit one decision with its source observation and predictions before external execution.",
    "S14": "Retain witnessed successful procedures as scoped candidates with applicability, checkpoints and counterexamples; revalidate before reuse.",
    "S15": "Respect action, time and reasoning budgets. Stop on unresolved transport or validation errors rather than fabricate actions.",
}
STATE_SKILLS = {
    "OBSERVE": ("S01", "S02", "S03", "S04", "S05"),
    "REVISE": ("S06", "S04", "S05", "S07", "S08", "S09", "S10", "S11", "S12"),
    "PROBE": ("S07", "S10", "S12", "S15"),
    "PLAN": ("S08", "S09", "S10", "S11", "S12", "S14"),
    "VERIFY": ("S03", "S10"), "UPDATE": ("S02", "S04"),
    "ACT": ("S12",), "COMMIT": ("S13",),
    "CONSOLIDATE": ("S14",), "RECOVER": ("S15",),
}


def instruction(state: str, schema: dict) -> str:
    import json
    return (
        "You solve an unknown visual game. Return ONE JSON object, no prose. "
        "Use only supplied observations, memory and legal actions. Never invent an observed result. "
        "Evidence IDs must refer to supplied observations. Facts are visible measurements, hypotheses are interpretations. "
        "Keep output concise and omit optional empty fields. Keep the existing goal text unchanged unless evidence requires a different goal. Empty goal means retain it. No tools.\n"
        + "\n".join(SKILLS[s] for s in STATE_SKILLS[state])
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
