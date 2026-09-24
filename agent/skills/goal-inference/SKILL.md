---
name: goal-inference
description: "Treat success conditions as hypotheses. Visual completion is not WIN; consider missing or simultaneous conditions."
---
# Goal Inference (S05)

1. Inspect the current observation and supplied memory. Identify evidence relevant to this skill.
2. Treat success conditions as hypotheses. Visual completion is not WIN; consider missing or simultaneous conditions.
3. Separate visible facts from hypotheses. Refer only to observation IDs supplied in the input.
4. If evidence is insufficient, report a specific unknown or a distinguishing experiment; do not invent a fact.
5. Call the current state's completion tool with the requested structured arguments. A skill must not directly execute game actions or modify budgets.

Read `references/evidence-contract.md` with `load_skill_resource` when checking evidence, coordinates or prediction semantics.

## Goal ownership

PLAN and REVISE consider goals as success conditions, which may be relations,
configurations, sequences or simultaneous constraints rather than an object. OBSERVE
never has to discover a goal. An unknown goal is valid: formulate an inquiry for PROBE
if evidence is needed. Retrieve past screens/comparisons before revising an assumption.
Store a proposed goal and its evidence_refs in Proposal.interpretation; use goal=""
with evidence to withdraw an old goal. Omit goal to retain it. VERIFY can report evidence
against an assumed condition; PLAN/REVISE use it to reconsider the objective.
