---
name: backward-planning
description: "Plan backward from the goal. Include prerequisites, staging space, preserved conditions, linked effects and noncommuting operations."
---
# Backward Planning (S08)

1. Inspect the current observation and supplied memory. Identify evidence relevant to this skill.
2. Plan backward from the goal. Include prerequisites, staging space, preserved conditions, linked effects and noncommuting operations.
3. Separate visible facts from hypotheses. Refer only to observation IDs supplied in the input.
4. If evidence is insufficient, report a specific unknown or a distinguishing experiment; do not invent a fact.
5. Return the requested state's JSON contract. A skill must not directly execute game actions or modify budgets.

Read `references/evidence-contract.md` with `load_skill_resource` when checking evidence, coordinates or prediction semantics.
