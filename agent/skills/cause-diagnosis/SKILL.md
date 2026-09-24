---
name: cause-diagnosis
description: "Diagnose input errors, identity, control mapping, delay and mode before changing rules. Revise only affected assumptions."
---
# Cause Diagnosis (S06)

1. Inspect the current observation and supplied memory. Identify evidence relevant to this skill.
2. Diagnose input errors, identity, control mapping, delay and mode before changing rules. Revise only affected assumptions.
3. Separate visible facts from hypotheses. Refer only to observation IDs supplied in the input.
4. If evidence is insufficient, report a specific unknown or a distinguishing experiment; do not invent a fact.
5. Call the current state's completion tool with the requested structured arguments. A skill must not directly execute game actions or modify budgets.

Read `references/evidence-contract.md` with `load_skill_resource` when checking evidence, coordinates or prediction semantics.
