---
name: budget-recovery
description: "Respect action, time and reasoning budgets. Stop on unresolved transport or validation errors rather than fabricate actions."
---
# Budget Recovery (S15)

1. Inspect the current observation and supplied memory. Identify evidence relevant to this skill.
2. Respect action, time and reasoning budgets. Stop on unresolved transport or validation errors rather than fabricate actions.
3. Separate visible facts from hypotheses. Refer only to observation IDs supplied in the input.
4. If evidence is insufficient, report a specific unknown or a distinguishing experiment; do not invent a fact.
5. Call the current state's completion tool with the requested structured arguments. A skill must not directly execute game actions or modify budgets.

Read `references/evidence-contract.md` with `load_skill_resource` when checking evidence, coordinates or prediction semantics.
