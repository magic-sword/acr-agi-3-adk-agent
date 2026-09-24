---
name: occlusion-memory
description: "Distinguish currently visible evidence from remembered hidden locations. Absence from the image is not proof of disappearance."
---
# Occlusion Memory (S02)

1. Inspect the current observation and supplied memory. Identify evidence relevant to this skill.
2. Distinguish currently visible evidence from remembered hidden locations. Absence from the image is not proof of disappearance.
3. Separate visible facts from hypotheses. Refer only to observation IDs supplied in the input.
4. If evidence is insufficient, report a specific unknown or a distinguishing experiment; do not invent a fact.
5. Call the current state's completion tool with the requested structured arguments. A skill must not directly execute game actions or modify budgets.

Read `references/evidence-contract.md` with `load_skill_resource` when checking evidence, coordinates or prediction semantics.
