---
name: procedure-reuse
description: "Retain witnessed successful procedures as scoped candidates with applicability, checkpoints and counterexamples; revalidate before reuse."
---
# Procedure Reuse (S14)

1. Inspect the current observation and supplied memory. Identify evidence relevant to this skill.
2. Retain witnessed successful procedures as scoped candidates with applicability, checkpoints and counterexamples; revalidate before reuse.
3. Separate visible facts from hypotheses. Refer only to observation IDs supplied in the input.
4. If evidence is insufficient, report a specific unknown or a distinguishing experiment; do not invent a fact.
5. Return the requested state's JSON contract. A skill must not directly execute game actions or modify budgets.

Read `references/evidence-contract.md` with `load_skill_resource` when checking evidence, coordinates or prediction semantics.
