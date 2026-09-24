---
name: hypothesis-maintenance
description: "Compare competing control, goal, dynamics or mode explanations; update only the claims actually supported or contradicted by an observation."
---
# Maintaining competing explanations

Use when several explanations lead to different predictions or an observation challenges a stored claim.

1. Separate the claim into a testable relationship and its conditions: input, affected object, mode and observable result. Avoid combining goal, control mapping and dynamics in one hypothesis.
2. Reuse an existing hypothesis ID when updating the same claim. Choose `level` scope for layout-dependent rules; use `game` scope only when evidence justifies transfer across layouts.
3. Compare the alternatives' predictions against evidence that could distinguish them. A result shared by both is compatible with both and selects neither. A repeated view of one observation is not independent support.
4. Set `supported` for observed support, `refuted` for an observable contradiction under the claim's conditions, and `suspended` when applicability has become uncertain. A single success is not certainty. Preserve unaffected claims.
5. Keep only decision-relevant competitors in the bounded context. State the missing discriminator in unknowns or an inquiry. When a claim changes, explain the evidence; dependent plans may need repair.

Example: UP moves an object left. Candidate explanations include a rotated control frame and control of another object. Testing UP again on the same ambiguous frame may support both. Observing a second distinguishable object's response or a second direction can separate them.

Describe uncertainty in existing hypothesis fields and summaries; do not invent confidence fields or claim to edit committed memory directly.
