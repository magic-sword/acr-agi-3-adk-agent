---
name: action-grounding
description: "Ground a planned input in the displayed controller, especially pixel CLICK targets and ambiguous directional controls; check that target selection matches the intended action."
---
# Grounding an intended action

Use when choosing an input requires resolving a target or control interpretation.

1. Match the intended intervention to `observation.available_actions`. UP/DOWN/LEFT/RIGHT identify buttons, not proven object directions. ACT and UNDO effects also require evidence. Inspect earlier action/result pairs if the mapping is uncertain.
2. For CLICK, locate the target in original board pixels: x is column, y is row, zero-based. Convert from the displayed board using `viewport.origin` and `viewport.scale`; padding and controller graphics are outside the game.
3. Call `move_cursor(x, y)` to preview the target. Inspect the reticle and printed coordinates; adjust it if it covers the wrong region. Previewing does not click or produce an external observation.
4. Use the nested action `{"action":"CLICK","reason":"..."}` without x/y. The host freezes the current cursor at action selection. A future CLICK node targets the cursor at execution, so do not queue clicks meant for several different targets. Reposition from a new observation for each target.
5. For any input, state its predicted effect or the competing predictions being tested. Legal input alone is not a sufficient reason to choose it. An uncertain mapping may justify an experiment rather than a plan.

A complete submission still uses the calling state's schema. The action shown here is only a nested field. Historical executed clicks contain x/y as evidence; those coordinates must not be copied as arguments into a new CLICK.

Example: a board is enlarged 6x with origin (32,44). A displayed center at (95,107) corresponds to original pixel (10,10), not (95,107). Preview the original coordinates before using the target.
