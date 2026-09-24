---
name: action-grounding
description: "Operate the pictured controller using UP/DOWN/LEFT/RIGHT/ACT/UNDO, move the host cursor, and propose CLICK at its current position without specifying click coordinates."
---
# Action Grounding (S12)

Use the button names printed in the current image and listed in `observation.available_actions`: `UP`, `DOWN`, `LEFT`, `RIGHT`, `ACT`, `CLICK`, `UNDO`. Dim buttons are unavailable. Directions identify controller inputs; actual game effects must be observed. RESET remains host-managed recovery.

## Button presses

The following is a NESTED action field, never the complete submission. In the completion tool arguments, use an action such as `{"action":"UP","reason":"test the up button"}`. Use the same shape in a plan node. Choose one legal external action per decision, with the required predicted effects or distinguishing experiment. Do not send internal ACTION numbers. `scripts/controls.py` maps displayed names to the engine at the validated execution boundary.

## Move, inspect, click

1. Call `move_cursor(x, y)` with integer **original game pixel** coordinates (x=column, y=row; zero-based). This moves only the host cursor and returns the current screen. It does not click, step the game, spend an external action, or create new evidence.
2. Inspect the reticle and its printed coordinates. Adjust with `move_cursor` if necessary. `observe_current` reads the current frame and cursor again without replaying animation.
3. Propose `{"action":"CLICK","reason":"click the visually checked target"}` within the requested contract. **Omit x and y.** The host resolves CLICK to the current cursor at action selection and records the concrete coordinates in the executed action. Missing/out-of-range cursors or unavailable CLICK are errors; no default target is invented.

A planned CLICK means "click wherever the cursor is when this node executes". It does not remember a future target. Do not queue clicks intended for different targets; obtain a new observation and position the cursor for each target. Host history can contain concrete x/y for already executed clicks; those fields are evidence, not arguments for a new CLICK. Once selected, an external click's coordinates are fixed for that decision and are not changed by later cursor movement.

Observation tools return views only. Submit the complete Proposal (observation_id, memory_revision, purpose, and experiment or plan) through the current completion tool, such as submit_experiment in PROBE; the host checks legality, preconditions and budgets before executing it once. If evidence is insufficient, report the specific unknown instead of choosing an arbitrary action.

Read `references/evidence-contract.md` with `load_skill_resource` for evidence and prediction semantics.
