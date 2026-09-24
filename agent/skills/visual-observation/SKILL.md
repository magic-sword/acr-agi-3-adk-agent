---
name: visual-observation
description: "Inspect rendered game frames with a controller and exact pixel cursor; retrieve recorded animation separately from the current final frame."
---
# Visual Observation (S01)

The host attaches the **final received frame**, rendered by `scripts/render_observation.py`, on every reasoning call. The controller and open cursor reticle are host UI, not game objects. Original palette pixels and the untouched grid remain the game evidence.

- Use `observe_current` to inspect the final frame again. Repeated reads do not advance the game or create new evidence.
- Coordinates are original pixels: x is column, y is row, zero-based. `viewport` gives the board origin and integer scale in the composed image. Never use controller/padding coordinates as click coordinates.
- Use `move_cursor(x, y)` to preview a target without clicking. Inspect the returned reticle and its printed coordinates; adjust if necessary. The reticle center is the target. Its overlay can obscure neighbors, so use the grid or move it away for inspection. Propose CLICK without x/y to click at the verified current cursor position. This tool spends no external action and does not prove a click occurred.
- Controller arrows describe conventional input buttons (UP/DOWN/LEFT/RIGHT); ACT is an action button, CLICK uses the cursor, and UNDO is the undo input. Dim controls are unavailable. The game's actual response to any button requires evidence; do not assume an object will move in that direction.

## Historical animation, requested separately

`animation` identifies the completed frame batch returned by one external action, with source step/action and frame count. The current screen is its last frame, not an animation still running. At initial observation the source action can be unknown.

Call `observe_animation(event_id, start_frame=0)` only when intermediate changes matter (motion, order, disappearance, or cause). It returns up to four consecutive frames as a labeled chronological image, read left-to-right then top-to-bottom. Follow `next_start_frame` until null if the whole event is needed. These are **historical replay frames**, not new current observations. Repeated reads of the same event must not count as repeated lava flows, repeated actions, elapsed time, or independent evidence. After analysis, reason about the final current frame. Actual animation durations are unavailable; do not infer speed from replay timing. The cursor on historical images is the archived host overlay, not an in-game pointer.

The latest 128 observations and their events are retained per game run, even without diagnostic logging. Evicted IDs return an explicit unavailable result. Historical cursor overlays are the positions recorded with those observations. With cognition logging enabled, raw batches are archived beside the final PNG for later explicit playback:

```bash
python agent/skills/visual-observation/scripts/render_observation.py current EVENT.json current.png
python agent/skills/visual-observation/scripts/render_observation.py replay EVENT.json history.gif
```

The GIF plays once without a loop extension, with synthetic 150 ms frame durations, and holds the final frame. The local image model receives ordered PNG frames because animated GIF playback is not supported by its image transport.

## Shared evidence access

OBSERVE is a host-only capture step. This skill is shared by VERIFY, PLAN, PROBE and
REVISE. Interpretation depends on the calling state's question, predictions and goals;
do not require scene recognition or a goal hypothesis before proceeding.

- `list_observations()` lists retained IDs, steps, boundary segments and source actions.
- `get_observation(observation_id)` retrieves an immutable recorded screen.
- `compare_observations(before_id, after_id, offset=0)` returns labeled before/after
  images, source actions and exact pixel changes in pages of 96. Follow next_offset
  to inspect remaining changes. It refuses comparisons across level/RESET boundaries.
- `observe_animation` also retrieves older retained events, not only the latest one.

Tool reads do not change observation IDs, memory revisions or external action counts.
Unavailable evidence must remain unknown. Retrieval does not itself establish an effect,
causal explanation, object identity or progress. Infer those only in the relevant reasoning state.

When relevant to your current question, extract visible objects and relations, preserving color IDs and original coordinates. Separate facts from hypotheses, cite only supplied observation IDs, and state specific unknowns. Return the requested JSON contract. Tools only inspect or position the host cursor; the host validates and executes game actions.

Read `references/evidence-contract.md` with `load_skill_resource` when checking evidence or predictions.
