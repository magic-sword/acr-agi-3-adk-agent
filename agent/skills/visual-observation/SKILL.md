---
name: visual-observation
description: "Inspect ambiguous objects or relations in a game frame; choose current, historical, difference or animation views to answer a specific visual question."
---
# Visual evidence inspection

Use when the attached final frame cannot answer a decision-relevant visual question. Identify the object, relation or location to inspect before requesting another view.

1. Locate the board using `viewport.origin` and `viewport.scale`. Read color IDs and original pixel coordinates; controller graphics, labels and cursor reticles are host overlays.
2. Choose the evidence source: the attached image for current layout; `get_observation(id)` for an earlier layout; `compare_observations(before_id, after_id, offset)` for a changed region; `observe_animation(event_id, start_frame)` for order or motion. `list_observations()` resolves missing IDs and source actions. Re-reading the same current image adds no evidence.
3. For a small target, use `move_cursor(x, y)` to verify original coordinates. Move the reticle away if it hides a boundary. Separate a color region's measured extent from its inferred role. Similar color alone does not establish object identity.
4. Inspect only the necessary pages. A difference page is not the whole difference set; follow `next_offset` if the relevant region has not appeared. Animation pages contain at most four ordered frames; follow `next_start_frame` if the needed transition lies later.
5. Report current visible measurements as facts and inferred roles as hypotheses. If a region is hidden, an ID is unavailable, or the timing is ambiguous, identify what cannot be observed instead of filling it in.

Example: a colored square disappears between frames. Compare its previous region and inspect intermediate frames before choosing among movement, covering and removal. A blank final region alone does not distinguish them.

For replay timing, boundary and coordinate pitfalls, read [references/views.md](references/views.md) only when those distinctions affect the decision.
