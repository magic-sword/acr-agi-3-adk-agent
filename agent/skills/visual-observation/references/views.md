# Choosing a view

Current view: the final received game frame. `observe_current` is useful after moving the cursor, not as a way to wait for a game effect. The overlay can hide a feature; move it away to inspect that feature.

Historical view: `get_observation` returns an archived screen with its archived cursor, not today's cursor. It does not establish that an old object remains in that position.

Differences: `compare_observations` labels before/after and returns pages of 96 pixel changes. Inspect source actions and boundaries before making a causal claim. Comparisons across RESET/level boundaries are refused. Status indicators can change even if a target does not move.

Animation: frames belong to an already completed external action. Their ordering is real; display timing is synthetic. Repeating playback is the same evidence. `next_start_frame` identifies the next page, not a future game step.

Only the latest 128 observations/events are retained. An evicted ID is unavailable; do not reconstruct its pixels from a summary. Use the known summary as remembered evidence with its uncertainty or choose an experiment that does not require the missing image.
