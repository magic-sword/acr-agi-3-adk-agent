---
name: occlusion-memory
description: "Track a previously seen object when it is covered or disappears; distinguish remembered positions, movement and removal without treating memory as current evidence."
---
# Reasoning under occlusion

Use when a relevant object is no longer fully visible or several objects could match an earlier one.

1. Retrieve its last visible observation and intervening actions. Record distinguishing shape, color, neighbors and position. A remembered coordinate is a last-seen coordinate, not a current measurement.
2. Identify possible occluders, viewport changes and actions capable of moving or removing it. Carry forward a hidden location only as a hypothesis conditioned on no relevant intervening change.
3. Compare reappearing candidates against both appearance and reachable displacement. If two identical objects could occupy the position, preserve both identities rather than swapping labels silently.
4. If the next action depends on the hidden object, choose an observation or reversible exposure experiment that distinguishes the candidates. If the action is valid for every candidate, proceed while retaining the uncertainty.
5. Keep current facts limited to visible evidence. Describe hidden-state alternatives in hypotheses/unknowns; the host already retains earlier facts with their evidence IDs. Do not resubmit an old fact as currently visible.

Example: a key was at (4, 2) before a panel covered that region. With no known motion it is plausibly still there, but a collection action during the occlusion leaves both collected and uncollected alternatives. Uncovering the region can distinguish them; repeatedly viewing the covered frame cannot.

A level or RESET boundary invalidates position continuity. Similar layouts across that boundary require fresh identity evidence.
