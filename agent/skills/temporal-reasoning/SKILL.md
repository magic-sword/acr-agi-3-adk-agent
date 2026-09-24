---
name: temporal-reasoning
description: "Reason about delayed effects or ordering windows using recorded external steps; separate historical animation order from game-time progression."
---
# Reasoning about delays and order

Use when effects may arrive after later actions, two processes need coordination, or a plan has an execution window.

1. Anchor the event to its executed action and source step. Use observation records to count subsequent external actions. Tool calls, model latency and historical replay do not advance this count.
2. Inspect animation only for within-action ordering. Frame order can show that one event preceded another; synthetic replay duration cannot establish elapsed game time, speed or a future delay.
3. Keep pending effects unknown before their justified deadline. If intervening actions could also cause the effect, preserve the causal ambiguity when it appears.
4. Represent justified windows with `delay_steps` and plan `earliest_step`/`latest_step`. These are external-action steps, not milliseconds. Do not insert WAIT: the available action set defines possible inputs, and even a seemingly idle input may have side effects.
5. Before choosing an action to advance a window, inspect its predicted effects and invariants. If no safe input is known, formulate an experiment about time advancement. Do not invent a synchronization model to fill the schema.

Example: a gate opens after a move following switch activation. Either the move advances a delayed switch effect or it independently opens the gate. Another replay of those frames will not distinguish these explanations; a controlled alternate input may.

An expired deadline with an unobservable result remains unresolved evidence. The host manages deadlines; this method supplies the interpretation and justified predictions.
