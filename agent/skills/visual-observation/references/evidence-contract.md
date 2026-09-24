# Evidence and action contract

- Coordinates use the original grid: x is column and y is row, starting at zero. ACTION6 needs integer x and y within width and height.
- Color IDs are measurements, not object meanings. Never infer a control's effect from its ACTION number.
- Cite the supplied observation_id. A hypothesis may reference only existing evidence IDs.
- frame_changed compares consecutive actual frames; it does not prove progress toward a goal.
- Unobservable and delayed effects are unknown, not contradicted. Time windows count external actions.
- Only the environment's WIN state establishes game completion. Levels and budget limits come from the host.
- Return proposals only. The host validates and executes at most one game action per decision.
