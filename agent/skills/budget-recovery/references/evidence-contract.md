# Evidence and action contract

- Coordinates use the original grid: x is column and y is row, starting at zero. move_cursor takes integer x and y within width and height. CLICK takes no coordinates and uses the current host cursor.
- Color IDs are measurements, not object meanings. Button names identify inputs, not proven game effects.
- Cite the supplied observation_id. A hypothesis may reference only existing evidence IDs.
- frame_changed compares consecutive actual frames; it does not prove progress toward a goal.
- Unobservable and delayed effects are unknown, not contradicted. Time windows count external actions.
- Only the environment's WIN state establishes game completion. Levels and budget limits come from the host.
- Return proposals only. The host validates and executes at most one game action per decision.
