---
name: notebook
description: Keep a shared puzzle notebook of scoped goals, hypotheses and plans with evidence links and bookmarks.
---

The opening pages already contain the current goal, latest host result and bookmark index.
Use read_notebook(reference) to open an ID or bookmark. With no reference it returns a six-item
index; query filters title/text, offset pages through it. include_previous searches earlier levels.
Read original screenshots/grids with get_observation when a recorded interpretation is uncertain.

Use write_note to create one short hypothesis, plan or interpretation at a time. Cite real
observation, experience or note IDs for hypotheses and interpretations; these remain interpretations,
not verified facts. Initial scene hypotheses may cite the current observation ID.
Revise a page by its note_id and expected_revision. The goal has ID goal: revise that page rather
than creating another source of truth. Keep the goal stable unless evidence warrants a change.
Do not repeat the initial scene description after every action. Only record useful changes.

set_bookmark names useful pages (e.g. experiment or skill_evidence). At most six custom bookmarks
can be open. current_goal and latest_result are maintained by the host. An empty note_id removes
a custom bookmark. erase_note withdraws a page with a reason; its past versions remain in the log.
Host-recorded results cannot be overwritten. Correct interpretations in a separate note.
Reset/level boundaries start a new segment and clear local bookmarks; older notes remain readable
by ID. Check scope before transferring an old rule to a new level. No notebook operation advances
game time or changes a skill's evaluation status. Avoid rereading pages already in this invocation.
