"""Each slow call answers one question; fast calls only choose a digit."""
from .state import Understanding, Backchain, Grounding, Reconciliation, FastSelection

TASKS = {
    'understand': ('submit_understanding', Understanding),
    'backchain': ('submit_backchain', Backchain),
    'ground': ('submit_grounding', Grounding),
    'reconcile': ('submit_reconciliation', Reconciliation),
    'choose_skill': ('single_token_choice', FastSelection),
    'execute_step': ('single_token_choice', FastSelection),
    'aim': ('single_token_choice', FastSelection),
}
SLOW_TASKS = ('understand', 'backchain', 'ground', 'reconcile')
OUTPUT_TOKENS = {'understand':1000, 'backchain':1200, 'ground':1800, 'reconcile':1000}
INSTRUCTIONS = {
    'understand': '''Inspect the board and the evidence. Identify only the few objects relevant to the
current unresolved goal, including the acting object and its proposed destination.
Describe shared concepts first, then relevant instances or groups using concept,
appearance, role_hypothesis and relations. Similar objects may share a concept while
having different roles. Roles are revisable hypotheses. No inventory of every cell and
no coordinates: locate clicked parts later from the current image. Reuse useful concept
knowledge from the catalogue; old descriptions are not current visual evidence.
Separate observed relations from causal hypotheses.
State a plausible goal and one uncertainty answerable by a concrete interaction, such as
whether clicking a chosen square changes that square. An unknown rule calls for a probe.
Choose backchain to find prerequisites, or ground for a one-action probe or an already
understood procedure. Preserve useful evidence; reinterpret targets when needed.
Submit a concise submit_understanding call. You are not writing the execution plan.''',
    'backchain': '''Work backwards from the goal hypothesis using the observed targets and causal notes.
Express goals as observable states, not intentions such as explore. target_query describes
the relevant instances through appearance, role and relations, rather than object IDs.
Ask what must hold just before the goal, and what action could establish that condition.
Update a small goal hierarchy with stable IDs: parent_id is the enclosing goal;
requires lists prerequisite goal IDs declared in goals or already retained; use [] when
none is needed. An action name is not a prerequisite goal. Preserve completed goals.
Select the current actionable leaf or the prerequisite whose uncertainty needs a probe.
Explain the dependency briefly in rationale. Submit submit_backchain, not action code.''',
    'ground': '''Turn the selected small goal into a short executable procedure, using current targets,
known effects and prior outcomes. For CLICK give target_query: identify the instance and
part to click using visible appearance and relations. The fast cursor will locate it on
the current board; no coordinates or target IDs. Distinguish similar instances enough
for execution. Other buttons need no target_query. Skills reuse semantic queries, not positions.
Specify a visible baseline, how the named targets should change, when to continue,
what observable relation finishes each step, and what result requires reconsideration.
A display change alone is not a target relation. Usually one skill with one or two steps.
Use intent=probe for an unresolved question: one acknowledged action and its observation
will return to reconciliation, even if nothing changed. Use intent=achieve for a goal.
New skills automatically become the choices. Put only existing skill names in reuse
after checking their target descriptions and conditions.
Return next=understand or backchain with plan=null if the missing information belongs
there; unknown interaction effects can instead be tested by a probe on the located target.
Otherwise next=execute with the grounded plan. Submit submit_grounding.''',
    'reconcile': '''Compare the named targets and the actual result with the expected effect and baseline.
A fast completion signal is a claim to assess, not proof. Probe completion only means
one action's result is available; it does not confirm the parent goal. Use invocation
and observation IDs to distinguish this attempt from an earlier use of the same skill.
Update concise causal notes with supported, contradicted and still-unknown relations.
Assess the reviewed goal as active, confirmed, or unknown using the board evidence.
Choose understand for a target/goal interpretation issue, backchain for prerequisites,
ground for a concrete procedure revision or next small goal, or resume to continue this
same unfinished achievement procedure. Retain unresolved goals instead of starting over.
Submit one concise submit_reconciliation call. Only the environment establishes victory.''',
    'choose_skill': '''Select the procedure matching the grounded small goal and its visible targets.
Return its one-digit label, or 8 if applicability needs reconciliation.''',
    'aim_locate': '''Find the instance and part described by target_query in the CURRENT board.
Choose the quadrant containing that part: 1 upper left, 2 upper right, 3 lower left,
4 lower right. Choose 8 if the description is ambiguous or the target is not visible.
Return one digit. This selects a search region, not a click. Scene roles are hypotheses;
use visible appearance and relations to distinguish similar instances.''',
    'aim': '''Locate target_query on the CURRENT board. The second image is a HOST cursor
preview of the same frozen observation, not a game object or a new game result.
Return one digit: 1 up, 2 down, 3 left, 4 right; 5 smaller moves, 6 larger moves;
7 click only if the yellow cursor encloses the requested instance and part;
8 if the target is missing, ambiguous or the plan needs reinterpretation.
Choose by visible alignment, not by assuming the initial center is the target.
Scene roles are prior hypotheses: locate the requested instance in the CURRENT image.
Cursor adjustments change only the host preview. They do not operate the game.''',
    'execute_step': '''Follow the current grounded step. Compare the named targets with its baseline,
expected effect and current result. Return one digit: an action while continue_when
holds, 7 when done_when is visible, or 8 for an unexpected result or uncertainty.
A prior invocation's result does not demonstrate progress in this invocation.
Repetition is allowed while useful. Judge the specified relation, not any pixel change.''',
}
