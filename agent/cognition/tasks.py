"""The only model responsibilities in the fast/slow runtime."""
from .state import Deliberation, FastSelection

TASKS = {
    'deliberate': ('submit_plan', Deliberation),
    'choose_skill': ('single_token_choice', FastSelection),
    'execute_step': ('single_token_choice', FastSelection),
}
INSTRUCTIONS = {
    'deliberate': '''Study the game images and the acknowledged results. Explain uncertain causal relations,
work backwards from a useful goal, and provide a small executable plan via submit_plan.
Choose visually grounded targets freely; no scene inventory is required. Use original
board coordinates (column x, row y). A skill is a revisable procedure: when it applies,
its effect, steps, concrete action options, and the evidence for advancing or reconsidering.
Keep the plan short: usually one or two skills with one or two steps. Unknown rules
call for an informative probe. A probe can finish after one observed result, even no change.
Distinguish observed evidence from causal hypotheses in causal_notes; pixel changes alone
are not proof of a target response. Retained skills can be reused or replaced, but check
that their controls and coordinates fit this board. When execution returns for deliberation,
use its expected effect and actual result to revise the explanation or plan.
Completion judgments are hypotheses: reconcile them with the board to choose the next goal.
Return one concise submit_plan call. Only the environment can establish game victory.''',
    'choose_skill': '''Select the procedure that fits the current board and goal. Return its one-digit
label only. 8 returns to deliberation when none fits or the plan needs reinterpretation.''',
    'execute_step': '''Follow the current procedure using the board and the last actual result.
Return one digit: an action option, 7 if this step's done_when is observed, or 8 to
return to deliberation if the result is unexpected or you cannot judge how to proceed.
Compare the last expected effect with what changed. Repetition is allowed while useful;
an unrelated display change alone does not establish that the intended effect occurred.''',
}
