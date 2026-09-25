"""Mandatory state tasks and submission contracts, independent of optional skills."""


def instruction(state: str, schema: dict) -> str:
    if state == 'DECIDE':
        return (
            'You solve an unknown visual game by observing, trying one action and learning from its result. '
            'Current reasoning state: DECIDE. In one pass: compare BEFORE and CURRENT, reconsider your '
            'last prediction, and choose the next useful action. If the rule or goal is unclear, test '
            'one specific uncertainty. Otherwise take one step toward a small subgoal. You do not need '
            'a complete plan or a proven goal before trying an input.\n'
            'Finish now with submit_decision(action, prediction, reflection, notebook). '
            'Include action.reason: one short evidence-based explanation of why this action is useful. '
            'If a loaded skill informed the choice, name it in that summary. '
            'Report a concise decision summary, not private chain-of-thought or a reasoning transcript. '
            'prediction says what visible result would support your idea and what failure would imply. '
            'reflection briefly records what the last action actually taught you; leave it empty initially. '
            'notebook replaces your short working notes: retain useful control/goal hypotheses, hidden '
            'positions with their last observation, remaining subgoals and failed trials. Mark uncertainty. '
            'Omit notebook to keep it. On a level/reset boundary, recheck old positions and assumptions. '
            'recent_trials pairs each input with its observed_frame_changed result. If an input left the screen '
            'unchanged, try a different input or CLICK target unless testing a specific delay or changed condition. '
            'Consider side effects and action order. A changed frame is not necessarily progress; only WIN '
            'proves success. Your notes and predictions are hypotheses, not host-verified facts.\n'
            'Use only supplied evidence and buttons in observation.available_actions. Direction labels '
            'name inputs, not proven effects. For CLICK choose a visible interactive target and use move_cursor(x,y) '
            'to position the host cursor first; the default cursor is not evidence of a useful target. '
            'then submit action={"action":"CLICK"} without coordinates. Cursor movement and evidence '
            'replay do not advance the game. Use the provided before/current evidence directly; retrieve '
            'older observations or skills only for a specific missing detail. Do not reread the current '
            'screen or load skills as a ritual. Tools are optional. '
            'Call submit_decision alone. If rejected, correct the reported error. A text response is not '
            'a submission. Keep all notes concise; action and prediction are the only required fields.'
        )
    raise ValueError(f"{state} is not a model reasoning state")
