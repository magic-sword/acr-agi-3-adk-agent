"""Mandatory state tasks and submission contracts, independent of optional skills."""

from .completion import COMPLETION_TOOLS


def instruction(state: str, schema: dict) -> str:
    import json
    from copy import deepcopy
    schema = deepcopy(schema)
    action_schema = schema.get('$defs', {}).get('Action', {})
    for field in ('x', 'y'):
        action_schema.get('properties', {}).pop(field, None)
    if 'action' in action_schema.get('properties', {}):
        action_schema['properties']['action']['enum'] = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'ACT', 'CLICK', 'UNDO']
    task = {
        'VERIFY': (
            'Interpret the result of the recorded action in relation to its pending predictions and experiment. '
            'Compare the referenced BEFORE observation with the current AFTER observation. '
            'Return Interpretation: current visible facts with evidence, supported or refuted hypotheses, '
            'specific remaining unknowns, and a short summary with evidence_refs. '
            'Do not invent a goal or an action. The host evaluates predicates and publishes updates. '
            'Pixel changes alone prove neither causation nor progress. Account for intervening actions, '
            'delayed effects, partial visibility and boundary changes.'),
        'PLAN': (
            'Consider plausible success CONDITIONS, not necessarily a goal object. The initial goal may be unknown. '
            'Interpret relevant current/past evidence and propose a short conditional plan if warranted. '
            'If an uncertainty changes the next decision, return status=need_evidence, purpose=plan, '
            'and a specific inquiry (no action, plan or experiment) to route to PROBE. '
            'Alternatively provide a complete discriminating experiment with purpose=probe. '
            'Unrelated unknowns do not prevent planning.'),
        'PROBE': (
            'Design ONE experiment addressing inquiry or a specific decision-relevant uncertainty. '
            'Use past observations/results to avoid repeating an already answered experiment. '
            'Return purpose=probe, action and experiment. experiment.alternatives is an OBJECT with '
            'at least two named ARRAYS of distinct predicate objects. Include question, discriminator and risk. '
            'A goal need not be known to test a control. Read before/after evidence if previous experiments exist.'),
        'REVISE': (
            'Reconsider the assumption contradicted by evidence, including goal conditions, control binding '
            'or dynamics. Preserve unrelated knowledge. Return a revised plan, a complete experiment, '
            'or status=need_evidence with a specific inquiry. Include evidence for revisions.'),
    }[state]
    return (
        f'You solve an unknown visual game. Current reasoning state: {state}. {task}\n'
        f'Finish by calling {COMPLETION_TOOLS[state]} with arguments matching the schema. '
        'A text response does not complete the state. Call the completion tool alone. '
        'If rejected, correct the reported error and resubmit within the request budget. '
        'Use only supplied observations and memory. '
        'The attached image is the current final frame. OBSERVE only records evidence; YOU interpret it for this task. '
        'Use list_observations, get_observation(observation_id), compare_observations(before_id,after_id,offset) '
        'and observe_animation(event_id,start_frame) to retrieve retained evidence. Replay never advances time. '
        'Facts are visible measurements, hypotheses are interpretations. Do not label an object as a confirmed goal. '
        'Only the environment WIN signal proves game success. Missing or ambiguous evidence may remain unknown. '
        'For Proposal put knowledge changes in interpretation; echo observation_id and memory_revision there too. '
        'For Interpretation omit goal/unknowns to retain them; goal="" withdraws a goal, unknowns=[] clears unknowns. '
        'Goal changes and summaries require evidence_refs; facts require the current observation ID as evidence. '
        'Hypotheses may cite any supplied observation ID; mode/dynamics/goal change requires change_evidence. '
        'Use ONLY buttons in observation.available_actions; the schema enum includes buttons unavailable in this game. '
        'UP/DOWN/LEFT/RIGHT name inputs, not proven effects. CLICK uses the host cursor: call move_cursor(x,y) '
        'to inspect the intended target before CLICK. CLICK has no x/y fields. Cursor movement is not an action. '
        'Predicates: cell needs x,y and integer color ID; fact needs key,value; state needs exact game state; '
        'levels_min needs integer; frame_changed needs boolean and does not imply useful progress. '
        'Plan dependencies must be acyclic; status must be todo; action nodes need completion and effects. '
        'delay_steps counts external actions. Hypothesis scope is level or game. '
        'Include observation_id and memory_revision copied from input. Both are REQUIRED even for need_evidence. '
        'For need_evidence include purpose=plan, status=need_evidence and inquiry, with no plan/action/experiment. '
        'An experiment is purpose=probe, never a plan node with empty completion/effects. '
        'Do not fill fields with template answers.\n'
        'JSON schema:\n' + json.dumps(schema, separators=(',', ':'))
    )

