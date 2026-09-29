"""Opt-in 2x2 action-update experiment; no action veto or new runtime states."""
from copy import deepcopy
import os

from .memory_comparison import SeparatedMemoryRuntime, MemoryReview
from .planning_comparison import OneActionPlan


UPDATE_INSTRUCTION = '''
Use this short decision procedure for the NEXT action:
1. Separate action acknowledgement, cursor self-confirmation and independently
verified target contact. Use available execution feedback; unknown contact is not
proof of a miss. Leave exact click alignment to the existing fast cursor.
2. Compare the tested prediction with the measured outcome. No visible change
weakens an immediate-visible-effect prediction under the tested conditions, not
every possible use of that object. Do not copy an old intended effect as a fact.
3. Choose what to change: target, control, or a relevant prerequisite. Alternatively
repeat a supported successful action, or perform a bounded repetition to test a
specific delay, accumulation or stochastic-effect hypothesis. Do not invent a
changed prerequisite when the observation supplies no evidence for it.
4. In rationale cite the previous trial/observation, state what changes (or the
specific purpose and maximum attempts for a repeat). In expected_effect state
the observable outcome that distinguishes the explanations. Make action and
target_query implement that choice. Brief concrete fields suffice; no long essay.
'''


def target_name(candidate):
    # Tracker identity is conservative and episode-scoped. Missing identities are
    # never rebound by color, location, or similarity of a generated description.
    return 'R-' + str(candidate.get('track_id') or candidate['id'])


def appearance(candidate, width, height):
    x0, y0, x1, y1 = candidate['bbox']
    horizontal = ('left', 'middle', 'right')[min(2, int(3 * (x0+x1+1) / (2*width)))]
    vertical = ('upper', 'middle', 'lower')[min(2, int(3 * (y0+y1+1) / (2*height)))]
    w, h = x1-x0+1, y1-y0+1
    extent = 'broad' if w*h > width*height/8 else 'small'
    shape = 'horizontal' if w > 2*h else 'vertical' if h > 2*w else 'compact'
    return f"{vertical}-{horizontal}, {extent} {shape} region; colors: {', '.join(candidate['colors'])}"


def named_objects(perception, grid):
    height, width = len(grid), len(grid[0])
    return [dict(name=target_name(c), candidate_ref=c['id'],
                 appearance=appearance(c, width, height), role='unknown')
            for c in perception['candidates']]


def execution_evidence(result):
    if result is None:
        return dict(acknowledged=False, cursor_report='not_executed', target_contact='unknown')
    click = result.get('action', {}).get('action') == 'ACTION6'
    return dict(acknowledged=bool(result.get('acknowledged')),
                cursor_report=('self_confirmed' if (result.get('binding') or {}).get('confirmed')
                               else 'unconfirmed') if click else 'not_applicable',
                target_contact='unknown' if click else 'not_applicable',
                note='Cursor self-confirmation is not independent verification of target contact.')


def trial_signature(trial):
    action = trial['action']
    target = tuple(trial['target_names'])
    if action['action'] != 'CLICK':
        target = ()
    elif not target:
        target = (action.get('target_query', ''),)
    return action['action'], target, trial['condition_frame_hash']


class ActionUpdateRuntime(SeparatedMemoryRuntime):
    def __init__(self, *args, update_arm=None, **kwargs):
        self.update_arm = update_arm or os.environ['COGNITION_ACTION_UPDATE_COMPARISON']
        if self.update_arm not in ('A', 'B', 'C', 'D'):
            raise ValueError('action update comparison requires A, B, C or D')
        self.trial_ledger = []
        self.proposed_trial = None
        super().__init__(*args, **kwargs)

    def _on_observation(self, boundary):
        if boundary:
            self.trial_ledger.clear()
            self.proposed_trial = None
        super()._on_observation(boundary)

    def _agent(self, work):
        fresh = work not in self.planners
        agent = super()._agent(work)
        if fresh and work == 'ground' and self.update_arm in ('C', 'D'):
            agent.instruction += UPDATE_INSTRUCTION
        return agent

    def _accept_stage(self, work, value):
        actual = deepcopy(self._current_result())
        super()._accept_stage(work, value)
        if isinstance(value, OneActionPlan):
            by_id = {c['id']: c for c in self.perception['candidates']}
            names = sorted({target_name(by_id[ref]) for t in value.targets for ref in t.candidate_refs})
            self.proposed_trial = dict(
                trial_id=self.memory.plan['plan_id'], observation_id=value.observation_id,
                condition_frame_hash=self.obs.get('frame_hash'), baseline=value.baseline,
                target_names=names, target_descriptions=[t.appearance for t in value.targets],
                action=value.action.model_dump(), tested_question=value.question,
                expected_effect=value.expected_effect, rationale=value.rationale)
            self._record('artifacts', 'update_plan', arm=self.update_arm, trial=deepcopy(self.proposed_trial))
        elif isinstance(value, MemoryReview) and actual is not None and self.proposed_trial is not None:
            trial = dict(**deepcopy(self.proposed_trial),
                         decision_id=actual['decision_id'], execution=execution_evidence(actual),
                         measured_result={k: deepcopy(actual.get(k)) for k in
                                          ('frame_changed', 'changed_cell_count', 'boundary')},
                         model_review=dict(assessment=value.assessment, evidence=value.evidence,
                                           causal_notes=value.causal_notes, next_question=value.next_question,
                                           verified_by_host=False))
            if not any(t['decision_id'] == trial['decision_id'] for t in self.trial_ledger):
                self.trial_ledger.append(trial)
                self._record('artifacts', 'update_trial', arm=self.update_arm, trial=trial)

    def _context(self, work=None):
        work = work or self.work
        context = super()._context(work)
        if work != 'ground' or self.update_arm not in ('B', 'D'):
            return context
        # Change presentation at planning only. Review, images, output schema,
        # perception, cursor and executor are identical in all four arms.
        context.pop('measurements', None)
        context.pop('evidence_format', None)
        context.pop('last_completed_trial', None)
        objects = named_objects(self.perception, self.obs['grid'])
        current_names = {o['name'] for o in objects}
        context['current_observation'] = dict(
            observation_id=self.obs['observation_id'], objects=objects,
            changed_pixels=self.perception.get('changed_pixels'),
            motion=[dict(name=target_name(c['after']), delta_xy=c.get('delta_xy'),
                         correspondence=c.get('correspondence')) for c in self.perception['changes']],
            unresolved_count=len(self.perception['unresolved']),
            uncovered_changed_pixels=self.perception.get('uncovered_changed_pixels'),
            proposal_coverage_incomplete=self.perception.get('proposal_coverage_incomplete'),
            note='Names identify measured regions, not semantic roles. Regions may overlap or include background. '
                 'Track continuity is a geometric hypothesis; absent names are unresolved. '
                 'Use current candidate_ref in output; describe the visible instance/part in target_query for the cursor.')
        ledger = []
        for trial in self.trial_ledger[-3:]:
            row = deepcopy(trial)
            frame = row.pop('condition_frame_hash')
            row['same_visible_conditions_now'] = frame == self.obs.get('frame_hash')
            row['missing_current_target_names'] = [n for n in row['target_names'] if n not in current_names]
            # Exact frame equality is not equality of hidden game state.
            row['same_observed_action_condition_count'] = sum(
                trial_signature(t) == trial_signature(trial) for t in self.trial_ledger)
            ledger.append(row)
        context['trial_ledger'] = ledger
        context['ledger_note'] = ('Last 3 completed trials; counts cover this episode. Same visible conditions '
                                  'do not imply identical hidden state. Targets were chosen by the planner, '
                                  'not independently verified at click time. No exact click coordinates here.')
        return context
