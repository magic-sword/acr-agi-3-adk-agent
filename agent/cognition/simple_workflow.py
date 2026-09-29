"""One model decision per action: interpret the last result and choose the next trial."""
from copy import deepcopy

from pydantic import Field
from google.genai import types

from agent.controls import ACTION_TO_BUTTON, controller_context
from .deliberation import StageTool
from .focused_workflow import FocusedRuntime
from .state import Contract
from .validation import validate_intent
from .region_masks import decode
from . import object_memory as objects


class NextAction(Contract):
    observation_id: str
    interpretation: str = Field(max_length=300,
        description='Interpret the last acknowledged result conditionally; empty on the first observation.')
    question: str = Field(min_length=1, max_length=200)
    action: str
    object_id: str | None
    expected_effect: str = Field(min_length=1, max_length=200)


def click_point(obj, observation):
    """A deterministic supported pixel near the mask centroid, never a box hole."""
    points = decode(obj.get('mask_runs', []))
    points = {(x, y) for x, y in points
              if 0 <= x < observation['width'] and 0 <= y < observation['height']}
    if not points:
        raise ValueError('selected object has no current clickable mask')
    sx, sy, n = sum(x for x, y in points), sum(y for x, y in points), len(points)
    return min(points, key=lambda p: ((p[0]*n-sx)**2 + (p[1]*n-sy)**2, p[1], p[0]))


def compact_result(result, target_ids):
    """Retain target evidence (including zero/unknown) and actual changes elsewhere."""
    targets = set(target_ids)
    rows = []
    for row in result.get('object_observations', []):
        delta = row.get('delta_xy')
        if (row['object_id'] not in targets
                and not row.get('changed_pixels_on_previous_support')
                and not (delta and any(delta))):
            continue
        rows.append({k: row.get(k) for k in ('object_id', 'correspondence',
            'changed_pixels_on_previous_support', 'delta_xy', 'clicked_previous_mask')})
    measured = {r['object_id'] for r in rows}
    rows.extend(dict(object_id=ident, correspondence='unmeasured',
                     changed_pixels_on_previous_support=None, delta_xy=None,
                     clicked_previous_mask=None) for ident in sorted(targets - measured))
    return {**{k: deepcopy(result.get(k)) for k in ('action', 'acknowledged', 'changed_cell_count')},
            'object_observations': rows}


class NextActionTool(StageTool):
    def _get_declaration(self):
        declaration = super()._get_declaration()
        schema = declaration.parameters_json_schema
        ids = self.runtime._clickable_ids()
        variants = []
        for action in self.runtime.obs['available_actions']:
            if action == 'RESET' or action not in ACTION_TO_BUTTON or (action == 'ACTION6' and not ids):
                continue
            branch = deepcopy(schema)
            branch['properties']['action'] = {'type': 'string', 'enum': [ACTION_TO_BUTTON[action]]}
            branch['properties']['object_id'] = ({'type': 'string', 'enum': ids} if action == 'ACTION6'
                                                   else {'type': 'null'})
            variants.append(branch)
        return types.FunctionDeclaration(name=self.name, description=self.description,
                                         parameters_json_schema={'anyOf': variants})


class SimpleRuntime(FocusedRuntime):
    stage_tool = NextActionTool
    slow_tasks = ('act',)
    stage_tasks = {'act': ('submit_next_action', NextAction)}
    stage_tokens = {'act': 650}
    stage_instructions = {'act': '''Interpret the latest acknowledged action result and
choose ONE next executable action in the same submission. Unknown effects are a
reason to experiment, not to reread an unchanged scene. Use the image and current
object_index. For CLICK select one current object_id: the host clicks a pixel inside
its mask. For other controls use object_id=null. Prefer a useful untested target or
control when the previous attempt had no effect. Explain a discriminating question
and expected effect. No change is a valid observation. A changed pixel elsewhere
does not prove an effect on the target; inspect clicked_previous_mask and target
observations. One trial suggests a conditional hypothesis, not a universal law.
Do not claim success from a hypothesis. Submit submit_next_action; do not request
reunderstanding, completion checks, cursor adjustment, or another planning stage.'''}

    def _clickable_ids(self):
        return [o['object_id'] for o in self.memory.object_memory['objects']
                if o['observation_id'] == self.obs['observation_id'] and o.get('mask_runs')]

    def _machine_transition(self, event):
        return super()._machine_transition('direct_accepted' if event == 'accepted' else event)

    def _snapshot_extra(self):
        return dict(object_index=objects.index(self.memory.object_memory),
                    action_trials=deepcopy(self.memory.trial_ledger[-4:]))

    def _on_observation(self, boundary):
        m = self.memory
        m.cursor = None
        self._measure_observation(boundary)
        if boundary:
            m.episode += 1
            m.trial_ledger.clear()
            m.causal_knowledge.clear()
            m.plan = m.active_skill = None
        elif self.outcome:
            target_ids = list((m.plan or {}).get('object_ids', []))
            trial = dict(observation_id=self.obs['observation_id'], step=self.obs['step'],
                         **compact_result(self.outcome, target_ids))
            self._record('artifacts', 'action_feedback', trial=trial)
            m.trial_ledger = (m.trial_ledger + [dict(
                invocation_id=(self.outcome.get('skill') or {}).get('invocation_id'),
                target_object_ids=target_ids,
                target_objects=[{k: deepcopy(o[k]) for k in ('object_id', 'identity_segment', 'bbox', 'color_ids')}
                                for o in (m.plan or {}).get('target_objects', [])],
                execution_status='acknowledged' if trial['acknowledged'] else 'not_executed_or_unacknowledged',
                expected_effect=(m.plan or {}).get('expected_effect'),
                question=(m.plan or {}).get('question'), actual_trials=[trial],
                interpretation=None)])[-16:]
        m.phase = 'act'
        self._machine_transition('direct_received')
        self._snapshot()

    def _context(self, work=None):
        return controller_context(dict(work='act', observation_id=self.obs['observation_id'],
            remaining_actions=self.obs['remaining_actions'],
            available_actions=self.obs['available_actions'],
            history_note='Object results retain planned targets including zero/unknown, plus changes elsewhere. '
                         'Counts refer to previous mask pixels, not causal proof. Omitted objects are not target evidence.',
            object_index=objects.index(self.memory.object_memory),
            recent_trials=[dict(question=t['question'], target_object_ids=t['target_object_ids'],
                expected_effect=t['expected_effect'],
                target_objects=[{k: o[k] for k in ('object_id', 'identity_segment', 'bbox', 'color_ids')}
                                for o in t['target_objects']],
                interpretation=t['interpretation'],
                results=[{k: r.get(k) for k in ('action', 'acknowledged', 'changed_cell_count', 'object_observations')}
                         for r in t['actual_trials']]) for t in self.memory.trial_ledger[-4:]]))

    def validate_stage(self, work, value):
        if value.observation_id != self.obs['observation_id']:
            raise ValueError('stale observation')
        intent = validate_intent(dict(action=value.action, target_query=value.object_id or ''), self.obs)
        if intent.action == 'ACTION6':
            if value.object_id not in self._clickable_ids():
                raise ValueError('CLICK requires a current object with a mask')
        elif value.object_id is not None:
            raise ValueError('non-click control requires object_id=null')

    def _accept_stage(self, work, value):
        self.validate_stage(work, value)
        m = self.memory
        if m.trial_ledger and self.outcome and self.outcome['acknowledged']:
            m.trial_ledger[-1]['interpretation'] = value.interpretation
        self.invocation_sequence += 1
        m.active_skill = dict(name='single-action', index=0, action_count=0, step_action_count=0,
            invocation_id=f'{m.run_id}:attempt:{self.invocation_sequence}')
        m.plan = dict(question=value.question, object_ids=[value.object_id] if value.object_id else [])
        m.plan.update(target_objects=objects.snapshot(m.object_memory, m.plan['object_ids']),
                      expected_effect=value.expected_effect)
        action = validate_intent(dict(action=value.action, target_query=value.object_id or ''), self.obs)
        self.job = dict(action={'action': action.action}, expected_effect=value.expected_effect)
        if action.action == 'ACTION6':
            obj = next(o for o in m.object_memory['objects'] if o['object_id'] == value.object_id)
            x, y = click_point(obj, self.obs)
            self.job['action'].update(x=x, y=y)
            self.job['binding'] = dict(observation_id=self.obs['observation_id'], object_id=value.object_id,
                x=x, y=y, method='current_mask_centroid_pixel', confirmed=True)
        self._record('artifacts', 'stage_accepted', work=work, result=value.model_dump())
        self._record('artifacts', 'direct_action_selected', job=self.job)
        self._snapshot()

    async def _think(self, ctx):
        if not self._clickable_ids() and set(self.obs['available_actions']) <= {'ACTION6', 'RESET'}:
            self._stop('no_grounded_action')
            return
        await self._deliberate(ctx, 'act')
        if self.result is None and self.job is None:
            self._stop('decision_time_exhausted' if self.time_left() <= 0 else 'stage_output_invalid')
