"""Default workflow: compact purpose, replaceable candidates, scoped evidence."""
import asyncio
import json
import os
import time

from agent.fast_choice import choose
from agent.local_vlm import LocalVisionLlm
from .simple_workflow import compact_result, click_point
from copy import deepcopy
from google.genai import types

from agent.controls import controller_context
from .candidates import CandidateBatch, CandidateGenerator, ModelCandidateGenerator, DEFAULT_VERBS
from .deliberation import StageTool, unique
from .focused_state import FocusedMemory, Scene, Subgoal, PlanChoice, FastSelection
from .perception import context_record
from .state import Reconciliation
from .validation import validate_intent
from .workflow import CognitiveRuntime
from . import object_memory as objects


class FocusedTool(StageTool):
    def _get_declaration(self):
        # Reuse the controller schema restrictions, but not legacy goal-tree rules.
        declaration = super()._get_declaration()
        schema = declaration.parameters_json_schema
        definitions = schema.get('$defs', {})
        if self.work == 'understand' and 'ObjectTarget' in definitions:
            target = definitions['ObjectTarget']
            known = deepcopy(target)
            ids = [o['object_id'] for o in self.runtime.memory.object_memory['objects']]
            known['required'] = list(dict.fromkeys(known['required'] + ['object_id']))
            known['properties']['object_id'] = {'type':'string', 'enum':ids}
            known['properties']['candidate_refs']['maxItems'] = 0
            known['properties']['candidate_refs']['description'] = 'Use [] with object_id; host supplies its measured references.'
            unresolved = deepcopy(target)
            unresolved['properties']['object_id'] = {'type':'null'}
            current_refs = [c['id'] for c in (self.runtime.perception or {}).get('candidates', [])]
            if current_refs:
                unresolved['properties']['candidate_refs']['items'] = {'type':'string', 'enum':current_refs}
            else:
                unresolved['properties']['candidate_refs']['maxItems'] = 0
            definitions['ObjectTarget'] = {'anyOf':([known] if ids else []) + [unresolved]}
        if 'ActionIntent' in definitions:
            from agent.controls import ACTION_TO_BUTTON
            allowed = [ACTION_TO_BUTTON[a] for a in self.runtime.obs['available_actions'] if a in ACTION_TO_BUTTON and a != 'RESET']
            definitions['ActionIntent']['properties']['action']['enum'] = allowed
            if allowed == ['CLICK']:
                definitions['ActionIntent']['properties']['target_query']['minLength'] = 1
                definitions['ActionIntent']['required'] = ['action', 'target_query']
        if self.work == 'reconcile':
            schema['properties']['goal_id']['enum'] = [self.runtime.memory.plan['goal_id']]
            if not any(t['acknowledged'] for t in self.runtime._attempt_results()):
                schema['properties']['goal_status']['enum'] = ['active', 'unknown']
            if self.runtime.memory.plan['intent'] == 'probe':
                schema['properties']['goal_status']['enum'] = ['active', 'unknown']
                schema['properties']['next']['enum'] = ['understand', 'backchain', 'ground']
        if self.work == 'ground':
            batch = self.runtime.memory.candidate_batch or {'candidates': []}
            schema['properties']['candidate_id']['enum'] = [None] + [c['id'] for c in batch['candidates']]
            # Encode route consistency in the sampling schema too. A nullable
            # procedure alone encourages small models to execute a null plan.
            definitions = schema.pop('$defs', {})
            variants = []
            for source in ('proposed', 'reuse'):
                ids = [c['id'] for c in batch['candidates'] if c['source'] == source]
                if not ids:
                    continue
                branch = deepcopy(schema)
                branch['properties']['next'] = {'type': 'string', 'enum': ['execute']}
                branch['properties']['blocked_by'] = {'type': 'null'}
                branch['properties']['candidate_id'] = {'type': 'string', 'enum': ids}
                branch['properties']['procedure'] = {'$ref': '#/$defs/Procedure'} if source == 'proposed' else {'type': 'null'}
                branch['properties']['probe_action_limit'] = (
                    {'type': 'integer', 'minimum': 1, 'maximum': 16}
                    if self.runtime.memory.subgoal['intent'] == 'probe' else {'type': 'null'})
                variants.append(branch)
            branch = deepcopy(schema)
            branch['properties']['next'] = {'type': 'string', 'enum': ['understand', 'backchain']}
            branch['properties']['candidate_id'] = {'type': 'null'}
            branch['properties']['procedure'] = {'type': 'null'}
            branch['properties']['probe_action_limit'] = {'type': 'null'}
            branch['properties']['blocked_by'] = {'type': 'string', 'enum': ['missing_target', 'unavailable_control']}
            branch['required'] = list(dict.fromkeys(branch['required'] + ['blocked_by']))
            variants.append(branch)
            schema = {'anyOf': variants, '$defs': definitions}
        if self.work == 'candidates':
            schema['properties']['goal_id']['enum'] = [self.runtime.memory.selected_goal_id]
            verbs = list(DEFAULT_VERBS) + [r['verb'] for r in self.runtime.memory.supported_skills.values()]
            schema['$defs']['Candidate']['properties']['verb']['enum'] = list(dict.fromkeys(verbs))
            target_ids = sorted({t['object_id'] for t in (self.runtime.memory.understanding or {}).get('targets', [])
                                 if t.get('object_id')})
            if target_ids:
                candidate_schema = schema['$defs']['Candidate']
                candidate_schema['properties']['object_refs'].update(minItems=1, uniqueItems=True,
                                                                      items={'type': 'string', 'enum': target_ids})
                candidate_schema['required'] = list(dict.fromkeys(candidate_schema.get('required', []) + ['object_refs']))
        return types.FunctionDeclaration(name=self.name, description=self.description, parameters_json_schema=schema)


class FocusedRuntime(CognitiveRuntime):
    slow_tasks = ('understand', 'backchain', 'candidates', 'ground', 'reconcile')
    probe_completes_on_receipt = False
    stage_tool = FocusedTool
    stage_tasks = {
        'understand': ('submit_scene', Scene),
        'backchain': ('submit_subgoal', Subgoal),
        'candidates': ('submit_candidates', CandidateBatch),
        'ground': ('submit_plan_choice', PlanChoice),
        'reconcile': ('submit_reconciliation', Reconciliation),
        'execute_step': ('choose_concrete_action', FastSelection),
    }
    stage_tokens = dict(understand=900, backchain=700, candidates=1100, ground=1600, reconcile=900)
    stage_instructions = {
        'execute_step': '''Translate the current abstract operation and procedure step
into ONE offered controller action for the current scene. Do not generate a new plan.
Return one offered digit. Use 7 only when the specified step is complete and offered.
Use 8 if target/control grounding is invalid. Unknown effects alone do not justify
returning: probes must act to obtain evidence. A prior invocation does not prove this
one complete. Repetition is allowed while useful. Pixel change alone is not success.''',
        'understand': '''Describe only the current relations and at most four useful targets.
Use concept as a short object name; do not build a concept taxonomy. Roles are hypotheses.
Identify a plausible final goal; earlier trial records are historical, not current pixels.
The final goal is an observable game outcome, not 'identify relations' or 'explore'.
Candidate indices, captions and measurement tables are host metadata, not game objects.
Use current measured references if available; leave references empty if unresolved.
Select object_id from object_index when supported, with candidate_refs=[]: the host
fills its references. Use object_id=null when selecting regions for a new grouping.
Whole/part/group entries are
hypotheses. Contact or synchronized movement alone does not prove physical unity.
Name targets by their visible colors, shape and position. Do not use a grouping
method (region, spatial_group), candidate refs or object IDs as an appearance/name.
Submit submit_scene. Do not propose controls or reproduce historical descriptions.''',
        'backchain': '''Work backwards from final_goal just far enough to choose ONE useful next
subgoal. Explain its link to the final goal in rationale. If a necessary causal relation
is unknown, choose intent=probe and a specific question with an observable discriminating
result. Otherwise choose achieve and question="". No full goal tree or optimal sequence.
Unknown effects, uncertain goals and a static initial image call for a probe of a
visible target using an available control. Do not wait for spontaneous movement.
Ask what changes after an intervention; compare changed AND unchanged outcomes.
Past skills/effects have scoped evidence, not universal truth. Submit submit_subgoal.''',
        'candidates': '''Generate up to three distinct verb + target candidates serving purpose.
Use temporally extended semantic actions such as move_to(the right wall), activate(the
left switch), attack(the obstacle), or a composite skill. Do not substitute a button
press for an abstract action. Each candidate needs an expected observable effect.
New candidates are source=proposed, skill_name=null. Reuse only a supplied supported
skill with its exact target query and effect, with source=reuse and skill_name.
Bind references only to the current targets. Use object_refs for target IDs and
candidate_refs for an acted part if needed. related_history includes correspondence
uncertainty and original conditions; a part's result does not apply to the whole.
Missing effects can be probed; ambiguous grouping can motivate inspect with a
discriminating expected observation. Inspect means an intervention on one visible
target followed by comparison, not merely looking at the same image again. Unknown
effects are not missing targets. Missing targets should produce an empty list
with missing_info. Submit submit_candidates.''',
        'ground': '''Use only purpose, candidates and evidence to choose one candidate and
implement it as a short procedure. Compare relevant past failures and results with
the expected effect. Repetition needs a reason why it remains useful; never assume
an acknowledged click established the effect. Set observable continue_when, done_when,
and reconsider_when. Fast execution can repeat actions and advance through steps.
For a probe start with 1 or 2 actions; set probe_action_limit sufficient
to observe its effect (1 for a single activation; more for movement/composite tests).
Reconciliation occurs by that limit even if nothing changes. For achieve use null.
For a proposed candidate supply procedure; for reuse set procedure=null to use the
stored procedure. CLICK uses a semantic target_query and target_object_id from the
chosen candidate's object_refs, never coordinates. The host resolves a current mask
pixel; there is no cursor adjustment model. A CLICK target must name ONE visible object,
not a question about several objects. Probe done_when must refer to the observation
AFTER the intervention, including no change, never the already-visible baseline.
Effect uncertainty and a static scene are reasons to execute a probe, not reroute.
For an ungroundable plan set candidate_id=null, procedure=null, and next=understand
or backchain ONLY with blocked_by=missing_target or unavailable_control. Name the
missing target or unavailable control in reason. For execute use blocked_by=null.
An executable choice requires next=execute. Submit submit_plan_choice.''',
        'reconcile': '''Compare this invocation's acknowledged trials, before/current observations
and expected relation. A fast completion signal and pixel change are not proof of the
effect. Confirm an achievement only with observed evidence from this invocation.
Without acknowledged trials, report uncertainty about execution, never evidence
against an effect. No change after a probe is evidence to choose a different test,
not a reason to wait for the scene to change by itself.
Probes do not confirm the goal. Explain supported/contradicted/unknown causal relations
conditionally; these are attributed interpretations, not universal rules. Choose resume
for an unfinished achievement, ground for a procedure revision, backchain for a new
subgoal, understand for a changed interpretation. Submit submit_reconciliation.''',
    }

    def __init__(self, *args, candidate_generator: CandidateGenerator | None = None, **kwargs):
        self.candidate_generator = candidate_generator if candidate_generator is not None else ModelCandidateGenerator()
        self._after_scene = 'backchain'
        self._goal_sequence = 0
        super().__init__(*args, **kwargs)
        self.memory = FocusedMemory(**self.memory.model_dump(exclude={'schema_version', 'phase'}))
        self.recent_trials = []
        self.semantic_answers = []
        self.plan_sequence = 0
        self.choice_model = LocalVisionLlm(model='qwen3-vl-4b-instruct',
            api_base=os.getenv('VLM_API_BASE', 'http://vlm:8080/v1'), max_output_tokens=1)


    def _retained_skills(self):
        return {name: record['procedure'] for name, record in self.memory.supported_skills.items()}


    def _snapshot_extra(self):
        if not isinstance(self.memory, FocusedMemory):
            return {}
        return dict(object_index=objects.index(self.memory.object_memory),
                    final_goal=self.memory.final_goal, subgoal=self.memory.subgoal,
                    understanding=self.memory.understanding, goals=self.memory.goals,
                    goal_status=self.memory.goal_status, review=self.memory.review,
                    skills=self.memory.skills, selected_candidate=self.memory.selected_candidate,
                    candidate_batch=self.memory.candidate_batch,
                    object_trials=[objects.trial_card(t) for t in self.memory.trial_ledger[-8:]])


    def _targets(self):
        scene = self.memory.understanding or {}
        if scene.get('observation_id') != self.obs['observation_id']:
            return []
        targets = deepcopy(scene.get('targets', []))
        for t in targets:
            t['part_refs'] = sorted(objects.allowed_refs(self.memory.object_memory, t.get('object_id')))
            t['related_history'] = objects.history(self.memory.object_memory, self.memory.trial_ledger,
                                                   [t['object_id']] if t.get('object_id') else [])
        return targets

    def _route_perception(self):
        m = self.memory
        if (m.phase == 'execute_step' and m.plan and m.plan['intent'] == 'probe'
                and self._current_result() and self.outcome.get('acknowledged')
                and m.active_skill['action_count'] >= m.plan['probe_action_limit']):
            self._queue_review('probe_result', 'The planned probe action limit was reached; assess its result including no change.',
                               'probe_observed')

    @staticmethod
    def _trial(t, target_ids=()):
        result = {**{k: deepcopy(t.get(k)) for k in
                   ('observation_id', 'acknowledged', 'frame_changed', 'changed_cell_count')},
                'action': (t.get('action') or {}).get('action')}
        measured = compact_result(t, target_ids)['object_observations']
        if measured:
            intended = [o for o in measured if o['object_id'] in target_ids]
            elsewhere = [o for o in measured if o['object_id'] not in target_ids and
                         (o['changed_pixels_on_previous_support'] or
                          o['delta_xy'] and any(o['delta_xy']))]
            result.update(intended_target_observations=deepcopy(intended),
                          other_object_observations=deepcopy(elsewhere[:4]),
                          other_object_observations_omitted=max(0, len(elsewhere)-4),
                          effect_note='Measured mask changes and hits, not evidence that an action caused the change.')
        return result

    def _attempt_results(self):
        active = self.memory.active_skill
        targets = (self.memory.selected_candidate or {}).get('object_refs', [])
        return [self._trial(t, targets) for t in self.recent_trials if active and
                (t.get('skill') or {}).get('invocation_id') == active['invocation_id']]

    def _evidence(self):
        m = self.memory
        sensor = self.perception or {}
        # Geometry stays with perception and fast execution. Keep the measured fact
        # of change without injecting rectangles, patterns or historical click positions.
        changes = [{**{k: c[k] for k in ('kind', 'changed', 'delta_xy') if k in c},
                    'before_id': (c.get('before') or {}).get('id'),
                    'after_id': (c.get('after') or {}).get('id')}
                   for c in sensor.get('changes', [])[:8]]
        scene = m.understanding or {}
        return dict(current=dict(observation_id=self.obs['observation_id'],
            scene=scene.get('observed') if scene.get('observation_id') == self.obs['observation_id'] else None,
            measured_status=sensor.get('status'), changed_pixels=sensor.get('changed_pixels'),
            changes=changes, changes_omitted=max(0, len(sensor.get('changes', []))-8),
            requires_review=sensor.get('requires_review')),
            past_trials=[objects.trial_card(t) for t in m.trial_ledger[-3:]],
            conditional_knowledge=deepcopy(m.causal_knowledge[-3:]),
            available_actions=self.obs['available_actions'])

    def _skill_cards(self):
        # Bounded provisional retrieval. Target applicability must be rechecked by
        # the generator; no old location or observed scene is transferred as fact.
        current = {o['object_id']: o['identity_segment'] for o in self.memory.object_memory['objects']}
        applicable = [(name, r) for name, r in self.memory.supported_skills.items()
                      if all(current.get(o['object_id']) == o['identity_segment'] for o in r.get('target_objects', []))]
        return [dict(name=name, verb=r['verb'], target_query=r['target_query'], effect=r['procedure']['effect'],
                     when_to_use=r['procedure']['when_to_use'], evidence=r['evidence'],
                     target_object_ids=[o['object_id'] for o in r.get('target_objects', [])],
                     source_observation_id=r['observation_id'], status='supported_in_one_context')
                for name, r in applicable[-4:]]

    def _context(self, work=None):
        work = work or self.work
        m = self.memory
        common = dict(work=work, observation_id=self.obs['observation_id'])
        if self.rejection:
            common['correction'] = self.rejection
        if work == 'understand':
            common.update(measured_objects=context_record(self.perception, inventory=True) if self.perception else None,
                          object_index=objects.index(m.object_memory),
                          previous_trial=[objects.trial_card(t) for t in m.trial_ledger[-1:]], question=m.handoff_question)
        elif work == 'backchain':
            common.update(final_goal=m.final_goal, current_scene=m.understanding,
                          evidence=self._evidence(), known_skills=self._skill_cards())
        elif work == 'candidates':
            common.update(purpose=m.subgoal, goal_id=m.selected_goal_id,
                          targets=self._targets(),
                          evidence=self._evidence(), known_skills=self._skill_cards())
        elif work == 'ground':
            cards = deepcopy((m.candidate_batch or {}).get('candidates', []))
            for c in cards:
                c.pop('candidate_refs', None)
                c['targets'] = [{k: t[k] for k in ('object_id', 'concept', 'appearance', 'role_hypothesis') if k in t}
                                for t in (m.understanding or {}).get('targets', [])
                                if t.get('object_id') in c.get('object_refs', [])]
                c['related_history'] = objects.history(m.object_memory, m.trial_ledger, c.get('object_refs', []), c['verb'])
                if c['source'] == 'reuse':
                    c['procedure'] = self._retained_skills()[c['skill_name']]
            common.update(purpose=m.subgoal, candidates=cards, evidence=self._evidence())
        elif work == 'reconcile':
            common.update(purpose=m.subgoal, plan=m.plan, candidate=m.selected_candidate,
                          review=m.review, attempt_results=self._attempt_results(),
                          evidence=self._evidence(), semantic_answers=self.semantic_answers)
            common['target_history'] = objects.history(m.object_memory, m.trial_ledger,
                (m.selected_candidate or {}).get('object_refs', []), (m.selected_candidate or {}).get('verb'))
        else:
            active = m.active_skill
            procedure = m.skills[active['name']]
            common.update(goal=m.goals.get(m.plan['goal_id']), intent=m.plan['intent'],
                question_to_test=m.plan['question'], baseline=m.plan['baseline'],
                active_skill=active, procedure_effect=procedure['effect'],
                abstract_action=m.selected_candidate,
                current_step={k:v for k,v in procedure['steps'][active['index']].items() if k!='options'},
                last_result=self._trial(self._current_result(), (m.selected_candidate or {}).get('object_refs', []))
                    if self._current_result() else None)
            # Fast control needs measured changes and the active step, not stale
            # global target hypotheses from the last planning observation.
            common.pop('scene_hypotheses', None)
            selected = (m.selected_candidate or {}).get('object_refs', [])
            current = [o for o in m.object_memory['objects'] if o['object_id'] in selected]
            common['current_target_objects'] = dict(observation_id=m.object_memory['observation_id'],
                targets=[{k: o[k] for k in ('object_id', 'bbox', 'color_ids', 'candidate_refs', 'identity_status')} for o in current],
                missing_object_ids=sorted(set(selected)-{o['object_id'] for o in current}),
                note='Current measured grounding for selected IDs. Masks may have holes; confirm the acted part in the current image.')
        return controller_context(common)



    async def _run_slow_stage(self, ctx, work):
        if work != 'candidates':
            await self._deliberate(ctx, work)
            return
        self.rejection = None
        request = deepcopy(self._context('candidates'))
        called = False
        async def propose():
            nonlocal called
            if called:
                raise ValueError('model proposal callback may be called only once per batch')
            called = True
            return await self._deliberate(ctx, 'candidates', accept_result=False)
        try:
            async with asyncio.timeout(max(.001, self.time_left())):
                result = await self.candidate_generator.generate(request, propose)
            if self.result is not None:
                return
            if result is None:
                raise ValueError('candidate provider returned no batch')
            # Validate custom providers as strictly as model tool submissions.
            result = CandidateBatch.model_validate(result.model_dump() if isinstance(result, CandidateBatch) else result)
            self._accept_stage('candidates', result)
        except Exception as exc:
            self._record('artifacts', 'candidate_provider_failed', error=str(exc)[:500])
            self._stop('candidate_provider_invalid')

    def validate_stage(self, work, value):
        m = self.memory
        if value.observation_id != self.obs['observation_id']:
            raise ValueError('stale observation_id')
        if work == 'understand':
            ids = {c['id'] for c in (self.perception or {}).get('candidates', [])}
            if any(ref not in ids for t in value.targets for ref in t.candidate_refs):
                raise ValueError('unknown current target reference')
            by_id = {o['object_id']: o for o in m.object_memory['objects']}
            for t in value.targets:
                if t.object_id is not None and (t.object_id not in by_id or
                        not set(t.candidate_refs) <= objects.allowed_refs(m.object_memory, t.object_id)):
                    raise ValueError('unknown object or incompatible target references')
        elif work == 'backchain':
            if value.intent == 'probe' and not value.question.strip():
                raise ValueError('probe requires a discriminating question')
        elif work == 'candidates':
            if value.goal_id != m.selected_goal_id:
                raise ValueError('stale goal_id')
            if (m.understanding or {}).get('observation_id') != value.observation_id:
                raise ValueError('candidate targets require a current scene')
            unique([c.id for c in value.candidates], 'candidate IDs')
            unique([(c.verb, c.target_query) for c in value.candidates], 'semantic candidates')
            refs = {ref for t in m.understanding['targets'] for ref in t['candidate_refs']}
            target_ids = {t['object_id'] for t in m.understanding['targets'] if t.get('object_id')}
            for ident in target_ids:
                refs.update(objects.allowed_refs(m.object_memory, ident))
            for c in value.candidates:
                if target_ids and not c.object_refs and not c.candidate_refs:
                    raise ValueError('bind a candidate to an offered object or measured part')
                if any(ref not in refs for ref in c.candidate_refs):
                    raise ValueError('candidate references must belong to current targets')
                if not set(c.object_refs) <= target_ids:
                    raise ValueError('candidate objects must belong to current targets')
                allowed = {ref for ident in c.object_refs for ref in objects.allowed_refs(m.object_memory, ident)}
                if c.object_refs and not set(c.candidate_refs) <= allowed:
                    raise ValueError('acted part must belong to the selected object')
                if c.verb.lower() in ('click', 'up', 'down', 'left', 'right', 'act', 'undo'):
                    raise ValueError('use a semantic verb such as activate or move_to, not a controller button')
                if c.source == 'reuse':
                    known = m.supported_skills.get(c.skill_name)
                    if not known or c.target_query != known['target_query'] or c.expected_effect != known['procedure']['effect']:
                        raise ValueError('reuse requires a supported skill with matching target and effect')
                    prior = known.get('target_objects', [])
                    if prior:
                        current = {o['object_id']: o for o in m.object_memory['objects']}
                        if set(c.object_refs) != {o['object_id'] for o in prior} or any(
                                o['object_id'] not in current or current[o['object_id']]['identity_segment'] != o['identity_segment']
                                for o in prior):
                            raise ValueError('reuse requires supported object identity; propose transfer as a new trial')
                elif c.skill_name is not None:
                    raise ValueError('proposed candidate cannot claim a supported skill')
            if not value.candidates and not value.missing_info.strip():
                raise ValueError('empty candidates require missing_info')
        elif work == 'ground':
            if value.next != 'execute':
                if value.blocked_by is None:
                    raise ValueError('rerouting requires missing_target or unavailable_control; unknown effects require a bounded probe')
                if value.candidate_id is not None or value.procedure is not None or value.probe_action_limit is not None:
                    raise ValueError('executable plan cannot reroute to understanding')
                return
            if value.blocked_by is not None:
                raise ValueError('an executable plan cannot claim an execution blocker')
            batch = m.candidate_batch
            if not batch or batch['observation_id'] != value.observation_id or batch['goal_id'] != m.selected_goal_id:
                raise ValueError('planning requires fresh candidates for the current goal')
            c = next((c for c in batch['candidates'] if c['id'] == value.candidate_id), None)
            if c is None:
                raise ValueError('choose an offered candidate')
            if (m.subgoal['intent'] == 'probe') != (value.probe_action_limit is not None):
                raise ValueError('provide a finite action limit for probes only')
            if c['source'] == 'reuse':
                if value.procedure is not None:
                    raise ValueError('reuse cannot overwrite a supported procedure')
                procedure = self._retained_skills()[c['skill_name']]
            else:
                if value.procedure is None:
                    raise ValueError('proposed candidate needs an executable procedure')
                procedure = value.procedure.model_dump()
            for step in procedure['steps']:
                for option in step['options']:
                    intent = validate_intent(option['action'], self.obs)
                    if intent.action == 'ACTION6':
                        self._click_object(option, c)
                    elif option.get('target_object_id') is not None:
                        raise ValueError('target_object_id belongs to CLICK')
        elif work == 'reconcile':
            if not m.plan or value.goal_id != m.plan['goal_id']:
                raise ValueError('stale reconciliation goal')
            if m.plan['intent'] == 'probe' and (value.goal_status == 'confirmed' or value.next == 'resume'):
                raise ValueError('probe cannot confirm a goal or resume after review')
            if value.next == 'resume' and (not m.active_skill or value.goal_status == 'confirmed'):
                raise ValueError('only an unfinished achievement can resume')
            if value.goal_status == 'confirmed' and value.assessment != 'matched':
                raise ValueError('confirmation requires a matched expected relation')
            if value.goal_status == 'confirmed' and not any(t['acknowledged'] for t in self._attempt_results()):
                raise ValueError('confirmation requires acknowledged evidence from this invocation')

    def _accept_stage(self, work, value):
        self.validate_stage(work, value)
        m = self.memory
        data = value.model_dump()
        if work == 'understand':
            data['targets'] = [objects.bind_target(m.object_memory, t) for t in data['targets']]
            m.understanding = data
            if self.proposal_perception is not None:
                self.proposal_perception.bind_targets(data)
            if not m.final_goal:
                m.final_goal = value.goal_hypothesis
            m.phase = self._after_scene
            self._after_scene = 'backchain'
            self._machine_transition('scene_candidates' if m.phase == 'candidates' else 'understood')
        elif work == 'backchain':
            self._goal_sequence += 1
            key = f'subgoal-{self._goal_sequence}'
            m.subgoal = data
            m.backchain = data
            m.selected_goal_id = key
            m.goals = {key: dict(id=key, parent_id=None, desired_state=value.desired_state,
                                target_query=value.target_query, requires=[])}
            m.goal_status = {key: dict(status='active', observation_id=value.observation_id)}
            m.candidate_batch = None
            m.phase = 'candidates'
            self._machine_transition('compact_backchained')
        elif work == 'candidates':
            for c in data['candidates']:
                if not c['object_refs'] and c['candidate_refs']:
                    matches = {
                        t['object_id'] for t in m.understanding['targets'] if t.get('object_id') and
                        set(c['candidate_refs']) <= objects.allowed_refs(m.object_memory, t['object_id'])}
                    if len(matches) == 1:
                        c['object_refs'] = sorted(matches)
            m.candidate_batch = data
            m.phase = 'ground' if value.candidates else 'understand'
            if not value.candidates:
                m.handoff_question = value.missing_info
            self._machine_transition('candidates_ready' if value.candidates else 'candidates_missing')
        elif work == 'ground':
            if value.next != 'execute':
                m.candidate_batch = None
                m.phase = value.next
                m.handoff_question = value.reason
                self._machine_transition('ground_' + value.next)
            else:
                c = deepcopy(next(c for c in m.candidate_batch['candidates'] if c['id'] == value.candidate_id))
                procedure = value.procedure.model_dump() if value.procedure else deepcopy(self._retained_skills()[c['skill_name']])
                m.selected_candidate = c
                # Execution storage is distinct from evidence-supported reusable skills.
                m.skills = {procedure['name']: procedure}
                self.plan_sequence += 1
                self.invocation_sequence += 1
                m.plan = dict(goal_id=m.selected_goal_id, intent=m.subgoal['intent'], question=m.subgoal['question'],
                              probe_action_limit=value.probe_action_limit,
                              target_query=c['target_query'], baseline=m.understanding['observed'],
                              candidates=[procedure['name']], observation_id=value.observation_id,
                              plan_id=f'{m.run_id}:plan:{self.plan_sequence}')
                m.active_skill = dict(name=procedure['name'], index=0,
                    invocation_id=f'{m.run_id}:attempt:{self.invocation_sequence}',
                    started_at=value.observation_id, step_started_at=value.observation_id,
                    action_count=0, step_action_count=0)
                m.active_target_binding = dict(invocation_id=m.active_skill['invocation_id'],
                    target_objects=objects.snapshot(m.object_memory, c.get('object_refs', [])),
                    acted_candidate_refs=list(c['candidate_refs']),
                    acted_part_ids=[o['object_id'] for o in m.object_memory['objects'] if o['kind'] == 'region' and
                                    set(o['candidate_refs']).intersection(c['candidate_refs'])])
                m.review = None
                m.phase = 'execute_step'
                self.plan_anchor = deepcopy(self.obs)
                self._record('artifacts', 'plan_created', plan=m.plan, skills=m.skills)
                self._record('artifacts', 'skill_started', skill=m.active_skill)
                self._machine_transition('focused_grounded')
        elif work == 'reconcile':
            active = m.active_skill
            trials = self._attempt_results()
            candidate = m.selected_candidate or {}
            binding = m.active_target_binding or {}
            if not active or binding.get('invocation_id') != active['invocation_id']:
                binding = {}
            record = dict(observation_id=value.observation_id, invocation_id=active['invocation_id'] if active else None,
                          purpose=m.subgoal, verb=candidate.get('verb'), target_query=candidate.get('target_query'),
                          expected_effect=candidate.get('expected_effect'), actual_trials=trials,
                          assessment=value.assessment, evidence=value.evidence, causal_interpretation=value.causal_notes)
            record.update(target_object_ids=list(candidate.get('object_refs', [])),
                          execution_status='acknowledged' if any(t['acknowledged'] for t in trials) else 'not_executed_or_unacknowledged',
                          actual_trials_omitted=max(0, (active or {}).get('action_count', 0)-len(trials)),
                          target_objects=deepcopy(binding.get('target_objects', [])),
                          acted_candidate_refs=deepcopy(binding.get('acted_candidate_refs', [])),
                          acted_part_ids=deepcopy(binding.get('acted_part_ids', [])),
                          baseline=deepcopy((m.plan or {}).get('baseline')),
                          object_observations=[deepcopy(t.get('object_observations', [])) for t in self.recent_trials
                                               if active and (t.get('skill') or {}).get('invocation_id') == active['invocation_id']])
            m.trial_ledger = (m.trial_ledger + [record])[-16:]
            if value.causal_notes.strip() and any(t['acknowledged'] for t in trials):
                m.causal_knowledge = (m.causal_knowledge + [dict(
                    target_query=candidate.get('target_query'), verb=candidate.get('verb'),
                    interpretation=value.causal_notes, assessment=value.assessment,
                    evidence=value.evidence, observation_id=value.observation_id,
                    invocation_id=record['invocation_id'], episode=m.episode,
                    target_object_ids=record['target_object_ids'],
                    status='conditional_model_interpretation')])[-16:]
            if active and value.assessment == 'matched' and value.goal_status == 'confirmed':
                procedure = deepcopy(m.skills[active['name']])
                m.supported_skills.pop(active['name'], None)
                m.supported_skills[active['name']] = dict(procedure=procedure, verb=candidate['verb'], target_query=candidate['target_query'],
                    evidence=value.evidence, observation_id=value.observation_id,
                    target_objects=deepcopy(record['target_objects']))
                while len(m.supported_skills) > 16:
                    m.supported_skills.pop(next(iter(m.supported_skills)))
            elif candidate.get('source') == 'reuse' and value.assessment == 'unexpected':
                m.supported_skills.pop(candidate['skill_name'], None)
            m.reconciliations = (m.reconciliations + [data])[-6:]
            m.goal_status[value.goal_id] = dict(status=value.goal_status, evidence=value.evidence,
                                               observation_id=value.observation_id)
            m.review = None
            m.handoff_question = value.next_question
            if value.next == 'resume':
                m.phase = 'execute_step'
                self._machine_transition('review_resume')
            else:
                m.active_skill = None
                m.active_target_binding = None
                m.candidate_batch = None
                # Re-ground current target descriptions before another candidate batch.
                self._after_scene = 'candidates' if value.next == 'ground' and value.goal_status != 'confirmed' else 'backchain'
                m.phase = 'understand'
                self._machine_transition('review_understand')
        self._record('artifacts', 'stage_accepted', work=work, result=data)
        self._snapshot()

    def _on_observation(self, boundary):
        m = self.memory
        self._measure_observation(boundary)
        if boundary:
            m.episode += 1
            m.final_goal = ''
            m.understanding = m.backchain = m.subgoal = m.plan = m.active_skill = m.review = None
            m.candidate_batch = m.selected_candidate = m.active_target_binding = None
            m.goals.clear(); m.goal_status.clear(); m.trial_ledger.clear()
            self.recent_trials.clear()
            self._after_scene = 'backchain'
            m.phase = 'understand'
        elif self.outcome:
            targets = (m.selected_candidate or {}).get('object_refs', [])
            trial = dict(observation_id=self.obs['observation_id'], step=self.obs['step'],
                **compact_result(self.outcome, targets), skill=deepcopy(self.outcome.get('skill')),
                frame_changed=self.outcome.get('frame_changed'))
            self._record('artifacts', 'action_feedback', trial=trial)
            if self.outcome['acknowledged']:
                self.recent_trials = (self.recent_trials + [trial])[-8:]
                if self._current_result():
                    m.active_skill['action_count'] += 1
                    m.active_skill['step_action_count'] += 1
        if m.phase == 'execute_step':
            self._route_perception()
        self._snapshot()

    def _current_result(self):
        active = self.memory.active_skill
        previous = (self.outcome or {}).get('skill') or {}
        if active and previous.get('invocation_id') == active['invocation_id'] and previous.get('index') == active['index']:
            return self.outcome
        return None

    def _queue_review(self, trigger, reason, transition):
        m = self.memory
        m.review = dict(trigger=trigger, reason=reason, observation_id=self.obs['observation_id'],
                        invocation=deepcopy(m.active_skill))
        m.phase = 'reconcile'
        self._record('artifacts', 'reconciliation_requested', **m.review)
        self._machine_transition(transition)
        self._snapshot()

    def _click_object(self, option, candidate=None):
        candidate = candidate or self.memory.selected_candidate or {}
        ids = candidate.get('object_refs', [])
        ident = option.get('target_object_id')
        if ident is None and len(ids) == 1:
            ident = ids[0]
        if ident not in ids:
            raise ValueError('CLICK must identify one object from the selected abstract action')
        obj = next((o for o in self.memory.object_memory['objects'] if o['object_id'] == ident
                    and o['observation_id'] == self.obs['observation_id']), None)
        if obj is None:
            raise ValueError('planned click target is missing in this observation')
        click_point(obj, self.obs)
        return obj

    def _visual_parts(self, slow=False):
        if self.work in ('backchain', 'ground'):
            return []
        # Execution resolves the abstract operation against the CURRENT frame.
        if self.obs.get('image_png_base64'):
            return [{'type':'text', 'text':'CURRENT board'},
                    {'type':'image_url', 'image_url':{'url':'data:image/png;base64,'+self.obs['image_png_base64']}}]
        return [{'type':'text', 'text':json.dumps(self.obs.get('grid'))}]

    async def _fast(self, options):
        self.work = 'execute_step'
        options = {**options, '8':dict(kind='reconsider', meaning='Target/control unavailable or grounding contradicted')}
        context = self._context('execute_step')
        context['choices'] = controller_context(options)
        instruction = self.stage_instructions['execute_step']
        self.calls += 1; self.memory.model_calls += 1
        self._record('states', 'state_entered', state='DECIDE', work=self.work, input=context)
        record = await choose(self.choice_model, instruction=instruction,
            parts=self._visual_parts()+[{'type':'text','text':json.dumps(context,separators=(',',':'))}],
            options=options, observe_request=self._request_record, timeout=self.time_left())
        self.http_requests += record['http_requests']
        self._record('model', 'model_decision', work=self.work, context=context, **record)
        self._record('states', 'state_exited', state='DECIDE', work=self.work, output=record)
        if not record['schema_valid']:
            self._queue_review('invalid_fast_output', record['error'], 'step_reconsider')
            return None
        choice = options[record['label']]
        self._record('artifacts', 'fast_selected', work=self.work, label=record['label'], selection=choice)
        if choice['kind'] == 'reconsider':
            self._queue_review('grounding_invalid', 'Check target/control grounding against the actual evidence.', 'step_reconsider')
            return None
        return choice

    async def _think(self, ctx):
        m = self.memory
        while self.result is None and self.job is None and self.time_left() > 0:
            if m.phase in self.slow_tasks:
                await self._run_slow_stage(ctx, m.phase)
                continue
            active = m.active_skill
            step = m.skills[active['name']]['steps'][active['index']]
            options = {str(i+1):dict(kind='action', **o) for i,o in enumerate(step['options'])}
            if m.plan['intent'] == 'achieve' or active['action_count'] > 0:
                options['7'] = dict(kind='advance', meaning='The current step done_when is satisfied')
            choice = await self._fast(options)
            if choice is None:
                continue
            if choice['kind'] == 'action':
                try:
                    intent = validate_intent(choice['action'], self.obs)
                    job = dict(action={'action':intent.action}, expected_effect=choice['expected_effect'])
                    if intent.action == 'ACTION6':
                        obj = self._click_object(choice)
                        x,y = click_point(obj, self.obs)
                        job['action'].update(x=x,y=y)
                        job['binding'] = dict(observation_id=self.obs['observation_id'], object_id=obj['object_id'],
                            x=x,y=y,method='current_mask_centroid_pixel',confirmed=True)
                except ValueError as exc:
                    self._queue_review('grounding_invalid', str(exc), 'step_reconsider')
                    continue
                self.job = job
                self._record('artifacts', 'concrete_action_selected', job=job, abstract_action=m.selected_candidate)
                break
            if active['index']+1 < len(m.skills[active['name']]['steps']):
                active['index'] += 1
                active['step_action_count'] = 0
                active['step_started_at'] = self.obs['observation_id']
                self._machine_transition('step_done')
            else:
                self._queue_review('completion_candidate', 'Assess the step completion against acknowledged evidence.', 'procedure_done')
        if self.result is None and self.job is None and self.time_left() <= 0:
            self._stop('decision_time_exhausted')
