"""Default workflow: compact purpose, replaceable candidates, scoped evidence."""
import asyncio
from copy import deepcopy
from google.genai import types

from agent.controls import controller_context
from .candidates import CandidateBatch, CandidateGenerator, ModelCandidateGenerator, DEFAULT_VERBS
from .deliberation import StageTool, unique
from .focused_state import FocusedMemory, Scene, Subgoal, PlanChoice
from .perception import context_record
from .state import Reconciliation
from .validation import validate_intent
from .workflow import CognitiveRuntime


class FocusedTool(StageTool):
    def _get_declaration(self):
        # Reuse the controller schema restrictions, but not legacy goal-tree rules.
        declaration = super()._get_declaration()
        schema = declaration.parameters_json_schema
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
            variants.append(branch)
            schema = {'anyOf': variants, '$defs': definitions}
        if self.work == 'candidates':
            schema['properties']['goal_id']['enum'] = [self.runtime.memory.selected_goal_id]
            verbs = list(DEFAULT_VERBS) + [r['verb'] for r in self.runtime.memory.supported_skills.values()]
            schema['$defs']['Candidate']['properties']['verb']['enum'] = list(dict.fromkeys(verbs))
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
    }
    stage_tokens = dict(understand=900, backchain=700, candidates=1100, ground=1600, reconcile=900)
    stage_instructions = {
        'understand': '''Describe only the current relations and at most four useful targets.
Use concept as a short object name; do not build a concept taxonomy. Roles are hypotheses.
Identify a plausible final goal; earlier trial records are historical, not current pixels.
The final goal is an observable game outcome, not 'identify relations' or 'explore'.
Candidate indices, captions and measurement tables are host metadata, not game objects.
Use current measured references if available; leave references empty if unresolved.
Submit submit_scene. Do not propose controls or reproduce historical descriptions.''',
        'backchain': '''Work backwards from final_goal just far enough to choose ONE useful next
subgoal. Explain its link to the final goal in rationale. If a necessary causal relation
is unknown, choose intent=probe and a specific question with an observable discriminating
result. Otherwise choose achieve and question="". No full goal tree or optimal sequence.
Past skills/effects have scoped evidence, not universal truth. Submit submit_subgoal.''',
        'candidates': '''Generate up to three distinct verb + target candidates serving purpose.
Use temporally extended semantic actions such as move_to(the right wall), activate(the
left switch), attack(the obstacle), or a composite skill. Do not substitute a button
press for an abstract action. Each candidate needs an expected observable effect.
New candidates are source=proposed, skill_name=null. Reuse only a supplied supported
skill with its exact target query and effect, with source=reuse and skill_name.
Bind references only to the current targets. Missing effects can be probed; missing
targets should produce an empty list with missing_info. Submit submit_candidates.''',
        'ground': '''Use only purpose, candidates and evidence to choose one candidate and
implement it as a short procedure. Compare relevant past failures and results with
the expected effect. Repetition needs a reason why it remains useful; never assume
an acknowledged click established the effect. Set observable continue_when, done_when,
and reconsider_when. Fast execution can repeat actions and advance through steps.
For a probe set probe_action_limit to a small number of controller actions sufficient
to observe its effect (1 for a single activation; more for movement/composite tests).
Reconciliation occurs by that limit even if nothing changes. For achieve use null.
For a proposed candidate supply procedure; for reuse set procedure=null to use the
stored procedure. CLICK uses a semantic target_query, never coordinates; cursor alignment
is the fast stage's responsibility. For an ungroundable plan set candidate_id=null,
procedure=null and next=understand or backchain with a specific reason.
An executable choice requires next=execute. Submit submit_plan_choice.''',
        'reconcile': '''Compare this invocation's acknowledged trials, before/current observations
and expected relation. A fast completion signal and pixel change are not proof of the
effect. Confirm an achievement only with observed evidence from this invocation.
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
        self.memory = FocusedMemory(**self.memory.model_dump(exclude={'schema_version'}))

    def _on_observation(self, boundary):
        if boundary:
            m = self.memory
            m.final_goal = ''
            m.subgoal = m.candidate_batch = m.selected_candidate = None
            m.trial_ledger.clear()
            self._after_scene = 'backchain'
        super()._on_observation(boundary)

    def _retained_skills(self):
        return {name: record['procedure'] for name, record in self.memory.supported_skills.items()}

    def _route_perception(self):
        m = self.memory
        if (m.phase == 'execute_step' and m.plan and m.plan['intent'] == 'probe'
                and self._current_result() and self.outcome.get('acknowledged')
                and m.active_skill['action_count'] >= m.plan['probe_action_limit']):
            self._queue_review('probe_result', 'The planned probe action limit was reached; assess its result including no change.',
                               'probe_observed')
        super()._route_perception()

    @staticmethod
    def _trial(t):
        return {**{k: deepcopy(t.get(k)) for k in
                   ('observation_id', 'acknowledged', 'frame_changed', 'changed_cell_count')},
                'action': (t.get('action') or {}).get('action')}

    def _attempt_results(self):
        active = self.memory.active_skill
        return [self._trial(t) for t in self.recent_trials if active and
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
            past_trials=deepcopy(m.trial_ledger[-3:]),
            conditional_knowledge=deepcopy(m.causal_knowledge[-3:]),
            available_actions=self.obs['available_actions'])

    def _skill_cards(self):
        # Bounded provisional retrieval. Target applicability must be rechecked by
        # the generator; no old location or observed scene is transferred as fact.
        return [dict(name=name, verb=r['verb'], target_query=r['target_query'], effect=r['procedure']['effect'],
                     when_to_use=r['procedure']['when_to_use'], evidence=r['evidence'],
                     source_observation_id=r['observation_id'], status='supported_in_one_context')
                for name, r in list(self.memory.supported_skills.items())[-4:]]

    def _context(self, work=None):
        work = work or self.work
        m = self.memory
        common = dict(work=work, observation_id=self.obs['observation_id'])
        if self.rejection:
            common['correction'] = self.rejection
        if work == 'understand':
            common.update(measured_objects=context_record(self.perception, inventory=True) if self.perception else None,
                          previous_trial=m.trial_ledger[-1:], question=m.handoff_question)
        elif work == 'backchain':
            common.update(final_goal=m.final_goal, current_scene=m.understanding,
                          evidence=self._evidence(), known_skills=self._skill_cards())
        elif work == 'candidates':
            common.update(purpose=m.subgoal, goal_id=m.selected_goal_id,
                          targets=(m.understanding or {}).get('targets', []),
                          evidence=self._evidence(), known_skills=self._skill_cards())
        elif work == 'ground':
            cards = deepcopy((m.candidate_batch or {}).get('candidates', []))
            for c in cards:
                c.pop('candidate_refs', None)
                if c['source'] == 'reuse':
                    c['procedure'] = self._retained_skills()[c['skill_name']]
            common.update(purpose=m.subgoal, candidates=cards, evidence=self._evidence())
        elif work == 'reconcile':
            common.update(purpose=m.subgoal, plan=m.plan, candidate=m.selected_candidate,
                          review=m.review, attempt_results=self._attempt_results(),
                          evidence=self._evidence(), semantic_answers=self.semantic_answers)
        else:
            common = super()._context(work)
            # Fast control needs measured changes and the active step, not stale
            # global target hypotheses from the last planning observation.
            common.pop('scene_hypotheses', None)
        return controller_context(common)

    def _views(self, slow=False):
        if self.work == 'reconcile':
            return super()._views(slow)
        return [('CURRENT board', self.obs)]

    def _visual_parts(self, slow=False):
        if self.work in ('backchain', 'ground'):
            return []
        # Scene/candidates see the current board; the controller and reviewer
        # retain before/current evidence, including when measurements are clean.
        if self.work in ('candidates', 'execute_step', 'reconcile'):
            # Use the common renderer while bypassing its old clean-frame skip.
            # (Rendering below also supports grid-only unit observations.)
            import json
            parts = []
            for label, observation in self._views(slow):
                parts.append({'type': 'text', 'text': label})
                if observation.get('image_png_base64'):
                    parts.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + observation['image_png_base64']}})
                else:
                    parts.append({'type': 'text', 'text': json.dumps(observation.get('grid'))})
            return parts
        return super()._visual_parts(slow)

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
            for c in value.candidates:
                if any(ref not in refs for ref in c.candidate_refs):
                    raise ValueError('candidate references must belong to current targets')
                if c.verb.lower() in ('click', 'up', 'down', 'left', 'right', 'act', 'undo'):
                    raise ValueError('use a semantic verb such as activate or move_to, not a controller button')
                if c.source == 'reuse':
                    known = m.supported_skills.get(c.skill_name)
                    if not known or c.target_query != known['target_query'] or c.expected_effect != known['procedure']['effect']:
                        raise ValueError('reuse requires a supported skill with matching target and effect')
                elif c.skill_name is not None:
                    raise ValueError('proposed candidate cannot claim a supported skill')
            if not value.candidates and not value.missing_info.strip():
                raise ValueError('empty candidates require missing_info')
        elif work == 'ground':
            if value.next != 'execute':
                if value.candidate_id is not None or value.procedure is not None or value.probe_action_limit is not None:
                    raise ValueError('executable plan cannot reroute to understanding')
                return
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
                    validate_intent(option['action'], self.obs)
        elif work == 'reconcile':
            super().validate_stage(work, value)
            if value.goal_status == 'confirmed' and value.assessment != 'matched':
                raise ValueError('confirmation requires a matched expected relation')
            if value.goal_status == 'confirmed' and not any(t['acknowledged'] for t in self._attempt_results()):
                raise ValueError('confirmation requires acknowledged evidence from this invocation')

    def _accept_stage(self, work, value):
        self.validate_stage(work, value)
        m = self.memory
        data = value.model_dump()
        if work == 'understand':
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
            record = dict(observation_id=value.observation_id, invocation_id=active['invocation_id'] if active else None,
                          purpose=m.subgoal, verb=candidate.get('verb'), target_query=candidate.get('target_query'),
                          expected_effect=candidate.get('expected_effect'), actual_trials=trials,
                          assessment=value.assessment, evidence=value.evidence, causal_interpretation=value.causal_notes)
            m.trial_ledger = (m.trial_ledger + [record])[-8:]
            if value.causal_notes.strip() and any(t['acknowledged'] for t in trials):
                m.causal_knowledge = (m.causal_knowledge + [dict(
                    target_query=candidate.get('target_query'), verb=candidate.get('verb'),
                    interpretation=value.causal_notes, assessment=value.assessment,
                    evidence=value.evidence, observation_id=value.observation_id,
                    invocation_id=record['invocation_id'], episode=m.episode,
                    status='conditional_model_interpretation')])[-16:]
            if active and value.assessment == 'matched' and value.goal_status == 'confirmed':
                procedure = deepcopy(m.skills[active['name']])
                m.supported_skills.pop(active['name'], None)
                m.supported_skills[active['name']] = dict(procedure=procedure, verb=candidate['verb'], target_query=candidate['target_query'],
                    evidence=value.evidence, observation_id=value.observation_id)
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
                m.candidate_batch = None
                # Re-ground current target descriptions before another candidate batch.
                self._after_scene = 'candidates' if value.next == 'ground' and value.goal_status != 'confirmed' else 'backchain'
                m.phase = 'understand'
                self._machine_transition('review_understand')
        self._record('artifacts', 'stage_accepted', work=work, result=data)
        self._snapshot()
