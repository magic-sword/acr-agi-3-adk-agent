"""Focused model tasks; host owns goal versions, routing and effectful execution."""
from copy import deepcopy
import os
from google.adk.agents import LlmAgent
from agent.local_vlm import LocalVisionLlm
from agent.controls import ACTION_TO_BUTTON, controller_context
from .completion import CompletionTool
from .tasks import (TASKS, COMMON, INSTRUCTIONS, Answer, GoalSelection, GoalAssessment,
                    ExperimentDesign, TargetInspection, EffectJudgment, MethodChoice,
                    SkillArguments, SkillDraft, MissingEvidence, TaskFailure)
from .state import ExperimentPlan, ExperimentReview, EffectFact, Draft, DesignRevision
from .experiments import RepeatedExperiment, action_key, pixels
from .validation import validate_action
from .routing import after_goal, recovery_route
from .machine import destination


class FocusedTasks:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiments.facts_only = bool(self.model)

    def _end_skill(self, outcome, reason):
        active = deepcopy(self.memory.active_skill)
        super()._end_skill(outcome, reason)
        if active and active['trial'] and len(self.library.get(active['id'])['trials']) >= 2:
            self.library.evaluate(active['id'])

    def _goal(self):
        page = self.notebook.pages.get(self.notebook.bookmarks.get('current_subgoal'))
        if page and page['segment'] == self.notebook.segment and page['data']['status'] == 'active':
            return page
        return None

    def _fact(self):
        page = self.notebook.pages.get(self.notebook.bookmarks.get('latest_review'))
        if not page or page['segment'] != self.notebook.segment:
            return None
        d = page['data']
        return {'experiment_id': page['id'], 'action': d['action'],
                'expected': d['plan']['expected'], 'measurement': d.get('measurement'),
                'verdict': d['review']['verdict'], 'finding': d['review']['finding'],
                'evidence_ids': d['review']['evidence_ids']}

    def _focused_work(self):
        if self.work == 'experiment_design':
            self.work = destination('existing_goal' if self._goal() else 'initial_goal')
        elif self.work == 'experiment_review':
            self.work = destination('semantic')

    def _agent(self):
        if self.work not in self.controllers:
            name, contract = TASKS[self.work]
            completion = [CompletionTool(self, name, contract),
                          CompletionTool(self, 'report_missing_evidence', MissingEvidence)]
            self.controllers[self.work] = LlmAgent(
                name=self.work, include_contents='none',
                model=LocalVisionLlm(model='qwen3-vl-4b-instruct',
                    api_base=os.getenv('VLM_API_BASE', 'http://vlm:8080/v1'), max_output_tokens=1800,
                    completion_tools=tuple(t.name for t in completion)),
                instruction=COMMON + INSTRUCTIONS[self.work],
                tools=[*completion, *([self.read_task_evidence] if self.work in (
                    'design_experiment','inspect_target','judge_effect','skill_creation','resolve_arguments') else [])],
                before_tool_callback=self._before_tool, after_tool_callback=self._after_tool,
                on_tool_error_callback=self._tool_error)
        return self.controllers[self.work]

    def _context(self):
        if not self.model:
            return super()._context()
        self._focused_work()
        goal = self._goal()
        c = {'task_id': getattr(self, 'task_id', '') if getattr(self, 'task_work', None) == self.work else '', 'work': self.work,
             'observation_id': self.obs.get('observation_id')}
        goal_view = None if not goal else {'id': goal['id'], 'revision': goal['revision'],
            'text': goal['text'], 'done_when': goal['data']['done_when']}
        if self.work == 'select_goal':
            c.update(parent_goal=self.notebook.pages['goal']['text'], latest_fact=self._fact(),
                     previous_goal=getattr(self, 'closed_goal', None))
        elif self.work == 'assess_goal':
            c.update(goal=goal_view, fact=self._fact(), rejection=getattr(self, 'rejection', None),
                     last_execution={k:v for k,v in (self.outcome or {}).items() if k in (
                         'action','acknowledged','experience_id','boundary','changed_cell_count')})
        elif self.work == 'design_experiment':
            c.update(goal=goal_view, question=getattr(self, 'remaining_question', ''),
                fact=self._fact(), rejection=getattr(self, 'rejection', None),
                target_assessment=getattr(self, 'target_assessment', None),
                legal_actions=[ACTION_TO_BUTTON[a] for a in self.obs.get('available_actions', [])
                               if a in ACTION_TO_BUTTON and a != 'RESET'])
        elif self.work == 'inspect_target':
            c.update(target=self.proposal.target, image_size={
                'width':self.obs.get('width',64), 'height':self.obs.get('height',64)})
        elif self.work == 'judge_effect':
            d = self.experiments.active['data']
            c.update(action=d['action'], expected=d['plan']['expected'],
                     measurement=d['measurement'], evidence_ids=[d['before_id'], d['after_id'],
                     d['outcome']['experience_id']], acknowledged=d['outcome']['acknowledged'])
        elif self.work == 'choose_method':
            c.update(goal=goal_view, skills=self.library.catalog(), learning_enabled=self.learning,
                     experiences=self._eligible_experiences())
        elif self.work == 'resolve_arguments':
            c.update(goal=goal_view, skill=self.read_skill(self.method.skill_id))
        elif self.work == 'skill_creation':
            c.update(request=self.learning_request,
                     experiences={k:self.library.experiences[k] for k in self.learning_request['evidence_ids']})
        if self.errors:
            c['field_error'] = self.errors[-1]
        if getattr(self, 'task_work', None) == self.work:
            c['evidence_refs'] = sorted(getattr(self, 'task_evidence', set()))
        return c

    def _eligible_experiences(self):
        # Receipt and unrelated screen changes alone do not establish a reusable effect.
        supported = {eid for p in self.notebook.pages.values() if p['kind'] == 'experiment'
                     and (p['data'].get('review') or {}).get('verdict') == 'supported'
                     for eid in p['data']['review']['evidence_ids']}
        return {k: {'id':k, 'action':e['action'], 'acknowledged':True,
                    'before_id':e['before']['observation_id'], 'after_id':e['after']['observation_id']}
                for k,e in list(self.library.experiences.items())[-8:]
                if k in supported and e['acknowledged'] and not e['boundary']}

    def _available_methods(self):
        methods = ['explore']
        catalog = self.library.catalog()
        if any(s['status'] == 'active' for s in catalog):
            methods.append('invoke')
        if self.learning and any(s['status'] == 'candidate' and s['fresh_trials'] < 8 for s in catalog):
            methods.append('trial')
        if self.learning and self._eligible_experiences():
            methods.append('learn')
        return methods

    def _apply_boundary(self):
        boundary = bool(self.boundary)
        super()._apply_boundary()
        if boundary:
            self.closed_goal = self.target_assessment = self.rejection = None
            self.remaining_question = ''

    def read_task_evidence(self, observation_id: str) -> dict:
        """Read only an observation explicitly attached to the current task; never advance time."""
        if self.submission is not None:
            return {'error':'the task has already been submitted'}
        if observation_id not in self.task_evidence:
            return {'error':'observation is outside this task snapshot'}
        try:
            return controller_context(self.evidence.view(observation_id))
        except ValueError:
            return {'error':'original evidence is no longer retained'}

    def _open_task(self):
        self.task_sequence = getattr(self, 'task_sequence', 0) + 1
        self.task_id = f'{self.session_id}:task:{self.task_sequence}'
        self.task_work = self.work
        self.task_corrections = 0
        self.task_evidence = {self.obs['observation_id']}
        if self.work == 'judge_effect':
            self.task_evidence.update(self.experiments.active['data'][k] for k in ('before_id','after_id'))
        if self.work == 'skill_creation':
            self.task_evidence.update(e[side]['observation_id'] for k,e in self.library.experiences.items()
                if k in self.learning_request['evidence_ids'] for side in ('before','after'))
        goal = self._goal()
        self.task_binding = (self.obs['observation_id'], None if not goal else (goal['id'], goal['revision']))

    def validate_job(self, job):
        if isinstance(job, (ExperimentReview, EffectFact)) and self.work == 'judge_effect':
            self.experiments.validate_review(job)
            return
        if not isinstance(job, (Answer, SkillDraft)):
            return super().validate_job(job)
        goal = self._goal()
        binding = (self.obs['observation_id'], None if not goal else (goal['id'], goal['revision']))
        if job.task_id != self.task_id or binding != self.task_binding:
            raise ValueError('stale task answer; use the current task_id and goal revision')
        if isinstance(job, TaskFailure) and self.submission is not job:
            raise ValueError('task failures are host-only')
        if not isinstance(job, (TASKS[self.work][1], MissingEvidence, TaskFailure)):
            raise ValueError('answer belongs to another task')
        if isinstance(job, GoalAssessment):
            allowed = set((self._fact() or {}).get('evidence_ids', [])) | {self.obs['observation_id']}
            if self.outcome:
                allowed.add(self.outcome['experience_id'])
            if not set(job.evidence_ids) <= allowed:
                raise ValueError('assessment must cite supplied facts')
        if isinstance(job, ExperimentDesign):
            validate_action(job.action, self.obs)
            self._plan(job)  # Validate structure; repeated tests route in RUN, not tool correction.
        if isinstance(job, TargetInspection) and job.region:
            r = job.region
            if r.x+r.width > self.obs.get('width',64) or r.y+r.height > self.obs.get('height',64):
                raise ValueError('target region is outside the image')
        if isinstance(job, EffectJudgment):
            data = self.experiments.active['data']
            if any(data[key] not in self.evidence.index for key in ('before_id','after_id')):
                raise ValueError('original experiment endpoints are unavailable')
            self.experiments.validate_review(self._review(job))
        if isinstance(job, MethodChoice):
            if job.method not in self._available_methods():
                raise ValueError('method is not available with the current evidence and skills')
            if job.method == 'learn':
                if not self.learning or not set(job.evidence_ids) <= self._eligible_experiences().keys():
                    raise ValueError('learning requires supplied acknowledged effect evidence')
            if job.skill_id:
                record = self.library.get(job.skill_id)
                if record['status'] != ('active' if job.method == 'invoke' else 'candidate'):
                    raise ValueError('skill status does not permit this method')
                if job.method == 'trial' and (not self.learning or len(record['trials']) >= 8):
                    raise ValueError('candidate trial unavailable')
        if isinstance(job, SkillArguments):
            from .library import step_action
            step_action(self.library.spec(self.method.skill_id), 0, job.arguments, self.obs)
        if isinstance(job, SkillDraft):
            if not self.learning:
                raise ValueError('learning disabled')
            from .library import SkillLibrary
            trial = SkillLibrary(self.memory.game_id)
            trial.records, trial.experiences = deepcopy(self.library.records), deepcopy(self.library.experiences)
            trial.sequence = self.library.sequence
            trial.draft(Draft.model_validate(job.model_dump(exclude={'task_id'})))

    def _plan(self, job):
        goal = self._goal()
        if not goal:
            raise ValueError('a fixed active goal is required')
        plan = ExperimentPlan(subgoal={'id':goal['id'], 'parent_id':goal['data']['parent_id'],
                'text':goal['text'], 'done_when':goal['data']['done_when']},
            **job.model_dump(exclude={'task_id','action','target'}))
        if job.retry_of:
            prior = self.experiments._prior(job.retry_of)
            plan.revision = prior['data']['plan'].get('revision')
            if plan.revision:
                plan.revision = DesignRevision.model_validate(plan.revision)
        return plan

    def _review(self, job):
        return EffectFact(experiment_id=self.experiments.active['id'], verdict=job.verdict,
                          finding=job.finding, evidence_ids=job.evidence_ids)

    def _recover(self, reason, details):
        event = {'skill_creation':'build_missing','resolve_arguments':'arguments_missing',
                 'choose_method':'method_missing'}.get(self.work,
                 'test_rejected' if reason=='same_test_unchanged_conditions' else 'target_rejected')
        self._machine_transition(event)
        self.recoveries = getattr(self, 'recoveries', 0) + 1
        self.rejection = {'reason':reason, 'details':details}
        self._record('artifacts', 'task_rejected', work=self.work, recovery=self.recoveries, **self.rejection)
        route = recovery_route(reason, attempts=self.recoveries)
        if route == 'stop':
            self._stop('recovery_budget_exhausted')
        else:
            self.work = route

    def _target_key(self, job):
        action = validate_action(job.action, self.obs).model_dump(exclude_none=True)
        return (self.obs['observation_id'], job.target, tuple(action_key(action).items()))

    def _commit_proposal(self):
        job = self.proposal
        plan = self._plan(job)
        action = validate_action(job.action,self.obs).model_dump(exclude_none=True)
        # Existing storage retains the old revision field for replay compatibility.
        # Its value is derived by the host, never supplied as a model claim.
        previous = self.notebook.pages.get(self.notebook.bookmarks.get('latest_review'))
        delta = {}
        if previous:
            d = previous['data']; old = d['plan']
            delta = {'action':action_key(action)!=action_key(d['action']),
                'observation_scope':plan.expected.model_dump()['region']!=old['expected']['region'],
                'context':(plan.model_dump()['context_region']!=old['context_region'] or
                    pixels(self.obs,old['expected']['region'])!=d['before_pixels'] or
                    pixels(self.obs,old['context_region'])!=d['before_context'] or
                    self.obs.get('levels_completed')!=d['before_level'])}
            if not plan.retry_of and d['review']['verdict'] != 'supported':
                change = next((k for k,v in delta.items() if v), None)
                if change is None:
                    self._recover('same_test_unchanged_conditions', self._fact())
                    return
                plan.revision = DesignRevision(experiment_id=previous['id'], review_revision=previous['revision'],
                    change=change, reconsidered_assumption='Host comparison of the frozen test.', reason=job.question)
        self._record('artifacts','plan_delta', changes=delta)
        try:
            experiment_id = self.experiments.start(plan, action, self.obs)
        except RepeatedExperiment as e:
            self._recover('same_test_unchanged_conditions', {'experiment_id':e.page['id'], 'action':action})
            return
        self._machine_transition('test_accepted')
        self._select(action, plan.expected.description, experiment_id=experiment_id)

    def _apply_focused(self, job):
        if not isinstance(job, (Answer, SkillDraft)):
            return False
        self.validate_job(job)
        if isinstance(job, TaskFailure):
            self.errors.append(job.reason)
            self._stop('invalid_task_output')
        elif isinstance(job, MissingEvidence):
            if self.work in ('skill_creation', 'resolve_arguments', 'choose_method'):
                self._recover('target_mismatch', {'missing_skill_evidence':job.reason})
            else:
                self._stop('missing_evidence: '+job.reason)
        elif isinstance(job, GoalSelection):
            key = f'focused-goal-{self.task_sequence}'
            self.notebook._commit(key,'subgoal',job.text[:80],job.text,[self.obs['observation_id']],
                data={'parent_id':'goal','done_when':job.done_when,'status':'active',
                      'goal_type':job.goal_type,'status_reason':job.reason})
            self.notebook.system_bookmark('current_subgoal',key)
            self.remaining_question = job.text
            self.work = destination('goal_methods' if len(self._available_methods()) > 1 else 'goal_selected')
        elif isinstance(job, GoalAssessment):
            goal = self._goal()
            status = {'continue':'active','completed':'completed','replace':'abandoned','deferred':'active'}[job.decision]
            self.notebook._commit(goal['id'],'subgoal',goal['title'],goal['text'],job.evidence_ids,
                data={**goal['data'],'status':status,'status_reason':job.reason})
            self.remaining_question = job.remaining_question
            self.work = after_goal(job.decision)
            if self.work == 'design_experiment' and not getattr(self, 'rejection', None) and len(self._available_methods()) > 1:
                self.work = destination('goal_choose_method')
            if self.work == 'select_goal':
                self.closed_goal = {'text':goal['text'],'done_when':goal['data']['done_when'],'reason':job.reason}
            if self.work == 'stop':
                self._stop('goal_evidence_insufficient')
        elif isinstance(job, ExperimentDesign):
            self.proposal = job
            self.work = destination('inspect_click')
            if (validate_action(job.action,self.obs).action != 'ACTION6' or
                    getattr(self, 'checked_target', None) == self._target_key(job)):
                self._machine_transition('direct_test')
                self._commit_proposal()
        elif isinstance(job, TargetInspection):
            action = validate_action(self.proposal.action, self.obs)
            region = job.region
            verdict = ('uncertain' if region is None else 'matched' if
                region.x <= action.x < region.x+region.width and region.y <= action.y < region.y+region.height
                else 'mismatched')
            self.target_assessment = {**job.model_dump(exclude={'task_id'}), 'verdict':verdict,
                                      'source':'host_contains_model_region'}
            self._record('artifacts', 'target_checked', **self.target_assessment)
            if verdict == 'matched':
                self.checked_target = self._target_key(self.proposal)
                self._machine_transition('target_found')
                self._commit_proposal()
            else:
                self._recover('target_mismatch', self.target_assessment)
        elif isinstance(job, EffectJudgment):
            self._finish_review(self._review(job), source='agent')
        elif isinstance(job, MethodChoice):
            self.method = job
            if job.method == 'learn':
                self.learning_request = {'evidence_ids':job.evidence_ids,'purpose':job.reason}
                self.work = destination('method_learn')
            else:
                self.work = destination('method_explore' if job.method == 'explore' else 'method_skill')
        elif isinstance(job, SkillArguments):
            self.memory.active_skill = {'id':self.method.skill_id,'arguments':job.arguments,
                'trial':self.method.method=='trial','index':0,'trace':[]}
            self._record('learning','skill_execution_started',skill_id=self.method.skill_id,
                trial=self.method.method=='trial',arguments=job.arguments)
            self._machine_transition('arguments_ready')
            self._advance_skill()
        elif isinstance(job, SkillDraft):
            self.job_result = self.library.draft(Draft.model_validate(job.model_dump(exclude={'task_id'})))
            self.work = destination('skill_built')
        return True
