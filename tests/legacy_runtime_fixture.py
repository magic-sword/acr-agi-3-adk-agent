"""Retired controller adapter used only to regress stored contracts and the shared engine.

Production imports CognitiveRuntime exclusively. This adapter is never bundled.
New task routing is covered by test_focused_tasks, not this retired protocol.
"""
from agent.cognition.workflow import *


class LegacyRuntime(ExecutionRuntime):
    def _note_operation(self, operation, *args, **kwargs):
        if self.submission is not None:
            return {'error': 'a job has already been submitted'}
        try:
            return controller_context(operation(*args, **kwargs))
        except ValueError as exc:
            return {'error': str(exc)}


    def read_notebook(self, reference: str = '', query: str = '', offset: int = 0,
                      include_previous: bool = False) -> dict:
        """Open a note/bookmark; without reference search six index entries. Never advances time."""
        return self._note_operation(self.notebook.read, reference, query, offset, include_previous)


    def write_note(self, kind: str, title: str, text: str, evidence_ids: list[str],
                   note_id: str = '', expected_revision: int = 0) -> dict:
        """Write goal/hypothesis/plan/interpretation. Cite observation/experience/note IDs.

        New notes use empty note_id and revision 0. Updates require the read revision.
        For goal, update note_id='goal' with the revision from the opening page.
        Title max 80; text max 400 for goal, otherwise 1000. Host result pages are read-only.
        """
        return self._note_operation(self.notebook.write, kind, title, text, evidence_ids,
                                    note_id, expected_revision)


    def erase_note(self, note_id: str, expected_revision: int, reason: str) -> dict:
        """Withdraw an obsolete note; keep its past versions. Goal and results cannot be erased."""
        return self._note_operation(self.notebook.erase, note_id, expected_revision, reason)


    def set_bookmark(self, name: str, note_id: str) -> dict:
        """Set a named note reference (max six custom bookmarks); empty note_id removes it."""
        return self._note_operation(self.notebook.bookmark, name, note_id)


    def _agent(self):
        if self.work not in self.controllers:
            completion = [CompletionTool(self, 'submit_decision', Decision)]
            instruction = COMMON + DESIGN + skill_instructions('design-experiment')
            note_tools = [self.write_note, self.erase_note, self.set_bookmark]
            if self.work == 'experiment_review':
                completion = [CompletionTool(self, 'submit_review', ExperimentReview)]
                instruction = COMMON + REVIEW + skill_instructions('review-experiment')
                note_tools = []
            elif self.work == 'skill_creation':
                completion = [CompletionTool(self, 'propose_skill', Draft),
                              CompletionTool(self, 'defer_skill', SkillDeferral)]
                instruction = COMMON + BUILD + skill_instructions('skill-creator')
            self.controllers[self.work] = LlmAgent(
                name=self.work,
                include_contents='none',
                model=LocalVisionLlm(model='qwen3-vl-4b-instruct',
                    api_base=os.getenv('VLM_API_BASE', 'http://vlm:8080/v1'), max_output_tokens=1800,
                    completion_tools=tuple(t.name for t in completion)),
                instruction=instruction,
                tools=[*completion, skill_toolset(), self.read_notebook, *note_tools,
                       self.list_observations, self.get_observation, self.compare_observations,
                       self.observe_animation, self.read_skill],
                before_tool_callback=self._before_tool, after_tool_callback=self._after_tool,
                on_tool_error_callback=self._tool_error)
        return self.controllers[self.work]


    def validate_job(self, job):
        if isinstance(job, ExperimentReview):
            if self.work != 'experiment_review':
                raise ValueError('reviews are only accepted in the result review job')
            self.experiments.validate_review(job)
            return
        if self.time_left() <= 0:
            raise ValueError('time budget exhausted')
        if isinstance(job, ExperimentRedesign):
            if self.work != 'experiment_design':
                raise ValueError('redesign belongs to experiment design')
            return
        if isinstance(job, SkillDeferral):
            if self.work != 'skill_creation':
                raise ValueError('only a skill builder can defer construction')
            return
        if isinstance(job, Draft):
            if not self.learning:
                raise ValueError('learning disabled')
            if self.work != 'skill_creation':
                raise ValueError('choose learn before constructing a skill')
            # Draft validation performs the full evidence check on an isolated library.
            trial = SkillLibrary(self.memory.game_id)
            trial.records, trial.experiences = deepcopy(self.library.records), deepcopy(self.library.experiences)
            trial.sequence = self.library.sequence
            trial.draft(job)
            return
        if self.work != 'experiment_design':
            raise ValueError('actions and skill selection belong to experiment design')
        if not set(job.evidence_ids) <= self.library.experiences.keys():
            raise ValueError('unknown or expired experience reference')
        if job.kind == 'learn':
            if not self.learning:
                raise ValueError('learning disabled')
            if self.work != 'experiment_design':
                raise ValueError('already constructing a skill')
            if not all(self.library.experiences[k]['acknowledged'] for k in job.evidence_ids):
                raise ValueError('learning requires acknowledged experiences')
        if job.kind == 'act':
            action = validate_action(job.action, self.obs)
            self.experiments.validate(job.experiment, action.model_dump(exclude_none=True), self.obs)
        if job.kind in ('trial', 'evaluate') and not self.learning:
            raise ValueError('learning disabled')
        if job.skill_id:
            r = self.library.get(job.skill_id)
            expected = 'active' if job.kind == 'invoke' else 'candidate'
            if r['status'] != expected:
                raise ValueError(f'{job.kind} requires a {expected} skill')
            if job.kind != 'evaluate':
                if job.kind == 'trial' and len(r['trials']) >= 8:
                    raise ValueError('candidate trial limit reached; revise from evidence')
                step_action(self.library.spec(job.skill_id), 0, job.arguments, self.obs)


    def _context(self):
        obs = {k: v for k, v in self.obs.items() if k not in VISUAL_PAYLOAD_KEYS | {'grid', 'cursor'}}
        context = {'work': self.work, 'observation': obs, 'notebook': self.notebook.opening(),
            'skills': self.library.catalog(), 'learning_enabled': self.learning,
            'last_tool_result': getattr(self, 'job_result', None), 'errors': self.errors[-1:],
            'remaining_seconds': round(self.time_left(), 2)}
        if self.work == 'skill_creation':
            context['learning_request'] = self.learning_request
        return controller_context(context)


    def _build_graph(self):
        @node(rerun_on_resume=True)
        async def decide(ctx: Context):
            self._record('states', 'state_entered', state='DECIDE', work=self.work, input=self._context())
            try:
                self.trace.append('DECIDE')
                if self.result is not None:
                    return
                self.job = None
                if self.work in ('experiment_review', 'judge_effect'):
                    reconsider = bool(self.experiments.active['data'].get('review_request'))
                    self.job = None if reconsider else self.experiments.automatic_review()
                    if self.job is None:
                        while (self.job is None and self.calls < self.max_calls and
                               self.http_requests < self.max_http_requests and self.time_left() > 0):
                            await self._ask(ctx)
                        if self.job is None:
                            self._stop('review_budget_exhausted')
                    return
                if self.obs['state']=='WIN':
                    self._stop('win')
                elif self.obs.get('evaluation_stop_reason'):
                    self._stop(self.obs['evaluation_stop_reason'])
                elif self.time_left()<=0 or self.obs['remaining_actions']<=0:
                    self._stop('budget_exhausted')
                elif self.obs['state'] in ('NOT_PLAYED','GAME_OVER'):
                    if self.memory.resets >= self.max_resets:
                        self._stop('reset_budget')
                    else:
                        self.memory.resets += 1
                        self._select({'action':'RESET','reason':'start or restart game'}, 'New episode boundary.')
                elif self._advance_skill():
                    pass
                elif not self.model:
                    self._offline_job()
                else:
                    self.job = None
                    while (self.job is None and self.calls < self.max_calls and
                           self.http_requests < self.max_http_requests and self.time_left()>0):
                        await self._ask(ctx)
                    if self.job is None:
                        self._stop('budget_exhausted' if self.time_left()<=0 else 'model_budget')

            finally:
                self._record('states', 'state_exited', state='DECIDE', work=self.work, output={
                    'job': self.job.model_dump() if self.job is not None else None,
                    'result': self.result, 'errors': self.errors})

        @node(rerun_on_resume=True)
        async def run(ctx: Context):
            self._record('states', 'state_entered', state='RUN', work=self.work, input={
                'job': self.job.model_dump() if self.job is not None else None,
                'selected_result': self.result,
                'active_skill': compact(self.memory.active_skill)})
            try:
                self.trace.append('RUN')
                if self.result is None:
                    try:
                        self.validate_job(self.job)
                        if hasattr(self, '_apply_focused') and self.model and self._apply_focused(self.job):
                            pass
                        elif isinstance(self.job, ExperimentReview):
                            self._finish_review(self.job, source='agent' if self.submission is self.job else 'host')
                        elif isinstance(self.job, ExperimentRedesign):
                            self.job_result = {'redesign_required': self.job.model_dump()}
                            self._record('artifacts', 'experiment_redesign_requested', state='RUN',
                                         **self.job.model_dump())
                            if self.experiments.reconsider(self.job.experiment_id):
                                self.work = 'experiment_review'
                            else:
                                self._stop('experiment_redesign_stalled')
                        elif isinstance(self.job, SkillDeferral):
                            self.job_result = {'missing_evidence': self.job.reason}
                            self.work, self.learning_request = 'experiment_design', None
                        elif isinstance(self.job, Draft):
                            self.job_result = self.library.draft(self.job)
                            self.work, self.learning_request = 'experiment_design', None
                        else:
                            job = self.job
                            if job.kind == 'learn':
                                self.work = 'skill_creation'
                                self.learning_request = {'evidence_ids': job.evidence_ids,
                                                         'purpose': job.purpose}
                            else:
                                self.work, self.learning_request = 'experiment_design', None
                            self._record('artifacts', 'decision_accepted', state='RUN', decision=job.model_dump())
                            if job.kind=='act':
                                action = validate_action(job.action, self.obs).model_dump(exclude_none=True)
                                experiment_id = self.experiments.start(job.experiment, action, self.obs)
                                self._select(action, job.experiment.expected.description, experiment_id=experiment_id)
                            elif job.kind=='stop':
                                self._stop('model_stopped')
                            elif job.kind=='learn':
                                self.job_result = {'work': self.work, **self.learning_request}
                            elif job.kind=='evaluate':
                                self.job_result = self.library.evaluate(job.skill_id)
                            else:
                                self.memory.active_skill = {'id':job.skill_id, 'arguments':job.arguments,
                                    'trial':job.kind=='trial', 'index':0, 'trace':[]}
                                self._record('learning','skill_execution_started', skill_id=job.skill_id,
                                             trial=job.kind=='trial', arguments=job.arguments)
                                self._advance_skill()
                    except ValueError as e:
                        self.errors.append(str(e)[:500])
                        self.job_result = {'error': str(e)}
                if self.result is None:
                    return Event(route='DECIDE')
                return Event(output=self.result, state={'cognition': self.memory.model_dump(mode='json')})

            finally:
                self._record('states', 'state_exited', state='RUN', work=self.work, output={
                    'result': self.result, 'tool_result': getattr(self, 'job_result', None),
                    'notebook': self.notebook.opening(),
                    'lifecycle': self.memory.lifecycle, 'errors': self.errors})

        return Workflow(name='skill_learning_loop', edges=[('START',decide), (decide,run), (run,{'DECIDE':decide})])


