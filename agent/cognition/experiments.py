"""One frozen experiment at a time, stored in the shared versioned notebook.

Region changes measure only the declared scope. They do not prove causation or
progress. Semantic verdicts belong to a separate model job with actual evidence.
"""
from copy import deepcopy

from .state import ExperimentReview, EffectFact

class RepeatedExperiment(ValueError):
    def __init__(self, page):
        super().__init__('same action and unchanged tested scope already gave no support; '
                         'redesign the intervention using its actual negative result')
        self.page = page


def pixels(observation, region):
    grid = observation.get('grid')
    if not grid or region is None:
        return None
    x, y, width, height = (region[k] for k in ('x', 'y', 'width', 'height'))
    if y + height > len(grid) or any(x + width > len(row) for row in grid[y:y+height]):
        return None
    return [row[x:x+width] for row in grid[y:y+height]]


def action_key(action):
    return {k: action[k] for k in ('action', 'x', 'y') if k in action and action[k] is not None}


class Experiments:
    def __init__(self, notebook, emit):
        self.notebook, self.emit = notebook, emit
        self.sequence = 0
        self.facts_only = False

    @property
    def active(self):
        key = self.notebook.bookmarks.get('active_experiment')
        return self.notebook.pages.get(key)

    def _prior(self, key):
        page = self.notebook.pages.get(key)
        if not page or page['kind'] != 'experiment' or page['segment'] != self.notebook.segment:
            raise ValueError('experiment must belong to the current level/reset segment')
        return page

    def validate(self, plan, action, obs):
        if self.active:
            raise ValueError('review the active experiment before another operation')
        goal = plan.subgoal
        parent = self.notebook.pages.get(goal.parent_id)
        if (not parent or parent['kind'] not in ('goal', 'subgoal') or
                parent['segment'] != self.notebook.segment or
                parent.get('data', {}).get('status', 'active') != 'active'):
            raise ValueError('subgoal parent must be an active current goal/subgoal')
        path = self.notebook.goal_path(goal.parent_id)
        if any(p.get('data', {}).get('status', 'active') != 'active' for p in path):
            raise ValueError('all goals in the parent path must be active')
        if len(path) >= 4:
            raise ValueError('keep at most four goals in the active path')
        if goal.id:
            old = self.notebook.pages.get(goal.id)
            if (not old or old['kind'] != 'subgoal' or old['segment'] != self.notebook.segment or
                    old['data']['status'] != 'active' or old['data']['parent_id'] != goal.parent_id or
                    old['text'] != goal.text or old['data']['done_when'] != goal.done_when):
                raise ValueError('reuse an active subgoal unchanged, or create a new scoped subgoal')
        for region in (plan.expected.region, plan.context_region):
            if region and (region.x + region.width > obs.get('width', 64) or
                           region.y + region.height > obs.get('height', 64)):
                raise ValueError('experiment region lies outside the observation')
        if plan.retry_of:
            old = self._prior(plan.retry_of)
            frozen = old['data']
            if self.notebook.pages[frozen['subgoal_id']]['data']['status'] != 'active':
                raise ValueError('cannot retry an experiment for a closed subgoal')
            latest = self.notebook.bookmarks.get('latest_review')
            if old['id'] != latest or not frozen.get('review'):
                raise ValueError('retry must reference the latest reviewed experiment')
            if frozen['attempt'] >= frozen['plan']['max_attempts']:
                raise ValueError('predeclared experiment attempt limit reached; redesign the test')
            compare = plan.model_dump()
            compare['retry_of'] = frozen['plan']['retry_of']
            if compare != frozen['plan'] or action_key(action) != action_key(frozen['action']):
                raise ValueError('a retry must preserve the original experiment plan and action')
            return
        # Deliberately ignore prose, hashes and unrelated pixels. Rewording the
        # same failed test or a changing status strip must not bypass this check.
        for old in reversed(list(self.notebook.pages.values())):
            if old['kind'] != 'experiment' or old['segment'] != self.notebook.segment:
                continue
            data = old['data']
            review = data.get('review')
            if not review or review['verdict'] == 'supported':
                continue
            expected = data['plan']['expected']
            scope = plan.expected.model_dump()
            context_region = plan.context_region.model_dump() if plan.context_region else None
            if (action_key(data['action']) == action_key(action) and
                    (self.facts_only or expected['kind'] == scope['kind']) and expected['region'] == scope['region'] and
                    data['before_pixels'] == pixels(obs, scope['region']) and
                    data['plan']['context_region'] == context_region and
                    data['before_context'] == pixels(obs, context_region) and
                    data['before_level'] == obs.get('levels_completed')):
                raise RepeatedExperiment(old)
        self.validate_revision(plan, action, obs)

    def validate_revision(self, plan, action, obs):
        """Check the declared change against evidence, not the novelty of its prose."""
        key = self.notebook.bookmarks.get('latest_review')
        prior = self.notebook.pages.get(key)
        revision = plan.revision
        required = prior and prior['data']['review']['verdict'] != 'supported'
        if revision is None:
            if required:
                raise ValueError('experiment.revision must cite notebook.handoff and explain a concrete change')
            return
        if (not prior or prior['segment'] != self.notebook.segment or
                revision.experiment_id != key or revision.review_revision != prior['revision']):
            raise ValueError('experiment.revision must cite the current handoff experiment_id and review_revision')
        if not revision.reconsidered_assumption.strip() or not revision.reason.strip():
            raise ValueError('revision must explain the reconsidered assumption and why the change is informative')
        d = prior['data']; old = d['plan']
        region = plan.expected.model_dump()['region']
        context = plan.context_region.model_dump() if plan.context_region else None
        changes = {
            'action': action_key(action) != action_key(d['action']),
            # Changing only the prose or criterion kind is not a new observed scope.
            'observation_scope': region != old['expected']['region'],
            'context': (context != old['context_region'] or
                        d['before_pixels'] != pixels(obs, old['expected']['region']) or
                        d['before_context'] != pixels(obs, old['context_region']) or
                        d['before_level'] != obs.get('levels_completed')),
        }
        if not changes[revision.change]:
            raise ValueError(f'declared revision change {revision.change} is absent from the action, scope or observed conditions')

    def start(self, plan, action, obs):
        self.validate(plan, action, obs)
        n = self.notebook
        self.sequence += 1
        key = f'experiment-{self.sequence}'
        goal = plan.subgoal
        goal_id = goal.id
        if not goal_id:
            goal_id = next((p['id'] for p in reversed(list(n.pages.values()))
                           if p['kind'] == 'subgoal' and p['segment'] == n.segment and
                           p['text'] == goal.text and p['data']['parent_id'] == goal.parent_id and
                           p['data']['done_when'] == goal.done_when and p['data']['status'] == 'active'), '')
        if not goal_id:
            goal_id = f'subgoal-{self.sequence}'
            n._commit(goal_id, 'subgoal', goal.text[:80], goal.text, [obs['observation_id']],
                      author='agent', data={'parent_id': goal.parent_id, 'done_when': goal.done_when, 'status': 'active'})
        n.system_bookmark('current_subgoal', goal_id)
        prior = self._prior(plan.retry_of)['data'] if plan.retry_of else None
        data = {'plan': plan.model_dump(), 'subgoal_id': goal_id,
                'goal_path': [{'id': p['id'], 'revision': p['revision']} for p in n.goal_path(goal_id)],
                'action': deepcopy(action), 'before_id': obs['observation_id'],
                'before_pixels': pixels(obs, plan.expected.model_dump()['region']),
                'before_context': pixels(obs, plan.context_region.model_dump() if plan.context_region else None),
                'before_level': obs.get('levels_completed'),
                'attempt': prior['attempt'] + 1 if prior else 1,
                'status': 'awaiting_observation', 'review': None}
        page = n._commit(key, 'experiment', plan.question[:80], plan.hypothesis,
                         [obs['observation_id']], author='agent', data=data)
        n.system_bookmark('active_experiment', key)
        self.emit('experiment_started', experiment=page)
        return key

    def observe(self, obs, outcome):
        page = self.active
        if not page:
            return
        data = deepcopy(page['data'])
        data.update(after_id=obs['observation_id'], outcome=deepcopy(outcome), status='awaiting_review')
        expected = data['plan']['expected']
        after = pixels(obs, expected['region'])
        before = data['before_pixels']
        comparable = before is not None and after is not None
        count = sum(a != b for r, s in zip(before, after) for a, b in zip(r, s)) if comparable else None
        data['measurement'] = {'region': expected['region'], 'changed_cells_in_region': count,
                               'before_level': data['before_level'], 'after_level': obs.get('levels_completed')}
        observed = self.notebook._commit(page['id'], 'experiment', page['title'], page['text'],
                                         [data['before_id'], data['after_id'], outcome['experience_id']],
                                         author='host', data=data)
        self.emit('experiment_observed', experiment=observed)

    def automatic_review(self):
        page = self.active
        if not page or page['data']['status'] != 'awaiting_review':
            return None
        d = page['data']; kind = d['plan']['expected']['kind']; outcome = d['outcome']
        verdict, finding = None, ''
        if not outcome['acknowledged']:
            verdict, finding = 'inconclusive', 'Execution acknowledgement is missing; no effect claim is possible.'
        elif outcome['boundary'] and not (outcome['boundary'] == 'level' and kind == 'level_increased'):
            verdict, finding = 'inconclusive', 'Environment boundary prevents the declared local comparison.'
        elif kind == 'region_changed':
            count = d['measurement']['changed_cells_in_region']
            if count is None:
                verdict, finding = 'inconclusive', 'Comparable pixels for the declared region are unavailable.'
            else:
                verdict = 'supported' if count else 'unsupported'
                finding = f'{count} cells changed in the declared region after this action; this does not establish causation.'
        elif kind == 'level_increased':
            a, b = d['measurement']['before_level'], d['measurement']['after_level']
            verdict = 'inconclusive' if a is None or b is None else 'supported' if b > a else 'unsupported'
            finding = f'Observed level count: {a} -> {b}.'
        if verdict is None:
            return None
        if self.facts_only:
            return EffectFact(experiment_id=page['id'], verdict=verdict, finding=finding,
                              evidence_ids=[outcome['experience_id']])
        attempts_left = d['plan']['max_attempts'] - d['attempt']
        update = ('Expected observation occurred. Decide whether the subgoal is answered or more evidence is needed.'
                  if verdict == 'supported' else
                  f'No support for this test: {action_key(d["action"])}. '
                  + (f'{attempts_left} predeclared attempts remain; retry or redesign.' if attempts_left else
                     'No retries remain. Choose a different informative intervention or check a different prerequisite.'))
        return ExperimentReview(experiment_id=page['id'], verdict=verdict, finding=finding,
            evidence_ids=[outcome['experience_id']], next_step='continue' if verdict == 'supported' else 'revise',
            update=update, understanding={
                'reconsider_assumption': 'Target identity, click placement and effect scope are unverified; the measurement only tests the declared criterion.',
                'open_question': ('Does this measured result answer the subgoal, or is more evidence needed?'
                                 if verdict == 'supported' else
                                 'Was the intended target reached, was the relevant effect observed, or is a prerequisite missing?'),
                'subgoal_reason': 'Kept active: a pixel/level measurement alone cannot decide the subgoal completion condition.'})

    def validate_review(self, review):
        page = self.active
        if not page or page['id'] != review.experiment_id or page['data']['status'] != 'awaiting_review':
            raise ValueError('review must close the active observed experiment')
        d = page['data']
        if d['outcome']['experience_id'] not in review.evidence_ids:
            raise ValueError('review must cite the actual resulting experience ID')
        if not set(review.evidence_ids) <= {d['before_id'], d['after_id'], d['outcome']['experience_id']}:
            raise ValueError('review evidence must come from this experiment')
        if isinstance(review, ExperimentReview) and any(not value.strip() for value in review.understanding.model_dump().values()):
            raise ValueError('review understanding must state an assumption, remaining question and subgoal reason')
        measured = self.automatic_review()
        if measured and review.verdict != measured.verdict:
            raise ValueError('the measured criterion verdict is fixed; explain its implications without changing it')

    def reconsider(self, key):
        """Escalate a host measurement once when design repeats the failed test.

        The model reviews implications for the subgoal, never alters the measured
        result. Original plans and earlier verdict versions remain immutable.
        """
        page = self._prior(key)
        data = deepcopy(page['data'])
        if data.get('reviewer') != 'host' or self.active:
            return False
        data.update(status='awaiting_review', review_request=
                    'The designer repeated this unsupported test. Separate measured facts from explanations. '
                    'Check target placement and effect scope before inferring a rule. State the assumption '
                    'to reconsider, the remaining question and why the subgoal continues or closes; do not choose an action.')
        revised = self.notebook._commit(key, 'experiment', page['title'], page['text'],
                                        page['evidence_ids'], author='host', data=data)
        self.notebook.system_bookmark('active_experiment', key)
        self.notebook.system_bookmark('current_subgoal', data['subgoal_id'])
        self.emit('experiment_review_requested', experiment=revised)
        return True

    def finish(self, review, *, source):
        self.validate_review(review)
        n = self.notebook; page = self.active; d = deepcopy(page['data'])
        d.update(status='reviewed', review=review.model_dump(), reviewer=source)
        result = n._commit(page['id'], 'experiment', page['title'], page['text'],
                           page['evidence_ids'], author=source, data=d)
        if isinstance(review, ExperimentReview):
            goal = n.pages[d['subgoal_id']]
            n._commit(goal['id'], 'subgoal', goal['title'], goal['text'], review.evidence_ids,
                      author=source, data={**goal['data'], 'status': review.subgoal_status,
                                           'latest_experiment': page['id'], 'update': review.update,
                                           'status_reason': review.understanding.subgoal_reason})
        n.system_bookmark('active_experiment', '')
        n.system_bookmark('latest_review', page['id'])
        self.emit('experiment_reviewed', experiment=result)
        return result

    def interrupt(self, reason):
        page = self.active
        if not page:
            return
        d = deepcopy(page['data'])
        d.update(status='interrupted', interruption=reason)
        result = self.notebook._commit(page['id'], 'experiment', page['title'], page['text'],
                                       page['evidence_ids'], author='host', data=d)
        self.notebook.system_bookmark('active_experiment', '')
        self.emit('experiment_interrupted', experiment=result)
