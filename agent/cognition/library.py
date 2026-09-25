"""Executable, bounded skill specifications. Only the host can promote versions.

Stored observations test recorded behavior, never counterfactual transitions.
Fresh trials run through the real driver and consume its normal action budget.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .state import Draft, SkillSpec
from .validation import validate_action

POLICY = 'observed-effects-v1:seed+2-fresh-distinct-trials+negative-guard+regression'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def bind(value, arguments):
    if isinstance(value, str):
        value = arguments.get(value[1:])
    if type(value) is not int or not 0 <= value <= 63:
        raise ValueError('coordinates/arguments must be integers in 0..63')
    return value


def check_arguments(spec, arguments):
    if set(arguments) != set(spec.parameters):
        raise ValueError('arguments must exactly match skill parameters')
    for value in arguments.values():
        bind(value, {})


def condition(c, before, after, arguments):
    """Three-valued, measured predicates. No model-provided facts are evaluated."""
    if c.kind == 'level_increased':
        if 'levels_completed' not in before or 'levels_completed' not in after:
            return None
        return after['levels_completed'] > before['levels_completed']
    a, b = before.get('grid'), after.get('grid')
    if not a or not b or len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        return None
    if c.kind == 'color_count_delta':
        return sum(row.count(c.value) for row in a) != sum(row.count(c.value) for row in b)
    x, y = bind(c.x, arguments), bind(c.y, arguments)
    if y >= len(b) or x >= len(b[y]):
        return None
    if c.kind == 'cell_is':
        return b[y][x] == c.value
    return a[y][x] != b[y][x]


def check_all(conditions, before, after, arguments):
    results = [condition(c, before, after, arguments) for c in conditions]
    return False if False in results else None if None in results else True


def step_action(spec, index, arguments, observation):
    check_arguments(spec, arguments)
    step = spec.steps[index]
    if check_all(step.before, observation, observation, arguments) is not True:
        raise ValueError('skill precondition is false or unknown')
    raw = {'action': step.action, 'reason': spec.description}
    if step.action == 'CLICK':
        raw.update(x=bind(step.x, arguments), y=bind(step.y, arguments))
    return validate_action(raw, observation).model_dump(exclude_none=True)


def matches(spec, arguments, trace):
    if len(trace) != len(spec.steps):
        return False
    for i, experience in enumerate(trace):
        if not experience['acknowledged'] or experience['boundary']:
            # Level completion can be a final, explicitly predicted effect.
            if not (experience['acknowledged'] and i == len(trace)-1 and
                    experience.get('boundary_kind') == 'level' and
                    any(c.kind == 'level_increased' for c in spec.steps[i].after)):
                return False
        if i and trace[i-1]['after']['observation_id'] != experience['before']['observation_id']:
            return False
        try:
            actual = experience['action']
            expected = step_action(spec, i, arguments, experience['before'])
            if any(actual.get(k) != expected.get(k) for k in ('action', 'x', 'y')):
                return False
            if check_all(spec.steps[i].after, experience['before'], experience['after'], arguments) is not True:
                return False
        except (ValueError, IndexError):
            return False
    return True


class SkillLibrary:
    def __init__(self, game_id, directory=None, source=None, emit=None):
        self.game_id, self.directory = game_id, Path(directory) if directory else None
        self.emit = emit or (lambda event, **data: None)
        self.records, self.experiences = {}, {}
        self.sequence = 0
        if source:
            data = json.loads(Path(source).read_text())
            if data.get('policy') != POLICY:
                raise ValueError('skill library evaluation policy mismatch')
            for key, record in data.get('records', {}).items():
                spec = SkillSpec.model_validate(record['spec'])
                if key != digest(spec.model_dump()) or record.get('id') != key:
                    raise ValueError('skill version hash mismatch')
                if record['status'] == 'active':
                    if not self._report_passes(record):
                        raise ValueError('active skill lacks a matching passing host evaluation')
                    self.records[key] = record
        self.save()

    def save(self):
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / 'library.json'
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps({'policy': POLICY, 'records': self.records}, ensure_ascii=False, indent=2))
            temp.replace(path)
            for key, r in self.records.items():
                folder = self.directory / key
                folder.mkdir(exist_ok=True)
                (folder / 'SKILL.md').write_text('---\nname: ' + r['spec']['name'] + '\ndescription: ' +
                    json.dumps(r['spec']['description']) + '\n---\n\n' +
                    'Use this immutable procedure only within its tested game scope.\n'
                    'The host checks procedure.json before each action and verifies each effect.\n')
                (folder / 'procedure.json').write_text(json.dumps(r['spec'], indent=2))
                (folder / 'evaluation.json').write_text(json.dumps(r.get('evaluation'), indent=2))

    def catalog(self):
        return [{'id': k, 'name': r['spec']['name'], 'description': r['spec']['description'],
                 'status': r['status'], 'parameters': r['spec']['parameters'],
                 'fresh_trials': len(r['trials'])}
                for k, r in self.records.items() if r['spec']['game_id'] == self.game_id]

    def get(self, key):
        if key not in self.records:
            raise ValueError('unknown skill_id')
        r = self.records[key]
        if r['spec']['game_id'] != self.game_id:
            raise ValueError('skill is not validated for this game version')
        if digest(r['spec']) != key:
            raise ValueError('skill specification was modified')
        return r

    def spec(self, key):
        return SkillSpec.model_validate(self.get(key)['spec'])

    def add_experience(self, before, action, after, *, acknowledged, boundary_kind=''):
        self.sequence += 1
        keys = ('observation_id', 'step', 'frame_hash', 'grid', 'width', 'height',
                'available_actions', 'levels_completed', 'state')
        experience = {'id': f"experience-{self.sequence}", 'sequence': self.sequence,
                      'before': {k: deepcopy(before[k]) for k in keys if k in before},
                      'after': {k: deepcopy(after[k]) for k in keys if k in after},
                      'action': deepcopy(action), 'acknowledged': acknowledged,
                      'boundary': bool(boundary_kind), 'boundary_kind': boundary_kind}
        self.experiences[experience['id']] = experience
        if len(self.experiences) > 128:
            del self.experiences[next(iter(self.experiences))]
        self.emit('experience_recorded', experience=experience)
        return experience

    def draft(self, proposal: Draft):
        spec = proposal.spec
        if spec.game_id != self.game_id:
            raise ValueError('candidate scope must be this game version')
        if proposal.parent_id:
            parent = self.get(proposal.parent_id)
            if parent['spec']['name'] != spec.name:
                raise ValueError('revision must keep the parent skill name')
        elif any(r['spec']['name'] == spec.name for r in self.records.values()):
            raise ValueError('existing name requires parent_id')
        trace = [self.experiences.get(k) for k in proposal.evidence_ids]
        if any(t is None for t in trace):
            raise ValueError('candidate needs retained real experience IDs')
        cases = []
        for arguments in proposal.examples:
            check_arguments(spec, arguments)
            for start in range(len(trace)-len(spec.steps)+1):
                sample = trace[start:start+len(spec.steps)]
                if matches(spec, arguments, sample):
                    cases.append({'arguments': arguments, 'trace': deepcopy(sample)})
                    break
            else:
                raise ValueError('each seed example must match a complete observed procedure')
        key = digest(spec.model_dump())
        if key in self.records:
            raise ValueError('identical version already exists; gather trials or revise it')
        if len(self.records) >= 32:
            raise ValueError('per-run library capacity reached')
        self.records[key] = {'id': key, 'spec': spec.model_dump(), 'status': 'candidate',
            'parent_id': proposal.parent_id, 'created_sequence': self.sequence,
            'seed_cases': cases, 'trials': [], 'evaluation': None}
        self.save()
        self.emit('skill_drafted', skill_id=key, parent_id=proposal.parent_id,
                  evidence_ids=proposal.evidence_ids, spec=spec.model_dump())
        return {'skill_id': key, 'status': 'candidate', 'next': 'Run fresh trials at distinct applicable starting states, then evaluate.'}

    def finish_trial(self, key, arguments, trace, outcome):
        r = self.get(key)
        verified = bool(trace) and matches(self.spec(key), arguments, trace)
        passed = outcome == 'pass' and verified and all(t['sequence'] > r['created_sequence'] for t in trace)
        r['trials'].append({'arguments': deepcopy(arguments), 'trace': deepcopy(trace),
                            'outcome': 'pass' if passed else outcome if outcome != 'pass' else 'fail'})
        r['evaluation'] = None
        self.save()
        self.emit('skill_trial_finished', skill_id=key, outcome=r['trials'][-1]['outcome'],
                  experience_ids=[t['id'] for t in trace])

    @staticmethod
    def _case_signature(case):
        return digest({'arguments': case['arguments'], 'grid': case['trace'][0]['before'].get('grid')})

    def _checks(self, r):
        spec = SkillSpec.model_validate(r['spec'])
        seeds, trials = r['seed_cases'], r['trials']
        seed_keys = {self._case_signature(c) for c in seeds}
        successful = [t for t in trials if t['outcome'] == 'pass' and
                      matches(spec, t['arguments'], t['trace']) and
                      all(e['sequence'] > r['created_sequence'] for e in t['trace'])]
        fresh = {self._case_signature(t) for t in successful} - seed_keys
        # Host-produced negative fixtures test exclusion, not invented game dynamics.
        excluded = []
        for case in seeds:
            before = deepcopy(case['trace'][0]['before'])
            guard = spec.steps[0].before[0]
            x, y = bind(guard.x, case['arguments']), bind(guard.y, case['arguments'])
            before['grid'][y][x] = (guard.value + 1) % 16
            excluded.append(check_all(spec.steps[0].before, before, before, case['arguments']) is False)
        parent = self.records.get(r.get('parent_id'))
        regression_cases = (parent['seed_cases'] + [c for c in parent['trials'] if c['outcome'] == 'pass']) if parent else []
        return {'seed_replay': bool(seeds) and all(matches(spec, c['arguments'], c['trace']) for c in seeds),
                'negative_guard': bool(excluded) and all(excluded),
                'fresh_behavior': len(fresh) >= 2,
                'no_failed_or_unknown_trials': bool(trials) and all(t['outcome'] == 'pass' for t in trials),
                'parent_regression': all(matches(spec, c['arguments'], c['trace']) for c in regression_cases)}

    def _report_passes(self, r):
        report = r.get('evaluation') or {}
        return (report.get('policy') == POLICY and report.get('candidate_hash') == digest(r['spec']) and
                report.get('status') == 'pass' and report.get('evidence_hash') == digest(
                    {'seeds': r['seed_cases'], 'trials': r['trials']}) and
                all(self._checks(r).values()))

    def evaluate(self, key):
        r = self.get(key)
        if r['status'] != 'candidate':
            raise ValueError('only candidate versions may be evaluated')
        checks = self._checks(r)
        # Existing versions are immutable and explicitly invoked by ID. Check their saved cases.
        checks['library_regression'] = all(digest(other['spec']) == k and self._report_passes(other)
            for k, other in self.records.items() if other['status'] == 'active')
        status = 'pass' if all(checks.values()) else 'unknown' if not checks['fresh_behavior'] and all(
            v for k, v in checks.items() if k not in ('fresh_behavior', 'no_failed_or_unknown_trials')) and not any(
                t['outcome'] == 'fail' for t in r['trials']) else 'fail'
        r['evaluation'] = {'policy': POLICY, 'candidate_hash': key,
            'evidence_hash': digest({'seeds': r['seed_cases'], 'trials': r['trials']}),
            'checks': checks, 'status': status,
            'scope': r['spec']['game_id'], 'limits': 'Observed local effects only; no proof of a causal law or universal generalization. Routing quality is measured in benchmarks.'}
        if status == 'pass':
            r['status'] = 'active'
            if r['parent_id'] and self.records[r['parent_id']]['status'] == 'active':
                self.records[r['parent_id']]['status'] = 'suspended'
        self.save()
        self.emit('skill_evaluated', skill_id=key, evaluation=r['evaluation'])
        if status == 'pass':
            self.emit('skill_promoted', skill_id=key, parent_id=r['parent_id'])
        return deepcopy(r['evaluation'])

    def suspend(self, key, reason):
        r = self.get(key)
        if r['status'] == 'active':
            r['status'] = 'suspended'
            self.save()
            self.emit('skill_suspended', skill_id=key, reason=reason)
