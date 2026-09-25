"""Evidence-backed persistent symbolic models and typed access to bundled skill code."""
from __future__ import annotations

from contextlib import closing
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
from typing import Literal

from google.adk.tools import BaseTool
from google.genai import types
from pydantic import Field, model_validator

from .state import Action, Contract
from .validation import validate_action
from agent.observation import render_current, png_base64

SKILL_ROOT = Path(__file__).resolve().parents[1] / 'skills'
SCRIPT_BINDINGS = {
    'cause-diagnosis': 'scripts/induction.py',
    'backward-planning': 'scripts/regression.py',
}


def _load(skill):
    path = SKILL_ROOT / skill / SCRIPT_BINDINGS[skill]
    spec = importlib.util.spec_from_file_location('arc_' + skill.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


learning, planning = _load('cause-diagnosis'), _load('backward-planning')


class Symbols(Contract):
    """Ground atoms; absence means unknown, not false. Values are never executed."""
    true: list[str] = Field(default_factory=list, max_length=32)
    false: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode='after')
    def consistent(self):
        if set(self.true) & set(self.false):
            raise ValueError('a symbol cannot be both true and false')
        if any(not p.strip() or len(p) > 100 for p in self.true + self.false):
            raise ValueError('symbols must be nonempty names of at most 100 characters')
        self.true = sorted(set(self.true))
        self.false = sorted(set(self.false))
        return self


class TransitionInput(Contract):
    id: str = Field(default='', max_length=100)
    before_id: str
    after_id: str
    action_name: str = Field(min_length=1, max_length=100)
    before: Symbols
    after: Symbols
    scope: Literal['level', 'game'] = 'level'
    confounded: bool = False


class RuleInput(Contract):
    id: str = Field(min_length=1, max_length=100)
    action_name: str = Field(min_length=1, max_length=100)
    actions: list[Action] = Field(min_length=1, max_length=12)
    preconditions: Symbols
    effects: Symbols
    scope: Literal['level', 'game'] = 'level'

    @model_validator(mode='after')
    def effect_required(self):
        if not self.effects.true and not self.effects.false:
            raise ValueError('a rule needs at least one effect')
        return self


class MemoryQuery(Contract):
    operation: Literal['read', 'update'] = 'read'
    transitions: list[TransitionInput] = Field(default_factory=list, max_length=8)
    rules: list[RuleInput] = Field(default_factory=list, max_length=8)
    delete_rule_ids: list[str] = Field(default_factory=list, max_length=16)
    delete_transition_ids: list[str] = Field(default_factory=list, max_length=16)
    induce: bool = True
    action_name: str | None = None
    observation_ids: list[str] = Field(default_factory=list, max_length=2)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=8, ge=1, le=16)

    @model_validator(mode='after')
    def read_is_readonly(self):
        if self.operation == 'read' and (self.transitions or self.rules or self.delete_rule_ids or self.delete_transition_ids):
            raise ValueError('use operation=update to modify causal memory')
        return self


class PlanQuery(Contract):
    observation_id: str
    current: Symbols
    goal: Symbols = Field(default_factory=lambda: Symbols(true=['stage_clear']))
    allow_candidates: bool = False
    max_depth: int = Field(default=12, ge=0, le=24)
    max_nodes: int = Field(default=2000, ge=1, le=5000)

    @model_validator(mode='after')
    def goal_required(self):
        if not self.goal.true and not self.goal.false:
            raise ValueError('a planning goal cannot be empty')
        return self


class CausalStore:
    """One SQLite document per full game/version ID; transactional updates and tombstones."""
    def __init__(self, game_id, directory):
        self.game_id = game_id
        self.path = Path(directory) / (hashlib.sha256(game_id.encode()).hexdigest() + '.sqlite3')

    def _empty(self):
        return {'schema_version': 1, 'game_id': self.game_id, 'revision': 0,
                'observations': {}, 'transitions': {}, 'rules': {}, 'deleted_rules': []}

    def _decode(self, row):
        data = json.loads(row[0]) if row else self._empty()
        if data['game_id'] != self.game_id or data['schema_version'] != 1:
            raise ValueError('incompatible causal memory namespace/schema')
        return data

    def read(self):
        if not self.path.exists():
            return self._empty()
        with closing(sqlite3.connect(self.path, timeout=2)) as db:
            return self._decode(db.execute('SELECT body FROM knowledge WHERE id=1').fetchone())

    def mutate(self, callback):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=2)) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS knowledge (id INTEGER PRIMARY KEY, body TEXT NOT NULL)')
            db.execute('BEGIN IMMEDIATE')
            data = self._decode(db.execute('SELECT body FROM knowledge WHERE id=1').fetchone())
            result = callback(data)
            if len(data['transitions']) > 1024 or len(data['rules']) > 256 or len(data['deleted_rules']) > 1024:
                raise ValueError('causal memory capacity reached; read and delete obsolete entries')
            data['revision'] += 1
            db.execute('INSERT OR REPLACE INTO knowledge VALUES (1, ?)', (json.dumps(data),))
            return result

    def archive(self, observation, *, run_id, source_action=None, boundary=False):
        keys = ('observation_id', 'step', 'grid', 'frame_hash', 'state', 'levels_completed',
                'available_actions', 'cursor', 'width', 'height')
        record = {k: observation[k] for k in keys if k in observation}
        if not record.get('grid') and observation.get('image_png_base64'):
            record['image_png_base64'] = observation['image_png_base64']
        record.update(run_id=run_id, source_action=source_action, boundary=boundary)
        def update(data):
            data['observations'][record['observation_id']] = record
            while len(data['observations']) > 128:
                del data['observations'][next(iter(data['observations']))]
        self.mutate(update)

    @staticmethod
    def relevant(rule, level):
        return rule['scope'] == 'game' or rule['level'] == level

    def overview(self):
        data = self.read()
        observations = list(data['observations'].values())
        return {'rules': len(data['rules']), 'transitions': len(data['transitions']),
                'revision': data['revision'], 'recent_evidence': [
                    {k: o.get(k) for k in ('observation_id', 'step', 'levels_completed', 'state', 'source_action')}
                    for o in observations[-3:]]}


def _clean_action(action):
    # Rule descriptions may refer to later inputs; validate vocabulary/coordinates, not current applicability.
    value = validate_action(action, {'available_actions': [f'ACTION{i}' for i in range(1, 8)],
                                     'width': 64, 'height': 64})
    return value.model_dump(exclude={'reason'}, exclude_none=True)


def _transition(item, data):
    observations = data['observations']
    if item.before_id not in observations or item.after_id not in observations:
        # Source endpoints for a correction survive eviction with the original transition.
        old = data['transitions'].get(item.id, {})
        observations = {**observations, **old.get('source_observations', {})}
    try:
        before, after = observations[item.before_id], observations[item.after_id]
    except KeyError as exc:
        raise ValueError('unknown/evicted observation ID; read retained causal evidence') from exc
    if before['run_id'] != after['run_id'] or not 0 < after['step'] - before['step'] <= 12:
        raise ValueError('a transition requires ordered observations in the same run, at most 12 actions')
    frames = sorted((o for o in observations.values() if o['run_id'] == before['run_id']
                     and before['step'] < o['step'] <= after['step']), key=lambda o: o['step'])
    if len(frames) != after['step'] - before['step']:
        raise ValueError('missing intermediate action evidence')
    actions = []
    for i, o in enumerate(frames):
        source = o.get('source_action')
        if not source or source['step'] != o['step'] - 1 or source['action']['action'] == 'RESET':
            raise ValueError('unacknowledged action or RESET is not a causal training transition')
        if o.get('boundary') and i != len(frames) - 1:
            raise ValueError('an abstract action cannot span intermediate level/reset boundaries')
        actions.append(_clean_action(source['action']))
    cleared = after.get('levels_completed', 0) > before.get('levels_completed', 0) or after['state'] == 'WIN'
    if after.get('boundary') and not cleared:
        raise ValueError('RESET/full-reset boundary is not causal effect evidence')
    b, a = item.before.model_dump(), item.after.model_dump()
    # Only environment evidence can label this reserved completion predicate.
    if 'stage_clear' in b['true'] or ('stage_clear' in a['true'] and not cleared) or ('stage_clear' in a['false'] and cleared):
        raise ValueError('stage_clear contradicts the observed environment outcome')
    b['false'] = sorted(set(b['false']) | {'stage_clear'})
    if cleared:
        if any(p != 'stage_clear' for p in a['true'] + a['false']):
            raise ValueError('after level completion, only stage_clear describes the old board; omit new-board facts')
        a = {'true': ['stage_clear'], 'false': []}
    else:
        a['false'] = sorted(set(a['false']) | {'stage_clear'})
    identifier = item.id or 'trial-' + hashlib.sha256(
        (item.before_id + item.after_id + item.action_name).encode()).hexdigest()[:20]
    old = data['transitions'].get(identifier)
    if old and (old['before_id'], old['after_id']) != (item.before_id, item.after_id):
        raise ValueError('a correction must retain its original source observations')
    if any(e['id'] != identifier and (e['before_id'], e['after_id'], e['action_name']) ==
           (item.before_id, item.after_id, item.action_name) for e in data['transitions'].values()):
        raise ValueError('this trial is already recorded; correct its existing ID instead of duplicating evidence')
    return {'id': identifier, 'before_id': item.before_id, 'after_id': item.after_id,
            'action_name': item.action_name, 'actions': actions, 'before': b, 'after': a,
            'scope': item.scope, 'level': before.get('levels_completed', 0) if item.scope == 'level' else None,
            'confounded': item.confounded, 'abstract_macro': len(actions) > 1,
            'source_observations': {o['observation_id']: o for o in [before] + frames}}


def assessed_rules(data, level):
    examples = list(data['transitions'].values())
    return [learning.assess(r, examples) for r in data['rules'].values() if CausalStore.relevant(r, level)]


def _stored_observation(data, oid):
    if oid in data['observations']:
        return data['observations'][oid]
    for example in data['transitions'].values():
        if oid in example['source_observations']:
            return example['source_observations'][oid]
    raise ValueError('observation unavailable; read the observation_index or recorded transition endpoints')


def _visual_evidence(observations):
    """Old evidence reaches the VLM as labeled images, not base64 embedded in JSON."""
    import base64
    import io
    from PIL import Image, ImageDraw
    images = []
    for o in observations:
        label = f'RECORDED step {o["step"]}: {o["observation_id"]}'
        if o.get('grid'):
            image = render_current(o['grid'], o.get('available_actions', []), o.get('cursor'), label=label)
        elif o.get('image_png_base64'):
            raw = Image.open(io.BytesIO(base64.b64decode(o['image_png_base64']))).convert('RGB')
            image = Image.new('RGB', (raw.width, raw.height + 32), 'white')
            image.paste(raw, (0, 32))
            ImageDraw.Draw(image).text((4, 4), label, fill='black')
        else:
            continue
        images.append(image)
    if not images:
        return None
    sheet = Image.new('RGB', (sum(i.width for i in images), max(i.height for i in images)), 'white')
    x = 0
    for image in images:
        sheet.paste(image, (x, 0))
        x += image.width
    return png_base64(sheet)


class CausalMemoryTool(BaseTool):
    def __init__(self, runtime):
        super().__init__(name='causal_memory', description=(
            'Read or transactionally update persistent, game-scoped causal knowledge. '
            'operation=update can record observed symbolic transitions, induce rules, upsert/delete rules '
            'and delete mistaken transitions together. Concrete actions and stage_clear labels come from '
            'real archived observations. Unknown facts are omitted, not false. '
            'Read with observation_ids to inspect old evidence, including final level/WIN transitions. '
            'Recorded symbols remain your interpretations; supported means sample consistency, not proof.'))
        self.runtime = runtime

    def _get_declaration(self):
        return types.FunctionDeclaration(name=self.name, description=self.description,
                                         parameters_json_schema=MemoryQuery.model_json_schema())

    async def run_async(self, *, args, tool_context):
        try:
            q = MemoryQuery.model_validate(args)
            runtime = self.runtime
            if runtime.turn.time_left() <= 0:
                raise ValueError('reasoning budget exhausted')
            learned = None
            if q.operation == 'update':
                def update(data):
                    for oid in q.observation_ids:
                        _stored_observation(data, oid)  # Validate read selectors before committing any writes.
                    for identifier in q.delete_transition_ids:
                        data['transitions'].pop(identifier, None)
                    for identifier in q.delete_rule_ids:
                        data['rules'].pop(identifier, None)
                        if identifier not in data['deleted_rules']:
                            data['deleted_rules'].append(identifier)
                    for item in q.transitions:
                        value = _transition(item, data)
                        data['transitions'][value['id']] = value
                    for item in q.rules:
                        value = item.model_dump()
                        value.update(actions=[_clean_action(a) for a in item.actions], origin='proposed',
                                     level=runtime.turn.memory.level if item.scope == 'level' else None)
                        data['rules'][item.id] = value
                        if item.id in data['deleted_rules']:
                            data['deleted_rules'].remove(item.id)
                    result = None
                    if q.induce:
                        result = learning.induce(list(data['transitions'].values()),
                                                 seconds=min(0.5, runtime.turn.time_left()))
                        for rule in result['rules']:
                            if rule['id'] not in data['deleted_rules']:
                                data['rules'][rule['id']] = rule
                    for oid in q.observation_ids:
                        _stored_observation(data, oid)
                    return result
                learned = runtime.causal.mutate(update)
            data = runtime.causal.read()
            rules = assessed_rules(data, runtime.turn.memory.level)
            if q.action_name:
                rules = [r for r in rules if r['action_name'] == q.action_name]
            examples = [e for e in data['transitions'].values()
                        if (not q.action_name or e['action_name'] == q.action_name)]
            observations = []
            for oid in q.observation_ids:
                observations.append(_stored_observation(data, oid))
            raw_index = [{k: o.get(k) for k in ('observation_id', 'step', 'state', 'levels_completed', 'source_action')}
                         for o in data['observations'].values()]
            if learned and len(learned['unresolved']) > 8:
                learned = {**learned, 'unresolved_count': len(learned['unresolved']), 'unresolved': learned['unresolved'][:8]}
            response = {'game_id': data['game_id'], 'revision': data['revision'],
                    'rule_count': len(rules), 'rules': rules[q.offset:q.offset + q.limit],
                    'transition_count': len(examples), 'transitions': [
                        {k: v for k, v in e.items() if k != 'source_observations'}
                        for e in examples[q.offset:q.offset + q.limit]],
                    'next_offset': q.offset + q.limit if max(len(rules), len(examples), len(raw_index)) > q.offset + q.limit else None,
                    'observation_index': raw_index[q.offset:q.offset + q.limit],
                    'observations': [{k: v for k, v in o.items() if k not in ('grid', 'image_png_base64')}
                                     for o in observations],
                    'learning': {k: v for k, v in learned.items() if k != 'rules'} if learned else None}
            if observations:
                response['_image_png_base64'] = _visual_evidence(observations)
            return response
        except (ValueError, OSError, sqlite3.Error) as exc:
            return {'error': str(exc)[:2000]}


class BackwardPlanTool(BaseTool):
    def __init__(self, runtime):
        super().__init__(name='plan_backward', description=(
            'Regress a goal (default: stage_clear) through stored causal models toward the CURRENT state. '
            'Returns a forward-checked action chain, knowledge gaps for experiments, or search_limit. '
            'Use explicit true/false current symbols and current observation_id. Missing is unknown. '
            'Candidate rules are excluded by default. Never executes the plan; re-ground only its next action.'))
        self.runtime = runtime

    def _get_declaration(self):
        return types.FunctionDeclaration(name=self.name, description=self.description,
                                         parameters_json_schema=PlanQuery.model_json_schema())

    async def run_async(self, *, args, tool_context):
        try:
            q = PlanQuery.model_validate(args)
            t = self.runtime.turn
            if q.observation_id != t.obs['observation_id']:
                raise ValueError('planning requires the current observation ID')
            if t.time_left() <= 0:
                raise ValueError('reasoning budget exhausted')
            current = q.current.model_dump()
            if 'stage_clear' in current['true'] and t.obs['state'] != 'WIN':
                raise ValueError('the current stage is not cleared; previous level completion is not this goal')
            if t.obs['state'] != 'WIN':
                current['false'] = sorted(set(current['false']) | {'stage_clear'})
            else:
                if 'stage_clear' in current['false']:
                    raise ValueError('WIN contradicts stage_clear=false')
                current['true'] = sorted(set(current['true']) | {'stage_clear'})
            data = self.runtime.causal.read()
            result = planning.backward_plan(current, q.goal.model_dump(), assessed_rules(data, t.memory.level),
                allow_candidates=q.allow_candidates, max_depth=q.max_depth, max_nodes=q.max_nodes,
                seconds=min(0.5, t.time_left()))
            return {**result, 'knowledge_revision': data['revision'], 'observation_id': q.observation_id}
        except (ValueError, OSError, sqlite3.Error) as exc:
            return {'error': str(exc)[:2000]}
