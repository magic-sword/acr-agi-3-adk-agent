"""Goal-directed play: Qwen hypothesises goal predicates, the program chains learned rules.

docs/backchain-review-ja.md §6.4: backward chaining is done by the program over simulatable
rules (V6: the program solved all tasks, Qwen 4B failed repeat counts), Qwen proposes goal
predicates (sampled, several hypotheses) and chooses what to probe when a causal link is
missing (V6: 17/18). The per-step understand/backchain/candidates/ground/reconcile stages
are not used; hypotheses are formed at a level's start and when all are falsified.
"""
import asyncio
from collections import Counter
from copy import deepcopy
import json
import os
import random
import time

from agent.controls import ACTION_TO_BUTTON
from agent.rendering import render_current, png_base64, ORIGIN, SCALE
from .focused_workflow import FocusedRuntime, FocusedTool
from .explorer import Explorer
from .goal_planner import Planner
from .goal_state import GoalHypotheses, GoalHypothesis, ProbeChoice
from .perception import COLORS
from .predicates import RELATIONS, BINARY, holds, describe
from .region_masks import decode, encode
from .rules import MAX_OBJECT_CELLS, components
from .simple_workflow import click_point
from .skill_library import build_skills

OBJECT_LIMIT = 64
HYPOTHESIS_PRESS_LIMIT = 30
PROBE_LIMIT = 6
VISIT_PRESS_LIMIT = 12
EXPLORE_STEPS_PER_HYPOTHESIS = 15   # explorer steps a hypothesis may steer without a plan
REHYPOTHESIZE_EVERY = 25            # minimum actions between hypothesis requests in a level
ORIENT_CLICKS = 6                   # random clicks at most, stopping at the first effective one (V9)
ORIENT_DIRECTIONS = 4               # every untried direction once, even after something moved
HYPOTHESIS_SAMPLES = 3              # sampled hypothesis sets; conditions proposed in more samples rank first
ATOM_GOALS = 6                      # single-condition goals kept from one hypothesis round
EXPERIMENTS_WITHOUT_NEW_FACTS = 8   # experiments before asking again although nothing new was measured
ACHIEVABLE_LISTED = 40              # relations the rules can produce, shown to one sample (V10)

HYPOTHESIZE = '''You play an unknown turn-based puzzle game. Nobody tells you the rules or the goal.
Look at the first screen of this level and propose up to three distinct hypotheses
of what the screen looks like when the level is won. Write each as conditions over the listed
object IDs using only the given relations; all conditions of a hypothesis must hold together.
Do not repeat a falsified hypothesis. A controllable object, if known, is marked. The program
will plan and act on your hypotheses and discard those that measurements contradict, so
propose different plausible ideas rather than one idea three times. Every hypothesis must be
false on the current screen: the level is not won yet.
Object numbers are drawn on the image at each object's top-left corner (the number after "o" in
its ID). bbox_2d coordinates are normalised to 0-1000 over the board. measured_effects are facts
the program measured by acting: use them. Write your reasoning in analysis first.
Before proposing, look at the image and decide:
1. Which object can be controlled or clicked, and what does acting do (measured_effects)?
2. Which objects look like a target, slot, frame, exit or container?
3. Which objects look like an example, key or pattern to copy or match (for instance a small icon
   showing a shape or colours)?
4. Which objects share a colour or shape but are not yet placed, aligned or matched together?
5. Which relation between objects of different roles would plausibly complete the level (for
   instance the controllable object inside a target, a piece aligned with its matching slot,
   tiles recoloured to match a shown pattern)? Status bars and counters along the edges are not goals.
Fill roles first, then write hypotheses that use those roles. If a falsified hypothesis held on the
screen without winning, keep what it got right and add the condition that was missing.
Submit submit_goal_hypotheses.'''

# V10: shown to every sample, the list pulled screen-driven guesses toward near, trivial relations
# (ls20 9/10 -> 5/10); for a goal the screen does not reveal it helped (vc33 0/10 -> 7/10). So one
# sample of each round sees it and the others judge from the screen alone.
ACHIEVABLE = '''
achievable_relations lists relations that the program can already make true with the measured
rules (nearest first, with the number of actions). The win condition is often one of them or a
combination of them: prefer hypotheses built from achievable relations when they fit what the
screen shows, and propose other relations only when the screen clearly suggests them.'''

PROBE = '''The program cannot reach the goal hypothesis with the rules measured so far: the listed
conditions depend on objects no known action has affected. Choose the one untested item most
likely to reveal how to change them. Submit submit_probe.'''


def color_names(ids):
    """Object memory stores colour IDs (SAM perception) or names (program perception)."""
    return sorted(COLORS[c] if isinstance(c, int) else c for c in ids)


class GoalTool(FocusedTool):
    def _get_declaration(self):
        declaration = super()._get_declaration()
        schema = declaration.parameters_json_schema
        defs = schema.get('$defs', {})
        rt = self.runtime
        if self.work == 'hypothesize' and 'Atom' in defs:
            ids = [o['id'] for o in rt._object_rows()]
            props = defs['Atom']['properties']
            for key in ('a', 'b'):
                p = props[key]
                branches = p.get('anyOf', [p])
                for branch in branches:
                    if branch.get('type') == 'string':
                        branch['enum'] = ids
            for branch in props['color'].get('anyOf', [props['color']]):
                if branch.get('type') == 'string':
                    branch['enum'] = COLORS
        if self.work == 'hypothesize' and 'RoleAssignment' in defs:
            defs['RoleAssignment']['properties']['id']['enum'] = [o['id'] for o in rt._object_rows()]
        if self.work == 'probe':
            schema['properties']['item']['enum'] = [u['id'] for u in rt._untested_items or []] or ['none']
        return declaration


class GoalRuntime(FocusedRuntime):
    stage_tool = GoalTool
    stage_tasks = {**FocusedRuntime.stage_tasks,
                   'hypothesize': ('submit_goal_hypotheses', GoalHypotheses),
                   'probe': ('submit_probe', ProbeChoice)}
    stage_tokens = {**FocusedRuntime.stage_tokens, 'hypothesize': 700, 'probe': 300}
    stage_instructions = {**FocusedRuntime.stage_instructions, 'hypothesize': HYPOTHESIZE, 'probe': PROBE}
    stage_temperature = {'hypothesize': 0.7}
    # The stage-pipeline note about candidate_refs does not apply here and diluted the prompt (V7 vs runtime).
    plain_stages = ('hypothesize', 'probe')

    def __init__(self, *args, **kwargs):
        self._untested_items = None
        self._missing = []
        self._hypothesis_serial = 0
        super().__init__(*args, **kwargs)
        self.goals = []          # dict(id, atoms, descriptors, status, source, presses, probes, rationale)
        self.falsified = []      # described atoms of discarded hypotheses (context for new ones)
        self._discarded_keys = set()
        self.carried = []        # descriptor-level hypotheses that won an earlier level
        self.probe = None        # running probe: dict(kind, action|object_id, presses)
        # Graph exploration is off by default: ARC-AGI-3 scores action counts, so brute-force coverage is
        # not the backbone (docs/goal-pipeline-ja.md). COGNITION_EXPLORER=1 enables it for experiments.
        self.use_explorer = os.getenv('COGNITION_EXPLORER', '0') == '1'
        self.explorer = Explorer()
        self._since_hypothesis = None   # actions since the last hypothesis request in this level
        self._node = None
        self._oriented = False          # the level's orientation tests are done
        self._orient_clicks = 0
        self._orient_rng = random.Random(0)
        self.achieved = []              # conditions reached in this level without ending it (kept in later plans)
        self._evidence_asked = None     # measured-rule key when hypotheses were last asked or retried
        self._experiments = 0           # experiments since then
        self._visited = set()           # visit keys already used as experiments in this level
        self.knowhow = []               # patterns found in earlier levels of this game (carried, not reset)
        self.hypothesis_samples = None   # tests may set 1
        self._tracks, self._objects, self._objects_for, self._track_serial = [], [], None, 0

    # --- objects and views ----------------------------------------------------------------
    def _current_objects(self):
        """Hypothesis objects: same-colour components with IDs carried across frames.

        V7 found goal hypotheses right only with this vocabulary (not overlapping perception
        regions). A component keeps the ID of the same-looking component at the nearest position.
        """
        if self._objects_for == self.obs['observation_id']:
            return self._objects
        previous, taken, objs = self._tracks, set(), []
        comps = sorted((c for c in components(self.obs['grid']) if len(c['cells']) <= MAX_OBJECT_CELLS),
                       key=lambda c: (c['origin'][1], c['origin'][0]))
        tracks = []
        for c in comps:
            same = [t for t in previous if t['sig'] == c['sig'] and t['id'] not in taken]
            if not same:  # recoloured or reshaped in place: same colour, overlapping position
                same = [t for t in previous if t['color'] == c['color'] and t['id'] not in taken and t['cells'] & c['cells']]
            if same:
                ident = min(same, key=lambda t: abs(t['origin'][0]-c['origin'][0]) + abs(t['origin'][1]-c['origin'][1]))['id']
            else:
                self._track_serial += 1
                ident = f'o{self._track_serial}'
            taken.add(ident)
            tracks.append(dict(id=ident, sig=c['sig'], color=c['color'], origin=c['origin'], cells=c['cells']))
            xs = [x for x, _ in c['cells']]
            ys = [y for _, y in c['cells']]
            objs.append(dict(object_id=ident, observation_id=self.obs['observation_id'], color_ids=[c['color']],
                             bbox=[min(xs), min(ys), max(xs), max(ys)], kind='region', mask_runs=encode(c['cells'])))
        self._tracks, self._objects, self._objects_for = tracks, objs, self.obs['observation_id']
        return objs

    def _current_object(self, ident):
        return next((o for o in self._current_objects() if o['object_id'] == ident), None)

    def _object_rows(self):
        mover_cells = set()
        if self.rules.movers():
            mover_cells, _ = self.rules.mover_cells(self.obs['grid'])
        rows = []
        for o in self._current_objects():
            cells = decode(o.get('mask_runs', []))
            if not cells or len(cells) > MAX_OBJECT_CELLS:
                continue
            rows.append(dict(id=o['object_id'], colors=color_names(o['color_ids']), bbox=o['bbox'], cells=len(cells), kind=o['kind'],
                             controllable=bool(mover_cells) and len(cells & mover_cells) * 2 > len(cells)))
        rows.sort(key=lambda r: (r['bbox'][1], r['bbox'][0]))   # screen order, as in V7
        return rows[:OBJECT_LIMIT]

    def _object_cells(self):
        return {o['object_id']: decode(o.get('mask_runs', [])) for o in self._current_objects()}

    def _view(self):
        return {o['object_id']: (decode(o.get('mask_runs', [])), set(color_names(o['color_ids'])))
                for o in self._current_objects()}

    def _anchors(self, atoms):
        boxes = {o['object_id']: o['bbox'] for o in self._current_objects()}
        return {ident: boxes.get(ident) for a in atoms for ident in (a.get('a'), a.get('b')) if ident}

    def _rebind(self, ident, colors, anchor):
        """Perception may give a changed object a new ID: follow it by colour and nearest position."""
        current = {o['object_id']: o for o in self._current_objects()}
        if ident in current:
            return ident
        same = [o for o in current.values() if color_names(o['color_ids']) == colors]
        if not same or anchor is None:
            return None
        cx, cy = (anchor[0]+anchor[2]) / 2, (anchor[1]+anchor[3]) / 2
        return min(same, key=lambda o: abs((o['bbox'][0]+o['bbox'][2])/2-cx) + abs((o['bbox'][1]+o['bbox'][3])/2-cy))['object_id']

    def _refresh(self, goal):
        """Keep a hypothesis bound to current objects; returns False if an object cannot be followed."""
        mapping = {}
        for ident, colors in goal['descriptors'].items():
            new = self._rebind(ident, colors, goal['anchors'].get(ident))
            if new is None:
                if any(a['relation'] == 'gone' and a['a'] == ident for a in goal['atoms']):
                    continue  # a vanished object is what 'gone' asks for
                return False
            mapping[ident] = new
        if any(k != v for k, v in mapping.items()):
            for a in goal['atoms']:
                a['a'] = mapping.get(a['a'], a['a'])
                if a.get('b'):
                    a['b'] = mapping.get(a['b'], a['b'])
            goal['descriptors'] = {mapping.get(k, k): v for k, v in goal['descriptors'].items()}
            self._record('artifacts', 'hypothesis_rebound', goal=goal['id'], mapping=mapping)
        goal['anchors'] = self._anchors(goal['atoms'])
        return True

    def _descriptors(self, atoms):
        colors = {o['object_id']: color_names(o['color_ids']) for o in self._current_objects()}
        return {ident: colors.get(ident) for a in atoms for ident in (a.get('a'), a.get('b')) if ident}

    # --- orientation: a few discriminating tests before hypothesising (docs/goal-pipeline-ja.md §4) --
    def _owner(self, cells):
        """Smallest listed object holding most of the given cells."""
        rows = self._object_rows()
        objs = self._object_cells()
        owners = [r for r in rows if len(cells & objs.get(r['id'], set())) * 2 > len(cells)]
        return min(owners, key=lambda r: r['cells'])['id'] if owners else None

    def _measured_facts(self):
        """What acting has shown in this level, named by current object IDs."""
        facts = []
        comps = components(self.obs['grid'])
        by_sig = {c['sig']: c for c in comps}
        movers = self.rules.movers()
        if movers:
            ids = sorted({i for s in movers if s in by_sig for i in [self._owner(by_sig[s]['cells'])] if i})
            for action, delta in sorted(self.rules.direction_deltas(movers).items()):
                facts.append(f"{ACTION_TO_BUTTON.get(action, action)} moves {', '.join(ids) or 'the controllable object'} by {delta}")
        # Per clicked object: same-looking objects may act differently by place.
        at = {(c['sig'], c['origin']): c for c in comps}
        name = lambda key: (self._owner(at[key]['cells']) if key in at else None) or COLORS[key[0][0]]
        for key, outcomes in self.rules.click_at.items():
            if key not in at:
                continue
            for _, moved, recolor in outcomes[-2:]:
                parts = [f"moved {self._owner(by_sig[m]['cells']) or COLORS[m[0]]} by {d}" for m, d in moved.items() if m in by_sig]
                if recolor is not None:
                    parts.append(f'turned it {COLORS[recolor]}')
                facts.append(f"clicking {name(key)} ({COLORS[key[0][0]]}): " + ('; '.join(parts) or 'no visible effect'))
        for a, b, _ in self.rules.inverse_pairs():
            facts.insert(0, f"{name(a)} and {name(b)} are opposite controls: they move the same objects in opposite directions")
        facts += [f'known from earlier levels: {k}' for k in self.knowhow]
        return facts[:12]

    def _orient_step(self):
        """Next orientation test, or None when the level's controls are understood enough."""
        available = [a for a in self.obs['available_actions'] if a != 'RESET']
        untried = [a for a in available if a != 'ACTION6' and not self.rules.tries[a]][:ORIENT_DIRECTIONS]
        if untried:
            # Each control once: which object moves, which way and how far is the cheapest strong clue.
            return dict(action={'action': untried[0]}, expected_effect='orientation: what does this control move?')
        effective = any(self.rules.effective[k] for k in self.rules.tries if isinstance(k, tuple))
        if 'ACTION6' in available and self._orient_clicks < ORIENT_CLICKS and effective:
            # A twin of an effective control may be its counterpart (for instance the opposite direction):
            # one click on an unclicked object of the same appearance tells.
            twin = self._twin_click()
            if twin:
                self._orient_clicks += 1
                return twin
        if 'ACTION6' in available and self._orient_clicks < ORIENT_CLICKS and not effective:
            job = self._random_click('orientation')
            if job:
                self._orient_clicks += 1
                return job
        return None

    def _twin_click(self):
        effective = {s for (s, _), o in self.rules.click_at.items() if any(m or r is not None for _, m, r in o)}
        tested = {s for s in effective if sum(1 for (t, _) in self.rules.click_at if t == s) > 1}
        for c in self._click_targets():
            if c['sig'] in effective - tested and (c['sig'], c['origin']) not in self.rules.click_at:
                x, y = min(c['cells'], key=lambda p: (p[1], p[0]))
                return dict(action={'action': 'ACTION6', 'x': x, 'y': y},
                            expected_effect=f'orientation: does this twin of an effective {COLORS[c["color"]]} control act the same?')
        return None

    def _random_click(self, purpose):
        """Click a never-clicked object in random order (V9: a biased guess cost more clicks)."""
        # Per object, not per appearance: identical-looking objects may act differently by place (V9, vc33).
        clicked = set(self.rules.click_at)
        comps = [c for c in self._click_targets() if (c['sig'], c['origin']) not in clicked]
        if not comps:
            return None
        c = self._orient_rng.choice(sorted(comps, key=lambda c: (c['origin'][1], c['origin'][0])))
        x, y = min(c['cells'], key=lambda p: (p[1], p[0]))
        return dict(action={'action': 'ACTION6', 'x': x, 'y': y},
                    expected_effect=f'{purpose}: what does clicking the {COLORS[c["color"]]} object do?')

    def _click_targets(self):
        """Clickable components: not the step counter, not a status bar along an edge."""
        tick = self.rules.tick_region()

        def status_bar(c):
            x0, y0, x1, y1 = (min(x for x, _ in c['cells']), min(y for _, y in c['cells']),
                              max(x for x, _ in c['cells']), max(y for _, y in c['cells']))
            return (y1 - y0 < 2 and x1 - x0 >= 15) or (x1 - x0 < 2 and y1 - y0 >= 15)
        return [c for c in components(self.obs['grid'])
                if len(c['cells']) <= MAX_OBJECT_CELLS and not (c['cells'] & tick) and not status_bar(c)]

    def _exploration_fallback(self):
        """The least-tried control, counting clicks per appearance (object IDs change between frames)."""
        available = [a for a in self.obs['available_actions'] if a != 'RESET']
        options = [(self.rules.tries[a], a, None) for a in available if a != 'ACTION6']
        if 'ACTION6' in available:
            options += [(self.rules.tries[('click', c['sig'])], 'ACTION6', c) for c in self._click_targets()]
        if not options:
            return None
        _, action, c = min(options, key=lambda o: (o[0], o[2]['origin'][::-1] if o[2] else (-1, -1)))
        job = dict(action={'action': action}, expected_effect='Host fallback: observe the effect of the least-tried action.')
        if c:
            x, y = min(c['cells'], key=lambda p: (p[1], p[0]))
            job['action'].update(x=x, y=y)
        return job

    def _evidence_key(self):
        """What the rules have measured to have an effect; asking again only pays when this changed."""
        movers = self.rules.movers()
        return (frozenset(self.rules.direction_deltas(movers).items()) if movers else frozenset(),
                # Which appearances clicks affect, not how many contexts were seen: repeating a known
                # click in another context is not a new kind of fact.
                frozenset((sig, frozenset(self.rules.click_affects[sig])) for sig, o in self.rules.click_outcomes.items()
                          if any(m or r is not None for _, m, r in o)),
                frozenset(k for k, o in self.rules.click_at.items() if any(m or r is not None for _, m, r in o)),
                frozenset(self.rules.blocking_colors(movers)) if movers else frozenset())

    def _experiment_step(self):
        """One experiment for a new fact when every hypothesis is spent: an unused control, a visit
        to an object the controllable object has not been at, or a never-clicked object."""
        available = [a for a in self.obs['available_actions'] if a != 'RESET']
        untried = [a for a in available if a != 'ACTION6' and not self.rules.tries[a]]
        if untried:
            return dict(action={'action': untried[0]}, expected_effect='experiment: what does this control move?')
        if self.rules.movers():
            tick = self.rules.tick_region()
            cells = self._object_cells()
            options = []
            for r in self._object_rows():
                key = ('visit', tuple(r['colors']), tuple(r['bbox']))
                if r['controllable'] or key in self._visited or cells.get(r['id'], set()) & tick:
                    continue
                try:
                    path = self.rules.plan_path(self.obs['grid'], tuple(r['bbox']), available=available)
                except ValueError:
                    continue
                if path:
                    options.append((len(path), r, key))
            if options:
                # The nearest unvisited object: the cheapest new contact with the level.
                _, r, key = min(options, key=lambda o: o[0])
                self._visited.add(key)
                self.probe = dict(kind='visit', object_id=r['id'], key=key, colors=r['colors'], anchor=r['bbox'],
                                  what=f"experiment: move the controllable object onto {r['id']}", presses=0)
                self._record('artifacts', 'experiment_selected', item=dict(kind='visit', object_id=r['id']))
                return self._probe_step()
        if 'ACTION6' in available:
            job = self._random_click('experiment')
            if job:
                return job
            # Every appearance was clicked once: a click that had an effect may act differently in
            # this state (click rules are conditioned on context), so repeat the least-tried one.
            tick = self.rules.tick_region()
            comps = [c for c in components(self.obs['grid']) if self.rules.effective[('click', c['sig'])]
                     and c['cells'] - tick]
            if comps:
                c = min(comps, key=lambda c: (self.rules.tries[('click', c['sig'])], c['origin'][1], c['origin'][0]))
                x, y = min(c['cells'] - tick, key=lambda p: (p[1], p[0]))
                return dict(action={'action': 'ACTION6', 'x': x, 'y': y},
                            expected_effect=f'experiment: does clicking the {COLORS[c["color"]]} object act differently now?')
        return None

    def _visual_parts(self, slow=False):
        if self.work != 'hypothesize':
            return super()._visual_parts(slow)
        # Set-of-Mark: the listed objects' numbers on the board, so IDs refer to what the model sees.
        from PIL import ImageDraw, ImageFont
        image = render_current(self.obs['grid'])
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.load_default(9)
        except TypeError:  # Pillow < 10.1 has no sized default font
            font = ImageFont.load_default()
        for row in self._object_rows():
            x, y = ORIGIN[0] + row['bbox'][0]*SCALE, ORIGIN[1] + row['bbox'][1]*SCALE
            label = row['id'].split('o')[-1]
            draw.rectangle(draw.textbbox((x, y), label, font=font), fill='black')
            draw.text((x, y), label, fill='yellow', font=font)
        return [{'type': 'text', 'text': 'CURRENT board with object numbers'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + png_base64(image)}}]

    # --- hypotheses as JSON output, sampled (V8: tool-call output fell to 1/60) ----------------------
    def _hypothesis_schema(self):
        from agent.local_vlm import json_schema
        schema = json_schema(GoalTool(self, 'hypothesize')._get_declaration().parameters_json_schema)
        schema['required'] = list(dict.fromkeys(['analysis'] + schema.get('required', [])))
        order = ['observation_id', 'analysis', 'roles', 'hypotheses']
        schema['properties'] = {k: schema['properties'][k] for k in order if k in schema['properties']}
        return schema

    async def _hypothesize(self):
        """Sample hypothesis sets as JSON output; hypotheses proposed repeatedly rank first."""
        self.work = 'hypothesize'
        self.rejection = None
        context = self._context('hypothesize')
        parts = self._visual_parts(True) + [{'type': 'text', 'text': json.dumps(context, separators=(',', ':'))}]
        schema = self._hypothesis_schema()
        from agent.local_vlm import LocalVisionLlm
        model = LocalVisionLlm(model='qwen3-vl-4b-instruct', api_base=os.getenv('VLM_API_BASE', 'http://vlm:8080/v1'))
        samples, votes, first_seen, roles = [], Counter(), {}, []
        atom_votes, atom_first = Counter(), {}
        planned = HYPOTHESIS_SAMPLES if self.hypothesis_samples is None else self.hypothesis_samples
        achievable = self._achievable() if planned > 1 else []
        n, repaired = 0, False
        while n < planned or (not samples and not repaired and self.rejection):
            if self.time_left() <= 0:
                break
            if n >= planned:
                # Every sample was invalid: one repair round that states the last error.
                repaired = True
                context = self._context('hypothesize')
                parts = self._visual_parts(True) + [{'type': 'text', 'text': json.dumps(context, separators=(',', ':'))}]
            n += 1
            system, user = self.stage_instructions['hypothesize'], parts
            listed = n == 1 and bool(achievable) and not repaired
            if listed:
                # First, so that its conditions win ties in the vote order.
                system += ACHIEVABLE
                user = parts[:-1] + [{'type': 'text', 'text': json.dumps(dict(context, achievable_relations=achievable),
                                                                          separators=(',', ':'))}]
            payload = {'model': 'qwen3-vl-4b-instruct', 'temperature': self.stage_temperature['hypothesize'],
                       'seed': random.randrange(2**31), 'max_tokens': 1400, 'stream': False,
                       'response_format': {'type': 'json_schema', 'json_schema': {'name': 'goals', 'schema': schema}},
                       'messages': [{'role': 'system', 'content': system},
                                    {'role': 'user', 'content': user}]}
            self.calls += 1
            self.memory.model_calls += 1
            record = dict(state='DECIDE', work='hypothesize', call_index=self.calls, sample=n, repair=repaired,
                          achievable_listed=len(achievable) if listed else 0,
                          context=context)
            started = time.monotonic()
            try:
                record['request_sha256'] = self._request_record(payload)
                model.timeout_seconds = max(.001, self.time_left())
                async with asyncio.timeout(max(.001, self.time_left())):
                    response = await asyncio.to_thread(model._complete, payload)
                record.update(usage=response.get('usage'), timings=response.get('timings'))
                value = GoalHypotheses.model_validate(json.loads(response['choices'][0]['message']['content']))
                self.validate_stage('hypothesize', value)
                samples.append(value)
                roles = roles or value.roles
                for h in value.hypotheses:
                    key = tuple(sorted(describe(a.model_dump()) for a in h.atoms))
                    votes[key] += 1
                    first_seen.setdefault(key, h)
                # A condition counts once per sample: whole hypotheses rarely recur, their parts do.
                for key, found in {describe(a.model_dump()): (a, h) for h in value.hypotheses for a in h.atoms}.items():
                    atom_votes[key] += 1
                    atom_first.setdefault(key, found)
                record.update(schema_valid=True, response=value.model_dump_json())
            except Exception as exc:
                record.update(schema_valid=False, error=f'{type(exc).__name__}: {exc}'[:500])
                self.errors.append(record['error'])
                self.rejection = str(exc)[:500]
            finally:
                record['seconds'] = time.monotonic() - started
                self._record('model', 'model_decision', **record)
        self.rejection = None
        if not samples:
            return False
        ranked = sorted(votes, key=lambda k: (-votes[k], list(first_seen).index(k)))
        # Each condition becomes its own goal: an unplannable condition bundled into a hypothesis
        # no longer takes a plausible one down with it; reached conditions stay required (self.achieved).
        view = self._view()
        atoms = [a for a in sorted(atom_votes, key=lambda a: (-atom_votes[a], list(atom_first).index(a)))
                 if not holds(atom_first[a][0].model_dump(), view)]
        singles = [GoalHypothesis(atoms=[atom_first[a][0]], rationale=atom_first[a][1].rationale)
                   for a in atoms[:ATOM_GOALS]]
        merged = GoalHypotheses(observation_id=self.obs['observation_id'], analysis=samples[0].analysis, roles=roles,
                                hypotheses=singles or [first_seen[k] for k in ranked][:6] or samples[0].hypotheses[:1])
        self._record('artifacts', 'hypothesis_votes', samples=len(samples),
                     votes=[dict(atoms=list(k), votes=votes[k]) for k in ranked],
                     atom_votes=[dict(atom=a, votes=atom_votes[a]) for a in atoms])
        self._accept_stage('hypothesize', merged)
        return True

    # --- stage contexts ---------------------------------------------------------------------
    def _context(self, work=None):
        work = work or self.work
        if work not in ('hypothesize', 'probe'):
            return super()._context(work)
        from agent.controls import controller_context
        common = dict(work=work, observation_id=self.obs['observation_id'])
        if self.rejection:
            common['correction'] = self.rejection
        rules = [dict(name=k.name, description=k.description) for k in build_skills(self.rules, self.memory.skill_stats)]
        if work == 'hypothesize':
            # V7 bridge test: listing colours as a list with a 'kind' field cut correct hypotheses 9/10 -> 4/10.
            w, h = self.obs.get('width', 64), self.obs.get('height', 64)
            # V8: Qwen3-VL's own 0-1000 grounding convention doubled correct hypotheses (11 -> 23 of 60).
            model_rows = [dict(id=r['id'], color=', '.join(r['colors']),
                               bbox_2d=[round(r['bbox'][0]*1000/(w-1)), round(r['bbox'][1]*1000/(h-1)),
                                        round(r['bbox'][2]*1000/(w-1)), round(r['bbox'][3]*1000/(h-1))],
                               cells=r['cells'], controllable=r['controllable']) for r in self._object_rows()]
            common.update(objects=model_rows, relations=RELATIONS,
                          controls=[ACTION_TO_BUTTON.get(a, a) for a in self.obs['available_actions'] if a != 'RESET'],
                          measured_effects=self._measured_facts(), falsified_hypotheses=self.falsified[-6:])
        else:
            goal = self._active()
            common.update(goal_hypothesis=[describe(a) for a in goal['atoms']],
                          unreachable_conditions=[describe(a) for a in self._missing],
                          untested_items=[{k: u[k] for k in ('id', 'what')} for u in self._untested_items],
                          measured_rules=rules)
        return controller_context(common)

    def validate_stage(self, work, value):
        if work == 'hypothesize':
            if value.observation_id != self.obs['observation_id']:
                raise ValueError('stale observation_id')
            ids = {o['id'] for o in self._object_rows()}
            view = self._view()
            for h in value.hypotheses:
                for a in h.atoms:
                    if a.a not in ids or (a.relation in BINARY and (a.b not in ids or a.b == a.a)):
                        raise ValueError(f'{a.relation} needs listed object IDs (two distinct ones for relations)')
                    if a.relation == 'color_is' and a.color not in COLORS:
                        raise ValueError('color_is needs a colour name')
            # Hypotheses already true are dropped one by one at acceptance; only an all-true batch is sent back.
            if all(all(holds(a.model_dump(), view) for a in h.atoms) for h in value.hypotheses):
                raise ValueError('every hypothesis already holds on the current screen, yet the level is not won; '
                                 'each win condition must require a change from the current screen')
            return
        if work == 'probe':
            if value.observation_id != self.obs['observation_id']:
                raise ValueError('stale observation_id')
            if value.item not in {u['id'] for u in self._untested_items or []}:
                raise ValueError('choose an offered untested item')
            return
        super().validate_stage(work, value)

    def _accept_stage(self, work, value):
        if work == 'hypothesize':
            self.validate_stage(work, value)
            seen = {tuple(sorted(describe(a) for a in g['atoms'])) for g in self.goals}
            added, view = [], self._view()
            for h in value.hypotheses:
                atoms = [a.model_dump() for a in h.atoms]
                key = tuple(sorted(describe(a) for a in atoms))
                if key in seen or key in self._discarded_keys:
                    continue
                if all(holds(a, view) for a in atoms):
                    self._discarded_keys.add(key)
                    self.falsified.append([describe(a) for a in atoms] + ['(already true on the screen)'])
                    self._record('artifacts', 'hypothesis_discarded', goal=None, reason='already true when proposed',
                                 status='trivial')
                    continue
                seen.add(key)
                self._hypothesis_serial += 1
                added.append(dict(id=f'H{self._hypothesis_serial}', atoms=atoms, descriptors=self._descriptors(atoms),
                                  anchors=self._anchors(atoms), probed=[], status='active', source='model',
                                  presses=0, probes=0, rationale=h.rationale))
            self.goals += added
            self._record('artifacts', 'goal_hypotheses', added=added, observation_id=value.observation_id)
            self._machine_transition('goal_hypothesized')
            return
        if work == 'probe':
            self.validate_stage(work, value)
            item = next(u for u in self._untested_items if u['id'] == value.item)
            self.probe = dict(item, presses=0)
            goal = self._active()
            goal['probes'] += 1
            goal['probed'].append(item.get('key'))
            self._record('artifacts', 'probe_selected', goal=goal['id'], item=item, rationale=value.rationale)
            self._machine_transition('probe_chosen')
            return
        super()._accept_stage(work, value)

    def _snapshot_extra(self):
        extra = super()._snapshot_extra()
        extra.update(goal_hypotheses=getattr(self, 'goals', []), goal_probe=getattr(self, 'probe', None))
        return extra

    # --- hypothesis bookkeeping ----------------------------------------------------------------
    def _active(self):
        return next((g for g in self.goals if g['status'] == 'active'), None)

    def _discard(self, goal, reason, status='falsified'):
        goal['status'] = status
        self._discarded_keys.add(tuple(sorted(describe(a) for a in goal['atoms'])))
        self.falsified.append([describe(a) for a in goal['atoms']] + [f'({reason})'])
        self._record('artifacts', 'hypothesis_discarded', goal=goal['id'], reason=reason, status=status)
        self._machine_transition('goal_falsified')

    def _reground(self, goal):
        """Rebind a hypothesis to current objects by colour descriptors (new level or restart)."""
        rows = self._object_rows()
        mapping = {}
        for ident, colors in goal['descriptors'].items():
            match = next((r['id'] for r in rows if sorted(r['colors']) == colors and r['id'] not in mapping.values()), None)
            if match is None:
                return None
            mapping[ident] = match
        atoms = [dict(a, a=mapping.get(a['a'], a['a']), b=mapping.get(a.get('b'), a.get('b'))) for a in goal['atoms']]
        self._hypothesis_serial += 1
        return dict(deepcopy(goal), id=f'H{self._hypothesis_serial}', atoms=atoms, status='active', presses=0, probes=0,
                    descriptors={mapping.get(k, k): v for k, v in goal['descriptors'].items()},
                    anchors=self._anchors(atoms), probed=[])

    def _on_observation(self, boundary):
        winner = self._active() if boundary == 'level' else None
        if boundary == 'level' and self.rules.inverse_pairs():
            note = 'same-looking buttons can come in pairs that move an object in opposite directions'
            if note not in self.knowhow:
                self.knowhow.append(note)
        kept = (self.rules, self.memory.skill_stats) if boundary == 'reset' else None
        super()._on_observation(boundary)
        if not boundary:
            return
        if kept:
            # A restart replays the same level: its rules and explored graph still hold.
            self.rules, self.memory.skill_stats = kept
        else:
            self.explorer = Explorer()
            self._since_hypothesis = None
            self._oriented = False
            self._orient_clicks = 0
            self.achieved, self._visited, self._evidence_asked, self._experiments = [], set(), None, 0
        self.probe = None
        if winner:
            self._record('artifacts', 'hypothesis_won', goal=winner['id'], atoms=winner['atoms'])
            self.carried = [dict(winner, source='carried')] + self.carried[:2]
        survivors = [g for g in self.goals if g['status'] == 'active'] if boundary == 'reset' else []
        self.goals = []
        for g in self.carried + survivors:
            rebound = self._reground(g)
            if rebound:
                self.goals.append(rebound)
        if self.goals:
            self._record('artifacts', 'goal_hypotheses', added=self.goals, observation_id=self.obs['observation_id'],
                         reason=f'regrounded after {boundary}')

    # --- untested items and probes ----------------------------------------------------------
    def _untested(self, goal):
        items = []
        probed = set(goal.get('probed', []))
        available = [a for a in self.obs['available_actions'] if a != 'RESET']
        for a in available:
            key = ('control', a)
            if a != 'ACTION6' and not self.rules.tries[a] and key not in probed:
                items.append(dict(kind='control', action=a, key=key,
                                  what=f'the {ACTION_TO_BUTTON.get(a, a)} control (never used)'))
        focus = [i for atom in goal['atoms'] for i in (atom.get('a'), atom.get('b')) if i]
        # A counter changes every step and would look untried forever; it is not a probe target.
        tick = self.rules.tick_region()
        cells = self._object_cells()
        rows = {r['id']: r for r in self._object_rows() if not (cells.get(r['id'], set()) & tick)}
        order = [i for i in dict.fromkeys(focus) if i in rows] + [i for i in rows if i not in focus]
        if 'ACTION6' in available:
            from .rules import components
            comps = components(self.obs['grid'])
            clicked = set(self.rules.click_at)
            for ident in order:
                obj = self._current_object(ident)
                try:
                    xy = click_point(obj, self.obs)
                except ValueError:
                    continue
                comp = next((c for c in comps if xy in c['cells']), None)
                key = ('click', comp['sig'], comp['origin']) if comp else None
                # The component under the actual click cell decides whether this was tried.
                if comp and (comp['sig'], comp['origin']) not in clicked and key not in probed and key not in {u.get('key') for u in items}:
                    items.append(dict(kind='click', object_id=ident, key=key,
                                      what=f"click {ident} ({', '.join(rows[ident]['colors'])}, never clicked)"))
                if len(items) >= 8:
                    break
        if self.rules.movers():
            for ident in order[:8]:
                key = ('visit', tuple(rows[ident]['colors']), tuple(rows[ident]['bbox']))
                if rows[ident]['controllable'] or key in probed:
                    continue
                try:
                    path = self.rules.plan_path(self.obs['grid'], tuple(rows[ident]['bbox']), available=available)
                except ValueError:
                    continue
                if path and len(items) < 10:
                    items.append(dict(kind='visit', object_id=ident, key=key, colors=rows[ident]['colors'],
                                      anchor=rows[ident]['bbox'],
                                      what=f"move the controllable object onto {ident} ({', '.join(rows[ident]['colors'])})"))
        for i, u in enumerate(items):
            u['id'] = f'U{i+1}'
        return items

    def _probe_step(self):
        p = self.probe
        if p['kind'] == 'control':
            self.probe = None
            return dict(action={'action': p['action']}, expected_effect=f"probe: {p['what']}")
        if p['kind'] == 'visit':
            followed = self._rebind(p['object_id'], p['colors'], p['anchor'])
            if followed:
                p['object_id'] = followed
        obj = self._current_object(p['object_id'])
        if obj is None:
            self.probe = None
            return None
        if p['kind'] == 'visit':
            p['anchor'] = obj['bbox']
        if p['kind'] == 'click':
            self.probe = None
            try:
                x, y = click_point(obj, self.obs)
            except ValueError:
                return None
            return dict(action={'action': 'ACTION6', 'x': x, 'y': y}, expected_effect=f"probe: {p['what']}")
        try:
            path = self.rules.plan_path(self.obs['grid'], tuple(obj['bbox']), available=self.obs['available_actions'])
        except ValueError:
            path = None
        if not path or p['presses'] >= VISIT_PRESS_LIMIT:
            self.probe = None
            return None
        p['presses'] += 1
        return dict(action={'action': path[0]}, expected_effect=f"probe: {p['what']}")

    # --- exploration backbone -----------------------------------------------------------------
    def _action_key(self, action):
        if action['action'] != 'ACTION6':
            return ('press', action['action'])
        xy = (action.get('x'), action.get('y'))
        comp = next((c for c in components(self.obs['grid']) if xy in c['cells']), None)
        return ('click', comp['color'], comp['origin']) if comp else None

    def _explore_step(self, goal):
        boost = set()
        if goal:
            cells = self._object_cells()
            for a in goal['atoms']:
                for ident in (a.get('a'), a.get('b')):
                    boost |= cells.get(ident, set())
        choice = self.explorer.choose(self._node, frozenset(boost))
        if choice is None:
            return None
        akey, action, how = choice
        self._record('artifacts', 'explore_step', how=how, action=action, goal=goal['id'] if goal else None,
                     boosted=bool(boost), **self.explorer.stats())
        return dict(action=dict(action), expected_effect=f'explore ({how})')

    # --- the loop ---------------------------------------------------------------------------
    async def _think(self, ctx):
        hypothesized = False
        if self.use_explorer:
            self._node = self.explorer.observe(self.obs['grid'], self.obs['available_actions'], self.rules.tick_region())
            if self._since_hypothesis is not None:
                self._since_hypothesis += 1
        while self.result is None and self.job is None and self.time_left() > 0:
            if self.probe:
                job = self._probe_step()
                if job:
                    self._press(job, 'probe_press')
                    break
                continue
            if not self._oriented:
                job = self._orient_step()
                if job:
                    self._press(job, 'orient_press')
                    break
                self._oriented = True
                self._record('artifacts', 'orientation_done', facts=self._measured_facts())
            goal = self._active()
            due = self._since_hypothesis is None or self._since_hypothesis >= REHYPOTHESIZE_EVERY
            if goal is None and self.use_explorer and (hypothesized or not due):
                job = self._explore_step(None)
                if job:
                    self._press(job, 'explore_press')
                else:
                    self._fallback('explored_everything')
                break
            if goal is None:
                if hypothesized:
                    # The fresh hypotheses were all unreachable at once: act for a new fact.
                    job = self._experiment_step()
                    if job:
                        self._experiments += 1
                        self._press(job, 'experiment_press')
                    elif self.probe:
                        continue
                    else:
                        self._fallback('no_goal_hypothesis')
                    break
                evidence = self._evidence_key()
                blocked = [g for g in self.goals if g['status'] == 'blocked']
                if self._evidence_asked is not None and evidence != self._evidence_asked and blocked:
                    # A newly measured rule may make an unreachable condition reachable: retry those first.
                    for g in blocked:
                        g.update(status='active', probes=0, probed=[])
                        self._discarded_keys.discard(tuple(sorted(describe(a) for a in g['atoms'])))
                    self._evidence_asked, self._experiments = evidence, 0
                    self._record('artifacts', 'hypotheses_retried', goals=[g['id'] for g in blocked])
                    continue
                if evidence == self._evidence_asked and self._experiments < EXPERIMENTS_WITHOUT_NEW_FACTS:
                    # Nothing new was measured since the last answer: asking again mostly repeats it.
                    # Act for a new fact instead (docs/goal-pipeline-ja.md §7).
                    job = self._experiment_step()
                    self._experiments += 1
                    if job:
                        self._press(job, 'experiment_press')
                        break
                    if self.probe:
                        continue
                    # No experiment left either: one host action, and ask again only after several.
                    self._fallback('no_new_fact')
                    break
                hypothesized = True
                self._since_hypothesis = 0
                self._evidence_asked, self._experiments = evidence, 0
                if not await self._hypothesize():
                    self._fallback('stage_output_invalid')
                    break
                continue
            if not self._refresh(goal):
                self._discard(goal, 'an object of the hypothesis can no longer be followed', status='blocked')
                continue
            view = self._view()
            if all(holds(a, view) for a in goal['atoms']):
                self.achieved += [a for a in goal['atoms'] if a not in self.achieved]
                self._discard(goal, 'all conditions hold but the level did not end')
                continue
            if goal['presses'] >= HYPOTHESIS_PRESS_LIMIT:
                self._discard(goal, 'press budget spent without reaching the conditions')
                continue
            planner = Planner(self.rules, self.obs['grid'], self._object_cells(), self.obs['available_actions'])
            plan = self._plan(planner, goal)
            if plan is None:
                # Another single-condition goal the rules can already reach goes first.
                other = next((g for g in self.goals if g['status'] == 'active' and g is not goal
                              and self._refresh(g) and self._plan(planner, g)), None)
                if other is not None:
                    self.goals.remove(other)
                    self.goals.insert(self.goals.index(goal), other)
                    continue
            if plan:
                action, idx = plan[0]
                job = dict(action={'action': action}, expected_effect=f"goal {goal['id']}: step 1 of {len(plan)}")
                if action == 'ACTION6':
                    x, y = planner.click_point(idx)
                    job['action'].update(x=x, y=y)
                goal['presses'] += 1
                self._record('artifacts', 'goal_plan', goal=goal['id'], length=len(plan),
                             plan=[(a, i) for a, i in plan[:12]])
                self._press(job, 'program_press')
                break
            if self.use_explorer:
                # No plan: explore, favouring the hypothesis' objects, instead of asking the model.
                goal['explore_steps'] = goal.get('explore_steps', 0) + 1
                if goal['explore_steps'] > EXPLORE_STEPS_PER_HYPOTHESIS:
                    self._discard(goal, 'no plan after steering exploration', status='blocked')
                    continue
                job = self._explore_step(goal)
                if job:
                    self._press(job, 'explore_press')
                    break
                self._discard(goal, 'nothing left to explore', status='blocked')
                continue
            # Achieve what can be achieved and see whether it is enough (VCGT pattern 1).
            unmet = [a for a in goal['atoms'] if not holds(a, view)]
            partial = [(p, a) for a in unmet for p in [planner.plan([a])] if p]
            if partial:
                steps, atom = min(partial, key=lambda x: len(x[0]))
                action, idx = steps[0]
                job = dict(action={'action': action},
                           expected_effect=f"goal {goal['id']}: achieve {describe(atom)} first ({len(steps)} steps)")
                if action == 'ACTION6':
                    x, y = planner.click_point(idx)
                    job['action'].update(x=x, y=y)
                goal['presses'] += 1
                self._record('artifacts', 'goal_plan', goal=goal['id'], length=len(steps), partial=describe(atom),
                             plan=[(a, i) for a, i in steps[:12]])
                self._press(job, 'program_press')
                break
            held = [a for a in goal['atoms'] if holds(a, view)]
            if held:
                # Evidence for revision: the achievable part is done, the level did not end.
                self._discard(goal, f"{', '.join(describe(a) for a in held)} achieved but the level did not end; "
                                    f"{', '.join(describe(a) for a in unmet)} could not be reached with the measured rules",
                              status='falsified')
                continue
            self._missing = planner.unaffected(goal['atoms']) or goal['atoms']
            self._untested_items = self._untested(goal)
            self._record('artifacts', 'missing_link', goal=goal['id'], missing=[describe(a) for a in self._missing],
                         untested=[u['what'] for u in self._untested_items])
            self._machine_transition('goal_missing_link')
            if not self._untested_items or goal['probes'] >= PROBE_LIMIT:
                self._discard(goal, 'no plan and nothing left to probe', status='blocked')
                continue
            self._stage_failed = None
            await self._deliberate(ctx, 'probe')
            if self._stage_failed or not self.probe:
                self._fallback('stage_output_invalid')
                break
        if self.result is None and self.job is None and self.time_left() <= 0:
            self._fallback('decision_time_exhausted')

    def _achievable(self):
        """Relations the measured rules can make true now, nearest first (Planner.reachable)."""
        planner = Planner(self.rules, self.obs['grid'], self._object_cells(), self.obs['available_actions'])
        return [f'{describe(a)} in {n}' for a, n in planner.reachable()[:ACHIEVABLE_LISTED]]

    def _plan(self, planner, goal):
        """Plan the goal while keeping conditions already reached (a win may need them together)."""
        view = self._view()
        keep = [a for a in self.achieved if a not in goal['atoms'] and self._refresh_atom(a) and holds(a, view)]
        return (planner.plan(keep + goal['atoms']) if keep else None) or planner.plan(goal['atoms'])

    def _refresh_atom(self, atom):
        ids = {r['id'] for r in self._object_rows()}
        return atom['a'] in ids and (atom.get('b') is None or atom['b'] in ids)

    def _press(self, job, event):
        self.job = job
        if self.use_explorer and self._node is not None:
            akey = self._action_key(job['action'])
            if akey is not None:
                self.explorer.sent(self._node, akey)
        self._record('artifacts', event, job=job)
        self._machine_transition('goal_press')
