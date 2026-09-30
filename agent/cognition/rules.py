"""Program-learned causal rules (context-action-result schemas) from measured transitions.

Validated offline in docs/backchain-review-ja.md §6.1 (V1). Objects are 4-connected
same-colour components; an appearance is (colour, shape). Nothing here calls a model.
"""
from collections import Counter, defaultdict, deque

from .perception import COLORS as COLOR_NAMES

# Components larger than this are backgrounds and walls, not movable objects.
MAX_OBJECT_CELLS = 400
# Status bars sit along the frame edge; lines this close to it may be masked as counters.
EDGE_BAND = 4
# ARC-AGI-3 convention for direction controls; used only as a prior for untried ones.
DIRECTION_PRIOR = {'ACTION1': (0, -1), 'ACTION2': (0, 1), 'ACTION3': (-1, 0), 'ACTION4': (1, 0)}


def components(grid):
    h, w = len(grid), len(grid[0])
    seen = [[False]*w for _ in range(h)]
    out = []
    for y in range(h):
        for x in range(w):
            if seen[y][x]:
                continue
            color, stack, cells = grid[y][x], [(x, y)], []
            seen[y][x] = True
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for nx, ny in ((cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)):
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and grid[ny][nx] == color:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            x0, y0 = min(c[0] for c in cells), min(c[1] for c in cells)
            shape = tuple(sorted((cx-x0, cy-y0) for cx, cy in cells))
            out.append(dict(color=color, sig=(color, shape), origin=(x0, y0), cells=set(cells)))
    return out


def changed(a, b):
    return {(x, y) for y, row in enumerate(b) for x, v in enumerate(row) if a[y][x] != v}


def moves(before, after):
    """Objects present exactly once (by appearance) in both frames at a different position."""
    ca, cb = Counter(c['sig'] for c in before), Counter(c['sig'] for c in after)
    at = {c['sig']: c for c in after}
    out = {}
    for c in before:
        if len(c['cells']) > MAX_OBJECT_CELLS or ca[c['sig']] != 1 or cb[c['sig']] != 1:
            continue
        o = at[c['sig']]['origin']
        d = (o[0]-c['origin'][0], o[1]-c['origin'][1])
        if d != (0, 0):
            out[c['sig']] = d
    return out


def object_changes(before, after, ignore=()):
    """Per-object changes between two component lists, identical-looking objects included.

    Returns [(kind, before component, detail)] with kind 'moved' (detail: delta), 'restyled'
    (detail: the after component at the same place) or 'vanished'. Objects of one appearance
    that stayed put match themselves; the remaining ones pair with the nearest unmatched
    instance of the same appearance, else with an overlapping new component.
    """
    small = lambda cs: [c for c in cs if len(c['cells']) <= MAX_OBJECT_CELLS and c['sig'] not in ignore]
    before, after = small(before), small(after)
    fixed = {(c['sig'], c['origin']) for c in after}
    gone = [c for c in before if (c['sig'], c['origin']) not in fixed]
    kept = {(c['sig'], c['origin']) for c in before}
    new = [c for c in after if (c['sig'], c['origin']) not in kept]
    out, used = [], set()
    for c in gone:
        same = [i for i, n in enumerate(new) if i not in used and n['sig'] == c['sig']]
        if same:
            i = min(same, key=lambda i: abs(new[i]['origin'][0]-c['origin'][0]) + abs(new[i]['origin'][1]-c['origin'][1]))
            used.add(i)
            out.append(('moved', c, (new[i]['origin'][0]-c['origin'][0], new[i]['origin'][1]-c['origin'][1])))
            continue
        over = [i for i, n in enumerate(new) if i not in used and len(n['cells'] & c['cells']) * 2 >= len(c['cells'])]
        if over:
            used.add(over[0])
            out.append(('restyled', c, new[over[0]]))
        else:
            out.append(('vanished', c, None))
    return out


def move_cells(comps, mv):
    by = {c['sig']: c for c in comps}
    cells = set()
    for sig, (dx, dy) in mv.items():
        if sig in by:
            old = by[sig]['cells']
            cells |= old ^ {(x+dx, y+dy) for x, y in old}
    return cells


def destination_colors(grid, cells, d, own=None):
    own = cells if own is None else own
    h, w = len(grid), len(grid[0])
    out = set()
    for x, y in cells:
        nx, ny = x+d[0], y+d[1]
        if (nx, ny) in own:
            continue
        out.add(grid[ny][nx] if 0 <= nx < w and 0 <= ny < h else 'edge')
    return out


def bbox(cells):
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return (min(xs), min(ys), max(xs), max(ys))


class RuleLearner:
    """Online schemas for one level: moves with enter/block context, click effects, and a tick."""

    def __init__(self):
        self.move_delta = {}                  # (action, sig) -> delta
        self.enter = defaultdict(set)         # (action, sig) -> colours entered
        self.block = defaultdict(set)         # (action, sig) -> colours that stopped it
        self.tries = Counter()                # action or ('click', sig) -> tries
        self.effective = Counter()            # same key -> tries with an object effect
        self.click_moves = {}                 # clicked sig -> {moved sig: delta} (last observed, for description)
        self.click_outcomes = defaultdict(list)  # clicked sig -> [(context, moves, recolor)]: conditioned, never overwritten
        self.click_affects = defaultdict(set)    # clicked sig -> appearances its clicks have moved
        # (clicked sig, origin) -> outcomes of that very object: same-looking buttons or tiles can act
        # differently by position (vc33's two blue buttons move the marker opposite ways; in ft09 only
        # one group of identical tiles reacts), so a rule generalises by appearance only when they agree.
        self.click_at = defaultdict(list)
        self.recolor = {}                     # (clicked shape, colour) -> colour after
        self.tick_seen = Counter()
        self.tick_rows, self.tick_cols = Counter(), Counter()   # edge lines with unexplained changes
        self.shape = None
        self.tick_steps = self.ticked = 0
        self.transitions = deque(maxlen=64)   # (before, action, xy, after) for replay validation
        # (action, mover top-left) where a learned move did nothing: a wall the colours cannot tell
        # (walls and floor may share a colour), so the position itself is the condition.
        self.stuck = set()
        # Contact schemas learned from changes no other rule explains: (kind, trigger appearance,
        # changed appearance) -> dict(effect, hits, trials). kind 'pushed' (moves with the mover),
        # 'moved', 'restyled' or 'vanished'; the trigger is what the mover entered or tried to enter,
        # or the clicked object. Validated like every rule: trials count each later contact.
        self.schemas = {}

    # --- learning -----------------------------------------------------------------
    def learn(self, before, after, action, xy=None):
        """Consolidate one transition; returns whether an object was affected (counter changes excluded)."""
        self.transitions.append((before, action, xy, after))
        comps = components(before)
        effect = False
        cells = changed(before, after)
        real = moves(comps, components(after))
        explained = move_cells(comps, real)
        if action == 'ACTION6':
            clicked = next((c for c in comps if xy in c['cells']), None)
            if clicked is not None:
                key = ('click', clicked['sig'])
                self.tries[key] += 1
                inside = (cells & clicked['cells']) - self.tick_region()
                effect = bool(real or inside)
                self.effective[key] += effect
                context = self.click_context(comps, clicked['sig'])
                self.click_moves[clicked['sig']] = dict(real)
                self.click_affects[clicked['sig']] |= set(real)
                recolor = None
                if inside:
                    x, y = next(iter(inside))
                    recolor = after[y][x]
                    self.recolor[(clicked['sig'][1], clicked['color'])] = recolor
                outcome = (context, dict(real), recolor)
                # The latest observation goes last: predictions use the most recent outcome in a context,
                # so an outcome seen again must supersede one seen in between.
                for seen in (self.click_outcomes[clicked['sig']], self.click_at[(clicked['sig'], clicked['origin'])]):
                    if outcome in seen:
                        seen.remove(outcome)
                    seen.append(outcome)
                explained |= clicked['cells']
        else:
            self.tries[action] += 1
            effect = bool(real)
            self.effective[action] += effect
            by = {c['sig']: c for c in comps}
            place = self._mover_origin(comps)
            learned_here = any((action, s) in self.move_delta for s in self.movers())
            if place is not None and (learned_here or action in DIRECTION_PRIOR):
                if any(s in real for s in self.movers()):
                    self.stuck.discard((action, place))
                else:
                    # A press that moved nothing is a fact about this place (a wall), not about the control.
                    self.stuck.add((action, place))
            self._follow_restyled_mover(comps, components(after), action, real)
            self._learn_attached_parts(comps, components(after), action, real)
            for sig, d in real.items():
                self.move_delta[(action, sig)] = d
                self.enter[(action, sig)] |= destination_colors(before, by[sig]['cells'], d)
            for (a, sig), d in list(self.move_delta.items()):
                if a == action and sig in by and sig not in real:
                    self.block[(a, sig)] |= destination_colors(before, by[sig]['cells'], d) - self.enter[(a, sig)]
        # Changes a contact explains are not a counter (the edge band, where status bars live, excepted).
        explained |= self._learn_schemas(before, after, comps, action, xy, explained)
        residual = cells - explained
        self.tick_steps += 1
        self.ticked += bool(residual)
        self.tick_seen.update(residual)
        self.shape = (len(before[0]), len(before))
        self.tick_rows.update({y for _, y in residual})
        self.tick_cols.update({x for x, _ in residual})
        return effect

    def _follow_restyled_mover(self, comps, after_comps, action, real):
        """A controllable object whose look changes (it turns to face its direction, say) stays the
        controllable object: a new appearance where a mover was, or moved to, inherits its rules."""
        sigs = self.movers()
        if not sigs:
            return
        after_sigs = {c['sig'] for c in after_comps}
        before_keys = {(c['sig'], c['origin']) for c in comps}
        for c in comps:
            if c['sig'] not in sigs or c['sig'] in after_sigs:
                continue
            d = self.move_delta.get((action, c['sig']), (0, 0))
            places = [c['cells'], {(x+d[0], y+d[1]) for x, y in c['cells']}]
            new = [n for n in after_comps if (n['sig'], n['origin']) not in before_keys and n['sig'] not in sigs
                   and len(n['cells']) <= MAX_OBJECT_CELLS and any(len(n['cells'] & p) * 2 >= len(n['cells']) for p in places)]
            if len(new) != 1:
                continue
            for (a, s), delta in list(self.move_delta.items()):
                if s == c['sig']:
                    self.move_delta.setdefault((a, new[0]['sig']), delta)
                    self.enter[(a, new[0]['sig'])] |= self.enter[(a, s)]
                    self.block[(a, new[0]['sig'])] |= self.block[(a, s)]

    def _learn_attached_parts(self, comps, after_comps, action, real):
        """Parts that moved with the core, touching it but not ahead of it, belong to the body (even
        when identical parts exist elsewhere, which the unique-appearance move detection skips)."""
        sigs = self.movers()
        d = next((real[s] for s in sigs if s in real), None)
        if d is None:
            return
        core = {p for c in comps if c['sig'] in sigs for p in c['cells']}
        near = {(x+dx, y+dy) for x, y in core for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))} - core
        ahead = {(x+d[0], y+d[1]) for x, y in core} - core
        for kind, c, delta in object_changes(comps, after_comps, ignore=sigs):
            if kind == 'moved' and delta == d and c['cells'] & near and not c['cells'] & ahead:
                self.move_delta.setdefault((action, c['sig']), d)

    def _contact(self, comps, action, xy):
        """(trigger components, mover cells, attempted delta) of one action."""
        if action == 'ACTION6':
            return [c for c in comps if xy in c['cells']], set(), None
        sigs = self.movers()
        deltas = self.direction_deltas(sigs)
        if action not in deltas:
            return [], set(), None
        mover = {p for c in self.controllable(comps) for p in c['cells']}
        d = deltas[action]
        front = {(x+d[0], y+d[1]) for x, y in mover} - mover
        return [c for c in comps if c['cells'] & front and c['sig'] not in sigs
                and len(c['cells']) <= MAX_OBJECT_CELLS], mover, d

    def _learn_schemas(self, before, after, comps, action, xy, explained):
        triggers, mover, d = self._contact(comps, action, xy)
        if not triggers:
            return set()
        triggers.sort(key=lambda t: len(t['cells']))   # the most specific thing touched first
        sigs = self.movers()
        after_comps = components(after)
        changes = object_changes(comps, after_comps, ignore=sigs)
        tick = self.tick_region()
        moved_to = {p for c in after_comps if c['sig'] in sigs for p in c['cells']}
        real = []
        for kind, c, detail in changes:
            if kind == 'moved':
                dest = {(x+detail[0], y+detail[1]) for x, y in c['cells']}
                # An identical dot covered here and uncovered where the mover stood is not a move.
                if dest <= mover or c['cells'] <= tick:
                    continue
            real.append((kind, c, detail))
        # Cells accounted for without a new rule: the counter, where movers and moved objects were
        # and are (a floor re-outlined around them, an object they now cover).
        trace = set(explained) | tick | mover | moved_to
        for kind, c, detail in real:
            if kind == 'moved':
                trace |= c['cells'] | {(x+detail[0], y+detail[1]) for x, y in c['cells']}
        observed, attributed = set(), set()
        for kind, c, detail in real:
            if kind == 'restyled':
                diff = (c['cells'] ^ detail['cells']) if detail['color'] == c['color'] else (c['cells'] | detail['cells'])
                if diff <= trace:
                    continue
            elif kind == 'vanished' and c['cells'] <= trace:
                continue
            if action == 'ACTION6' and c in triggers and kind != 'vanished':
                continue    # the clicked object's own move or colour is a click rule already
            touched = next((t for t in triggers if t is c), None) or triggers[0]
            if kind == 'moved' and detail == d and touched is not c:
                continue    # moved along with the controls without contact: a control effect, not a contact one
            if kind == 'moved':
                kind, effect = ('pushed', 'moves with the controllable object') if (
                    touched is c and detail == d) else ('moved', detail)
            elif kind == 'restyled':
                effect = (detail['color'], detail['sig'][1] != c['sig'][1])
            else:
                effect = None
            h, w = len(before), len(before[0])
            attributed |= {(x, y) for x, y in c['cells'] | (detail['cells'] if kind == 'restyled' else set())
                           if EDGE_BAND <= x < w - EDGE_BAND and EDGE_BAND <= y < h - EDGE_BAND}
            key = (kind, touched['sig'], c['sig'])
            rec = self.schemas.setdefault(key, dict(effect=effect, hits=0, trials=0, action=action))
            rec['effect'] = effect
            observed.add(key)
        touched = {t['sig'] for t in triggers}
        for key, rec in self.schemas.items():
            if key[1] in touched and (action == 'ACTION6') == (rec['action'] == 'ACTION6'):
                rec['trials'] += 1
                rec['hits'] += key in observed
        return attributed

    def schema_text(self, key, name=None):
        """One-line description of a contact schema; name(sig) names an appearance."""
        name = name or (lambda sig: f'{COLOR_NAMES[sig[0]]} object of {len(sig[1])} cells')
        kind, trigger, target = key
        rec = self.schemas[key]
        cause = f"clicking the {name(trigger)}" if rec['action'] == 'ACTION6' else f"moving the controllable object into the {name(trigger)}"
        if kind == 'pushed':
            what = f"pushes it: it moves with the controllable object"
        elif kind == 'moved':
            what = f"moves the {name(target)} by {rec['effect']}"
        elif kind == 'restyled':
            colour, reshaped = rec['effect']
            what = f"turns the {name(target)} {COLOR_NAMES[colour]}" + (' and changes its shape' if reshaped else '')
        else:
            what = f"removes the {name(target)}"
        return f"{cause} {what} ({rec['hits']} of {rec['trials']} contacts)"

    def confirmed(self, key):
        """A contact schema seen at least twice and on at least half of the contacts."""
        r = self.schemas[key]
        return r['hits'] >= 2 and r['hits'] * 2 >= r['trials']

    def pushable(self):
        """Appearances the mover pushes: a 'pushed' schema confirmed at least as often as refuted."""
        return {k[2] for k, r in self.schemas.items() if k[0] == 'pushed' and r['hits'] * 2 >= r['trials']}

    # --- derived knowledge ------------------------------------------------------------
    def movers(self):
        """Appearances that every direction control moved together: the controllable object."""
        by_action = defaultdict(set)
        for (a, sig) in self.move_delta:
            by_action[a].add(sig)
        if not by_action:
            return set()
        common = set.intersection(*by_action.values())
        return common or max(by_action.values(), key=len)

    def direction_deltas(self, sigs):
        out = {}
        for (a, sig), d in self.move_delta.items():
            if sig in sigs:
                out[a] = d
        return out

    def blocking_colors(self, sigs):
        return set().union(*(self.block[(a, s)] for (a, s) in self.move_delta if s in sigs)) - {'edge'}

    def entered_colors(self, sigs):
        return set().union(*(self.enter[(a, s)] for (a, s) in self.move_delta if s in sigs)) - {'edge'}

    def tick_region(self):
        """Cells that keep changing whatever the action (a counter); never a click effect.

        A shrinking bar changes a different cell each step, so besides cells seen changing twice,
        whole lines near the frame edge with unexplained changes in two transitions are masked
        (status bars, as in Graph-Based Exploration for ARC-AGI-3).
        """
        out = {c for c, n in self.tick_seen.items() if n >= 2}
        if self.shape:
            w, h = self.shape
            for y, n in self.tick_rows.items():
                if n >= 2 and (y < EDGE_BAND or y >= h - EDGE_BAND):
                    out |= {(x, y) for x in range(w)}
            for x, n in self.tick_cols.items():
                if n >= 2 and (x < EDGE_BAND or x >= w - EDGE_BAND):
                    out |= {(x, y) for y in range(h)}
        return out

    def ticks(self):
        return self.tick_steps >= 2 and self.ticked / self.tick_steps > 0.5

    # --- prediction and validation -------------------------------------------------------
    def clicked(self, grid, xy):
        return next((c for c in components(grid) if xy in c['cells']), None)

    def click_context(self, comps, sig):
        """Condition of a click rule: where the objects its clicks have moved currently are."""
        affected = self.click_affects[sig]
        return tuple(sorted((c['sig'][0], len(c['sig'][1]), c['origin']) for c in comps if c['sig'] in affected))

    def state_dependent(self, sig):
        return len({(tuple(sorted(m.items())), r) for _, m, r in self.click_outcomes[sig]}) > 1

    def position_dependent(self, sig):
        """Objects of this appearance, clicked at different places, had different latest effects."""
        effects = {(tuple(sorted(o[-1][1].items())), o[-1][2]) for (s, _), o in self.click_at.items() if s == sig and o}
        return len(effects) > 1

    def inverse_pairs(self):
        """Pairs of clicked objects whose latest effects move the same objects by opposite amounts."""
        latest = {k: o[-1][1] for k, o in self.click_at.items() if o and o[-1][1]}
        keys = sorted(latest)
        out = []
        for i, a in enumerate(keys):
            for b in keys[i+1:]:
                common = set(latest[a]) & set(latest[b])
                if common and all(latest[a][m] == tuple(-v for v in latest[b][m]) for m in common):
                    out.append((a, b, {m: latest[a][m] for m in common}))
        return out

    def click_rule(self, sig, origin, context):
        """(moves, recolor) expected from clicking the object of this appearance at origin, or None.

        The object's own record comes first; the appearance's record stands in only when every
        object of that appearance clicked so far agreed. Within a record, an outcome seen in the
        same context wins; a record with differing outcomes predicts only in a context it has seen.
        """
        outcomes = self.click_at.get((sig, origin))
        if not outcomes:
            if self.position_dependent(sig):
                return None
            outcomes = self.click_outcomes.get(sig)
        if not outcomes:
            return None
        match = [o for o in outcomes if o[0] == context]
        if match:
            return match[-1][1], match[-1][2]
        if len({(tuple(sorted(m.items())), r) for _, m, r in outcomes}) == 1:
            return outcomes[-1][1], outcomes[-1][2]
        return None

    def predict_outcome(self, grid, action, xy=None):
        """What the learned rules expect from one action, or None when no rule applies.

        A click rule whose outcomes differ between observations predicts only in a context it
        has seen (Drescher-style conditioning); elsewhere it abstains instead of guessing.
        """
        if action == 'ACTION6':
            comps = components(grid)
            c = next((c for c in comps if xy in c['cells']), None)
            rule = self.click_rule(c['sig'], c['origin'], self.click_context(comps, c['sig'])) if c else None
            if rule is None:
                return None
            moved, recolor = rule
            return dict(kind='click', clicked=c['sig'], moves=dict(moved), recolor=recolor)
        predicted = self.predict(grid, action)
        return dict(kind='move', moves=predicted) if predicted else None

    def check(self, prediction, before, after, action, xy=None):
        """Compare a prediction with the measured transition; returns (hit, mismatches)."""
        actual = moves(components(before), components(after))
        misses = []
        for sig, d in prediction['moves'].items():
            got = actual.get(sig)
            if got != d:
                misses.append(dict(object=sig[0], cells=len(sig[1]), predicted=d, measured=got))
        if prediction['kind'] == 'click':
            c = self.clicked(before, xy)
            inside = (changed(before, after) & c['cells']) - self.tick_region()
            measured = sorted({after[y][x] for x, y in inside})
            expected = [] if prediction['recolor'] is None else [prediction['recolor']]
            if measured != expected:
                misses.append(dict(object=c['color'], cells=len(c['cells']), predicted_color=expected or None,
                                   measured_color=measured or None))
        return not misses, misses

    def replay_consistency(self):
        """Fraction of stored transitions the current rules reproduce (Twin-style replay check)."""
        checked = hits = 0
        for before, action, xy, after in self.transitions:
            prediction = self.predict_outcome(before, action, xy)
            if prediction is None:
                continue
            checked += 1
            hits += self.check(prediction, before, after, action, xy)[0]
        return hits / checked if checked else None

    # --- tools over a current frame ---------------------------------------------------
    def controllable(self, comps):
        """Components of the controllable object: the parts every direction control moves (the
        core), plus touching parts that some direction control has moved. A body that changes
        its look with the direction (tu93 turns to face its way) moves under one control per look,
        so the core alone can be a single eye cell, and paths planned for it pass gaps the body cannot."""
        sigs = self.movers()
        core = [c for c in comps if c['sig'] in sigs]
        if not core:
            return []
        moved = {s for (_, s) in self.move_delta}
        body, cells = list(core), {p for c in core for p in c['cells']}
        grown = True
        while grown:
            grown = False
            near = {(x+dx, y+dy) for x, y in cells for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))}
            for c in comps:
                if c in body or c['sig'] not in moved or len(c['cells']) > MAX_OBJECT_CELLS or not (c['cells'] & near):
                    continue
                body.append(c)
                cells |= c['cells']
                grown = True
        return body

    def _mover_origin(self, comps):
        cells = [p for c in self.controllable(comps) for p in c['cells']]
        return bbox(cells)[:2] if cells else None

    def mover_cells(self, grid):
        return {cell for c in self.controllable(components(grid)) for cell in c['cells']}, self.movers()

    def predict(self, grid, action):
        """One press from the current frame: which learned objects move, or stay blocked."""
        comps = components(grid)
        out = {}
        stuck = (action, self._mover_origin(comps)) in self.stuck
        movers = self.movers()
        for c in comps:
            key = (action, c['sig'])
            d = self.move_delta.get(key)
            if d is None:
                continue
            dest = destination_colors(grid, c['cells'], d)
            if dest & self.block[key] or 'edge' in dest or (stuck and c['sig'] in movers):
                out[c['sig']] = None
            else:
                out[c['sig']] = d
        return out

    def assumed_deltas(self, sigs, available):
        """Untried direction controls, assumed to follow the convention with the learned step size."""
        learned = self.direction_deltas(sigs)
        if not learned:
            return {}
        step = max(max(abs(dx), abs(dy)) for dx, dy in learned.values())
        # A control that moved nothing keeps the prior until it failed at several places: a single
        # failure is usually a wall there (recorded in stuck), not a dead control (tu93 starts boxed in).
        failed_at = Counter(a for a, _ in self.stuck)
        return {a: (dx*step, dy*step) for a, (dx, dy) in DIRECTION_PRIOR.items()
                if a in available and a not in learned and not self.effective[a] and failed_at[a] < 3}

    def plan_path(self, grid, target_box, limit=4000, available=()):
        """Shortest direction sequence that brings the mover inside (or over) target_box.

        Unknown colours are passable (optimism, as in WorldCoder); learned blockers are not.
        Untried direction controls among available are assumed by convention (see assumed_deltas).
        Returns None when unreachable, or raises ValueError when the target is the mover itself.
        """
        cells, sigs = self.mover_cells(grid)
        deltas = {**self.assumed_deltas(sigs, available), **self.direction_deltas(sigs)}
        if not cells or not deltas:
            return None
        if any(target_box[0] <= x <= target_box[2] and target_box[1] <= y <= target_box[3] for x, y in cells):
            raise ValueError('target overlaps the controllable block itself')
        blocking = self.blocking_colors(sigs)
        h, w = len(grid), len(grid[0])
        x0, y0, x1, y1 = bbox(cells)
        tx0, ty0, tx1, ty1 = target_box

        def reached(dx, dy):
            bx0, by0, bx1, by1 = x0+dx, y0+dy, x1+dx, y1+dy
            fits = tx0 <= bx0 and ty0 <= by0 and bx1 <= tx1 and by1 <= ty1
            covers = bx0 <= tx0 and by0 <= ty0 and tx1 <= bx1 and ty1 <= by1
            return fits or covers

        def free(dx, dy):
            for x, y in cells:
                nx, ny = x+dx, y+dy
                if not (0 <= nx < w and 0 <= ny < h):
                    return False
                if (nx, ny) not in cells and grid[ny][nx] in blocking:
                    return False
            return True

        prev, queue = {(0, 0): None}, deque([(0, 0)])
        while queue and len(prev) < limit:
            p = queue.popleft()
            if reached(*p):
                path = []
                while prev[p] is not None:
                    p, a = prev[p]
                    path.append(a)
                return path[::-1]
            for a, (dx, dy) in deltas.items():
                q = (p[0]+dx, p[1]+dy)
                if q not in prev and (a, (x0+p[0], y0+p[1])) not in self.stuck and free(*q):
                    prev[q] = (p, a)
                    queue.append(q)
        return None
