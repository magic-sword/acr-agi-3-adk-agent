"""Program-learned causal rules (context-action-result schemas) from measured transitions.

Validated offline in docs/backchain-review-ja.md §6.1 (V1). Objects are 4-connected
same-colour components; an appearance is (colour, shape). Nothing here calls a model.
"""
from collections import Counter, defaultdict, deque

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
        self.recolor = {}                     # (clicked shape, colour) -> colour after
        self.tick_seen = Counter()
        self.tick_rows, self.tick_cols = Counter(), Counter()   # edge lines with unexplained changes
        self.shape = None
        self.tick_steps = self.ticked = 0
        self.transitions = deque(maxlen=64)   # (before, action, xy, after) for replay validation
        # (action, mover top-left) where a learned move did nothing: a wall the colours cannot tell
        # (walls and floor may share a colour), so the position itself is the condition.
        self.stuck = set()

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
                if outcome in self.click_outcomes[clicked['sig']]:
                    self.click_outcomes[clicked['sig']].remove(outcome)
                self.click_outcomes[clicked['sig']].append(outcome)
                explained |= clicked['cells']
        else:
            self.tries[action] += 1
            effect = bool(real)
            self.effective[action] += effect
            by = {c['sig']: c for c in comps}
            place = self._mover_origin(comps)
            if place is not None and any((action, s) in self.move_delta for s in self.movers()):
                if any(s in real for s in self.movers()):
                    self.stuck.discard((action, place))
                else:
                    self.stuck.add((action, place))
            for sig, d in real.items():
                self.move_delta[(action, sig)] = d
                self.enter[(action, sig)] |= destination_colors(before, by[sig]['cells'], d)
            for (a, sig), d in list(self.move_delta.items()):
                if a == action and sig in by and sig not in real:
                    self.block[(a, sig)] |= destination_colors(before, by[sig]['cells'], d) - self.enter[(a, sig)]
        residual = cells - explained
        self.tick_steps += 1
        self.ticked += bool(residual)
        self.tick_seen.update(residual)
        self.shape = (len(before[0]), len(before))
        self.tick_rows.update({y for _, y in residual})
        self.tick_cols.update({x for x, _ in residual})
        return effect

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

    def predict_outcome(self, grid, action, xy=None):
        """What the learned rules expect from one action, or None when no rule applies.

        A click rule whose outcomes differ between observations predicts only in a context it
        has seen (Drescher-style conditioning); elsewhere it abstains instead of guessing.
        """
        if action == 'ACTION6':
            comps = components(grid)
            c = next((c for c in comps if xy in c['cells']), None)
            outcomes = self.click_outcomes.get(c['sig']) if c else None
            if not outcomes:
                return None
            context = self.click_context(comps, c['sig'])
            match = [o for o in outcomes if o[0] == context]
            if match:
                _, moved, recolor = match[-1]
            elif not self.state_dependent(c['sig']):
                _, moved, recolor = outcomes[-1]
            else:
                return None
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
    def _mover_origin(self, comps):
        sigs = self.movers()
        cells = [p for c in comps if c['sig'] in sigs for p in c['cells']]
        return bbox(cells)[:2] if cells else None

    def mover_cells(self, grid):
        sigs = self.movers()
        return {cell for c in components(grid) if c['sig'] in sigs for cell in c['cells']}, sigs

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
        # Only never-pressed controls get the prior: a press that moved nothing refutes it.
        return {a: (dx*step, dy*step) for a, (dx, dy) in DIRECTION_PRIOR.items()
                if a in available and a not in learned and not self.tries[a]}

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
