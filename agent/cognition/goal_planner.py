"""Program backward chaining over learned rules (docs/backchain-review-ja.md §6.4, V6).

The model proposes goal predicates; this module finds the press sequence that makes them
true by simulating RuleLearner's operators on a component-level state, or reports which
goal conditions no known operator can affect (the missing causal links to probe).
"""
from collections import deque

from .perception import COLORS
from .predicates import box, describe, holds
from .rules import MAX_OBJECT_CELLS, bbox, components


def object_components(comps, objects):
    """Map object IDs to the components that make them up (most of a component inside the object)."""
    out = {}
    for ident, cells in objects.items():
        out[ident] = [i for i, c in enumerate(comps)
                      if len(c['cells'] & cells) * 2 > len(c['cells'])]
    return out


class Planner:
    def __init__(self, rules, grid, objects, available, max_nodes=6000, max_depth=24):
        self.rules, self.grid, self.available = rules, grid, set(available)
        self.max_nodes, self.max_depth = max_nodes, max_depth
        self.comps = components(grid)
        self.members = object_components(self.comps, objects)
        movers = rules.movers()
        self.mover_idx = [i for i, c in enumerate(self.comps) if c['sig'] in movers]
        self.deltas = {}
        if self.mover_idx:
            learned = rules.direction_deltas(movers)
            self.deltas = {a: d for a, d in {**rules.assumed_deltas(movers, available), **learned}.items() if a in available}
        self.blocking = rules.blocking_colors(movers) if movers else set()
        affected = set().union(*rules.click_affects.values()) if rules.click_affects else set()
        self.click_idx = [i for i, c in enumerate(self.comps)
                          if len(c['cells']) <= MAX_OBJECT_CELLS and rules.click_outcomes.get(c['sig'])]
        self.mutable = sorted(set(self.mover_idx) | set(self.click_idx) |
                              {i for i, c in enumerate(self.comps) if c['sig'] in affected})

    # state: tuple over self.mutable of (dx, dy, colour)
    def start(self):
        return tuple((0, 0, self.comps[i]['color']) for i in self.mutable)

    def _get(self, state, i):
        return state[self.mutable.index(i)] if i in self.mutable else (0, 0, self.comps[i]['color'])

    def sig(self, state, i):
        return (self._get(state, i)[2], self.comps[i]['sig'][1])

    def cells(self, state, i):
        dx, dy, _ = self._get(state, i)
        return {(x+dx, y+dy) for x, y in self.comps[i]['cells']}

    def view(self, state, idents=None):
        out = {}
        for ident, idx in self.members.items():
            if idents is not None and ident not in idents:
                continue
            cells, colors = set(), set()
            for i in idx:
                cells |= self.cells(state, i)
                colors.add(COLORS[self._get(state, i)[2]])
            out[ident] = (cells, colors)
        return out

    def _move(self, state, delta, action=None):
        mover = set().union(*(self.cells(state, i) for i in self.mover_idx))
        if action is not None and (action, bbox(mover)[:2]) in self.rules.stuck:
            return None
        h, w = len(self.grid), len(self.grid[0])
        start_mover = set().union(*(self.comps[i]['cells'] for i in self.mover_idx))
        for x, y in mover:
            nx, ny = x+delta[0], y+delta[1]
            if not (0 <= nx < w and 0 <= ny < h):
                return None
            if (nx, ny) not in mover and (nx, ny) not in start_mover and self.grid[ny][nx] in self.blocking:
                return None
        s = list(state)
        for i in self.mover_idx:
            k = self.mutable.index(i)
            dx, dy, col = s[k]
            s[k] = (dx+delta[0], dy+delta[1], col)
        return tuple(s)

    def _click(self, state, i):
        sig = self.sig(state, i)
        affected = self.rules.click_affects[sig]
        context = tuple(sorted((self.sig(state, j)[0], len(self.comps[j]['cells']),
                                (self.comps[j]['origin'][0]+self._get(state, j)[0],
                                 self.comps[j]['origin'][1]+self._get(state, j)[1]))
                               for j in range(len(self.comps)) if self.sig(state, j) in affected))
        dx, dy, _ = self._get(state, i)
        rule = self.rules.click_rule(sig, (self.comps[i]['origin'][0]+dx, self.comps[i]['origin'][1]+dy), context)
        if rule is None:
            return None
        moved, recolor = rule
        s = list(state)
        for msig, (mx, my) in moved.items():
            for j in self.mutable:
                if self.sig(state, j) == msig:
                    k = self.mutable.index(j)
                    dx, dy, col = s[k]
                    s[k] = (dx+mx, dy+my, col)
        if recolor is not None and i in self.mutable:
            k = self.mutable.index(i)
            dx, dy, _ = s[k]
            s[k] = (dx, dy, recolor)
        return tuple(s)

    def actions(self, state):
        for a, d in self.deltas.items():
            nxt = self._move(state, d, a)
            if nxt is not None:
                yield (a, None), nxt
        if 'ACTION6' in self.available:
            for i in self.click_idx:
                nxt = self._click(state, i)
                if nxt is not None and nxt != state:
                    yield ('ACTION6', i), nxt

    def plan(self, atoms):
        """Shortest action list [(action, component index or None)] satisfying atoms, else None."""
        start = self.start()
        idents = {i for a in atoms for i in (a.get('a'), a.get('b')) if i}
        prev, depth, queue = {start: None}, {start: 0}, deque([start])
        while queue:
            state = queue.popleft()
            view = self.view(state, idents)
            if all(holds(a, view) for a in atoms):
                out = []
                while prev[state] is not None:
                    state, act = prev[state]
                    out.append(act)
                return out[::-1]
            if depth[state] >= self.max_depth or len(prev) >= self.max_nodes:
                continue
            for act, nxt in self.actions(state):
                if nxt not in prev:
                    prev[nxt] = (state, act)
                    depth[nxt] = depth[state] + 1
                    queue.append(nxt)
        return None

    def unaffected(self, atoms):
        """Goal atoms whose objects no known operator can change: the missing causal links."""
        out = []
        for atom in atoms:
            idents = [atom.get('a')] + ([atom.get('b')] if atom.get('b') else [])
            touched = any(i in self.mutable for ident in idents for i in self.members.get(ident, []))
            if not holds(atom, self.view(self.start())) and not touched:
                out.append(atom)
        return out

    def click_point(self, i):
        """A cell of component i in the current frame (plans are executed one step at a time)."""
        return min(self.comps[i]['cells'], key=lambda c: (c[1], c[0]))

    def reachable(self, max_states=2000):
        """Relations the measured rules can make true that are false now: [(atom, steps)], nearest first.

        A breadth-first pass over the same states as plan(): each changeable object is compared
        with every object by bounding box (inside, overlaps, same_column, same_row), contact
        (adjacent) and colour (same_color, color_is). Shapes are not changed by the known rules.
        """
        start = self.start()
        full = self.view(start)
        moving = [ident for ident, idx in self.members.items() if any(i in self.mutable for i in idx)]
        if not moving:
            return []
        boxes = {ident: box(cells) for ident, (cells, _) in full.items() if cells}
        found = {}

        def relations(view, bxs):
            out = []
            for a in moving:
                if a not in bxs:
                    continue
                ca, cola = view[a]
                ax0, ay0, ax1, ay1 = bxs[a]
                if len(cola) == 1:
                    out.append(dict(relation='color_is', a=a, b=None, color=next(iter(cola))))
                for b, (bx0, by0, bx1, by1) in bxs.items():
                    if b == a:
                        continue
                    cb, colb = view[b]
                    if bx0 <= ax0 and by0 <= ay0 and ax1 <= bx1 and ay1 <= by1:
                        out.append(dict(relation='inside', a=a, b=b, color=None))
                    if ax0 <= bx1 and bx0 <= ax1 and ay0 <= by1 and by0 <= ay1:
                        out.append(dict(relation='overlaps', a=a, b=b, color=None))
                    if ax0 <= bx1 and bx0 <= ax1:
                        out.append(dict(relation='same_column', a=a, b=b, color=None))
                    if ay0 <= by1 and by0 <= ay1:
                        out.append(dict(relation='same_row', a=a, b=b, color=None))
                    if ax0 - 1 <= bx1 and bx0 - 1 <= ax1 and ay0 - 1 <= by1 and by0 - 1 <= ay1 and any(
                            (x+dx, y+dy) in cb for x, y in ca for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                        out.append(dict(relation='adjacent', a=a, b=b, color=None))
                    if cola == colb:
                        out.append(dict(relation='same_color', a=a, b=b, color=None))
            return out

        now = {describe(a) for a in relations(full, boxes)}
        prev, depth, queue = {start}, {start: 0}, deque([start])
        while queue:
            state = queue.popleft()
            if state != start:
                view = dict(full)
                view.update(self.view(state, set(moving)))
                bxs = dict(boxes)
                bxs.update({i: box(view[i][0]) for i in moving if view[i][0]})
                for atom in relations(view, bxs):
                    key = describe(atom)
                    if key not in now and key not in found:
                        found[key] = (atom, depth[state])
            if depth[state] >= self.max_depth or len(prev) >= max_states:
                continue
            for _, nxt in self.actions(state):
                if nxt not in prev:
                    prev.add(nxt)
                    depth[nxt] = depth[state] + 1
                    queue.append(nxt)
        return sorted(found.values(), key=lambda f: f[1])
