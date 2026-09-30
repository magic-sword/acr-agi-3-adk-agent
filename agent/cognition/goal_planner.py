"""Program backward chaining over learned rules (docs/backchain-review-ja.md §6.4, V6).

The model proposes goal predicates; this module finds the press sequence that makes them
true by simulating RuleLearner's operators on a component-level state, or reports which
goal conditions no known operator can affect (the missing causal links to probe).
"""
from collections import deque

from .perception import COLORS
from .predicates import holds
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
        outcomes = self.rules.click_outcomes.get(sig)
        if not outcomes:
            return None
        affected = self.rules.click_affects[sig]
        context = tuple(sorted((self.sig(state, j)[0], len(self.comps[j]['cells']),
                                (self.comps[j]['origin'][0]+self._get(state, j)[0],
                                 self.comps[j]['origin'][1]+self._get(state, j)[1]))
                               for j in range(len(self.comps)) if self.sig(state, j) in affected))
        match = [o for o in outcomes if o[0] == context]
        if match:
            _, moved, recolor = match[-1]
        elif not self.rules.state_dependent(sig):
            _, moved, recolor = outcomes[-1]
        else:
            return None
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
