"""Systematic state-graph exploration: the backbone when no goal plan exists.

After Rudakov et al., Graph-Based Exploration for ARC-AGI-3 (2025): states are frames with
the changing counter masked, actions are direction controls and one click per connected
component, untested actions are tried in salience order, and when a state is exhausted the
explorer walks the shortest known path to the nearest state with untested actions. Goal
hypotheses do not replace this search; they only raise the priority of actions on the objects
they mention (docs/goal-inference-literature-ja.md §6).
"""
from collections import deque
import hashlib

from .rules import MAX_OBJECT_CELLS, components

CLICK_TIERS = ((16, 1), (64, 2), (MAX_OBJECT_CELLS, 3))   # smaller components first
BOOST = -1                                                 # actions on hypothesis objects


def state_key(grid, masked):
    rows = [[(-1 if (x, y) in masked else v) for x, v in enumerate(row)] for y, row in enumerate(grid)]
    return hashlib.sha1(repr(rows).encode()).hexdigest()[:16]


class Explorer:
    def __init__(self):
        self.edges = {}      # node -> {action key: next node}
        self.actions = {}    # node -> {action key: (priority, action dict, cells)}
        self.tested = {}     # node -> set of action keys
        self.last = None     # (node, action key) awaiting its outcome
        self.uses = {}       # (node, action key) -> times sent

    def _enumerate(self, grid, available, masked):
        out = {}
        for a in available:
            if a not in ('ACTION6', 'RESET'):
                out[('press', a)] = (0, {'action': a}, set())
        if 'ACTION6' in available:
            for c in components(grid):
                cells = c['cells'] - masked
                if not cells or len(c['cells']) > MAX_OBJECT_CELLS:
                    continue
                tier = next(t for size, t in CLICK_TIERS if len(c['cells']) <= size)
                x, y = min(cells, key=lambda p: (p[1], p[0]))
                out[('click', c['color'], c['origin'])] = (tier, {'action': 'ACTION6', 'x': x, 'y': y}, cells)
        return out

    def observe(self, grid, available, masked):
        """Register the current frame; closes the edge of the previous action."""
        node = state_key(grid, masked)
        if node not in self.actions:
            self.actions[node] = self._enumerate(grid, available, masked)
            self.edges[node] = {}
            self.tested[node] = set()
        if self.last is not None:
            prev, akey = self.last
            self.edges[prev][akey] = node
            self.tested[prev].add(akey)
            self.last = None
        return node

    def _priority(self, node, akey, boost_cells):
        tier, _, cells = self.actions[node][akey]
        return BOOST if boost_cells and cells & boost_cells else tier

    def untested(self, node):
        return [k for k in self.actions[node] if k not in self.tested[node]]

    def choose(self, node, boost_cells=frozenset()):
        """Next action: an untested one here, else the first step toward the nearest such state."""
        for p in range(BOOST, 4):
            here = [k for k in self.untested(node) if self._priority(node, k, boost_cells) <= p]
            if here:
                akey = min(here, key=lambda k: (self._priority(node, k, boost_cells),
                                                len(self.actions[node][k][2]), str(k)))  # smaller first within a tier
                return akey, self.actions[node][akey][1], 'untested_here'
            # Boost cells describe the current frame; elsewhere only the tier counts.
            prev, queue = {node: None}, deque([node])
            while queue:
                n = queue.popleft()
                if n != node and any(self.actions[n][k][0] <= p for k in self.untested(n)):
                    step = n
                    while prev[step][0] != node:
                        step = prev[step][0]
                    return prev[step][1], self.actions[node][prev[step][1]][1], 'walk_to_frontier'
                for akey, nxt in self.edges[n].items():
                    if nxt not in prev and nxt in self.actions:
                        prev[nxt] = (n, akey)
                        queue.append(nxt)
        # No known path to untested actions (e.g. irreversible clicks): keep moving, as in
        # Go-Explore, with the least-used action that leads somewhere else.
        moving = [k for k, nxt in self.edges[node].items() if nxt != node] or list(self.actions[node])
        if not moving:
            return None
        akey = min(moving, key=lambda k: (self.uses.get((node, k), 0), str(k)))
        return akey, self.actions[node][akey][1], 'wander'


    def sent(self, node, akey):
        self.last = (node, akey)
        self.uses[(node, akey)] = self.uses.get((node, akey), 0) + 1

    def stats(self):
        return dict(states=len(self.actions), untested=sum(len(self.untested(n)) for n in self.actions))
