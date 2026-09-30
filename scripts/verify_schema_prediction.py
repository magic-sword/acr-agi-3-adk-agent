"""V1 (docs/backchain-review-ja.md §6): can the host learn causal schemas from measured
transitions and predict the next step, without a model?

Replays SDK recordings of evaluation runs. Each predictor learns online inside one run and
level, predicting transition t only from transitions before t (as the agent would).

  P0  nothing changes (baseline)
  P1  the same control repeats its last observed effect (like the current control tally)
  P2  context-action-result schemas (Drescher): a move is predicted only when its
      destination holds colours the same object has entered before and no colour it
      failed to enter. A click predicts (a) moves of other objects previously caused by
      clicking the same appearance (non-local effect) and (b) changes inside the clicked
      object seen when that appearance was clicked. A learned "tick" region (cells that
      change regardless of the action, e.g. a step counter) predicts that the board changes.

Cell metrics exclude the tick region each predictor-independent learner has found so far,
so object effects are not credited or blamed for a counter.

usage: python scripts/verify_schema_prediction.py outputs/evaluations/RUN [...]
"""
from collections import Counter, defaultdict
import glob
import json
from pathlib import Path
import sys


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
        if len(c['cells']) > 400 or ca[c['sig']] != 1 or cb[c['sig']] != 1:
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
            new = {(x+dx, y+dy) for x, y in old}
            cells |= old ^ new
    return cells


def destination_colors(grid, comp, d):
    own = comp['cells']
    h, w = len(grid), len(grid[0])
    out = set()
    for x, y in comp['cells']:
        nx, ny = x+d[0], y+d[1]
        if (nx, ny) in own:
            continue
        if not (0 <= nx < w and 0 <= ny < h):
            out.add('edge')
        else:
            out.add(grid[ny][nx])
    return out


def transitions(path):
    rows = [json.loads(l)['data'] for l in open(path)]
    for prev, cur in zip(rows, rows[1:]):
        act = cur.get('action_input') or {}
        if act.get('id') in (None, 'RESET') or cur.get('levels_completed') != prev.get('levels_completed') or cur.get('full_reset'):
            yield None  # level or episode boundary: forget
            continue
        if not prev.get('frame') or not cur.get('frame'):
            continue
        data = act.get('data') or {}
        yield dict(action=act['id'], xy=(data.get('x'), data.get('y')) if act['id'] == 'ACTION6' else None,
                   before=prev['frame'][-1], after=cur['frame'][-1])


class Learner:
    def __init__(self):
        self.last_moves = {}                       # P1: action -> last move dict
        self.last_click = {}                       # P1/P2: (sig, origin) -> absolute changed cells
        self.enter = defaultdict(set)              # P2: (action, sig) -> colours entered
        self.block = defaultdict(set)              # P2: (action, sig) -> colours that stopped it
        self.move_delta = {}                       # P2: (action, sig) -> delta
        self.click_rel = {}                        # P2: sig -> in-object changes relative to clicked origin
        self.click_moves = {}                      # P2: clicked sig -> moves of other objects
        self.recolor = {}                          # P2: (clicked shape, colour) -> colour after the click

    def familiar(self, t, comps):
        """Whether this kind of intervention was observed before (the learnable part)."""
        if t['action'] != 'ACTION6':
            return any(a == t['action'] for a, _ in self.move_delta) or t['action'] in self.last_moves
        c = self.clicked(comps, t['xy'])
        return c is not None and (c['sig'] in self.click_rel or (c['sig'][1], c['color']) in self.recolor)

    def clicked(self, comps, xy):
        return next((c for c in comps if xy in c['cells']), None)

    def predict(self, t, comps, kind):
        if t['action'] == 'ACTION6':
            c = self.clicked(comps, t['xy'])
            if c is None:
                return set(), {}
            if kind == 'P1':
                return set(self.last_click.get((c['sig'], c['origin'])) or ()), {}
            ox, oy = c['origin']
            local = {(ox+dx, oy+dy) for dx, dy in self.click_rel.get(c['sig'], ())}
            if not local and self.recolor.get((c['sig'][1], c['color']), c['color']) != c['color']:
                local = set(c['cells'])  # same shape recoloured before: appearance changes, rule does not
            mv = dict(self.click_moves.get(c['sig'], {}))
            return local | move_cells(comps, mv), mv
        if kind == 'P1':
            mv = dict(self.last_moves.get(t['action'], {}))
        else:
            mv = {}
            for c in comps:
                d = self.move_delta.get((t['action'], c['sig']))
                if d is None:
                    continue
                dest = destination_colors(t['before'], c, d)
                key = (t['action'], c['sig'])
                if dest & self.block[key] or not dest <= self.enter[key]:
                    continue
                mv[c['sig']] = d
        return move_cells(comps, mv), mv

    def learn(self, t, comps, actual_cells, actual_moves):
        if t['action'] == 'ACTION6':
            c = self.clicked(comps, t['xy'])
            if c is not None:
                self.last_click[(c['sig'], c['origin'])] = set(actual_cells)
                ox, oy = c['origin']
                self.click_rel[c['sig']] = {(x-ox, y-oy) for x, y in actual_cells & c['cells']}
                self.click_moves[c['sig']] = dict(actual_moves)
                inside = actual_cells & c['cells']
                if inside:
                    x, y = next(iter(inside))
                    self.recolor[(c['sig'][1], c['color'])] = t['after'][y][x]
            return
        self.last_moves[t['action']] = dict(actual_moves)
        by = {c['sig']: c for c in comps}
        for sig, d in actual_moves.items():
            self.move_delta[(t['action'], sig)] = d
            self.enter[(t['action'], sig)] |= destination_colors(t['before'], by[sig], d)
        for (action, sig), d in list(self.move_delta.items()):
            if action == t['action'] and sig in by and sig not in actual_moves:
                self.block[(action, sig)] |= destination_colors(t['before'], by[sig], d) - self.enter[(action, sig)]


class Tick:
    """Cells that change without being explained by object moves or the clicked object."""
    def __init__(self):
        self.seen = Counter()
        self.steps = self.ticked = 0

    def region(self):
        return {c for c, n in self.seen.items() if n >= 2}

    def learn(self, comps, t, actual, real_moves):
        explained = move_cells(comps, real_moves)
        if t['action'] == 'ACTION6':
            explained |= next((c['cells'] for c in comps if t['xy'] in c['cells']), set())
        residual = actual - explained
        self.steps += 1
        self.ticked += bool(residual)
        self.seen.update(residual)

    def expects_change(self):
        return self.steps >= 2 and self.ticked / self.steps > 0.5


def score(pred, actual):
    tp = len(pred & actual)
    return tp, len(pred), len(actual)


def main(runs):
    totals = defaultdict(lambda: defaultdict(float))
    for run in runs:
        for rec in glob.glob(str(Path(run) / '*' / 'recordings' / '*' / '*.jsonl')):
            game = Path(rec).parts[-4][:4]
            learners = {k: Learner() for k in ('P1', 'P2')}
            tick = Tick()
            for t in transitions(rec):
                if t is None:
                    learners = {k: Learner() for k in ('P1', 'P2')}
                    tick = Tick()
                    continue
                comps = components(t['before'])
                full = changed(t['before'], t['after'])
                hud = tick.region()
                actual = full - hud
                real_moves = moves(comps, components(t['after']))
                g = totals[game]
                g['steps'] += 1
                seen = learners['P2'].familiar(t, comps)
                g['familiar_steps'] += seen
                for kind in ('P0', 'P1', 'P2'):
                    pred, mv = (set(), {}) if kind == 'P0' else learners[kind].predict(t, comps, kind)
                    pred -= hud
                    tp, np_, na = score(pred, actual)
                    g[f'{kind}.tp'] += tp; g[f'{kind}.pred'] += np_; g[f'{kind}.act'] += na
                    predicts_change = bool(pred) or (kind == 'P2' and tick.expects_change())
                    g[f'{kind}.change_ok'] += (predicts_change == bool(full))
                    if seen:
                        g[f'{kind}.ftp'] += tp; g[f'{kind}.fpred'] += np_; g[f'{kind}.fact'] += na
                    if t['action'] != 'ACTION6':
                        g['move_steps'] += kind == 'P0'
                        g[f'{kind}.moves_exact'] += (mv == real_moves)
                for learner in learners.values():
                    learner.learn(t, comps, actual, real_moves)
                tick.learn(comps, t, full, real_moves)
    print('|game|steps (familiar)|predictor|board-change accuracy|changed-cell precision|recall|F1|F1 on familiar steps|exact moved objects+direction (non-click)|')
    print('|---|---:|---|---:|---:|---:|---:|---:|---:|')
    for game, g in sorted(totals.items()):
        for kind in ('P0', 'P1', 'P2'):
            p = g[f'{kind}.tp']/g[f'{kind}.pred'] if g[f'{kind}.pred'] else 0
            r = g[f'{kind}.tp']/g[f'{kind}.act'] if g[f'{kind}.act'] else 0
            f1 = 2*p*r/(p+r) if p+r else 0
            mx = f"{g[f'{kind}.moves_exact']/g['move_steps']:.2f}" if g['move_steps'] else '-'
            fp = g[f'{kind}.ftp']/g[f'{kind}.fpred'] if g[f'{kind}.fpred'] else 0
            fr = g[f'{kind}.ftp']/g[f'{kind}.fact'] if g[f'{kind}.fact'] else 0
            ff = 2*fp*fr/(fp+fr) if fp+fr else 0
            print(f"|{game}|{int(g['steps'])} ({int(g['familiar_steps'])})|{kind}|{g[f'{kind}.change_ok']/g['steps']:.2f}|{p:.2f}|{r:.2f}|{f1:.2f}|{ff:.2f}|{mx}|")


if __name__ == '__main__':
    main(sys.argv[1:])
