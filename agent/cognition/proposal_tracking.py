"""Conservative pixel correspondence for persistent region proposals.

Boxes are half-open internally. Multiple exact matches are unresolved, even
when one lies at the old position. A region ID is not proof of object identity.
"""
from .geometry import extract


class ProposalTracker:
    def __init__(self, radius=8):
        self.radius = radius
        self.previous = None
        self.tracks = []
        self.serial = 0
        self.shape = None
        self.program = None

    def add(self, boxes, source):
        h, w = self.shape
        # Validate the whole provider response before mutating state.
        for box in boxes:
            if (len(box) != 4 or any(type(x) is not int for x in box) or
                    not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h)):
                raise ValueError('invalid proposal box')
        existing = {tuple(t['box']): t for t in self.tracks}
        for box in boxes:
            if tuple(box) in existing:
                if source not in existing[tuple(box)]['sources']:
                    existing[tuple(box)]['sources'].append(source)
                continue
            self.serial += 1
            track = dict(id=self.serial, box=list(box), sources=[source])
            self.tracks.append(track)
            existing[tuple(box)] = track

    def advance(self, grid):
        import numpy as np
        current = np.asarray(grid, dtype=np.uint8)
        self.shape = current.shape
        h, w = self.shape
        if self.previous is not None and self.previous.shape != current.shape:
            raise ValueError('reset tracker across frame size boundaries')
        changed = np.zeros(self.shape, bool) if self.previous is None else current != self.previous
        unchanged = self.previous is not None and not changed.any()
        # Identical grids also have identical program proposals; avoid repeated
        # combinatorial geometry extraction on stationary patterned boards.
        if not unchanged:
            self.program = extract(grid, 'p')
        boxes = [[o['bbox'][0], o['bbox'][1], o['bbox'][2]+1, o['bbox'][3]+1]
                 for o in self.program['instances']]
        old = self.tracks
        self.tracks = []
        links, unresolved = [], []
        if unchanged:
            self.tracks = old
            links = [dict(id=t['id'], before=t['box'], after=t['box']) for t in old]
        elif self.previous is not None:
            found = {}
            for t in old:
                x1, y1, x2, y2 = t['box']
                bw, bh = x2-x1, y2-y1
                template = self.previous[y1:y2, x1:x2]
                left, top = max(0, x1-self.radius), max(0, y1-self.radius)
                right, bottom = min(w, x2+self.radius), min(h, y2+self.radius)
                windows = np.lib.stride_tricks.sliding_window_view(
                    current[top:bottom, left:right], (bh, bw))
                matches = np.argwhere(np.all(windows == template, axis=(-2, -1)))
                if len(matches) != 1:
                    unresolved.append(dict(track=t, reason='no_exact_match' if not len(matches) else 'multiple_exact_matches'))
                    continue
                y, x = matches[0]
                box = (int(x+left), int(y+top), int(x+left+bw), int(y+top+bh))
                found.setdefault(box, []).append(t)
            for box, matches in found.items():
                if len(matches) != 1:
                    unresolved.extend(dict(track=t, reason='competing_tracks') for t in matches)
                    continue
                t = matches[0]
                self.tracks.append(dict(t, box=list(box)))
                links.append(dict(id=t['id'], before=t['box'], after=list(box)))
        self.add(boxes, 'program')
        covered = np.zeros(self.shape, bool)
        for t in old+self.tracks:
            x1, y1, x2, y2 = t['box']
            if (x2-x1)*(y2-y1) <= 256:
                covered[y1:y2, x1:x2] = True
        residual = [[int(x), int(y)] for y, x in np.argwhere(changed & ~covered)]
        self.previous = current.copy()
        return dict(links=links, unresolved=unresolved, changed_pixels=int(changed.sum()),
                    uncovered_positions=residual, unchanged=unchanged,
                    extraction_limited=bool(self.program.get('extraction_limited')))
