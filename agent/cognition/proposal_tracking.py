"""Conservative pixel correspondence for persistent region proposals.

Boxes are half-open internally. Multiple exact matches are unresolved, even
when one lies at the old position. A region ID is not proof of object identity.
"""
from .geometry import extract, components
from .region_masks import encode, support, translate


class ProposalTracker:
    def __init__(self, radius=8):
        self.radius = radius
        self.previous = None
        self.tracks = []
        self.serial = 0
        self.shape = None
        self.program = None

    def add(self, boxes, source, masks=None):
        h, w = self.shape
        # Validate the whole provider response before mutating state.
        for box in boxes:
            if (len(box) != 4 or any(type(x) is not int for x in box) or
                    not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h)):
                raise ValueError('invalid proposal box')
        if masks is not None:
            if len(masks) != len(boxes):
                raise ValueError('proposal masks must align with boxes')
            for box, runs in zip(boxes, masks):
                if any(len(r) != 3 or any(type(v) is not int for v in r) or
                       not (box[1] <= r[0] < box[3] and box[0] <= r[1] < r[2] <= box[2]) for r in runs):
                    raise ValueError('invalid proposal mask')
        def key(box, mask):
            return tuple(box), None if mask is None else tuple(tuple(r) for r in mask)
        existing = {key(t['box'], t.get('mask_runs')): t for t in self.tracks}
        for i, box in enumerate(boxes):
            mask = masks[i] if masks is not None else None
            k = key(box, mask)
            if k in existing:
                if source not in existing[k]['sources']:
                    existing[k]['sources'].append(source)
                continue
            self.serial += 1
            track = dict(id=self.serial, box=list(box), sources=[source])
            if mask is not None:
                track['mask_runs'] = [list(r) for r in mask]
            self.tracks.append(track)
            existing[k] = track

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
            # Keep original parts even when the geometry extractor proposes a whole.
            self.parts = components(grid)
        instances = self.program['instances']
        boxes = [[o['bbox'][0], o['bbox'][1], o['bbox'][2]+1, o['bbox'][3]+1] for o in instances]
        masks = [encode(support(o)) for o in instances]
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
                # Different masks within an identical box remain distinct proposals.
                mask = t.get('mask_runs')
                moved_mask = translate(mask, box[0]-x1, box[1]-y1) if mask is not None else None
                key = box, None if moved_mask is None else tuple(tuple(r) for r in moved_mask)
                found.setdefault(key, []).append(t)
            for (box, mask), matches in found.items():
                if len(matches) != 1:
                    unresolved.extend(dict(track=t, reason='competing_tracks') for t in matches)
                    continue
                t = matches[0]
                moved = dict(t, box=list(box))
                if mask is not None:
                    moved['mask_runs'] = [list(r) for r in mask]
                self.tracks.append(moved)
                links.append(dict(id=t['id'], before=t['box'], after=list(box)))
        self.add(boxes, 'program', masks)
        parts = self.parts[:256]
        self.add([[p['box'][0], p['box'][1], p['box'][2]+1, p['box'][3]+1] for p in parts],
                 'component', [encode(p['pixels']) for p in parts])
        covered = np.zeros(self.shape, bool)
        for t in old+self.tracks:
            x1, y1, x2, y2 = t['box']
            if (x2-x1)*(y2-y1) <= 256:
                covered[y1:y2, x1:x2] = True
        residual = [[int(x), int(y)] for y, x in np.argwhere(changed & ~covered)]
        self.previous = current.copy()
        return dict(links=links, unresolved=unresolved, changed_pixels=int(changed.sum()),
                    uncovered_positions=residual, unchanged=unchanged,
                    extraction_limited=bool(self.program.get('extraction_limited') or len(self.parts) > 256))
