"""Native pixel support. Runs are [y, x_start, x_end_exclusive]."""
from itertools import groupby


def encode(points):
    runs = []
    for y, row in groupby(sorted(points, key=lambda p: (p[1], p[0])), key=lambda p: p[1]):
        xs = [p[0] for p in row]
        start = end = xs[0]
        for x in xs[1:]:
            if x != end + 1:
                runs.append([y, start, end + 1])
                start = x
            end = x
        runs.append([y, start, end + 1])
    return runs


def decode(runs):
    return {(x, y) for y, left, right in runs for x in range(left, right)}


def translate(runs, dx, dy):
    return [[y + dy, left + dx, right + dx] for y, left, right in runs]


def support(candidate):
    if candidate.get('mask_runs') is not None:
        return decode(candidate['mask_runs'])
    if 'children' in candidate:
        return {(x, y) for c in candidate['children']
                for y in range(c['bbox'][1], c['bbox'][3] + 1)
                for x in range(c['bbox'][0], c['bbox'][2] + 1)}
    # A box-only SAM provider has no foreground mask. Never fill its box here.
    if candidate.get('shape') == 'region_proposal':
        return set()
    pattern = candidate.get('pattern', [])
    if isinstance(pattern, dict):
        pattern = [[v for v, n in row for _ in range(n)] for row in pattern['row_runs']]
    x0, y0 = candidate['bbox'][:2]
    return {(x0 + x, y0 + y) for y, row in enumerate(pattern)
            for x, value in enumerate(row) if value is not None}


def native_mask(mask, height, width, scale=6):
    """Majority coverage per source cell; resizing is an explicit hypothesis."""
    import numpy as np
    values = np.asarray(mask, dtype=bool)
    if values.shape != (height * scale, width * scale):
        raise ValueError('unexpected SAM mask size')
    native = values.reshape(height, scale, width, scale).sum(axis=(1, 3)) >= (scale * scale / 2)
    yy, xx = native.nonzero()
    return encode(set(zip(xx.tolist(), yy.tolist())))
