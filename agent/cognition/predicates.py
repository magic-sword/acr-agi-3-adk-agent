"""State predicates over measured objects: the shared vocabulary of goals and plans.

docs/backchain-review-ja.md §6.2 (V2) found that the public games' win conditions are
conjunctions of position, colour, adjacency and shape relations. Every relation here is
computed by the program from object cells and colours; the model only names hypotheses.
"""
RELATIONS = {
    'inside': 'every cell of a lies within the bounding box of b',
    'overlaps': 'the bounding boxes of a and b intersect',
    'same_column': 'a and b share at least one column',
    'same_row': 'a and b share at least one row',
    'adjacent': 'a cell of a touches a cell of b (4-neighbourhood)',
    'same_color': 'a and b have the same set of colours',
    'color_is': 'a is entirely the given colour',
    'same_shape': 'a and b have the same shape up to position',
    'gone': 'a is no longer present',
}
BINARY = {'inside', 'overlaps', 'same_column', 'same_row', 'adjacent', 'same_color', 'same_shape'}


def box(cells):
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return min(xs), min(ys), max(xs), max(ys)


def shape(cells):
    x0, y0, _, _ = box(cells)
    return frozenset((x-x0, y-y0) for x, y in cells)


def holds(atom, view):
    """Evaluate one atom on a view {object_id: (cells, colours)}; missing objects count as gone."""
    rel, a, b = atom['relation'], atom.get('a'), atom.get('b')
    va, vb = view.get(a), view.get(b)
    if rel == 'gone':
        return va is None or not va[0]
    if va is None or not va[0] or (rel in BINARY and (vb is None or not vb[0])):
        return False
    ca, cola = va
    if rel == 'color_is':
        return cola == {atom.get('color')}
    cb, colb = vb
    ax0, ay0, ax1, ay1 = box(ca)
    bx0, by0, bx1, by1 = box(cb)
    if rel == 'inside':
        return all(bx0 <= x <= bx1 and by0 <= y <= by1 for x, y in ca)
    if rel == 'overlaps':
        return ax0 <= bx1 and bx0 <= ax1 and ay0 <= by1 and by0 <= ay1
    if rel == 'same_column':
        return ax0 <= bx1 and bx0 <= ax1
    if rel == 'same_row':
        return ay0 <= by1 and by0 <= ay1
    if rel == 'adjacent':
        return any((x+dx, y+dy) in cb for x, y in ca for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    if rel == 'same_color':
        return cola == colb
    if rel == 'same_shape':
        return shape(ca) == shape(cb)
    raise ValueError(f'unknown relation {rel}')


def describe(atom):
    rel, a, b = atom['relation'], atom.get('a'), atom.get('b')
    if rel == 'color_is':
        return f"color_is({a}, {atom.get('color')})"
    return f'{rel}({a}, {b})' if rel in BINARY else f'{rel}({a})'
