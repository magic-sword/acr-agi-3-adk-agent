"""Build a standalone canvas viewer for exact, co-registered temporal overlays.

No VLM calls, object tracking, correspondence inference, or game actions.
Synthetic targets are specified only to construct controls, never to render masks.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def changed_pixels(before, after):
    if not before or not before[0] or len(before) != len(after):
        raise ValueError('incompatible grids')
    width = len(before[0])
    if any(len(row) != width for grid in (before, after) for row in grid):
        raise ValueError('incompatible grids')
    return [[x, y] for y, row in enumerate(before) for x, v in enumerate(row)
            if v != after[y][x]]


def build_cases(source):
    old = {c['name']: c for c in json.loads(source.read_text())}
    cases = []

    def add(c, name, title, note):
        c = deepcopy(c)
        cases.append(dict(name=name, title=title, note=note, before=c['before'],
                          after=c['after'], palette=c['palette']))

    add(old['ls20_up'], 'ls20_up', '実ログ：ls20の操作前後',
        '実際の観測。移動ブロックと下部の帯に変化があります。画像の位置合わせは元の盤面座標のままです。')
    base = old['synthetic_static']
    block = [(x, y) for y in range(32, 38) for x in range(30, 35)]
    cross = [(x, y) for y, row in enumerate(base['before']) for x, v in enumerate(row) if v == 0]

    def moved(name, title, moves):
        c = deepcopy(base)
        for points, dx, dy in moves:
            for x, y in points:
                c['after'][y][x] = 3
        for points, dx, dy in moves:
            for x, y in points:
                assert 0 <= x+dx < 64 and 0 <= y+dy < 64
                c['after'][y+dy][x+dx] = c['before'][y][x]
        add(c, name, title, '合成した表示確認用の例です。ゲームの実観測ではありません。')

    for n in [1, 2]:
        moved(f'block_{n}px', f'合成：二色ブロックが{n}画素上へ', [(block, 0, -n)])
        moved(f'cross_{n}px', f'合成：白い十字が{n}画素右へ', [(cross, n, 0)])
    moved('opposite_1px', '合成：二色ブロックは上・十字は下へ各1画素', [(block, 0, -1), (cross, 0, 1)])
    for name, title in [('synthetic_reshape', '合成：L字への変形'),
                        ('synthetic_split', '合成：二色ブロックの分離'),
                        ('synthetic_new', '合成：緑の三角形が出現'),
                        ('synthetic_static', '対照：前後が完全に同じ')]:
        add(old[name], name, title, '過去の比較で使用した合成例。役割や因果関係を与えるものではありません。')
    c = deepcopy(base)
    for x, y in cross:
        c['after'][y][x] = 12
    add(c, 'color_only', '対照：十字は移動せず白から橙へ',
        '色だけの変化も差分として強調されます。差分は移動の証明ではありません。')
    for c in cases:
        c['changed'] = changed_pixels(c['before'], c['after'])
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path('outputs/object-gate-20260928/cases.json'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    cases = build_cases(args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    template = Path(__file__).with_suffix('.html').read_text()
    data = json.dumps(cases, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    (args.output/'preview.html').write_text(template.replace('__CASES_JSON__', data))
    (args.output/'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2)+'\n')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {'source': str(args.source), 'source_sha256': sha(args.source),
                'cases_sha256': sha(args.output/'cases.json'),
                'generator_sha256': sha(Path(__file__)), 'template_sha256': sha(Path(__file__).with_suffix('.html')),
                'changed_pixels': {c['name']: len(c['changed']) for c in cases},
                'model_evaluation': 'not performed',
                'mask': 'exact co-registered color-ID inequality; optional Chebyshev dilation only for context brightness',
                'geometry': 'no displacement amplification, interpolation, motion grouping, or object matching',
                'rendering': 'deterministic HTML canvas from raw grids; supplementary visualization, not an observation'}
    (args.output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
