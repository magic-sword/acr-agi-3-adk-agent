"""Apply a documented scoring-only erratum without changing frozen requests.

The original substring test included 'across' as a cross. Only the two known
real bar annotations are removed; synthetic annotations and all boxes stay fixed.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_native_grounding import verify, evaluate, edges, save, digest, sha


def correct(out):
    _, original, _ = verify(out)
    cases = deepcopy(original)
    changes = []
    for name, gold_id, source in [('vc33', 'g1', 'vc33-pink-bar'),
                                  ('ft09', 'g3', 'ft09-orange-bar')]:
        c = next(c for c in cases if c['name'] == name)
        g = next(g for g in c['gold'] if g['id'] == gold_id)
        assert not c['fully_annotated'] and c['cross_gold_ids'] == [gold_id]
        assert g['source_case'] == source and 'across' in g['kind']
        changes.append(dict(case=name, removed_id=gold_id, description=g['kind']))
        c['cross_gold_ids'] = []
    real = [c for c in cases if not c['fully_annotated']]
    assert [(c['name'], c['cross_gold_ids']) for c in real] == [('ls20', ['g7']), ('vc33', []), ('ft09', [])]
    assert [c for c in cases if c['fully_annotated']] == [c for c in original if c['fully_annotated']]
    rows = [json.loads(l) for l in (out/'responses.jsonl').read_text().splitlines()]
    result = json.loads((out/'summary.json').read_text())
    for arm in result['arms']:
        rr = [r for r in rows if r['arm'] == arm['arm']]
        arm['real']['cross_recall'] = evaluate(rr, real, .5, cross=True, precision_allowed=False)
        if arm['arm'] == 'normalized_cross':
            for t in (.5, .75):
                arm['real'][str(t)] = evaluate(rr, real, t, cross=True, precision_allowed=False)
    program = [dict(case=c['name'], valid=True,
                    objects=[dict(board_box=edges(o['bbox'])) for o in c['program']['instances']]) for c in real]
    result['program']['real']['cross'] = evaluate(program, real, .5, cross=True, precision_allowed=False)
    save(out/'corrected-cases.json', cases)
    save(out/'corrected-summary.json', result)
    gallery = (out/'gallery.html').read_text().replace('Green: frozen gold.', 'Green: gold after documented cross-label erratum (annotation-erratum.json).')
    for change in changes:
        name = change['case']
        im = Image.open(out/(name+'-board.png')).convert('RGB')
        d = ImageDraw.Draw(im)
        r = next(r for r in rows if r['case'] == name and r['arm'] == 'normalized_cross')
        for o in r.get('objects', []) if r['valid'] else []:
            d.rectangle(tuple(v*6 for v in o['board_box']), outline='#ff33ff', width=2)
        old = name+'-normalized_cross-overlay.png'
        new = name+'-normalized_cross-corrected-overlay.png'
        im.save(out/new)
        gallery = gallery.replace(old, new)
    (out/'corrected-gallery.html').write_text(gallery)
    save(out/'annotation-erratum.json', dict(
        reason="Substring 'cross' incorrectly matched 'across' in two real bar descriptions.",
        changes=changes, original_real_cross_count=3, corrected_real_cross_count=1,
        original_cases_digest=digest(original), corrected_cases_digest=digest(cases),
        correction_source_sha256=sha(Path(__file__)),
        unchanged='Frozen cases, jobs, requests, images, raw responses, original summary/gallery; all synthetic scores and all-object scores.',
        inference_repeated=False))
    print('Corrected real cross denominator: 3 -> 1; original measurements preserved.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    correct(p.parse_args().output)
