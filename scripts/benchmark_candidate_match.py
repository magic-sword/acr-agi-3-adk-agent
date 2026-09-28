"""Independent M/N/U quizzes, token probabilities, and whole-inventory latency.

Reuse frozen target-binding frames, annotate every candidate before inference, and
keep calibration frames separate from evaluation. No game or runtime mutations.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import sys
import time
from urllib.parse import urlparse

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_target_binding import compact, support, digest, sha, save, http
from scripts.benchmark_recognition import encode

SOURCE = Path('outputs/target-binding-quiz-20260928')
MODES = ('uncached', 'reuse')
ENCODINGS = ({'1': 'M', '2': 'N', '8': 'U'}, {'1': 'N', '2': 'U', '8': 'M'})
MEANINGS = {'M': 'MATCH', 'N': 'NONMATCH', 'U': 'INSUFFICIENT'}
THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
SYSTEM = (
    'Judge only the focused measured candidate against the target description. '
    'Each complete candidate satisfying the description counts independently; multiple candidates may match. '
    'Do not choose a winner among candidates. '
    'MATCH: the candidate represents one ENTIRE object satisfying ALL stated appearance and spatial conditions. '
    'NONMATCH: visible evidence contradicts a required condition, or the candidate is only a fragment of the described whole. '
    'INSUFFICIENT: the supplied evidence cannot settle whether the candidate matches. '
    'Functional roles must not be guessed from appearance. No action history or established roles are supplied. '
    'Use the current image for surrounding spatial context and the measured description to identify the focused candidate. '
    'Coordinates are original pixels, x rightward, y downward, inclusive bounds. '
    'Answer with exactly one offered digit, without explanation.'
)


def annotate(source_cases):
    boards = sorted({digest(c['grid']) for c in source_cases if c['origin'] == 'synthetic'})
    random.Random(929041).shuffle(boards)
    calibration = set(boards[:14])
    cases = deepcopy(source_cases)
    for c in cases:
        c['split'] = 'calibration' if c['origin'] == 'synthetic' and digest(c['grid']) in calibration else 'test'
        regions = [{tuple(p) for p in region} for region in c['annotated_supports']]
        c['candidate_gold'] = {
            o['id']: ('U' if c['gold_reason'] == 'role_unknown' else
                      'M' if any(support(o) == region for region in regions) else 'N')
            for o in c['candidates']}
        # The prior task required a UNIQUE object; this task asks membership.
        if c['gold_reason'] == 'unique':
            assert [k for k, v in c['candidate_gold'].items() if v == 'M'] == [c['gold_candidate']]
        elif c['gold_reason'] == 'ambiguous':
            assert sum(v == 'M' for v in c['candidate_gold'].values()) == len(regions)
        elif c['gold_reason'] in ('fragmented', 'absent'):
            assert set(c['candidate_gold'].values()) == {'N'}
    assert not ({digest(c['grid']) for c in cases if c['split'] == 'calibration'} &
                {digest(c['grid']) for c in cases if c['split'] == 'test'})
    return cases


def request_for(c, candidate, encoding, image, cache):
    options = ENCODINGS[encoding]
    common = ('Target condition: an object satisfying this description: '+c['query']+'\n'
              + '\n'.join(k+'. '+MEANINGS[v] for k, v in options.items()))
    return dict(model='qwen3-vl-4b-instruct', temperature=0, seed=929041,
                max_tokens=1, stream=False, cache_prompt=cache,
                logprobs=True, top_logprobs=20, post_sampling_probs=False,
                grammar='root ::= "1" | "2" | "8"',
                messages=[dict(role='system', content=SYSTEM),
                          dict(role='user', content=[deepcopy(image), dict(type='text', text=common),
                               dict(type='text', text='Focused candidate '+candidate['id']+': '
                                    +compact(candidate)+'\nDoes this candidate match?')])])


def prepare(out):
    out.mkdir(parents=True, exist_ok=False); (out/'sources').mkdir()
    cases = annotate(json.loads((SOURCE/'cases.json').read_text()))
    jobs = []; blocks = []
    for c in cases:
        source_image = SOURCE/(c['name']+'.png'); shutil.copyfile(source_image, out/source_image.name)
        image = encode(Image.open(source_image).convert('RGB'))
        for encoding in range(2):
            candidates = c['candidates'].copy()
            random.Random(int(digest(c['name'])[:8], 16)+encoding).shuffle(candidates)
            for mode in MODES:
                block = dict(id=len(blocks), case=c['name'], mode=mode, encoding=encoding, job_ids=[])
                for position, candidate in enumerate(candidates):
                    payload = request_for(c, candidate, encoding, image, mode == 'reuse' and position > 0)
                    job = dict(id=len(jobs), block_id=block['id'], case=c['name'], candidate=candidate['id'],
                               mode=mode, encoding=encoding, position=position, payload=payload,
                               payload_digest=digest(payload))
                    block['job_ids'].append(job['id']); jobs.append(job)
                blocks.append(block)
    save(out/'cases.json', cases); save(out/'jobs.json', jobs); save(out/'blocks.json', blocks)
    paths = [Path(__file__), Path('scripts/benchmark_target_binding.py'), Path('scripts/benchmark_recognition.py')]
    for p in paths: shutil.copyfile(p, out/'sources'/p.name)
    save(out/'plan.json', dict(created=datetime.now(timezone.utc).isoformat(), cases=len(cases),
         candidates=sum(len(c['candidates']) for c in cases), requests=len(jobs), blocks=len(blocks),
         cases_digest=digest(cases), jobs_digest=digest(jobs), blocks_digest=digest(blocks),
         source_hashes={str(p): sha(p) for p in paths}, source_cases_sha256=sha(SOURCE/'cases.json'),
         image_hashes={p.name: sha(p) for p in out.glob('*.png')}, encodings=ENCODINGS,
         thresholds=THRESHOLDS,
         design='Each of 568 candidates across 66 frozen target conditions receives an independent MATCH/NONMATCH/INSUFFICIENT quiz. Two digit encodings and two cache modes. All candidates evaluated, no oracle shortlist. Same image and target prefix, only focused candidate changes. First request of every block has cache disabled; reuse mode then enables cache. Randomized block order, no prior answers in context.',
         gold='Exact annotated object support = M, visible mismatch or fragment = N, unestablished role-only description = U for all candidates. Multiple complete matching objects may independently be M. A group of fragments is not assembled. No complete matching candidate is not a claim that the visual target is absent.',
         primary='Reuse mode, encoding 0, held-out-by-frame test split. Report all-candidate set AND aggregate status exactness, class metrics and false matches. Raw labels plus fixed 0.9 gate and threshold selected on calibration only. Candidate accuracy alone is secondary because NONMATCH dominates.',
         calibration='14 synthetic board hashes selected with seed 929041, producing 17 conditions/72 candidates. Other 49 conditions/496 candidates are evaluation, including 18 previously seen real-frame conditions. No frame overlap. Candidate thresholds chosen to maximize calibration task exactness, then minimize false matches, maximize resolved cases, then lower threshold.',
         limits='Reused questions and synthetic templates; test split is held out from threshold fitting, not unseen development data. One call per mode/encoding/candidate; cache pairs have identical messages but are not exact repeat requests. Cache correctness and digit encoding stability are checked separately. Current images plus focal measurement differ from prior full-inventory prompt, so historical comparison does not isolate response format. No gameplay or runtime integration.'))
    print('Prepared', len(jobs), 'requests in', len(blocks), 'blocks', flush=True)


def verify_inputs(out):
    load = lambda name: json.loads((out/name).read_text())
    plan, cases, jobs, blocks = (load(n) for n in ('plan.json', 'cases.json', 'jobs.json', 'blocks.json'))
    for name, data in [('cases', cases), ('jobs', jobs), ('blocks', blocks)]:
        assert digest(data) == plan[name+'_digest']
    for path, expected in plan['source_hashes'].items():
        assert sha(path) == expected and sha(out/'sources'/Path(path).name) == expected
    assert sha(SOURCE/'cases.json') == plan['source_cases_sha256']
    for name, expected in plan['image_hashes'].items(): assert sha(out/name) == expected
    for j in jobs: assert digest(j['payload']) == j['payload_digest']
    return plan, cases, jobs, blocks


def run(out, base):
    parsed = urlparse(base)
    assert parsed.scheme == 'http' and parsed.hostname in ('localhost', '127.0.0.1')
    plan, cases, jobs, blocks = verify_inputs(out)
    assert not (out/'responses.jsonl').exists()
    save(out/'server-before.json', http(base+'/props'))
    audit = {s: http(base+'/tokenize', dict(content=s, add_special=False)) for s in ('1', '2', '8')}
    assert all(len(v['tokens']) == 1 for v in audit.values())
    save(out/'token-audit.json', audit)
    order = blocks.copy(); random.Random(929081).shuffle(order)
    save(out/'run-order.json', [b['id'] for b in order])
    completed = 0; run_started = time.monotonic()
    with (out/'responses.jsonl').open('w') as log, (out/'block-times.jsonl').open('w') as times:
        for b in order:
            block_started = time.monotonic()
            for jid in b['job_ids']:
                j = jobs[jid]; started = time.monotonic()
                r = {k: j[k] for k in ('id', 'block_id', 'case', 'candidate', 'mode', 'encoding', 'position', 'payload_digest')}
                try:
                    response = http(base+'/v1/chat/completions', j['payload'])
                    r.update(response=response, answer=response['choices'][0]['message']['content'])
                except Exception as exc: r.update(error=f'{type(exc).__name__}: {exc}', answer='')
                r['seconds'] = time.monotonic()-started
                log.write(json.dumps(r)+'\n'); log.flush(); completed += 1
            times.write(json.dumps(dict(block_id=b['id'], case=b['case'], mode=b['mode'], encoding=b['encoding'],
                                        requests=len(b['job_ids']), seconds=time.monotonic()-block_started))+'\n')
            times.flush()
            if completed//100 != (completed-len(b['job_ids']))//100:
                print(completed, '/', len(jobs), round(time.monotonic()-run_started, 1), 'seconds', flush=True)
    save(out/'server-after.json', http(base+'/props')); save(out/'health-after.json', http(base+'/health'))


def probabilities(choice, encoding):
    records = (choice.get('logprobs') or {}).get('content') or []
    entries = records[0].get('top_logprobs', []) if records else []
    bytoken = {e['token']: e['logprob'] for e in entries}
    if records: bytoken.setdefault(records[0]['token'], records[0]['logprob'])
    missing = [k for k in ENCODINGS[encoding] if k not in bytoken or not math.isfinite(bytoken[k])]
    if missing: return dict(complete=False, missing=missing, normalized=None, offered_mass=None)
    logits = {v: bytoken[k] for k, v in ENCODINGS[encoding].items()}
    largest = max(logits.values()); weights = {k: math.exp(v-largest) for k, v in logits.items()}
    total = sum(weights.values())
    return dict(complete=True, missing=[], normalized={k: v/total for k, v in weights.items()},
                offered_mass=sum(math.exp(v) for v in logits.values()))


def gated(row, threshold=None):
    if not row['valid']: return 'U'
    if threshold is None or row['decision'] == 'U': return row['decision']
    p = row['probabilities']['normalized']
    return row['decision'] if p is not None and p[row['decision']] >= threshold else 'U'


def aggregate(decisions):
    matches = sorted(k for k, v in decisions.items() if v == 'M')
    uncertain = sorted(k for k, v in decisions.items() if v == 'U')
    status = ('information_insufficient' if uncertain else 'multiple' if len(matches) > 1
              else 'single' if matches else 'no_complete_match')
    return dict(matches=matches, uncertain=uncertain, status=status)


def evaluate(rows, cases, threshold=None):
    per_class = {}
    for label in 'MNU':
        tp = sum(r['gold'] == gated(r, threshold) == label for r in rows)
        pred = sum(gated(r, threshold) == label for r in rows)
        gold = sum(r['gold'] == label for r in rows)
        per_class[label] = dict(tp=tp, predicted=pred, gold=gold,
            precision=tp/pred if pred else None, recall=tp/gold if gold else None,
            f1=2*tp/(pred+gold) if pred+gold else None)
    tasks = []
    for c in cases:
        rr = [r for r in rows if r['case'] == c['name']]
        assert len(rr) == len(c['candidates']) and len({r['candidate'] for r in rr}) == len(rr)
        actual = aggregate({r['candidate']: gated(r, threshold) for r in rr}); gold = aggregate(c['candidate_gold'])
        exact = actual['matches'] == gold['matches'] and actual['status'] == gold['status']
        tasks.append(dict(case=c['name'], origin=c['origin'], group=c['group'], split=c['split'],
                          gold_reason=c['gold_reason'], gold=gold, actual=actual, exact=exact,
                          set_exact=actual['matches'] == gold['matches'],
                          false_matches=sorted(set(actual['matches'])-set(gold['matches'])),
                          missed_matches=sorted(set(gold['matches'])-set(actual['matches']))))
    return dict(threshold=threshold, candidate_correct=sum(r['gold'] == gated(r, threshold) for r in rows),
                candidate_total=len(rows), classes=per_class,
                macro_f1=statistics.mean(v['f1'] for v in per_class.values() if v['f1'] is not None),
                task_exact=sum(t['exact'] for t in tasks), task_total=len(tasks),
                set_exact=sum(t['set_exact'] for t in tasks),
                tasks_with_false_matches=sum(bool(t['false_matches']) for t in tasks),
                false_matches=sum(len(t['false_matches']) for t in tasks),
                missed_matches=sum(len(t['missed_matches']) for t in tasks),
                resolved=sum(t['actual']['status'] != 'information_insufficient' for t in tasks),
                unsafe_resolved=sum(t['actual']['status'] != 'information_insufficient' and not t['exact'] for t in tasks),
                by_reason={g: dict(correct=sum(t['exact'] for t in tasks if t['gold_reason'] == g),
                                   total=sum(t['gold_reason'] == g for t in tasks)) for g in sorted({t['gold_reason'] for t in tasks})},
                tasks=tasks)


def probability_audit(rows):
    complete = [r for r in rows if r['valid'] and r['probabilities']['complete']]
    bins = []
    for i in range(10):
        rr = [r for r in complete if min(9, int(r['probabilities']['normalized'][r['decision']]*10)) == i]
        bins.append(dict(lower=i/10, upper=(i+1)/10, total=len(rr),
            mean_confidence=statistics.mean(r['probabilities']['normalized'][r['decision']] for r in rr) if rr else None,
            accuracy=statistics.mean(r['gold'] == r['decision'] for r in rr) if rr else None))
    return dict(complete=len(complete), total=len(rows),
        mean_offered_mass=statistics.mean(r['probabilities']['offered_mass'] for r in complete) if complete else None,
        multiclass_brier=statistics.mean(sum((r['probabilities']['normalized'][k]-(r['gold'] == k))**2 for k in 'MNU') for r in complete) if complete else None,
        ece=sum(b['total']*abs(b['mean_confidence']-b['accuracy']) for b in bins if b['total'])/len(complete) if complete else None,
        bins=bins)


def analyze(out):
    plan, cases, jobs, blocks = verify_inputs(out)
    raw = [json.loads(l) for l in (out/'responses.jsonl').read_text().splitlines()]
    times = [json.loads(l) for l in (out/'block-times.jsonl').read_text().splitlines()]
    assert Counter(r['id'] for r in raw) == Counter(j['id'] for j in jobs)
    assert Counter(t['block_id'] for t in times) == Counter(b['id'] for b in blocks)
    bycase = {c['name']: c for c in cases}; rows = []
    for r in raw:
        j = jobs[r['id']]; c = bycase[r['case']]
        assert all(r[k] == j[k] for k in ('case', 'candidate', 'mode', 'encoding', 'block_id', 'position', 'payload_digest'))
        usage = r.get('response', {}).get('usage', {})
        choice = (r.get('response', {}).get('choices') or [{}])[0]
        valid = r['answer'] in ENCODINGS[r['encoding']] and usage.get('completion_tokens') == 1
        rows.append({k: v for k, v in r.items() if k != 'response'} | dict(valid=valid,
             decision=ENCODINGS[r['encoding']].get(r['answer'], 'U'), gold=c['candidate_gold'][r['candidate']],
             split=c['split'], origin=c['origin'], probabilities=probabilities(choice, r['encoding']),
             prompt_tokens=usage.get('prompt_tokens'),
             cached_tokens=usage.get('prompt_tokens_details', {}).get('cached_tokens')))
    calibration = [c for c in cases if c['split'] == 'calibration']
    test = [c for c in cases if c['split'] == 'test']
    fitting = [r for r in rows if r['mode'] == 'reuse' and r['encoding'] == 0 and r['split'] == 'calibration']
    fits = [evaluate(fitting, calibration, t) for t in THRESHOLDS]
    best = max(fits, key=lambda v: (v['task_exact'], -v['false_matches'], v['resolved'], -v['threshold']))
    selected = best['threshold']
    save(out/'calibration.json', dict(selected_threshold=selected, split_cases=[c['name'] for c in calibration],
                                    criterion=plan['calibration'], results=fits))
    arms = []
    for mode in MODES:
        for encoding in range(2):
            rr = [r for r in rows if r['mode'] == mode and r['encoding'] == encoding]
            tt = [r for r in rr if r['split'] == 'test']
            bt = [t for t in times if t['mode'] == mode and t['encoding'] == encoding]
            previous = json.loads((SOURCE/'summary.json').read_text())['rows']
            legacy = {r['case']: r for r in previous if r['arm'] == 'compact_image' and r['permutation'] == encoding and r['repeat'] == 0}
            raw_result = evaluate(rr, cases)
            collapsed = {t['case']: (t['actual']['matches'][0] if t['actual']['status'] == 'single' else None) for t in raw_result['tasks']}
            arms.append(dict(mode=mode, encoding=encoding, all_raw=raw_result,
                test_raw=evaluate(tt, test), test_fixed_09=evaluate(tt, test, 0.9), test_calibrated=evaluate(tt, test, selected),
                probability_test=probability_audit(tt),
                valid=sum(r['valid'] for r in rr), calls=len(rr),
                missing_probabilities=sum(not r['probabilities']['complete'] for r in rr),
                median_call_seconds=statistics.median(r['seconds'] for r in rr),
                median_block_seconds=statistics.median(t['seconds'] for t in bt),
                total_http_seconds=sum(r['seconds'] for r in rr),
                median_prompt_tokens=statistics.median(r['prompt_tokens'] or 0 for r in rr),
                median_cached_tokens=statistics.median(r['cached_tokens'] or 0 for r in rr),
                by_origin_time={o: dict(median_block_seconds=statistics.median(t['seconds'] for t in bt if bycase[t['case']]['origin'] == o),
                                       candidate_count=len(next(c for c in cases if c['origin'] == o)['candidates']) if o != 'synthetic' else None)
                                for o in sorted({c['origin'] for c in cases})},
                historical_unique_contract=dict(correct=sum(collapsed[c['name']] == c['gold_candidate'] for c in cases),
                      total=len(cases), previous_correct=sum(legacy[c['name']]['correct'] for c in cases),
                      note='Historical comparison on old unique-or-X contract; different prompt and input context. Not multi-match performance.')))
    index = {(r['mode'], r['encoding'], r['case'], r['candidate']): r for r in rows}
    stability = dict(cache_disagreements=sum(index[('uncached', e, c['name'], o['id'])]['decision'] != index[('reuse', e, c['name'], o['id'])]['decision'] for e in range(2) for c in cases for o in c['candidates']),
                     encoding_disagreements={m: sum(index[(m, 0, c['name'], o['id'])]['decision'] != index[(m, 1, c['name'], o['id'])]['decision'] for c in cases for o in c['candidates']) for m in MODES})
    save(out/'summary.json', dict(selected_threshold=selected, arms=arms, stability=stability, rows=rows))
    assert json.loads((out/'server-before.json').read_text()) == json.loads((out/'server-after.json').read_text())
    assert json.loads((out/'health-after.json').read_text())['status'] == 'ok'
    save(out/'verification.json', dict(passed=True, requests=len(rows), valid=sum(r['valid'] for r in rows),
         complete_probabilities=sum(r['probabilities']['complete'] for r in rows), same_server=True,
         frozen_inputs_and_code=True, all_jobs_once=True, disjoint_calibration_test_frames=True))
    gallery = ['<!doctype html><meta charset="utf-8"><title>Independent candidate quizzes</title><style>body{font:15px sans-serif;max-width:1250px;margin:auto}article{border-top:1px solid #aaa;padding:20px}img{width:250px}td,th{border:1px solid #bbb;padding:5px}table{border-collapse:collapse}pre{white-space:pre-wrap}</style><h1>Candidate MATCH / NONMATCH / INSUFFICIENT</h1><p>Gold and individual predictions. Reuse/encoding 0 displays normalized M,N,U token scores; these are not calibrated truth probabilities.</p>']
    for c in cases:
        gallery.append('<article><h2>'+html.escape(c['name'])+'</h2><p>'+html.escape(c['query']+' | '+c['split'])+'</p><img src="'+c['name']+'.png"><table><tr><th>Candidate</th><th>Gold</th><th>Cold e0</th><th>Reuse e0</th><th>Cold e1</th><th>Reuse e1</th><th>Reuse e0: M / N / U</th></tr>')
        for o in c['candidates']:
            rs = [index[(m, e, c['name'], o['id'])] for e in range(2) for m in MODES]
            ps = rs[1]['probabilities']['normalized']
            gallery.append('<tr><td>'+html.escape(o['id']+' '+compact(o))+'</td><td>'+c['candidate_gold'][o['id']]+'</td>'
                           +''.join('<td>'+r['decision']+(' ✓' if r['decision'] == r['gold'] else ' ✗')+'</td>' for r in rs)
                           +'<td>'+(' / '.join(f'{ps[k]:.4f}' for k in 'MNU') if ps else 'missing')+'</td></tr>')
        gallery.append('</table></article>')
    (out/'gallery.html').write_text('\n'.join(gallery))
    for a in arms:
        print(a['mode'], a['encoding'], 'test', a['test_raw']['task_exact'], '/', len(test),
              'all', a['all_raw']['task_exact'], '/', len(cases), 'block median', round(a['median_block_seconds'], 3),
              'calibrated', a['test_calibrated']['task_exact'], 'threshold', selected)
    print('Stability', stability)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'run', 'analyze'])
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--base', default='http://127.0.0.1:8080')
    args = parser.parse_args()
    if args.command == 'run': run(args.output, args.base)
    else: globals()[args.command](args.output)
