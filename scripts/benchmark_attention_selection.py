"""Fixed-frame comparison: all candidates, rule selection, fast model selection.

Only run contacts the local model. No environment actions or runtime mutations.
Host measurements remain immutable; model judgments reference observation-local IDs.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import random
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.motion_evidence import analyze, components, adjacent, bounds
from scripts.benchmark_parallel import digest, read_lines
from scripts.benchmark_recognition import save_json, decode

ARMS = ('all', 'rule', 'fast')
CHANGES = ('up', 'down', 'left', 'right', 'unchanged', 'shrunk', 'grew', 'shape_changed', 'uncertain')
SLOW = '''Compare BEFORE and AFTER using the supplied candidate groups. IDs refer
only to this observation pair. Candidate grouping and shape correspondence are
hypotheses, not proven physical objects. Host pixel measurements are fixed facts;
interpret their relevance without rewriting coordinates or measured displacement.
Positive dy is downward. Separate a visible change from a functional game role.
Identify the parts relevant to the action response and reassess previous roles.
Include changed parts, small count changes, and useful unchanged reference parts.
Previous Player/Exit/Health descriptions are hypotheses, not evidence. One action
does not establish controllability, health, or an exit. Give concise assessments
using only supplied IDs and integrate their relations. If selected candidates are
insufficient, set needs_more_candidates=true to request the remaining candidates.
Do not infer movement from the action name alone. Use the full images for context.
Submit exactly one submit_assessments tool call.'''
FAST = '''Choose exactly three distinct candidate IDs most useful for examining
what responded to the recorded action and reassessing prior role hypotheses.
Prefer informative changed groups, small count changes, and an unchanged reference
that can challenge a previous interpretation. Consider the images and measured
evidence; a role name in the prior is not a fact. Do not explain or classify roles.
Return only a JSON array of three integers from the supplied candidate IDs.'''


def now():
    return datetime.now(timezone.utc).isoformat()


def candidates(before, after, palette):
    """Disjoint adjacent small parts sharing measured displacement form proposals."""
    evidence = analyze(before, after)
    parts = components(before)
    facts = []
    for r in evidence['regions']:
        f = {'before_box': r['before_box'], 'before_pixels': r['before_pixels'],
             'correspondence': r['correspondence'], 'after_box': None,
             'dx': None, 'dy': None, 'pixel_count_delta': None, 'kind': 'uncertain'}
        matches = r.get('exact_shape_matches', [])
        stationary = next((m for m in matches if (m['dx'], m['dy']) == (0, 0)), None)
        if stationary is not None and r['old_support_unchanged']:
            f.update(stationary, kind='unchanged', pixel_count_delta=0)
        elif len(matches) == 1:
            f.update(matches[0], kind='translation', pixel_count_delta=0)
        else:
            overlaps = r.get('same_color_overlaps', [])
            if len(overlaps) == 1:
                o = overlaps[0]; delta = o['pixel_count_delta']
                f.update(after_box=o['after_box'], pixel_count_delta=delta,
                         kind='grew' if delta > 0 else 'shrunk' if delta < 0 else
                         'unchanged' if r['old_support_unchanged'] else 'shape_changed')
                if f['kind'] == 'unchanged':
                    f.update(dx=0, dy=0)
        if len(matches) > 1:
            f['alternative_shape_matches'] = matches
        facts.append(f)
    eligible = {i for i, f in enumerate(facts) if parts[i]['area'] <= 64
                and f['kind'] in ('unchanged', 'translation')}
    groups = []
    while eligible:
        i = min(eligible); eligible.remove(i); group = [i]; queue = [i]
        while queue:
            j = queue.pop()
            for k in sorted(eligible):
                if ((facts[j]['dx'], facts[j]['dy']) == (facts[k]['dx'], facts[k]['dy'])
                        and adjacent(parts[j]['pixels'], parts[k]['pixels'])):
                    eligible.remove(k); group.append(k); queue.append(k)
        groups.append(sorted(group))
    used = {i for g in groups for i in g}
    groups.extend([i] for i in range(len(parts)) if i not in used)
    groups.sort(key=min)
    out = []
    for number, group in enumerate(groups):
        pts = set().union(*(parts[i]['pixels'] for i in group))
        colors = sorted({parts[i]['color'] for i in group})
        if len(group) == 1:
            f = deepcopy(facts[group[0]])
        else:
            dx, dy = facts[group[0]]['dx'], facts[group[0]]['dy']; box = bounds(pts)
            f = {'before_box': box, 'after_box': [v + (dx if j % 2 == 0 else dy) for j, v in enumerate(box)],
                 'before_pixels': len(pts), 'dx': dx, 'dy': dy, 'pixel_count_delta': 0,
                 'kind': 'translation' if (dx, dy) != (0, 0) else 'unchanged',
                 'correspondence': 'adjacent_parts_share_displacement_not_proof_of_one_object',
                 'part_measurements': [facts[i] for i in group]}
        # Local boundary contrast is a ranking heuristic, never a role label.
        differences = []
        def luminance(c):
            s = palette[str(c)].lstrip('#'); rgb = [int(s[j:j+2], 16) for j in (0, 2, 4)]
            return sum(v * w for v, w in zip(rgb, (.2126, .7152, .0722)))
        for x, y in pts:
            for nx, ny in ((x-1,y), (x+1,y), (x,y-1), (x,y+1)):
                if 0 <= ny < len(before) and 0 <= nx < len(before[0]) and (nx, ny) not in pts:
                    differences.append(abs(luminance(before[y][x]) - luminance(before[ny][nx])))
        out.append({'id': number, 'colors': [{'id': c, 'rgb': palette[str(c)]} for c in colors],
                    'measurement': f, 'component_indices': group,
                    'boundary_contrast': round(statistics.mean(differences), 3) if differences else 0})
    return out


def rule_select(catalog):
    def rank(c):
        f = c['measurement']
        priority = (0 if f['kind'] == 'translation' else 1 if f['kind'] in ('shrunk', 'grew')
                    else 2 if f['kind'] == 'shape_changed' and f['before_pixels'] <= 256
                    else 3 if f['kind'] == 'unchanged' and f['before_pixels'] <= 64 else 4)
        return priority, -c['boundary_contrast'], f['before_pixels'], c['id']
    return [c['id'] for c in sorted(catalog, key=rank)[:3]]


def public(catalog):
    return [{k: v for k, v in c.items() if k not in ('component_indices', 'boundary_contrast')}
            for c in catalog]


def schema(ids):
    fields = {'candidate_id': {'type': 'integer', 'enum': ids},
              'visible_change': {'type': 'string', 'enum': list(CHANGES)},
              'role_hypothesis': {'type': 'string', 'maxLength': 100},
              'role_status': {'type': 'string', 'enum': ['unknown', 'hypothesis', 'confirmed']},
              'evidence': {'type': 'string', 'maxLength': 140}}
    return {'type': 'object', 'additionalProperties': False, 'properties': {
        'assessments': {'type': 'array', 'minItems': 1, 'maxItems': 6, 'items': {
            'type': 'object', 'additionalProperties': False, 'properties': fields, 'required': list(fields)}},
        'relations': {'type': 'string', 'maxLength': 240},
        'needs_more_candidates': {'type': 'boolean'}},
        'required': ['assessments', 'relations', 'needs_more_candidates']}


def payload(case, catalog, fast=False, previous=None, max_tokens=700):
    ctx = deepcopy(case['context']); ctx['candidates'] = public(catalog)
    ctx['candidate_scope'] = 'all' if len(catalog) == len(case['catalog']) else 'selected; remaining candidates can be requested'
    if previous is not None:
        ctx['previous_assessment'] = previous
        ctx['instruction'] = 'Additional candidates are now supplied. Return a complete replacement assessment.'
    result = {'model': 'qwen3-vl-4b-instruct', 'messages': [
        {'role': 'system', 'content': FAST if fast else SLOW},
        {'role': 'user', 'content': [
            {'type': 'text', 'text': 'BEFORE'}, case['images'][0],
            {'type': 'text', 'text': 'AFTER'}, case['images'][1],
            {'type': 'text', 'text': json.dumps(ctx, separators=(',', ':'))}]}],
        'temperature': 0, 'seed': 123, 'cache_prompt': True, 'id_slot': 0, 'stream': False,
        'max_tokens': 32 if fast else max_tokens}
    ids = [c['id'] for c in catalog]
    if fast:
        result['grammar'] = ('root ::= "[" ws choice ws "," ws choice ws "," ws choice ws "]"\n'
                             'choice ::= ' + ' | '.join(json.dumps(str(i)) for i in ids) +
                             '\nws ::= [ \\t\\n]*')
    else:
        result.update(tools=[{'type': 'function', 'function': {'name': 'submit_assessments',
                      'description': 'Interpret selected visible groups and their relations.', 'parameters': schema(ids)}}],
                      tool_choice='required', parallel_tool_calls=False)
    return result


def parse(response, ids, fast=False):
    choice = response['choices'][0]
    if choice.get('finish_reason') == 'length':
        raise ValueError('truncated response')
    message = choice['message']
    if fast:
        result = json.loads(message['content'])
        if (not isinstance(result, list) or len(result) != 3 or any(type(i) is not int or i not in ids for i in result)
                or len(set(result)) != 3):
            raise ValueError('expected three distinct supplied IDs')
        return result
    calls = message.get('tool_calls', [])
    if len(calls) != 1 or calls[0]['function']['name'] != 'submit_assessments':
        raise ValueError('expected one submit_assessments')
    value = calls[0]['function']['arguments']; value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(value, dict) or set(value) != {'assessments', 'relations', 'needs_more_candidates'}:
        raise ValueError('invalid assessment fields')
    if not isinstance(value['relations'], str) or len(value['relations']) > 240 or type(value['needs_more_candidates']) is not bool:
        raise ValueError('invalid relation or expansion request')
    if not isinstance(value['assessments'], list) or not 1 <= len(value['assessments']) <= 6:
        raise ValueError('invalid assessment count')
    for a in value['assessments']:
        if not isinstance(a, dict) or set(a) != {'candidate_id', 'visible_change', 'role_hypothesis', 'role_status', 'evidence'}:
            raise ValueError('invalid region fields')
        if type(a['candidate_id']) is not int or a['candidate_id'] not in ids or a['visible_change'] not in CHANGES:
            raise ValueError('invalid candidate ID or change')
        if a['role_status'] not in ('unknown', 'hypothesis', 'confirmed'):
            raise ValueError('invalid role status')
        for key, limit in (('role_hypothesis', 100), ('evidence', 140)):
            if not isinstance(a[key], str) or len(a[key]) > limit:
                raise ValueError('invalid text field')
    if len({a['candidate_id'] for a in value['assessments']}) != len(value['assessments']):
        raise ValueError('duplicate candidate assessment')
    return value


def coverage(case, ids):
    refs = case['reference_ids']
    return {key: bool(set(v) & set(ids)) for key, v in refs.items()}


def score(case, value, ids):
    selected = coverage(case, ids)
    assessments = {a['candidate_id']: a for a in value.get('assessments', [])}
    result = {'selected_' + k: v for k, v in selected.items()}
    for key, refids in case['reference_ids'].items():
        expected = 'unchanged' if case['name'] == 'same_frame' or key == 'cross' else 'up' if key == 'motion' else 'shrunk'
        result[key] = any(i in assessments and assessments[i]['visible_change'] == expected for i in refids)
    result['all_three'] = all(result[k] for k in ('motion', 'cross', 'bar'))
    result['no_change'] = bool(assessments) and all(a['visible_change'] == 'unchanged' for a in assessments.values())
    result['unsupported_confirmed_roles'] = sum(a['role_status'] == 'confirmed' for a in assessments.values())
    result['measurement_conflicts'] = 0
    for c in case['catalog']:
        if c['id'] not in assessments:
            continue
        f = c['measurement']; kind = f['kind']; expected = kind
        if kind == 'translation':
            expected = 'up' if f['dx'] == 0 and f['dy'] < 0 else 'down' if f['dx'] == 0 else 'right' if f['dy'] == 0 and f['dx'] > 0 else 'left' if f['dy'] == 0 else 'uncertain'
        answer = assessments[c['id']]['visible_change']
        if expected != 'uncertain' and answer not in (expected, 'uncertain'):
            result['measurement_conflicts'] += 1
    return result


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    frames = json.loads((source/'frames.json').read_text())
    originals = json.loads((source/'cases.json').read_text())
    cases = []
    for name in ('up', 'action_hidden', 'same_frame'):
        original = next(c for c in originals if c['name'] == name and c['arm'] == 'raw_prior')
        parts = original['payload']['messages'][1]['content']; ctx = json.loads(parts[-1]['text'])
        ctx['question'] = 'Which groups responded, which reference stayed unchanged, and how should prior roles be treated?'
        before = frames['before']; after = before if name == 'same_frame' else frames['after']
        catalog = candidates(before, after, ctx['palette'])
        # Evaluation annotations never enter model payloads or the selection heuristic.
        refboxes = {'motion': [34,45,38,49], 'cross': [20,31,22,33], 'bar': [13,61,54,62]}
        references = {k: [c['id'] for c in catalog if c['measurement']['before_box'] == b] for k, b in refboxes.items()}
        assert all(len(v) == 1 for v in references.values()), references
        cases.append({'name': name, 'context': ctx, 'images': [p for p in parts if p['type'] == 'image_url'],
                      'catalog': catalog, 'reference_ids': references,
                      'source_request': original['source_request']})
    save_json(output/'cases.json', cases); save_json(output/'frames.json', frames)
    save_json(output/'experiment.json', {'created_at': now(), 'case_digest': digest(cases),
        'source': str(source), 'source_cases_sha256': hashlib.sha256((source/'cases.json').read_bytes()).hexdigest(),
        'arms': ARMS, 'warmups': 1, 'repeats': 3, 'wall_budget_seconds': 20,
        'fast_selection_timeout_seconds': 4, 'selection_size': 3, 'fast_max_tokens': 32,
        'deliberation_total_max_tokens': 700,
        'fallback': 'Invalid fast selection uses all candidates. At most one requested expansion, within remaining time and token budget.',
        'order': 'Catalog shuffled per repetition with seed 9500+rep; same catalog order in all arms. Jobs shuffled with seed 9600+rep.',
        'scope': 'One real ls20 transition; action-hidden and duplicate-frame diagnostics. Not independent game samples.',
        'primary_metrics': ['selection coverage of three reference groups', 'three correctly classified changes by candidate ID',
                            'field conflicts', 'unsupported confirmed roles', 'time including selection and preprocessing'],
        'review': 'Free-text coherence reviewed separately by assistant. Structured factual retrieval is not independent visual understanding.'})
    save_json(output/'rubric.json', {'created_at': now(), 'reference_boxes': refboxes,
        'up_and_hidden': {'motion': 'up', 'cross': 'unchanged', 'bar': 'shrunk'},
        'same_frame': 'all assessed regions unchanged; coverage reported separately',
        'strict_success': 'all three correct and no measured-field conflicts, no unsupported confirmed role; free-text review required separately'})
    for key in ('before', 'after'):
        (output/(key+'.png')).write_bytes((source/(key+'.png')).read_bytes())
    for path in [Path(__file__), Path(__file__).with_name('motion_evidence.py')]:
        (output/path.name).write_bytes(path.read_bytes())
    for c in cases:
        print(c['name'], len(c['catalog']), 'rule', rule_select(c['catalog']), 'references', c['reference_ids'], flush=True)


def http(url, value=None, timeout=5):
    req = urllib.request.Request(url, data=None if value is None else json.dumps(value).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=max(.01, timeout)) as response:
        return json.load(response)


def call(base, p, deadline, trace, stage, timeout_cap=None):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('shared wall budget exhausted')
    entry = {'stage': stage, 'request': p, 'request_digest': digest(p), 'started_at': now()}; trace.append(entry)
    t = time.monotonic()
    try:
        entry['response'] = http(base+'/chat/completions', p, min(remaining, timeout_cap) if timeout_cap else remaining)
        return entry['response']
    except Exception as exc:
        entry['error'] = f'{type(exc).__name__}: {exc}'; raise
    finally:
        entry['seconds'] = time.monotonic() - t


def run_one(case, arm, rep, base, frames):
    row = {'name': case['name'], 'arm': arm, 'repeat': rep, 'warmup': rep == -1, 'started_at': now(), 'trace': []}
    start = time.monotonic(); deadline = start + 20
    before = frames['before']; after = before if case['name'] == 'same_frame' else frames['after']
    catalog = candidates(before, after, case['context']['palette'])
    assert catalog == case['catalog'], 'candidate generation drifted'
    random.Random(9500+rep).shuffle(catalog)
    row['preprocess_seconds'] = time.monotonic() - start
    all_ids = [c['id'] for c in catalog]; ids = all_ids
    if arm == 'rule':
        ids = rule_select(catalog)
    elif arm == 'fast':
        try:
            response = call(base, payload(case, catalog, fast=True), deadline, row['trace'], 'select', 4)
            ids = parse(response, all_ids, fast=True); row['selection_valid'] = True
        except Exception as exc:
            row.update(selection_valid=False, selection_error=f'{type(exc).__name__}: {exc}')
            ids = all_ids
    row['initial_ids'] = ids[:]; row['initial_coverage'] = coverage(case, ids)
    selected = [c for c in catalog if c['id'] in ids]
    try:
        response = call(base, payload(case, selected), deadline, row['trace'], 'integrate')
        value = parse(response, ids); row['initial_parsed'] = value
        used_tokens = response.get('usage', {}).get('completion_tokens', 700)
        if value['needs_more_candidates'] and len(ids) < len(all_ids):
            row['expansion_requested'] = True
            if 700-used_tokens >= 150 and deadline-time.monotonic() >= 2:
                response = call(base, payload(case, catalog, previous=value, max_tokens=700-used_tokens),
                                deadline, row['trace'], 'expand')
                value = parse(response, all_ids); ids = all_ids; row['expanded'] = True
            else:
                row['expansion_unavailable'] = 'remaining time or token budget insufficient'
        row.update(parsed=value, valid=True)
    except Exception as exc:
        row.update(valid=False, error=f'{type(exc).__name__}: {exc}')
    row['final_ids'] = ids; row['seconds'] = time.monotonic() - start
    row['within_budget'] = row['seconds'] <= 20
    row['score'] = score(case, row.get('parsed', {}), ids)
    s = row['score']
    row['strict_structured_success'] = bool(row['valid'] and row['within_budget'] and s['all_three']
        and s['measurement_conflicts'] == 0 and s['unsupported_confirmed_roles'] == 0)
    return row


def run(output, base):
    cases = json.loads((output/'cases.json').read_text()); frames = json.loads((output/'frames.json').read_text())
    props = http(base.removesuffix('/v1')+'/props'); assert props['total_slots'] == 1
    assert not (output/'measurements.jsonl').exists(), 'use a new output directory'
    paths = [Path(__file__), Path(__file__).with_name('motion_evidence.py')]
    save_json(output/'environment.json', {'server': props, 'case_digest': digest(cases),
        'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})
    with (output/'measurements.jsonl').open('w') as log:
        for rep in (-1, 0, 1, 2):
            jobs = [(c, arm) for c in cases for arm in ARMS]; random.Random(9600+rep).shuffle(jobs)
            for case, arm in jobs:
                row = run_one(case, arm, rep, base, frames)
                log.write(json.dumps(row, ensure_ascii=False)+'\n'); log.flush()
                print(rep, row['name'], arm, round(row['seconds'], 2), row['valid'],
                      'selected', row['initial_ids'], 'score', row['score'], row.get('error', ''), flush=True)
    save_json(output/'server-after.json', http(base.removesuffix('/v1')+'/props')); report(output)


def report(output):
    rows = [r for r in read_lines(output/'measurements.jsonl') if not r['warmup']]
    summary = []; unique = {}
    for name in ('up', 'action_hidden', 'same_frame'):
        for arm in ARMS:
            rs = [r for r in rows if (r['name'], r['arm']) == (name, arm)]
            if not rs: continue
            summary.append({'case': name, 'arm': arm, 'n': len(rs), 'valid': sum(r['valid'] for r in rs),
                'initial_all_three': sum(all(r['initial_coverage'].values()) for r in rs),
                'motion': sum(r['score']['motion'] for r in rs), 'cross': sum(r['score']['cross'] for r in rs),
                'bar': sum(r['score']['bar'] for r in rs),
                'strict_structured_success': sum(r['strict_structured_success'] for r in rs),
                'conflicts': sum(r['score']['measurement_conflicts'] for r in rs),
                'confirmed_roles': sum(r['score']['unsupported_confirmed_roles'] for r in rs),
                'no_change': sum(r['score']['no_change'] for r in rs),
                'expansions': sum(r.get('expanded', False) for r in rs),
                'selection_failures': sum(r.get('selection_valid') is False for r in rs),
                'seconds_median': statistics.median(r['seconds'] for r in rs),
                'selection_seconds_median': statistics.median(sum(t['seconds'] for t in r['trace'] if t['stage'] == 'select') for r in rs),
                'preprocess_ms_median': 1000*statistics.median(r['preprocess_seconds'] for r in rs),
                'prompt_tokens_median': statistics.median(sum(t.get('response', {}).get('usage', {}).get('prompt_tokens', 0) for t in r['trace']) for r in rs),
                'completion_tokens_median': statistics.median(sum(t.get('response', {}).get('usage', {}).get('completion_tokens', 0) for t in r['trace']) for r in rs)})
    for r in rows:
        value = r.get('parsed', {'error': r.get('error')}); key = digest(value)
        unique.setdefault(key, {'id': key, 'output': value, 'members': []})['members'].append(
            {k: r[k] for k in ('name', 'arm', 'repeat', 'initial_ids', 'score', 'valid')})
    save_json(output/'summary.json', summary); save_json(output/'review-candidates.json', list(unique.values()))
    lines = ['# 候補選択と統合の比較', '', '同一ls20遷移の診断。IDによる変化分類と、文章の意味の確認は別評価。', '',
        '|ケース|条件|n|形式|3対象選択|移動|十字|帯|構造上の成功|矛盾|確定役割|秒|選択秒|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append('|'+ '|'.join(str(s[k]) for k in ('case', 'arm', 'n', 'valid', 'initial_all_three', 'motion', 'cross', 'bar', 'strict_structured_success', 'conflicts', 'confirmed_roles')) + f'|{s["seconds_median"]:.2f}|{s["selection_seconds_median"]:.2f}|')
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    page = ['<!doctype html><html lang="ja"><meta charset="utf-8"><title>候補選択比較</title>',
            '<style>body{font:16px system-ui;margin:24px}pre{white-space:pre-wrap;background:#f2f4f6;padding:12px}article{border:1px solid #aaa;padding:12px;margin:16px 0}</style>',
            '<h1>候補選択比較</h1><img src="before.png"><img src="after.png">']
    cases = json.loads((output/'cases.json').read_text())
    for c in cases:
        page.append('<details><summary>'+c['name']+' candidates</summary><pre>'+html.escape(json.dumps(c['catalog'], ensure_ascii=False, indent=2))+'</pre></details>')
    for r in unique.values():
        page.append('<article><pre>'+html.escape(json.dumps(r, ensure_ascii=False, indent=2))+'</pre></article>')
    (output/'comparison.html').write_text('\n'.join(page)+'</html>')
    print('Summarized', len(rows), 'measurements;', len(unique), 'unique responses', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('command', choices=['prepare', 'run', 'report'])
    p.add_argument('--source', type=Path, default=Path('outputs/motion-comparison-20260927'))
    p.add_argument('--output', type=Path, required=True); p.add_argument('--base', default='http://127.0.0.1:8080/v1')
    a = p.parse_args()
    if a.command == 'prepare': prepare(a.source, a.output)
    elif a.command == 'run': run(a.output, a.base)
    else: report(a.output)


if __name__ == '__main__':
    main()
