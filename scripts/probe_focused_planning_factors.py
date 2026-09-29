"""Frozen focused-runtime requests: candidates x history x purpose, no game actions."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from probe_incremental_planning_inputs import execute
from probe_repeated_click_inputs import context_part, digest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/focused-workflow-smoke-20260929-validated'


def build():
    log = next(SOURCE.glob('*/cognition/*.requests.jsonl'))
    rows = [r for r in map(json.loads, log.read_text().splitlines())
            if r.get('work') == 'ground' and r['step'] in (1, 2)]
    assert len(rows) == 2
    jobs = []
    for row in rows:
        original = row['request']
        assert digest(original) == row['request_sha256']
        _, baseline = context_part(original)
        for candidates in ('original', 'diverse'):
            for history in ('full', 'facts', 'none'):
                for purpose in ('original', 'untested_effect'):
                    for reverse in (False, True):
                        payload = deepcopy(original)
                        part, c = context_part(payload)
                        if candidates == 'diverse':
                            # Preserve the actually repeated candidate verbatim. Supply
                            # two visible alternative regions; no affordance is claimed.
                            alternatives = [
                                ('the red square at the top middle of the upper-left group',
                                 'red square in the upper-left group'),
                                ('the blue square at the top right of the upper-right group',
                                 'blue square in the upper-right group')]
                            c['candidates'] = [deepcopy(c['candidates'][0])] + [
                                dict(id=f'c{i+2}', verb='activate', target_query=query,
                                     expected_effect=f'Determine whether activating the {name} produces a visible board change.',
                                     rationale='A visible alternative region whose activation effect is unknown.',
                                     source='proposed', skill_name=None)
                                for i, (query, name) in enumerate(alternatives)]
                        if history != 'full':
                            c['evidence']['conditional_knowledge'] = []
                            c['evidence']['past_trials'] = ([dict(
                                observation_id=t['observation_id'], target_query=t['target_query'],
                                verb=t['verb'], actual_trials=deepcopy(t['actual_trials']))
                                for t in baseline['evidence']['past_trials']] if history == 'facts' else [])
                        if purpose == 'untested_effect':
                            c['purpose'] = dict(observation_id=c['observation_id'],
                                desired_state='The effect of a previously untested interaction is observed.',
                                target_query='Visible regions whose activation effects have not yet been tested.',
                                intent='probe', question='What effect does a previously untested interaction have?',
                                expected_observation='An acknowledged interaction followed by a measured change or no change.',
                                rationale='Testing an unresolved interaction distinguishes possible game rules.')
                        if reverse:
                            c['candidates'].reverse()
                            # Counterbalance both presentation position and c1/c3 labels.
                            for i, candidate in enumerate(c['candidates']): candidate['id'] = f'c{i+1}'
                        if c != baseline:
                            part['text'] = json.dumps(c, separators=(',', ':'))
                        assert payload['tools'] == original['tools']
                        assert payload['messages'][0] == original['messages'][0]
                        assert c['evidence']['current'] == baseline['evidence']['current']
                        assert c['evidence']['available_actions'] == ['CLICK']
                        variant = f'{candidates}-{history}-{purpose}'
                        jobs.append(dict(case=f'after-{row["step"]}-failures', variant=variant,
                            repetition=int(reverse), order='reverse_relabel' if reverse else 'forward',
                            candidates=candidates, history=history, purpose=purpose,
                            offered=deepcopy(c['candidates']), source_request_sha256=row['request_sha256'],
                            payload=payload))
    random.Random(2026093001).shuffle(jobs)
    assert len(jobs) == 48
    return jobs


def order_controls():
    jobs = []
    for base in build():
        if base['candidates'] != 'diverse' or base['history'] != 'full' or base['order'] != 'forward':
            continue
        for mode in ('order_only', 'ids_only'):
            job = deepcopy(base)
            part, c = context_part(job['payload'])
            if mode == 'order_only':
                c['candidates'].reverse()
            else:
                for i, candidate in enumerate(c['candidates']): candidate['id'] = f'c{3-i}'
            part['text'] = json.dumps(c, separators=(',', ':'))
            job.update(variant=job['variant']+'_'+mode, order=mode, offered=deepcopy(c['candidates']))
            jobs.append(job)
    random.Random(2026093002).shuffle(jobs)
    assert len(jobs) == 8
    return jobs


def analyze(out):
    import jsonschema
    results = json.loads((out / 'results.json').read_text())
    groups = {}
    for row in results:
        a = row.get('answer')
        valid = False
        try:
            payload = json.loads((out / row['request_file']).read_text())
            jsonschema.validate(a, payload['tools'][0]['function']['parameters'])
            valid = row.get('tool') == 'submit_plan_choice'
        except (ValueError, jsonschema.ValidationError):
            pass
        chosen = next((c for c in row['offered'] if isinstance(a, dict) and c['id'] == a.get('candidate_id')), None)
        target = chosen['target_query'] if chosen else None
        options = [o['action'] for s in (a.get('procedure') or {}).get('steps', []) for o in s['options']] if valid else []
        record = dict(case=row['case'], order=row['order'], valid=valid,
            selected_id=a.get('candidate_id') if isinstance(a, dict) else None,
            selected_target=target, selected_verb=chosen['verb'] if chosen else None, actions=options,
            exact_target_handoff=bool(options) and all(o.get('target_query') == target for o in options),
            route=a.get('next') if isinstance(a, dict) else None,
            reason=a.get('reason') if isinstance(a, dict) else None,
            request_file=row['request_file'])
        groups.setdefault(row['variant'], []).append(record)
    summary = {}
    for variant, records in groups.items():
        summary[variant] = dict(n=len(records), valid=sum(r['valid'] for r in records),
            white=sum(r['valid'] and r['selected_target'] == 'central white square' for r in records),
            other=sum(r['valid'] and r['selected_target'] not in (None, 'central white square') for r in records),
            reroute=sum(r['valid'] and r['route'] != 'execute' for r in records),
            exact_target_handoff=sum(r['exact_target_handoff'] for r in records), details=records)
    (out / 'analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    lines = ['# Frozen planning input factors', '',
        '| Candidates | History | Purpose | Valid | White target | Other target | Reroute |',
        '|---|---|---|---:|---:|---:|---:|']
    for variant, r in sorted(summary.items()):
        lines.append('| '+' | '.join(variant.split('-'))+f" | {r['valid']}/{r['n']} | {r['white']} | {r['other']} | {r['reroute']} |")
    (out / 'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='details'} for k,v in sorted(summary.items())}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--analyze', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--order-controls', action='store_true')
    args = p.parse_args()
    if args.analyze: return analyze(args.output)
    jobs = order_controls() if args.order_controls else build()
    if args.dry_run:
        print(f'{len(jobs)} requests; invariance checks passed'); return
    execute(jobs, args.output, dict(
        adaptive_followup=('Order-only and ID-only crossed controls; original schema enum order retained.' if args.order_controls else None),
        design='Full factorial: 2 candidate sets x 3 historical evidence forms x 2 purposes x '
               '2 orders/labels x 2 frozen post-failure contexts. Same model, temperature, prompt, schema and current evidence. '
               'Original/full/original/forward exactly reproduces recorded requests.',
        candidates='Preserve the repeated candidate; replace other two with researcher-supplied visible red/blue regions. '
                   'Effects unknown, not claims of successful game actions.',
        history='Full versus factual action/result records without old plans/interpretations versus empty historical fields. '
                'Current scene, purpose and candidate descriptions remain fixed within each comparison; '
                'none is not total removal of all failure-related cues.',
        purpose='Replace the whole purpose card with testing an untested effect; no candidate IDs prescribed.',
        limitations='One unchanged board, two nested histories, deterministic counterbalanced prompts, not independent games. '
                    'Candidate intervention changes descriptions/effects together. Orders and labels change together. '
                    'Planning outputs only; no cursor/game actions, attention weights, success or generalization claims.',
        source=str(SOURCE.relative_to(ROOT)), probe_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    analyze(args.output)


if __name__ == '__main__': main()
