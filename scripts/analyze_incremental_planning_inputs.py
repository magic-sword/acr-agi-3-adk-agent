"""Verify saved probe requests and summarize responses without executing actions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import jsonschema

from probe_incremental_planning_inputs import build_jobs, read_cases
from probe_planning_fact_representation import build_refinement_jobs
from probe_repeated_click_inputs import digest


def selection(answer):
    if not isinstance(answer, dict):
        return '', '', []
    if 'action' in answer:
        return (answer['action'].get('target_query', ''),
                answer['action'].get('action', ''),
                [ref for t in answer.get('targets', []) for ref in t.get('candidate_refs', [])])
    text = answer.get('next_step', '')
    target = answer.get('target', '')
    refs = sorted(set(answer.get('candidate_refs', []) + re.findall(r'\bf\dt\d+\b', text + ' ' + target)))
    return text + (' | target=' + target if target else ''), answer.get('status', ''), refs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output
    plan = json.loads((out / 'plan.json').read_text())
    results = json.loads((out / 'results.json').read_text())
    cases = read_cases()
    expected = (build_refinement_jobs(cases, plan['repeats']) if plan['phase'] == 'refinement'
                else build_jobs(cases, plan['repeats'], plan['phase']))
    checks, evidence, groups, unique = [], [], defaultdict(list), {}
    for index, row in enumerate(results):
        job = expected[index]
        payload = json.loads((out / row['request_file']).read_text())
        assert row['request_sha256'] == digest(payload) == digest(job['payload'])
        assert all(row[k] == job[k] for k in ('case', 'variant', 'repetition', 'groups'))
        answer = row.get('answer')
        schema_errors = []
        if payload.get('tools') and 'error' not in row:
            if row.get('tool') != payload['tools'][0]['function']['name']:
                schema_errors.append('Wrong or missing tool')
            schema = payload['tools'][0]['function']['parameters']
            schema_errors += [e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(answer)]
        text, status, refs = selection(answer)
        has_candidates = row['variant'].startswith('bridge_') or 'C' in (row['groups'] or [])
        current_refs = {r[0] for r in cases[row['case']]['groups']['C']['candidate_index']}
        invalid_refs = [r for r in refs if r not in current_refs] if has_candidates else None
        item = dict(case=row['case'], variant=row['variant'], repetition=row['repetition'],
                    status=status, selection=text, refs=refs,
                    lexical_white=bool(re.search(r'\bwhite\b', text, re.I)),
                    proposes_click=bool(re.search(r'\bclick\b', text, re.I)) or status == 'CLICK',
                    invalid_current_refs=invalid_refs, schema_errors=schema_errors,
                    answer_sha256=digest(answer), request_sha256=row['request_sha256'],
                    error=row.get('error'))
        evidence.append(item)
        groups[(row['case'], row['variant'])].append(item)
        unique[digest(answer)] = answer
        checks.append(dict(case=row['case'], variant=row['variant'], repetition=row['repetition'],
                           schema_errors=schema_errors, error=row.get('error'),
                           finish_reason=row.get('response', {}).get('choices', [{}])[0].get('finish_reason')))
    summary = []
    for (case, variant), items in groups.items():
        summary.append(dict(case=case, variant=variant, n=len(items),
                            need_information=sum(r['status'] == 'need_information' for r in items),
                            white_click_lexical=sum(r['lexical_white'] and r['proposes_click'] and
                                                    r['status'] != 'need_information' for r in items),
                            unique_answers=len({r['answer_sha256'] for r in items}),
                            unique_requests=len({r['request_sha256'] for r in items}),
                            refs=[r['refs'] for r in items], selections=[r['selection'] for r in items]))
    summary.sort(key=lambda r: (r['variant'], r['case']))
    for name, value in [('evidence.json', evidence), ('summary.json', summary), ('unique-answers.json', unique),
                        ('validation.json', dict(planned=len(expected), completed=len(results),
                                                 request_reconstruction='All saved requests equal allowlisted construction',
                                                 checks=checks))]:
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Input probe responses', '',
             'Lexical white+click counts are screening only, not semantic target correctness.', '',
             '| Variant | Case | n | Need info | White+click | Distinct answers |',
             '|---|---|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f"| {s['variant']} | {s['case'][:2]} | {s['n']} | {s['need_information']} | "
                     f"{s['white_click_lexical']} | {s['unique_answers']} |")
    lines += ['', '## Full selections', '']
    for s in summary:
        lines += [f"### {s['case']} {s['variant']}", '']
        lines += [f'- {x}' for x in s['selections']]
        lines.append('')
    (out / 'responses.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(planned=len(expected), completed=len(results),
                          errors=sum('error' in r for r in results),
                          schema_failures=sum(bool(c['schema_errors']) for c in checks)), indent=2))


if __name__ == '__main__':
    main()
