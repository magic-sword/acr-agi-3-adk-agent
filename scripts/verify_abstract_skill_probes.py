"""Audit paired interventions, request provenance, and repeat sensitivity."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from probe_abstract_skill_planning import build as baseline_build, solve
from probe_abstract_skill_inputs import build as followup_build, prose
from probe_skill_backchain_component import build as component_build
from probe_repeated_click_inputs import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    all_jobs, all_rows = {}, {}
    for name, builder, script in (
        ('abstract-skill-planning', baseline_build, 'probe_abstract_skill_planning.py'),
        ('abstract-skill-inputs', followup_build, 'probe_abstract_skill_inputs.py'),
        ('skill-backchain-component', component_build, 'probe_skill_backchain_component.py')):
        out = ROOT / 'outputs' / (name + '-20260929')
        jobs = builder()
        rows = json.loads((out / 'results.json').read_text())
        plan = json.loads((out / 'plan.json').read_text())
        assert plan['probe_script_sha256'] == hashlib.sha256((ROOT / 'scripts' / script).read_bytes()).hexdigest()
        assert plan['script_sha256'] == hashlib.sha256((ROOT / 'scripts/probe_incremental_planning_inputs.py').read_bytes()).hexdigest()
        assert len(jobs) == len(rows) == plan['requests']
        for i, (job, row) in enumerate(zip(jobs, rows)):
            assert digest(job['payload']) == row['request_sha256'] == plan['jobs'][i]['request_sha256']
            assert digest(json.loads((out / row['request_file']).read_text())) == row['request_sha256']
        all_jobs[name], all_rows[name] = jobs, rows
    base = all_jobs['abstract-skill-planning']
    key = lambda j: (j['scenario'], j['naming'], j['repetition'], j['variant'])
    lookup = {key(j): j for j in base}
    for job in base:
        other = lookup[job['scenario'], job['naming'], job['repetition'], 'direct']
        candidate = deepcopy(job['payload'])
        candidate['messages'][0]['content'][0] = deepcopy(other['payload']['messages'][0]['content'][0])
        assert candidate == other['payload']  # hint intervention changes question only
        other = lookup[job['scenario'], job['naming'], 1-job['repetition'], job['variant']]
        task = json.loads(job['payload']['messages'][0]['content'][1]['text'])
        flipped = deepcopy(task)
        flipped['available_skills'].reverse()
        assert flipped == json.loads(other['payload']['messages'][0]['content'][1]['text'])
        if job['scenario'] == 'other_goal':
            original = json.loads(lookup['closed_gate', job['naming'], job['repetition'], job['variant']]['payload']['messages'][0]['content'][1]['text'])
            original['goal'] = task['goal']
            assert original == task
    for job in all_jobs['abstract-skill-inputs']:
        task = job['task']
        text = job['payload']['messages'][0]['content'][1]['text']
        assert text == (json.dumps(task, separators=(',', ':')) if job['representation'] == 'json' else prose(task))
        if job['scope'] == 'relevant_only':
            original = json.loads(lookup[job['scenario'], 'descriptive', job['repetition'], 'direct']['payload']['messages'][0]['content'][1]['text'])
            assert all(s in original['available_skills'] for s in task['available_skills'])
            assert len(solve(original)) == len(solve(task))
            original['available_skills'] = task['available_skills']
            assert task == original
    for job in all_jobs['skill-backchain-component']:
        original = lookup[job['scenario'], 'descriptive', job['repetition'], 'direct']['payload']
        assert original['messages'][0]['content'][1] == job['payload']['messages'][0]['content'][1]
    baseline_rows = {r['request_sha256']: r for r in all_rows['abstract-skill-planning']}
    repeats = []
    for row in all_rows['abstract-skill-inputs']:
        previous = baseline_rows.get(row['request_sha256'])
        if previous is None:
            continue
        def answers(r):
            # Ignore generated call IDs, preserve every returned argument.
            return [c['function'] for c in r['response']['choices'][0]['message'].get('tool_calls', [])]
        repeats.append(dict(case=row['case'], order=row['repetition'],
                            same_all_function_arguments=answers(row) == answers(previous),
                            previous_error=previous.get('error'), repeat_error=row.get('error')))
    assert len(repeats) == 8
    rows = [r for rs in all_rows.values() for r in rs]
    verification = dict(requests=len(rows), request_hashes_match=True, probe_sources_unchanged=True,
                        matched_interventions_checked=True, exact_repeat_requests=repeats,
                        finish_reasons=dict(Counter(r['response']['choices'][0]['finish_reason'] for r in rows)),
                        max_prompt_tokens=max(r['response']['usage']['prompt_tokens'] for r in rows),
                        response_errors=sum('error' in r for r in rows),
                        runtime_changes=False,
                        sources={s: hashlib.sha256((ROOT / 'scripts' / s).read_bytes()).hexdigest() for s in (
                            'probe_abstract_skill_planning.py', 'probe_abstract_skill_inputs.py',
                            'probe_skill_backchain_component.py', 'analyze_abstract_skill_planning.py',
                            'verify_abstract_skill_probes.py')})
    (ROOT / 'outputs/abstract-skill-planning-20260929/verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
