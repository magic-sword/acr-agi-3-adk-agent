"""Local output-sensitivity XAI on the deployed Q4 model; not attention weights."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import re

from probe_incremental_planning_inputs import execute
from probe_repeated_click_inputs import context_part

ROOT=Path(__file__).resolve().parents[1]
ARMS=('baseline','omit_purpose','omit_trials','omit_current','mask_first_description',
      'mask_gold_description','omit_candidate_rationales','reverse_trial_rows',
      'swap_trial_results','attach_facts_to_candidates','reverse_schema_ids','omit_system')


def build(stability=False):
    jobs=[]
    for count in (2,3):
        source=ROOT/f'outputs/candidate-count-20260929/n{count}-p{count}-full_plan-r0.request.json'
        original=json.loads(source.read_text())
        for arm in (("baseline", "attach_facts_to_candidates") if stability else ARMS):
            q=deepcopy(original);part,c=context_part(q)
            if arm=='omit_purpose':c.pop('purpose')
            elif arm=='omit_trials':c['evidence']['past_trials']=[]
            elif arm=='omit_current':c['evidence'].pop('current')
            elif arm in ('mask_first_description','mask_gold_description'):
                card=c['candidates'][0 if arm=='mask_first_description' else -1]
                for k in ('target_query','expected_effect','rationale'):card[k]='[withheld]'
            elif arm=='omit_candidate_rationales':
                for card in c['candidates']:card.pop('rationale')
            elif arm=='reverse_trial_rows':c['evidence']['past_trials'].reverse()
            elif arm=='swap_trial_results':
                facts=c['evidence']['past_trials'];a=next(t for t in facts if t['target_query']=='the amber switch')
                b=next(t for t in facts if t['target_query']==c['candidates'][0]['target_query'])
                for k in ('completed_activations','acknowledged','changed_cell_count'):a[k],b[k]=b[k],a[k]
            elif arm=='attach_facts_to_candidates':
                facts={f['target_query']:f for f in c['evidence']['past_trials']}
                for card in c['candidates']:
                    card['trial_fact']={k:v for k,v in facts[card['target_query']].items() if k!='target_query'}
                c['evidence']['past_trials']=[]
            elif arm=='reverse_schema_ids':
                for b in q['tools'][0]['function']['parameters']['anyOf']:
                    x=b['properties']['candidate_id']
                    if 'enum' in x:x['enum'].reverse()
            elif arm=='omit_system':q['messages']=q['messages'][1:]
            if arm!='baseline':part['text']=json.dumps(c,separators=(',',':'))
            q.update(logprobs=True,top_logprobs=20)
            jobs.append(dict(case=f'n{count}',variant=arm,repetition=0,gold_digit=str(count),
                source=str(source.relative_to(ROOT)),payload=q))
    random.Random(2026092910).shuffle(jobs)
    return jobs


def score(row):
    items=row.get('response',{}).get('choices',[{}])[0].get('logprobs',{}).get('content',[])
    prefix=''
    for item in items:
        if re.search(r'"candidate_id"\s*:\s*"c$',prefix):
            probs={x['token']:x['logprob'] for x in item['top_logprobs']}
            if '1' not in probs or row['gold_digit'] not in probs:
                return dict(score_available=False,error='comparison token not in returned top probabilities')
            a,b=probs['1'],probs[row['gold_digit']]
            return dict(score_available=True,chosen_token=item['token'],first_logp=a,gold_logp=b,
                margin=a-b,first_probability=math.exp(a),gold_probability=math.exp(b),
                pairwise_first_share=1/(1+math.exp(b-a)))
        prefix+=item['token']
    return dict(score_available=False,error='choice-token boundary not found')


def analyze(out):
    rows=json.loads((out/'results.json').read_text());scored=[]
    for r in rows:
        a=r.get('answer') or {}
        scored.append(dict(case=r['case'],arm=r['variant'],choice=a.get('candidate_id'),
                           reason=a.get('reason'),request_file=r['request_file'],**score(r)))
    baseline={r['case']:r['margin'] for r in scored if r['arm']=='baseline' and r['score_available']}
    for r in scored:
        if r['score_available']:r['delta_margin']=r['margin']-baseline[r['case']]
    (out/'analysis.json').write_text(json.dumps(scored,ensure_ascii=False,indent=2)+'\n')
    for r in sorted(scored,key=lambda r:(r['case'],r.get('delta_margin',0))):
        print(r['case'],r['arm'],r['choice'],round(r.get('margin',float('nan')),3),round(r.get('delta_margin',float('nan')),3))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--analyze',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--stability',action='store_true');a=p.parse_args()
    if a.analyze:return analyze(a.output)
    jobs=build(a.stability)
    if a.dry_run:print(len(jobs));return
    execute(jobs,a.output,dict(stability_repeat=a.stability,design='2 fixed failing full-plan contexts x 12 input interventions; deployed Q4 model. '
        'Collect served next-token log probabilities where candidate_id digit is generated. '
        'Report log P(c1 digit) - log P(correct last-candidate digit), and change from baseline.',
        interpretation='Local output sensitivity, not attention weights, not integrated gradients, not SHAP. '
        'Negative delta means intervention weakens the first-over-correct preference. '
        'Deletion may remove necessary information; effect sizes cannot be interpreted as additive causal shares.',
        controls='Reverse factual-row order without changing facts; swap outcomes without changing candidates; '
                 'move exactly the same facts into candidate cards; reverse allowed-ID enum order; omit system instruction.',
        limitations='Only two synthetic histories. Model likelihood is not correctness confidence. '
                    'Margins compare tokens at the generated prefix; other prior generated tokens can change. '
                    'Top-20 omissions reported as unscorable, not zero probability. No image attribution: these prompts are text-only.',
        probe_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    analyze(a.output)


if __name__=='__main__':main()
