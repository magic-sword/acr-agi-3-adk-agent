"""Small follow-up: require an explicit reference field when prose ignores it."""
import argparse,json,shutil,sys
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_relation_overlay as core
from scripts.benchmark_parallel import digest
from scripts.benchmark_recognition import save_json
from scripts.benchmark_official_relations import sha

SOURCE=Path('outputs/relation-overlay-20260928-final')
METHODS=['reference_repeat','reference_edges']
INSTRUCTION='''In reference, name ONE specific visible neighboring object, corner, or boundary used for this object's spatial comparison. Choose the same reference in both images. Use "none" if no reliable reference can be identified. The entire game board, "center", and "upper left" alone are not specific references.
In before, describe gap, contact, overlap, side, or alignment to that named reference in BEFORE (at most 12 words).
In after, inspect AFTER and describe the SAME relation to the SAME reference (at most 12 words).
In description, state which relation changed or stayed the same and any color/shape change (at most 12 words). Check whether the reference itself moved; a relative change alone does not establish which object moved. If evidence is insufficient, use uncertain. Do not merely repeat the inventory.'''

def prepare(out):
    out.mkdir(exist_ok=False);(out/'sources').mkdir()
    chosen=['ls20_up','cross_1px'];jobs=[]
    for original in json.loads((SOURCE/'jobs.json').read_text()):
        if original['case'] not in chosen or not original['variant'].startswith('relation_'):continue
        j=deepcopy(original);j.update(id=len(jobs),variant=j['variant'].replace('relation_','reference_'),method='reference')
        j['arm']=j['backend']+'_'+j['variant'];t=j['payload']['messages'][1]['content'][-1]['text']
        t=t.replace(core.EVIDENCE['relation'],INSTRUCTION).replace('id, before, after, description, kind','id, reference, before, after, description, kind')
        t=t.replace('"id":"A","before":','"id":"A","reference":"specific visible reference or none","before":')
        j['payload']['messages'][1]['content'][-1]['text']=t;j['payload_digest']=digest(j['payload']);jobs.append(j)
    assert len(jobs)==8
    for name in ['cases','catalogs']:
        rows=[c for c in json.loads((SOURCE/f'{name}.json').read_text()) if c.get('case',c.get('name')) in chosen];save_json(out/f'{name}.json',rows)
    for c in chosen:
        for fr in ['before','after','after_edges']:
            name=f'{c}-{fr}.png';shutil.copyfile(SOURCE/name,out/name)
    save_json(out/'jobs.json',jobs);save_json(out/'input-manifest.json',{p.name:sha(p) for p in out.glob('*.png')})
    plan=dict(created_at=core.now(),requests=8,warmups=4,jobs_digest=digest(jobs),script_sha256=sha('scripts/benchmark_relation_overlay.py'),probe_script_sha256=sha(__file__),arms=[b+'_'+m for b in ['llama','official'] for m in METHODS],
        design='Post-hoc instruction-compliance probe: real ls20 and white-cross 1px, same inventories/images as primary run; add explicit reference field; repeat vs boundary image; two backends; 1024 tokens.',
        reason='Primary native relation prompts produced coarse board locations instead of a named local reference. Test a mandatory output field once; do not tune on later outcomes.',limitations='Only two selected pairs, not a held-out confirmation. Adds a field and changes the instruction together. Compare separately from the 56 primary requests.')
    save_json(out/'plan.json',plan)
    for p in [Path(__file__),Path('scripts/benchmark_relation_overlay.py')]:shutil.copyfile(p,out/'sources'/p.name)
    print('Prepared reference-field probe',len(jobs))

def run(out,backend):
    assert sha(__file__)==json.loads((out/'plan.json').read_text())['probe_script_sha256']
    core.METHODS=METHODS;original_parse=core.parse
    def parse(row,answer,stopped,cat):
        original_parse(row,answer,stopped,cat)
        if row.get('valid'):
            row['reference_schema_valid']=all(list(o)==['id','reference','before','after','description','kind'] and all(isinstance(o[k],str) for k in ['reference','before','after']) for o in row['parsed']['observations'])
            row['valid']=row['valid'] and row['reference_schema_valid']
    core.parse=parse;core.run(out,backend)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True);p.add_argument('--backend',choices=['llama','official']);a=p.parse_args()
    if a.command=='prepare':prepare(a.output)
    else:run(a.output,a.backend)
