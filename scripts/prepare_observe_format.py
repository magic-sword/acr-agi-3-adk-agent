"""Post-hoc format control, retaining initial-run failures unchanged."""
import json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_parallel import digest
from scripts.benchmark_recognition import save_json
from scripts.benchmark_official_relations import sha
from scripts.benchmark_attention_selection import now

source=Path('outputs/observe-compare-20260928');out=Path('outputs/observe-format-20260928')
out.mkdir(exist_ok=False);(out/'sources').mkdir()
# Five XAI diagnostic scenes with inventory 2 (frequent format failures), plus
# the real scene with inventory 1. This is an explicitly selected diagnostic set.
selection={c:2 for c in ['block_1px','cross_1px','color_only','synthetic_static','synthetic_new']};selection['ls20_up']=1
jobs=[]
for j in json.loads((source/'jobs.json').read_text()):
    if selection.get(j['case'])!=j['repeat']:continue
    j['id']=len(jobs)
    c=j['payload']['messages'][1]['content'][-1]
    schema='{"id":"A","kind":"unchanged","description":"brief visual evidence"}' if j['method']=='direct' else '{"id":"A","before":"visible first-frame state","after":"visible last-frame state","description":"observed agreement or difference","kind":"unchanged"}'
    assert schema in c['text']
    c['text']=c['text'].replace(schema,'{"observations":['+schema+']}')
    c['text']+='\nThe top-level JSON object must contain an observations array. Include every listed ID and X as separate rows in that array. A single row object is not a complete answer.'
    j['payload_digest']=digest(j['payload']);jobs.append(j)
assert len(jobs)==24
cats=[c for c in json.loads((source/'catalogs.json').read_text()) if selection.get(c['case'])==c['repeat']]
cases=[c for c in json.loads((source/'cases.json').read_text()) if c['name'] in selection]
save_json(out/'jobs.json',jobs);save_json(out/'catalogs.json',cats);save_json(out/'cases.json',cases)
shutil.copyfile(source/'input-manifest.json',out/'input-manifest.json')
plan=json.loads((source/'plan.json').read_text());plan.update(created_at=now(),requests=24,jobs_digest=digest(jobs),cases_digest=digest(cases),catalog_digest=digest(cats),
    design='Post-hoc explicit-array format control: five XAI synthetic scenes inventory2 plus real ls20 inventory1; 6 inputs x 2 prompts x 2 backends; unchanged images and 1024-token limit.',
    selection=selection,post_hoc=True,reason='Initial native evidence answers sometimes stopped after a single row. Make the outer array explicit in BOTH prompts. Selected diagnostic set; do not pool with primary 132 requests or claim held-out confirmation.',
    preparation_script_sha256=sha(__file__))
save_json(out/'plan.json',plan)
for name in ['benchmark_observe_compare.py','prepare_observe_format.py']:shutil.copyfile(Path('scripts')/name,out/'sources'/name)
print('Prepared',len(jobs),'post-hoc format controls')
