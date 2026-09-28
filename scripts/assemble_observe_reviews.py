"""Assemble explicitly reviewed records; never infer semantic labels here."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_parallel import read_lines
from scripts.summarize_temporal_overlay import audit_key

out=Path(sys.argv[1]);ledger=json.loads((out/'review-ledger.json').read_text())
rows=[r for r in read_lines(out/'measurements.jsonl') if not r['warmup']]
assert sorted(r['id'] for r in rows)==sorted(r['id'] for r in ledger['rows'])
byid={r['id']:r for r in rows};unique={}
for r in ledger['rows']:
    key=audit_key(byid[r['id']]);assert key==r['key']==r['review']['key']
    if key in unique:
        for field in ['recovered_ids','rejected_ids','contradiction','artifact_hallucination']:
            assert unique[key].get(field)==r['review'].get(field)
    unique[key]=r['review']
(out/'semantic-review.json').write_text(json.dumps(dict(reviewer=ledger['reviewer'],rows=list(unique.values())),ensure_ascii=False,indent=2)+'\n')
print('Assembled',len(unique),'reviewed answer groups')
