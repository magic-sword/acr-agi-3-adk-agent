"""Exploratory follow-up: remove only overview panel, preserve both local tiles.

Prepared after inspecting partial overview-condition outputs. It is not a
pre-registered independent confirmation. No original measurements are rewritten.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_region_descriptions import prepare,save,sha,digest,encode


def main(out,reference):
    prepare(out)
    cases=json.loads((out/'cases.json').read_text());jobs=json.loads((out/'description-jobs.json').read_text())
    for c in cases:
        for o in c['candidates']:
            path=out/o['card'];im=Image.open(path).crop((408,0,944,432));im.save(path)
            for tile in o['tiles']:tile['offset'][0]-=408
            j=next(j for j in jobs if j['stage']=='describe' and j['case']==c['name'] and j['candidate_id']==o['id'])
            p=j['payload'];p['messages'][0]['content']=p['messages'][0]['content'].replace('The overview and NEIGHBORHOOD show','The NEIGHBORHOOD shows')
            p['messages'][1]['content'][0]=encode(im);j['payload_digest']=digest(p)
    save(out/'cases.json',cases);save(out/'description-jobs.json',jobs)
    plan=json.loads((out/'plan.json').read_text());plan.update(cases_digest=digest(cases),description_jobs_digest=digest(jobs),
        images={str(p.relative_to(out)):sha(p) for p in (out/'cards').glob('*.png')})
    path=Path(__file__).resolve();plan['sources'][str(path)]=sha(path);(out/'sources'/path.name).write_bytes(path.read_bytes())
    plan['exploratory_amendment']=dict(created=datetime.now(timezone.utc).isoformat(),reference=str(reference),
        trigger='Partial overview-condition outputs sometimes describe another object or call a visually complete object a part.',
        change='Remove left overview panel only. Same clean candidate pixels, same neighborhood pixels, same tile scale, same output schema and gate, same matching and geometry. Remove overview mention from system instruction.',
        limitations='Prepared after seeing some first-arm responses; not held-out validation. Overall image dimensions and visual token processing also change. No gold-driven candidate filtering.')
    save(out/'plan.json',plan)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    a=p.parse_args();main(a.out,a.reference)
