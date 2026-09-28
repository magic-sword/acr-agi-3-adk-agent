"""Independent bitset-DP audit of frozen SAM proposal results and input bytes."""
import base64
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_sam_proposals import verify, sha, save


def independent_count(boxes,gold,padding,area_cap):
    # All reachable subsets of matched gold, rather than augmenting-path matching.
    reachable={0}
    for b in boxes:
        left=max(0,b[0]*64/1000-padding); top=max(0,b[1]*64/1000-padding)
        right=min(64,b[2]*64/1000+padding); bottom=min(64,b[3]*64/1000+padding)
        eligible=[]
        for j,g in enumerate(gold):
            pixels=np.asarray(g['support'],dtype=float)
            overlap=np.clip(np.minimum(pixels+1,[right,bottom])-np.maximum(pixels,[left,top]),0,1)
            covered=np.prod(overlap,axis=1).sum()/len(pixels)
            x1,y1,x2,y2=g['bbox']
            if covered+1e-9>=.9 and (right-left)*(bottom-top)<=(x2-x1+1)*(y2-y1+1)*area_cap+1e-9:
                eligible.append(j)
        reachable |= {state | (1<<j) for state in reachable for j in eligible if not state & (1<<j)}
    return max(state.bit_count() for state in reachable)


def run(out):
    verify(out,model=True)
    cases={c['name']:c for c in json.loads((out/'cases.json').read_text())}
    rows=[json.loads(line) for name in ('sam-results.jsonl','qwen-results.jsonl')
          for line in (out/name).read_text().splitlines()]
    lookup={(r['case'],r['arm']):r for r in rows}
    mask_count=0
    for r in rows:
        if not r['arm'].startswith('sam_'):continue
        assert sha(out/r['masks'])==r['masks_sha256']
        with np.load(out/r['masks']) as masks:
            assert len(masks.files)==len(r['records'])
            last=None
            for record,box in zip(r['records'],r['boxes']):
                mask=masks[str(record['index'])]
                coords=np.argwhere(mask)
                y1,x1=coords.min(axis=0);y2,x2=coords.max(axis=0)+1
                expected=np.array([x1,y1,x2,y2])/np.array([mask.shape[1],mask.shape[0]]*2)*1000
                assert np.allclose(box,expected,rtol=0,atol=1e-10)
                assert record['area']==int(mask.sum())
                key=(-record['predicted_iou'],-record['stability_score'],record['index'])
                assert last is None or last<=key
                last=key; mask_count+=1
    input_count=0
    for name,p in json.loads((out/'native-payloads.json').read_text()).items():
        images=[part for msg in p['messages'] if isinstance(msg['content'],list)
                for part in msg['content'] if part.get('type')=='image_url']
        assert len(images)==1
        decoded=Image.open(io.BytesIO(base64.b64decode(images[0]['image_url']['url'].split(',',1)[1]))).convert('RGB')
        actual=Image.open(out/(name+'.png')).convert('RGB')
        assert decoded.size==actual.size and decoded.tobytes()==actual.tobytes()
        input_count+=1
    checks=0
    for r in json.loads((out/'scored.json').read_text()):
        c=cases[r['case']]
        if r['arm']=='program':
            boxes=[[b[0]/64*1000,b[1]/64*1000,(b[2]+1)/64*1000,(b[3]+1)/64*1000]
                   for o in c['program']['instances'] for b in [o['bbox']]]
        else: boxes=lookup[(r['case'],r['arm'])]['boxes']
        for key,padding,area_cap in [('main',1,9),('raw',0,9),('cap4',1,4)]:
            actual=independent_count(boxes[:r['cap']],c['gold'],padding,area_cap)
            assert actual==r[key]['tp'], (r['case'],r['arm'],r['cap'],key,actual,r[key]['tp'])
            checks+=1
    result=dict(status='passed',mask_boxes_verified=mask_count,input_images_verified=input_count,
                independent_score_checks=checks,source_and_checkpoint_hashes='passed')
    save(out/'independent-audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': run(Path(sys.argv[1] if len(sys.argv)>1 else 'outputs/sam-proposals-20260928'))
