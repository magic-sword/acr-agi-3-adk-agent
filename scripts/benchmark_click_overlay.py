"""Compare post-page selection with fixed first-round winners, text vs in-context outlines."""
import json
import random
import time
from pathlib import Path
from PIL import Image, ImageDraw
from scripts import benchmark_click_choices as b
from scripts.benchmark_click_pages import pages

OUT=Path('outputs/click-overlay-20260930')
SOURCE=Path('outputs/click-pages-20260929')


def contextual_image(grid,mapping):
    # Separate full-frame panels preserve absolute location and avoid overlapping masks.
    scale=4;size=280
    sheet=Image.new('RGB',(size*4,size*((len(mapping)+3)//4)),'#18202b')
    for i,(label,obj) in enumerate(mapping.items()):
        panel=Image.new('RGB',(size,size),'#18202b')
        board=b.frame_image(grid).resize((256,256),Image.Resampling.NEAREST)
        panel.paste(board,(12,20));d=ImageDraw.Draw(panel)
        d.text((12,4),'OPTION '+label+' - full board',fill='white')
        support=b.decode(obj['mask_runs'])
        for x,y in support:
            xx,yy=12+x*scale,20+y*scale
            for dx,dy,edge in ((-1,0,(xx,yy,xx,yy+3)),(1,0,(xx+3,yy,xx+3,yy+3)),
                              (0,-1,(xx,yy,xx+3,yy)),(0,1,(xx,yy+3,xx+3,yy+3))):
                if (x+dx,y+dy) not in support:d.line(edge,fill='#00ffff',width=1)
        sheet.paste(panel,((i%4)*size,(i//4)*size))
    return sheet


def request(grid,query,options,order,arm):
    body,mapping=b.request_for(grid,query,options,'choice_text',order)
    if arm=='overlay':
        body['messages'][1]['content'][1:1]=[
            dict(type='text',text='Each numbered panel repeats the SAME full board. Cyan outlines mark ONLY that option mask; they are host annotations, not game objects. Compare its position and extent to the requested part. The clean board above has no annotations.'),
            b.image_part(contextual_image(grid,mapping))]
    return body,mapping


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    cases=json.loads((SOURCE/'cases.json').read_text());old=json.loads((SOURCE/'scored.json').read_text())
    jobs=[]
    for c in cases:
        for order in range(2):
            prior=next(r for r in old if r['case']==c['name'] and r['order']==order and r['arm']=='pages_text')
            jobs.append((c,order,prior['rounds'][0]['winners']))
    random.Random(b.SEED).shuffle(jobs)
    b.save(OUT/'cases.json',cases);b.save(OUT/'jobs.json',jobs)
    sources={}
    for path in (Path(__file__).resolve(),Path(b.__file__).resolve(),Path('scripts/benchmark_click_pages.py')):
        (OUT/path.name).write_bytes(path.read_bytes());sources[path.name]=b.sha(path)
    b.save(OUT/'plan.json',dict(cases_hash=b.digest(cases),jobs_hash=b.digest(jobs),sources=sources,
        source_scores_hash=b.sha(SOURCE/'scored.json'),arms=['text','overlay'],primary_order=0,
        protocol='Fixed first-round winners from prior pages_text. Repeat remaining tournament with text vs full-board outlined panels. '
        'No gold retrieval. Same candidate order and text. No further calls if <=1 fixed winner. '
        'Later-round candidate pools may diverge by selection. Record first comparison paired results. No gameplay.',
        scoring='Recall and purity >=0.8. Actual annotated pixel hit separately. Sequential elapsed seconds for post-page stage only.'))
    base='http://vlm:8080';b.save(OUT/'server-before.json',b.http(base,'/props'))
    rows=[];start=time.monotonic()
    with (OUT/'requests.jsonl').open('w') as rq,(OUT/'responses.jsonl').open('w') as rp:
        for index,(c,order,ids) in enumerate(jobs):
            objects={o['id']:o for o in c['candidates']};arms=['text','overlay'];random.Random(b.SEED+index).shuffle(arms)
            for arm in arms:
                current=[objects[i] for i in ids];round_id=1;calls=0;valid=True;rounds=[];begin=time.monotonic()
                while len(current)>1:
                    winners=[]
                    for page_id,options in enumerate(pages(current,order+round_id)):
                        body,mapping=request(c['grid'],c['query'],options,order,arm)
                        ident=f'{c["name"]}/{order}/{arm}/{round_id}/{page_id}'
                        rq.write(json.dumps(dict(id=ident,payload=body,digest=b.digest(body)))+'\n');rq.flush()
                        row=dict(id=ident,mapping={k:o['id'] for k,o in mapping.items()},digest=b.digest(body));t=time.monotonic()
                        try:
                            response=b.http(base,'/v1/chat/completions',body);answer=response['choices'][0]['message']['content']
                            row.update(response=response,answer=answer);assert answer=='X' or answer in mapping
                            if answer!='X':winners.append(mapping[answer])
                        except Exception as exc:valid=False;row['error']=str(exc)
                        row['seconds']=time.monotonic()-t;rp.write(json.dumps(row)+'\n');rp.flush();calls+=1
                        if arm=='overlay' and order==0 and c['name'].startswith('ft09'):
                            contextual_image(c['grid'],mapping).save(OUT/f'{c["name"]}-{round_id}-{page_id}.png')
                    rounds.append(dict(offered=[o['id'] for o in current],winners=[o['id'] for o in winners]))
                    assert len(winners)<len(current);current=winners;round_id+=1
                gold=b.decode(c['gold_runs']);support=set().union(*(b.decode(o['mask_runs']) for o in current))
                rows.append(dict(case=c['name'],order=order,arm=arm,origin=c['origin'],positive=bool(gold),valid=valid,
                    eligible=len(ids)>1,calls=calls,seconds=time.monotonic()-begin,rounds=rounds,selected=[o['id'] for o in current],
                    available=any(b.region_score(b.decode(objects[i]['mask_runs']),gold)['correct'] for i in ids),
                    region=valid and b.region_score(support,gold)['correct'],centroid=valid and b.point_for(support,'centroid') in gold,
                    interior=valid and b.point_for(support,'interior') in gold,abstention=valid and not gold and not current,
                    false_click=valid and not gold and bool(current)))
            b.save(OUT/'scored.json',rows)
            if (index+1)%4==0:print(index+1,'/52',round(time.monotonic()-start,1),flush=True)
    b.save(OUT/'server-after.json',b.http(base,'/props'))
    assert json.loads((OUT/'server-before.json').read_text())==json.loads((OUT/'server-after.json').read_text())
    summarize()


def summarize():
    import statistics
    rows=json.loads((OUT/'scored.json').read_text());assert len(rows)==104
    result={}
    for arm in ('text','overlay'):
        result[arm]={}
        for name,predicate in [('primary',lambda r:r['order']==0),('real',lambda r:r['order']==0 and r['origin']=='real'),
                               ('all_orders',lambda r:True),('eligible',lambda r:r['order']==0 and r['eligible'])]:
            rs=[r for r in rows if r['arm']==arm and predicate(r)];pos=[r for r in rs if r['positive']]
            result[arm][name]=dict(cases=len(rs),positive=len(pos),negative=len(rs)-len(pos),
                **{k:sum(r[k] for r in rs) for k in ('region','centroid','interior','abstention','false_click','calls')},
                available=sum(r['available'] for r in pos),median_stage_seconds=statistics.median(r['seconds'] for r in rs))
    b.save(OUT/'summary.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':run()
