"""Compare free description, joint binary judgments, and one-object one-token gates."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sys
import time

from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_attention_selection import http, now
from scripts.benchmark_parallel import digest, read_lines
from scripts.benchmark_recognition import encode, save_json
from scripts.benchmark_action_cue import render_board

SYSTEM='Use only supplied images as evidence. Object descriptions are BEFORE-only hypotheses, not game rules. No images means visual evidence is unavailable. Names do not prove physical identity.'
CHANGE='''A visible change includes position, shape, color, size, connectivity,
appearance or disappearance. Changes need not follow any action direction.
For a target that is a large region or group, a change to its contents counts.
For a specific object, ignore unrelated changes elsewhere. Judge visible change,
not its cause or game role. Do not require precise coordinates or distances.'''
OUTSIDE='X: Has any object changed or appeared that is NOT individually listed? A generic whole-board or large-region entry does not count as an individual listing.'


def payload(content, prompt, rep, tokens, grammar=None):
    p={'model':'qwen3-vl-4b-instruct','messages':[{'role':'system','content':SYSTEM},
        {'role':'user','content':content+[{'type':'text','text':prompt}]}],
        'temperature':0,'top_p':1,'top_k':0,'min_p':0,'repeat_penalty':1,
        'presence_penalty':0,'seed':3407+rep,'max_tokens':tokens,'cache_prompt':False,'id_slot':0,'stream':False}
    if grammar:p['grammar']=grammar
    return p


def load_cases():
    old=Path('outputs/object-labels-20260927');cs=[c for c in json.loads((old/'cases.json').read_text()) if c['name']!='synthetic_ambiguous_exchange']
    c=deepcopy(next(c for c in cs if c['name']=='synthetic_static'))
    c['palette']['14']='#4FCC30'
    c['name']='synthetic_new';c['truth']={'category':'appearance','targets':['new green triangular shape below and left of center']}
    for dy in range(5):
        for dx in range(-dy,dy+1):c['after'][38+dy][20+dx]=14
    c['provenance']='Synthetic appearance control: new green triangle added; all prior foreground objects unchanged.'
    cs.append(c);return cs


def catalog(c,rep,inventory):
    items=[{'id':chr(65+i),'description':o['description']} for i,o in enumerate(inventory)]
    change=c['truth']['category']!='static';truth={o['id']:'N' for o in items};outside=False
    if c['kind']=='synthetic':
        assert len(items)==5
        truth['A']='Y' if change else 'N'
        if c['name'] in ('synthetic_cross','synthetic_opposite'):truth['B']='Y'
        if c['name'] in ('synthetic_translate','synthetic_opposite','synthetic_reshape','synthetic_rotate','synthetic_split'):truth['D']='Y'
        outside=c['name']=='synthetic_new'
        leaves=['B','C','D','E'];movers= {'synthetic_translate':['D'],'synthetic_cross':['B'],'synthetic_opposite':['B','D']}.get(c['name'],[])
    elif c['name']=='ls20_up':
        assert len(items)==8
        for k in ['A','E',('D','C','F')[rep]]:truth[k]='Y'
        leaves=['B','C','D','E','F','G'] if rep<2 else ['B','C','D','E']
        movers=[('D','C',None)[rep]];outside=rep==2
    else:
        leaves=['D','E','G'] if rep==0 else ['C'] if rep==1 else ['C','D','F']
        movers=[]
    truth['X']='Y' if outside else 'N'
    return {'items':items,'truth':truth,'leaf_ids':leaves,'mover_ids':movers,'outside_truth':outside,
        'note':'Truth is scorer-only. Foreground leaves exclude broad regions, duplicate region entries and console. A missing mover is retained as null for end-to-end coverage.'}


def inventory_text(cat):return '\n'.join(o['id']+': '+o['description'] for o in cat['items'])


def prepare(out):
    assert not (out/'jobs.json').exists()
    cs=load_cases();source=Path('outputs/object-labels-20260927')
    inventories=json.loads((source/'inventories.json').read_text());by={(r['scene'],r['repeat']):r['objects'] for r in inventories}
    template=Image.open('outputs/action-cue-20260927/ui-template.png').convert('RGB');jobs=[];catalogs=[]
    gallery=['<!doctype html><meta charset="utf-8"><title>Object gates</title><style>body{font:16px system-ui;margin:24px}img{width:40%;max-width:640px;image-rendering:pixelated}</style>']
    def add(c,rep,arm,target,p,cat):
        jobs.append({'id':len(jobs),'case':c['name'],'repeat':rep,'arm':arm,'target':target,'catalog_key':cat,'payload':p,'payload_digest':digest(p)})
    for c in cs:
        media=[];gallery.append('<h2>'+c['name']+'</h2>')
        for fr in ['before','after']:
            im=render_board(c,fr,template);im.save(out/f'{c["name"]}-{fr}.png')
            gallery.append(f'<img src="{c["name"]}-{fr}.png">');media.extend([{'type':'text','text':fr.upper()},encode(im)])
        for rep in range(3):
            cat=catalog(c,rep,by[c['scene'],rep]);key=f'{c["name"]}:{rep}';cat.update(key=key,case=c['name'],repeat=rep);catalogs.append(cat)
            listing=inventory_text(cat);common=CHANGE+'\nBEFORE-only inventory:\n'+listing+'\n'+OUTSIDE
            free=common+'\nCompare BEFORE and AFTER. For every listed entry, state whether and how it changed, referencing its ID. Also report X if applicable. Correct a mistaken inventory if necessary. If unchanged say so. At most 180 words.'
            add(c,rep,'free',None,payload(media,free,rep,512),key)
            n=len(cat['items'])+1
            joint=common+'\nReturn only '+str(n)+' letters, one Y (changed) or N (unchanged) per entry, in the listed order, followed by the answer for X. No spaces or explanations.'
            add(c,rep,'joint',None,payload(media,joint,rep,32,'root ::= '+ ' '.join(['[YN]']*n)),key)
            for item in cat['items']:
                q=CHANGE+'\nTARGET '+item['id']+': '+item['description']+'\nCompare this target in BEFORE and AFTER. Has it visibly changed? Return only Y for yes or N for no.'
                add(c,rep,'individual',item['id'],payload(media,q,rep,1,'root ::= [YN]'),key)
            add(c,rep,'individual','X',payload(media,common+'\nAnswer only question X with Y for yes or N for no.',rep,1,'root ::= [YN]'),key)
    # Same one-target task without visual evidence. Forced binary answers are diagnostic,
    # never treated as evidence that the model can appropriately abstain.
    example=catalogs[0]
    for rep in range(3):
        q=CHANGE+'\nTARGET D: small orange and blue stacked block. Has this object changed? Return only Y or N.'
        add({'name':'no_images'},rep,'control','D',payload([],q,rep,1,'root ::= [YN]'),None)
    save_json(out/'cases.json',cs);save_json(out/'catalogs.json',catalogs);save_json(out/'jobs.json',jobs)
    save_json(out/'inventories.json',inventories);(out/'gallery.html').write_text('\n'.join(gallery))
    files=[Path(__file__),Path('scripts/benchmark_attention_selection.py'),Path('scripts/benchmark_parallel.py'),Path('scripts/benchmark_recognition.py'),Path('scripts/benchmark_action_cue.py')]
    (out/'sources').mkdir(exist_ok=True)
    hashes={}
    for f in files:hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest();shutil.copyfile(f,out/'sources'/f.name)
    inputs={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [source/'cases.json',source/'inventories.json',Path('outputs/action-cue-20260927/ui-template.png')]}
    save_json(out/'input-manifest.json',inputs)
    save_json(out/'plan.json',{'created_at':now(),'cases_digest':digest(cs),'catalog_digest':digest(catalogs),'jobs_digest':digest(jobs),'sources':hashes,
        'requests':len(jobs),'arms':['free','joint','individual'],'repeats':3,
        'primary':'Changed listed foreground-object sensitivity and false-positive rate. Broad region positives reported separately, never counted as identifying a moving leaf. Missing mover inventory coverage recorded separately.',
        'gate':'Y-only gate: joint and individual each pass selected IDs plus X to one explanatory call per case/rep; if all N, no call and pipeline output none. Invalid gate marks uncertainty and all entries passed as fallback, recorded separately.',
        'explanation_rubric':'Correct main changed target(s) and change class without explicit contradictions or invented changes; no numeric coordinates or exact direction. ls20 needs orange-blue mover, not only strip. Opposite needs both objects. New case needs new green shape. Static requires none.',
        'catalog':'Reuse frozen model-generated BEFORE inventories across all arms. No corrections to inventories. Scorer truth and leaf/mover maps are not in requests.',
        'generation':'Same Qwen3-VL4B Q4_K_M; greedy temperature0 top_p1 top_k0 min_p0 repeat_penalty1 presence_penalty0; different output budgets only. No native reasoning_effort switch supported. Seeds vary but greedy replicas are not independent stochastic trials.',
        'single_token':'Y/N tokenizer verified one token each; exact Y/N with one completion token is valid even finish_reason=length. Joint grammar constrains only format, not label choices.',
        'budgets':'Individual1token, joint32tokens fixed-letter grammar, free and explanation512tokens; each request20seconds; cache_prompt false, serial one slot.',
        'cost':'Sum per-call seconds for each logical pipeline, excludes pre-existing static inventory cost; not equal input compute.',
        'scope':'2 real ls20 transitions, 8 synthetic pairs incl appearance; full identical two images, no crops or direction grouping. No game actions. YNU not tested in this pilot.',
        'warmups':'One per primary arm excluded. Two format-only preflight calls separately saved.',
        'review':'Free and gated explanations arm-hidden shuffled semantic review by implementing assistant; not independent human judging.'})
    print('Prepared',len(jobs),'requests',flush=True)


def execute(out,base,jobs,filename,warm=()):
    path=out/filename
    assert not path.exists()
    js=list(jobs);random.Random(28149).shuffle(js)
    with path.open('w') as f:
        for i,(j,iswarm) in enumerate([(j,True) for j in warm]+[(j,False) for j in js]):
            r={k:v for k,v in j.items() if k!='payload'};r.update(warmup=iswarm,started_at=now());start=time.monotonic()
            try:
                result=http(base+'/chat/completions',j['payload'],20);r['response']=result
                choice=result['choices'][0];answer=choice['message']['content'].strip();r['answer']=answer
                if j['arm'] in ('individual','control'):
                    valid=answer in ('Y','N') and result['usage']['completion_tokens']==1
                elif j['arm']=='joint':
                    expected=j['payload']['grammar'].count('[YN]');valid=bool(re.fullmatch('[YN]{'+str(expected)+'}',answer))
                else:valid=bool(answer) and choice['finish_reason']=='stop'
                r['valid']=valid
            except Exception as e:r.update(valid=False,error=f'{type(e).__name__}: {e}')
            r['seconds']=time.monotonic()-start;r['within_budget']=r['seconds']<=20
            f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
            print(i,j['case'],j['arm'],j.get('target'),r['valid'],round(r['seconds'],2),flush=True)


def run(out,base):
    plan=json.loads((out/'plan.json').read_text());jobs=json.loads((out/'jobs.json').read_text())
    assert digest(jobs)==plan['jobs_digest']
    for path,h in plan['sources'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    save_json(out/'server-before.json',http(base.removesuffix('/v1')+'/props'))
    warm=[next(j for j in jobs if j['arm']==arm) for arm in ['free','joint','individual']]
    execute(out,base,jobs,'measurements.jsonl',warm)
    rows=[r for r in read_lines(out/'measurements.jsonl') if not r['warmup']]
    cats=json.loads((out/'catalogs.json').read_text());gates=[];explain=[]
    for cat in cats:
        for arm in ['joint','individual']:
            rs=[r for r in rows if r['catalog_key']==cat['key'] and r['arm']==arm]
            ok=all(r['valid'] and r['within_budget'] for r in rs)
            if arm=='joint':answers=dict(zip([o['id'] for o in cat['items']]+['X'],rs[0].get('answer',''))) if ok else {}
            else:answers={r['target']:r.get('answer') for r in rs} if ok else {}
            selected=[o for o in cat['items'] if not ok or answers.get(o['id'])=='Y'];outside=not ok or answers.get('X')=='Y'
            gate={'case':cat['case'],'repeat':cat['repeat'],'arm':arm,'catalog_key':cat['key'],'answers':answers,'valid':ok,
                'selected':[o['id'] for o in selected],'outside':outside,'seconds':sum(r['seconds'] for r in rs),'explanation_id':None}
            if selected or outside:
                source=next(j for j in jobs if j['catalog_key']==cat['key'] and j['arm']=='free')
                content=deepcopy(source['payload']['messages'][1]['content'][:4])
                prompt=CHANGE+'\nA preliminary classifier flagged these entries, but may be wrong:\n'+ '\n'.join(o['id']+': '+o['description'] for o in selected)
                if outside:prompt+='\nIt also flagged X: possible new or unlisted changed objects.'
                prompt+='\nVerify the flags against BEFORE and AFTER. For the selected entries and any flagged X, describe whether and how they changed, with current appearance. Correct false flags, use none if unchanged, and uncertain if correspondence cannot be established. No game roles. At most 180 words.'
                p=payload(content,prompt,cat['repeat'],512);jid=len(explain);gate['explanation_id']=jid
                explain.append({'id':jid,'case':cat['case'],'repeat':cat['repeat'],'arm':arm+'_explain','catalog_key':cat['key'],'target':None,'payload':p,'payload_digest':digest(p)})
            gates.append(gate)
    save_json(out/'gates.json',gates);save_json(out/'explanation-jobs.json',explain);save_json(out/'explanation-digest.json',{'digest':digest(explain),'created_at':now()})
    execute(out,base,explain,'explanations.jsonl')
    save_json(out/'server-after.json',http(base.removesuffix('/v1')+'/props'))
    candidates=[dict(r,review_key='free:'+str(r['id'])) for r in rows if r['arm']=='free']
    candidates += [dict(r,review_key='explain:'+str(r['id'])) for r in read_lines(out/'explanations.jsonl')]
    random.Random(41893).shuffle(candidates)
    save_json(out/'review-key.json',{str(i):r['review_key'] for i,r in enumerate(candidates)})
    save_json(out/'review-candidates.json',[{'review_id':i,'case':r['case'],'answer':r.get('answer',''),'valid':r['valid'] and r['within_budget']} for i,r in enumerate(candidates)])
    print('Completed; explanation requests',len(explain),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['prepare','run']);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--base',default='http://127.0.0.1:8080/v1');a=ap.parse_args()
    if a.command=='prepare':prepare(a.output)
    else:run(a.output,a.base)
