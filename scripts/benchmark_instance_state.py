"""Concept/instance/state ablation on fixed pixels; local Qwen only."""
import argparse,json,random,shutil,sys,time
from copy import deepcopy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageDraw
from scripts.instance_state import extract,compare
from scripts.benchmark_recognition import save_json,encode
from scripts.benchmark_parallel import digest,read_lines
from scripts.benchmark_official_relations import sha
from scripts.benchmark_attention_selection import http,now

SOURCE=Path('outputs/relation-overlay-20260928-final/cases.json')
ARMS=['direct_pair','independent_text','measured_text','measured_images']
PALETTE='Color IDs: 0 white, 1 light gray, 2 gray, 3 dark gray, 4 charcoal, 5 black, 6 magenta, 7 pink, 8 red, 9 blue, 10 cyan, 11 yellow, 12 orange, 13 maroon, 14 green, 15 purple.'
OBS='''Observe ONLY this single game frame. No previous observation is supplied. Separate reusable structural classes from individual instances and their current measured state. Different copies of a repeated structure share a class but remain separate instances. Roles are unknown. Do not invent player, goal, hint or controllability.
List distinct small visible objects, multicolor assemblies and repeated structured panels. For a regular grid panel, record it as one instance and give its cell-color matrix in row-major order. Exclude the uniform background. Do not merge objects merely because they share color or class. Use original 0..63 pixel coordinates from the axes, inclusive bbox [left,top,right,bottom]; use null if uncertain. Keep color arrangement and geometry, not just a coarse location phrase. A separate later process will compare these records.
Return only JSON with this structure: {"classes":["structural name"],"instances":[{"id":"local1","class_name":"structural name","appearance":"short visible description","bbox":null,"pattern":null,"role":"unknown"}]}. Include all discernible instances; IDs are local to this frame. Pattern is a color-ID matrix or a concise shape/color arrangement. No explanations outside JSON.
'''+PALETTE
JUDGE='''Compare BEFORE and AFTER observations. Classes denote shared structure, instances denote separate copies, and each frame records a new state. Local IDs are NOT persistent identity. Same class does not prove same instance. Match using available shape/color pattern, position and context. Retain ambiguous correspondences as uncertain rather than guessing. Objects may move in opposite directions. Unchanged appearance does not imply unchanged position. Changes of position, shape, size, internal color/pattern and appearance/disappearance are distinct. No game rules or roles are provided.
Records, if supplied, are independent per-frame observations. Visual estimates can be wrong; deterministic pixel measurements have exact coordinates but heuristic grouping. Do not assume the two lists have the same order or match by ID number. When images are absent, reason from the supplied records; if evidence is missing or ambiguous set needs_image=true. With images, check contradictions. Shared classes and matching internal patterns are relevant even when instances occupy different places.
Return one JSON object: {"changes":[],"shared_classes":[],"same_pattern_after":[],"needs_image":false}.
For each actually changed object add a changes row with keys before_id, after_id, object, kind, description. IDs are the supplied local IDs, or null when absent; object is a brief appearance-based reference. kind is moved, appearance_changed, resized, shape_changed, appeared, disappeared, or uncertain. Include no unchanged rows. An empty changes list means no observable changes.
shared_classes rows have class_name, before, after (counts). same_pattern_after lists pairs of separate AFTER instances of repeated structured panels with identical internal cell patterns, using IDs or short unique descriptions if no IDs are given. Do not list unrelated plain rectangles as matching panels. Do not infer roles or action causes. Keep descriptions short; exact motion direction and distance are not required.
'''+PALETTE

def extra_cases(palette):
    p0=[[9,8,9],[8,0,9],[9,9,8]];p1=[[9,8,9],[8,0,8],[9,8,9]];p2=[[8,9,9],[8,0,8],[9,8,9]];p3=deepcopy(p0);p3[2][2]=9
    def panels(patterns):
        g=[[5]*64 for _ in range(64)]
        for (x0,y0),matrix in zip([(8,8),(40,8),(8,40),(40,40)],patterns):
            for y in range(3):
                for x in range(3):
                    for dy in range(4):
                        for dx in range(4):g[y0+y*6+dy][x0+x*6+dx]=matrix[y][x]
        return g
    b=panels([p0,p1,p2,p3]);a=panels([p0,p1,p2,p0])
    result=[dict(name='four_panels_change',title='合成：4つの3×3パネル、右下の1セル変化',before=b,after=a,palette=palette,note='Constructed structural analogue, not the attached game or its rules.'),dict(name='four_panels_static',title='対照：同じ4パネルで無変化',before=b,after=deepcopy(b),palette=palette,note='Same concept and distinct copies, no changes.')]
    frames=[]
    for centers in [[16,40],[24,32]]:
        g=[[3]*64 for _ in range(64)]
        for cx in centers:
            for dx,dy in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:g[24+dy][cx+dx]=0
        frames.append(g)
    result.append(dict(name='ambiguous_copies',title='診断：同形の十字2つ、対応は一意に決まらない',before=frames[0],after=frames[1],palette=palette,note='Two identical instances change occupied positions; individual trajectories are not determined by these frames.'))
    return result

def render(grid,palette):
    im=Image.new('RGB',(592,592),'#18212b');d=ImageDraw.Draw(im);origin=48;scale=8
    for y,row in enumerate(grid):
        for x,color in enumerate(row):d.rectangle([origin+x*scale,origin+y*scale,origin+(x+1)*scale-1,origin+(y+1)*scale-1],fill=palette[str(color)])
    for k in list(range(0,64,8))+[63]:
        d.text((origin+8*k,28),str(k),fill='white');d.text((20,origin+8*k),str(k),fill='white')
    # Thin cell boundaries in the enlarged view, preserving the interior source color.
    for y,row in enumerate(grid):
        for x,color in enumerate(row):
            rgb=tuple(int(palette[str(color)][i:i+2],16) for i in [1,3,5]);line=tuple(int(v*.8) for v in rgb)
            d.line((origin+x*8,origin+y*8,origin+x*8+7,origin+y*8),fill=line);d.line((origin+x*8,origin+y*8,origin+x*8,origin+y*8+7),fill=line)
    return im

def payload(content,limit):
    return dict(model='qwen3-vl-4b-instruct',messages=[dict(role='system',content='Use only supplied observations. Keep measured facts separate from hypotheses. Return the requested JSON without commentary.'),dict(role='user',content=content)],temperature=0,max_tokens=limit,stream=False,cache_prompt=False)

def prepare(out):
    out.mkdir(exist_ok=False);(out/'sources').mkdir();cases=json.loads(SOURCE.read_text());cases+=extra_cases(cases[0]['palette']);jobs=[];states={};program=[]
    for c in cases:
        states[c['name']]={}
        for frame,label in [('before','b'),('after','a')]:
            render(c[frame],c['palette']).save(out/f'{c["name"]}-{frame}.png');states[c['name']][frame]=extract(c[frame],label)
            jobs.append(dict(id=len(jobs),stage='observe',case=c['name'],frame=frame,arm='independent_observation'))
        result=compare(states[c['name']]['before'],states[c['name']]['after']);program.append(dict(case=c['name'],arm='deterministic',result=result))
    for c in cases:
        for arm in ARMS:jobs.append(dict(id=len(jobs),stage='judge',case=c['name'],arm=arm))
    save_json(out/'cases.json',cases);save_json(out/'states.json',states);save_json(out/'program-results.json',program);save_json(out/'jobs.json',jobs)
    save_json(out/'plan.json',dict(created_at=now(),requests=60,warmups=2,cases_digest=digest(cases),jobs_digest=digest(jobs),states_digest=digest(json.loads((out/'states.json').read_text())),source_hashes={p:sha(p) for p in ['scripts/benchmark_instance_state.py','scripts/instance_state.py']},arms=ARMS+['deterministic'],
        design='10 fixed pairs. 20 independent single-frame observations, 40 judgments (direct pair; independent-observation text; exact measured-state text; measured states + same two images). Deterministic correspondence on measured snapshots separately. Single local quantized Qwen backend.',
        controls='Same two 592x592 coordinate-grid images in direct_pair and measured_images. Same comparison instruction in all four arms; only observation source and images differ. No previous state in single-frame extraction. No differences, correspondences, game roles, actions or truth injected into measured states.',
        extractor='Limited single-frame geometric heuristic: small connected components, touching filled rectangular assemblies, regular 3x3 arrays of solid square cells. All candidate IDs frame-local. Exact pixel facts do not prove segmentation or physical identity.',
        limitations='Small diagnostic dataset; panel examples are constructed analogues not the attached game. No equal total compute claim: independent_text uses two extra VLM calls. Original cases reused but rendering, prompts, schema and inventory changed; historical results are not isolated controls. Program comparison is assisted perception, not improvement of raw Qwen vision.'))
    save_json(out/'input-hashes.json',{p.name:sha(p) for p in out.glob('*.png')})
    for p in ['scripts/benchmark_instance_state.py','scripts/instance_state.py']:shutil.copyfile(p,out/'sources'/Path(p).name)
    print('Prepared',len(jobs),'model requests',flush=True)

def request_for(j,out,states,observations):
    content=[]
    if j['stage']=='observe':
        content=[encode(Image.open(out/f'{j["case"]}-{j["frame"]}.png').convert('RGB')),dict(type='text',text=OBS)]
        return payload(content,900)
    if j['arm'] in ['direct_pair','measured_images']:
        for frame in ['before','after']:content.extend([dict(type='text',text=frame.upper()),encode(Image.open(out/f'{j["case"]}-{frame}.png').convert('RGB'))])
    text=JUDGE
    if j['arm']!='direct_pair':
        if j['arm']=='independent_text':
            s={fr:dict(valid=observations[(j['case'],fr)]['valid'],data=observations[(j['case'],fr)].get('parsed',observations[(j['case'],fr)].get('answer'))) for fr in ['before','after']}
            text+='\nIndependent visual estimates (may be inaccurate):\n'+json.dumps(s,ensure_ascii=False,separators=(',',':'))
        else:text+='\nIndependent deterministic pixel measurements (grouping is hypothetical):\n'+json.dumps(states[j['case']],ensure_ascii=False,separators=(',',':'))
    content.append(dict(type='text',text=text));return payload(content,700)

def valid(parsed,stage):
    if stage=='observe':
        assert isinstance(parsed['classes'],list) and isinstance(parsed['instances'],list)
        ids=[]
        for x in parsed['instances']:
            for key in ['id','class_name','appearance','bbox','pattern','role']:assert key in x
            ids.append(x['id'])
            if x['bbox'] is not None:assert len(x['bbox'])==4 and all(isinstance(v,(int,float)) and 0<=v<=63 for v in x['bbox'])
        assert len(ids)==len(set(ids))
    else:
        assert all(isinstance(parsed[k],list) for k in ['changes','shared_classes','same_pattern_after']) and isinstance(parsed['needs_image'],bool)
        for c in parsed['changes']:
            assert all(k in c for k in ['before_id','after_id','object','kind','description'])
            assert c['kind'] in ['moved','appearance_changed','resized','shape_changed','appeared','disappeared','uncertain']

def run(out):
    plan=json.loads((out/'plan.json').read_text());jobs=json.loads((out/'jobs.json').read_text());states=json.loads((out/'states.json').read_text())
    assert digest(jobs)==plan['jobs_digest'] and digest(states)==plan['states_digest']
    for p,h in plan['source_hashes'].items():assert sha(p)==h
    for p,h in json.loads((out/'input-hashes.json').read_text()).items():assert sha(out/p)==h
    assert not (out/'measurements.jsonl').exists();save_json(out/'server-before.json',http('http://127.0.0.1:8080/props'));observations={}
    obs=[j for j in jobs if j['stage']=='observe'];judges=[j for j in jobs if j['stage']=='judge'];random.Random(28919).shuffle(obs);random.Random(28920).shuffle(judges)
    warm=[next(j for j in jobs if j['stage']=='observe'),next(j for j in jobs if j['arm']=='direct_pair')]
    with (out/'measurements.jsonl').open('w') as log,(out/'requests.jsonl').open('w') as req:
        for j,iswarm in [(j,True) for j in warm]+[(j,False) for j in obs+judges]:
            p=request_for(j,out,states,observations);d=digest(p);req.write(json.dumps(dict(id=j['id'],warmup=iswarm,payload_digest=d,payload=p),ensure_ascii=False)+'\n');req.flush()
            r=dict(j,warmup=iswarm,payload_digest=d,started_at=now());start=time.monotonic()
            try:
                response=http('http://127.0.0.1:8080/v1/chat/completions',p,180);choice=response['choices'][0];r.update(response=response,answer=choice['message']['content'],output_tokens=response['usage']['completion_tokens'],prompt_tokens=response['usage']['prompt_tokens'])
                parsed=json.loads(r['answer']);r['parsed']=parsed;valid(parsed,j['stage']);r['valid']=choice['finish_reason']=='stop'
            except Exception as e:r.update(valid=False,error=f'{type(e).__name__}: {e}')
            r['seconds']=time.monotonic()-start;log.write(json.dumps(r,ensure_ascii=False)+'\n');log.flush()
            if not iswarm and j['stage']=='observe':observations[(j['case'],j['frame'])]=r
            print(j['id'],j['case'],j.get('frame',j['arm']),r['valid'],round(r['seconds'],2),r.get('error',''),flush=True)
    save_json(out/'server-after.json',http('http://127.0.0.1:8080/props'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.command=='prepare':prepare(a.output)
    else:run(a.output)
