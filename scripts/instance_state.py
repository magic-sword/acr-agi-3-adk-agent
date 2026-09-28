"""Single-frame geometric candidates, separate from temporal correspondence.

No game IDs, actions, truth regions, old frames, or movement grouping are used.
This deliberately limited 64x64 extractor is an experimental sensor, not a
general object detector. Large/irregular regions remain available separately.
"""
from collections import Counter
from itertools import combinations,product
import hashlib,json
from scripts.motion_evidence import components,bounds,adjacent

def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:12]

def extract(grid,frame):
    parts=components(grid);small=[];large=[]
    for p in parts:
        x0,y0,x1,y1=p['box'];w=x1-x0+1;h=y1-y0+1
        if p['area']<=128 and (max(w,h)<=12 or (p['area']==w*h and max(w,h)>=4*min(w,h))):small.append(p)
        else:large.append(dict(color=p['color'],bbox=p['box'],area=p['area']))
    # Groups are proposals based only on current contact and filled rectangular support.
    proposals=[]
    for size in [2,3]:
        for ids in combinations(range(len(small)),size):
            seen={ids[0]}
            while True:
                more={j for j in ids if any(adjacent(small[i]['pixels'],small[j]['pixels']) for i in seen)}
                if more<=seen:break
                seen|=more
            if len(seen)!=size:continue
            points=set().union(*(small[i]['pixels'] for i in ids));box=bounds(points);area=len(points)
            if area<=128 and area==(box[2]-box[0]+1)*(box[3]-box[1]+1):proposals.append((area,ids,points))
    used=set();objects=[]
    for area,ids,points in sorted(proposals,reverse=True):
        if used.intersection(ids):continue
        used.update(ids);objects.append(points)
    objects.extend(p['pixels'] for i,p in enumerate(small) if i not in used)
    records=[]
    for pts in objects:
        b=bounds(pts);w=b[2]-b[0]+1;h=b[3]-b[1]+1
        pattern=[[grid[y][x] if (x,y) in pts else None for x in range(b[0],b[2]+1)] for y in range(b[1],b[3]+1)]
        full=len(pts)==w*h;colors=dict(sorted(Counter(grid[y][x] for x,y in pts).items()))
        cls='solid_rectangle' if full and len(colors)==1 else 'multicolor_rectangle' if full else 'pixel_shape'
        # Retain a readable compact description plus exact signature; thin bars need not expand.
        detail=pattern if w*h<=100 else {'row_runs':[[[color,len(list(group))] for color,group in __import__('itertools').groupby(row)] for row in pattern]}
        records.append(dict(class_name=cls,bbox=b,size=[w,h],area=len(pts),colors=colors,pattern=detail,
                            appearance_signature=signature(pattern),shape_signature=signature([[v is not None for v in row] for row in pattern])))
    # Discover repeated 3x3 regular arrays of same-sized solid square cells in one frame.
    squares={}
    for i,r in enumerate(records):
        if r['class_name']=='solid_rectangle' and r['size'][0]==r['size'][1] and r['size'][0]>=2:squares.setdefault(r['size'][0],{})[(r['bbox'][0],r['bbox'][1])]=i
    panel_candidates=[]
    for size,lookup in squares.items():
        xs=sorted({x for x,y in lookup});ys=sorted({y for x,y in lookup})
        xt=[t for t in combinations(xs,3) if t[1]-t[0]==t[2]-t[1] and size<t[1]-t[0]<=3*size]
        yt=[t for t in combinations(ys,3) if t[1]-t[0]==t[2]-t[1] and size<t[1]-t[0]<=3*size]
        for xx,yy in product(xt,yt):
            if not all((x,y) in lookup for x,y in product(xx,yy)):continue
            ids=[lookup[(x,y)] for y in yy for x in xx];matrix=[[next(iter(records[lookup[(x,y)]]['colors'])) for x in xx] for y in yy]
            box=[xx[0],yy[0],xx[-1]+size-1,yy[-1]+size-1]
            panel_candidates.append((ids,dict(class_name='regular_grid_3x3',bbox=box,size=[box[2]-box[0]+1,box[3]-box[1]+1],area=9*size*size,cell_size=size,cell_spacing=[xx[1]-xx[0],yy[1]-yy[0]],pattern=matrix,colors=dict(Counter(v for row in matrix for v in row)),appearance_signature=signature(matrix),shape_signature=signature(['3x3',size,xx[1]-xx[0],yy[1]-yy[0]]))))
    cell_used=set();panels=[]
    for ids,panel in panel_candidates:
        if cell_used.intersection(ids):continue
        cell_used.update(ids);panel['children']=[dict(bbox=records[i]['bbox'],color=next(iter(records[i]['colors']))) for i in ids];panels.append(panel)
    records=[r for i,r in enumerate(records) if i not in cell_used]+panels
    records.sort(key=lambda r:(r['bbox'][1],r['bbox'][0],r['class_name']))
    for i,r in enumerate(records):r['id']=frame+str(i+1);r['role']='unknown'
    return dict(frame=frame,coordinate_convention='original pixel x,y; inclusive bbox endpoints',classes=sorted({r['class_name'] for r in records}),instances=records,large_regions=large,
                limitation='Geometric grouping hypotheses; local IDs are not temporal identities. Missing small candidates or changes in large regions require image review.')

def compare(before,after):
    """Conservative correspondences. All same-pattern alternatives stay visible."""
    old=before['instances'];new=after['instances'];events=[];paired_old=set();paired_new=set();matches=[]
    # Equal visible state at equal position is observation equality, not physical identity proof.
    for a in old:
        bs=[b for b in new if b['bbox']==a['bbox'] and b['appearance_signature']==a['appearance_signature'] and b['class_name']==a['class_name']]
        if len(bs)==1:paired_old.add(a['id']);paired_new.add(bs[0]['id']);matches.append([a['id'],bs[0]['id'],'same_observed_state'])
    for a in old:
        if a['id'] in paired_old:continue
        bs=[b for b in new if b['class_name']==a['class_name'] and b['appearance_signature']==a['appearance_signature']]
        reverse=lambda b:[x for x in old if x['class_name']==b['class_name'] and x['appearance_signature']==b['appearance_signature']]
        if len(bs)==1 and len(reverse(bs[0]))==1 and bs[0]['id'] not in paired_new:
            b=bs[0];kind='moved';reason='unique unchanged local appearance; position differs'
        else:
            at=[b for b in new if b['id'] not in paired_new and b['bbox']==a['bbox'] and b['shape_signature']==a['shape_signature'] and b['class_name']==a['class_name']]
            if len(at)==1:b=at[0];kind='appearance_changed';reason='same support and position; internal appearance differs'
            elif a['class_name']=='solid_rectangle' and len([x for x in old if x['class_name']=='solid_rectangle' and set(x['colors'])==set(a['colors'])])==1 and len(resized:=[x for x in new if x['class_name']=='solid_rectangle' and set(x['colors'])==set(a['colors']) and x['id'] not in paired_new])==1 and a['bbox'][0]<=resized[0]['bbox'][2] and resized[0]['bbox'][0]<=a['bbox'][2] and a['bbox'][1]<=resized[0]['bbox'][3] and resized[0]['bbox'][1]<=a['bbox'][3]:
                b=resized[0];kind='resized';reason='unique overlapping same-color solid rectangle; extent differs'
            elif bs:
                events.append(dict(before_id=a['id'],after_id=None,kind='uncertain',object=a['class_name'],candidates=[b['id'] for b in bs],description='Multiple/competing appearance correspondences; do not force identity'));continue
            else:
                events.append(dict(before_id=a['id'],after_id=None,kind='uncertain',object=a['class_name'],description='No confident correspondence: disappearance, shape change or grouping change'));continue
        paired_old.add(a['id']);paired_new.add(b['id']);matches.append([a['id'],b['id'],reason]);events.append(dict(before_id=a['id'],after_id=b['id'],kind=kind,object=a['class_name'],description=reason))
    for b in new:
        if b['id'] in paired_new:continue
        if not any(a['class_name']==b['class_name'] and a['appearance_signature']==b['appearance_signature'] for a in old):
            events.append(dict(before_id=None,after_id=b['id'],kind='appeared',object=b['class_name'],description='Unmatched new candidate; could also reflect changed grouping'))
    equal=[]
    for a,b in combinations(new,2):
        if a['class_name']=='regular_grid_3x3' and a['class_name']==b['class_name'] and a['appearance_signature']==b['appearance_signature']:equal.append([a['id'],b['id']])
    return dict(changes=events,same_pattern_after=equal,shared_classes=[dict(class_name=k,before=sum(r['class_name']==k for r in old),after=sum(r['class_name']==k for r in new)) for k in sorted(set(before['classes'])&set(after['classes']))],needs_image=any(e['kind']=='uncertain' for e in events),correspondences=matches)
