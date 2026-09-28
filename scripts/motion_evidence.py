"""Role-free exact shape correspondences for small color-ID boards.

Correspondences are hypotheses, not tracked object identities. Multiple exact
matches remain ambiguous. Large components are retained as measurements but are
not translation candidates. No game names, actions, or semantic labels are used.
"""
from collections import Counter, deque


def bounds(points):
    return [min(x for x,y in points),min(y for x,y in points),
            max(x for x,y in points),max(y for x,y in points)]


def components(grid):
    h=len(grid);w=len(grid[0]);seen=set();out=[]
    if any(len(row)!=w for row in grid):raise ValueError('ragged grid')
    for y in range(h):
        for x in range(w):
            if (x,y) in seen:continue
            color=grid[y][x];queue=deque([(x,y)]);seen.add((x,y));points=set()
            while queue:
                px,py=queue.popleft();points.add((px,py))
                for nx,ny in [(px-1,py),(px+1,py),(px,py-1),(px,py+1)]:
                    if 0<=nx<w and 0<=ny<h and (nx,ny) not in seen and grid[ny][nx]==color:
                        seen.add((nx,ny));queue.append((nx,ny))
            box=bounds(points)
            out.append({'color':color,'pixels':points,'box':box,'area':len(points),
                        'shape':frozenset((px-box[0],py-box[1]) for px,py in points)})
    return out


def adjacent(a,b):
    return any((x+dx,y+dy) in b for x,y in a for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)])


def analyze(before,after):
    if not before or not after or len(before)!=len(after) or len(before[0])!=len(after[0]):
        raise ValueError('incompatible grids')
    h=len(before);w=len(before[0]);old=components(before);new=components(after)
    changed=[(x,y) for y in range(h) for x in range(w) if before[y][x]!=after[y][x]]
    records=[];moving={};limit=max(16,w*h//16)
    for i,a in enumerate(old):
        row={'region':f'b{i}','color_id':a['color'],'before_box':a['box'],'before_pixels':a['area']}
        if all(after[y][x]==a['color'] for x,y in a['pixels']):
            # This measures the old support, not persistence of the entire component.
            row.update(old_support_unchanged=True)
        else:row.update(old_support_unchanged=False)
        matches=[b for b in new if a['area']<=limit and b['color']==a['color'] and b['shape']==a['shape']]
        if matches:
            row['exact_shape_matches']=[{'after_box':b['box'],'dx':b['box'][0]-a['box'][0],
                                        'dy':b['box'][1]-a['box'][1]} for b in matches]
            row['correspondence']='unique_shape_candidate' if len(matches)==1 else 'ambiguous_shape_candidates'
            if len(matches)==1 and not row['old_support_unchanged']:
                m=row['exact_shape_matches'][0]
                if (m['dx'],m['dy'])!=(0,0):moving[i]=(m['dx'],m['dy'])
        else:
            row['correspondence']='no_exact_small_component_match'
            overlapping=[b for b in new if b['color']==a['color'] and a['pixels']&b['pixels']]
            if overlapping:
                row['same_color_overlaps']=[{'after_box':b['box'],'after_pixels':b['area'],
                    'pixel_count_delta':b['area']-a['area']} for b in overlapping]
        records.append(row)
    groups=[];remaining=set(moving)
    while remaining:
        i=min(remaining);remaining.remove(i);members=[i];queue=[i]
        while queue:
            j=queue.pop()
            for k in sorted(remaining):
                if moving[j]==moving[k] and adjacent(old[j]['pixels'],old[k]['pixels']):
                    remaining.remove(k);members.append(k);queue.append(k)
        if len(members)>1:
            pts=set().union(*(old[j]['pixels'] for j in members));dx,dy=moving[i];box=bounds(pts)
            groups.append({'members':[f'b{j}' for j in sorted(members)],
                'colors':sorted({old[j]['color'] for j in members}),'before_box':box,
                'after_box':[box[0]+dx,box[1]+dy,box[2]+dx,box[3]+dy],
                'dx':dx,'dy':dy,'status':'adjacent_parts_share_displacement_not_proof_of_one_object'})
    transitions=Counter((before[y][x],after[y][x]) for x,y in changed)
    return {'width':w,'height':h,'changed_pixels':len(changed),
            'color_replacements':[{'before':a,'after':b,'pixels':n} for (a,b),n in sorted(transitions.items())],
            'regions':records,'common_motion_candidates':groups,
            'interpretation_limit':'Exact shape matches are correspondence candidates, not object identity or causal proof. Static support does not prove the component did not grow. No elapsed-time velocity is inferred.'}
