"""Obstacle-aware SVG layout; runtime declarations supply nodes and all edges."""
from functools import lru_cache
from heapq import heappop, heappush
from html import escape
import json

WIDTH, HEIGHT, STEP = 230, 72, 10
COLORS={'normal':'#64748b','skill':'#0f766e','recovery':'#c56816','stop':'#b91c1c'}


def ports(node):
    _,_,x,y=node
    return [(round((x-20)/STEP),round((y+HEIGHT/2)/STEP),x,y+HEIGHT/2),
            (round((x+WIDTH+20)/STEP),round((y+HEIGHT/2)/STEP),x+WIDTH,y+HEIGHT/2),
            (round((x+WIDTH/2)/STEP),round((y-20)/STEP),x+WIDTH/2,y),
            (round((x+WIDTH/2)/STEP),round((y+HEIGHT+20)/STEP),x+WIDTH/2,y+HEIGHT)]


@lru_cache(maxsize=8)
def routes(serialized):
    data=json.loads(serialized);states=data['states'];occupied={};blocked=set()
    for _,_,x,y in states.values():
        for a in range((x-10)//STEP,(x+WIDTH+10)//STEP+1):
            for b in range((y-10)//STEP,(y+HEIGHT+10)//STEP+1):blocked.add((a,b))
    def search(start,end):
        sx,sy=start[:2];ex,ey=end[:2];initial=(sx,sy,4)
        queue=[(0,0,initial)];best={initial:0};previous={}
        while queue:
            _,cost,state=heappop(queue);x,y,d=state
            if cost!=best[state]:continue
            if (x,y)==(ex,ey):
                path=[(x,y)]
                while state in previous:
                    state=previous[state];path.append(state[:2])
                return cost,list(reversed(path))
            for direction,(dx,dy) in enumerate(((1,0),(-1,0),(0,1),(0,-1))):
                nx,ny=x+dx,y+dy
                if not (0<=nx<=138 and 0<=ny<=72) or (nx,ny) in blocked:continue
                new=(nx,ny,direction)
                score=cost+1+(3 if d!=4 and d!=direction else 0)+occupied.get((nx,ny),0)*2
                if score>=best.get(new,float('inf')):continue
                best[new]=score;previous[new]=state
                heappush(queue,(score+abs(ex-nx)+abs(ey-ny),score,new))
        return float('inf'),[]
    result=[];badges=[]
    for event,(source,target,label,kind) in data['transitions'].items():
        pairs=sorted(((abs(a[0]-b[0])+abs(a[1]-b[1]),a,b) for a in ports(states[source]) for b in ports(states[target])),key=lambda p:p[0])
        candidates=[]
        for _,a,b in pairs[:4]:
            cost,path=search(a,b)
            if path:candidates.append((cost,a,b,path))
        if not candidates:raise ValueError('Cannot lay out transition: '+event)
        _,a,b,path=min(candidates,key=lambda p:p[0])
        for point in path:occupied[point]=occupied.get(point,0)+1
        points=[(a[2],a[3])]+[(x*STEP,y*STEP) for x,y in path]+[(b[2],b[3])]
        simplified=[points[0]]
        for i in range(1,len(points)-1):
            before,here,after=points[i-1:i+2]
            if (before[0]==here[0]==after[0]) or (before[1]==here[1]==after[1]):continue
            simplified.append(here)
        simplified.append(points[-1])
        # Number sits halfway along the routed path, away from the source/destination boxes.
        candidates=sorted(range(1,len(points)-1),key=lambda i:abs(i-len(points)/2))
        badge=next((points[i] for i in candidates if all(abs(points[i][0]-x)+abs(points[i][1]-y)>28 for x,y in badges)),points[len(points)//2])
        badges.append(badge)
        result.append((event,source,target,label,kind,simplified,badge))
    return result


def graph_svg(data, machine):
    states=data['states'];active={'end':'stop'}.get(machine.get('node'),machine.get('node'))
    if active not in states:active=machine.get('work')
    paths=[];boxes=[];legend=[]
    for number,(event,source,target,label,kind,points,(tx,ty)) in enumerate(routes(json.dumps({'states':states,'transitions':data['transitions']})),1):
        color=COLORS[kind];dash=' stroke-dasharray="7 4"' if kind=='recovery' else ''
        path='M'+' L'.join(f'{x},{y}' for x,y in points)
        title=escape(states[source][0]+' → '+states[target][0]+'：'+label)
        paths.append(f'<g class="transition edge-{kind}" data-transition="{escape(event)}"><title>{number}. {title}</title><path d="{path}" fill="none" stroke="{color}" stroke-width="1.5"{dash} marker-end="url(#arrow-{kind})"/>'
                     f'<circle cx="{tx}" cy="{ty}" r="10" fill="white" stroke="{color}"/><text x="{tx}" y="{ty+4}" text-anchor="middle" font-size="10" fill="{color}">{number}</text></g>')
        legend.append(f'<div style="padding:5px;border-left:3px solid {color}"><b>{number}. {escape(label)}</b><br><small>{escape(states[source][0])} → {escape(states[target][0])}</small></div>')
    for key,(title,kind,x,y) in states.items():
        selected=key==active;fill={'llm':'#eff6ff','host':'#ecfdf5','terminal':'#fef2f2'}[kind]
        stroke='#2563eb' if selected else {'llm':'#93b4df','host':'#80b9a4','terminal':'#f0aaaa'}[kind]
        boxes.append(f'<g data-state="{key}" data-active="{str(selected).lower()}"><rect x="{x}" y="{y}" width="{WIDTH}" height="{HEIGHT}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="{4 if selected else 1.5}"/>'
            f'<text x="{x+12}" y="{y+20}" font-size="11" fill="#475569">{("● 現在位置 · " if selected else "")+("LLM" if kind=="llm" else "HOST")}</text>'
            f'<text x="{x+12}" y="{y+43}" font-size="17" font-weight="600" fill="#172554">{escape(title)}</text>'
            f'<text x="{x+12}" y="{y+61}" font-size="11" fill="#475569">{escape(key)}</text></g>')
    markers=''.join(f'<marker id="arrow-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{v}"/></marker>' for k,v in COLORS.items())
    gates=''.join('<li>'+escape(line)+'</li>' for line in data['gates'])
    return ('<div class="machine-graph"><div style="padding:8px"><b>強調する経路: </b><label><input type="checkbox" class="show-skill" checked> スキル</label>　<label><input type="checkbox" class="show-recovery" checked> 差し戻し</label>　<label><input type="checkbox" class="show-stop" checked> 停止</label></div><div style="overflow:auto"><svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="最新実装の全ステートと条件付き遷移" viewBox="0 0 1400 740" style="width:100%;min-width:1000px;font-family:system-ui">'
            +'<style>.machine-graph:has(.show-skill:not(:checked)) .edge-skill,.machine-graph:has(.show-recovery:not(:checked)) .edge-recovery,.machine-graph:has(.show-stop:not(:checked)) .edge-stop{opacity:.08;pointer-events:none}.transition:hover path{stroke-width:4}.transition:hover circle{stroke-width:3}</style><defs>'+markers+'</defs>'+''.join(paths)+''.join(boxes)+'</svg></div>'
            +'<div style="padding:12px;background:#fff1f2;border-radius:10px"><b>共通の終了・検証条件</b><ul>'+gates+'</ul></div>'
            +'<details><summary>矢印の番号と遷移条件（矢印にカーソルを置いても確認できます）</summary><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:8px">'+''.join(legend)+'</div></details></div>')
