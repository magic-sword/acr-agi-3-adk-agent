"""Conservative exact-pixel ROI tracker shared by all benchmark arms.

No game roles, annotations, actions or expected motion are consumed. This is a
baseline tracker, not a general solution for occlusion/deformation/identity.
"""
from agent.cognition.geometry import extract


def program_boxes(grid):
    result=extract(grid,'p')
    return [[o['bbox'][0],o['bbox'][1],o['bbox'][2]+1,o['bbox'][3]+1] for o in result['instances']],bool(result.get('extraction_limited'))


def native_boxes(normalized):
    import math
    return [[max(0,math.floor(b[0]*.064)),max(0,math.floor(b[1]*.064)),
             min(64,math.ceil(b[2]*.064)),min(64,math.ceil(b[3]*.064))] for b in normalized]


class PixelTracker:
    def __init__(self,radius=8):
        self.radius=radius;self.previous=None;self.tracks=[];self.serial=0

    def add(self,boxes,source):
        existing={tuple(t['box']) for t in self.tracks}
        for b in boxes:
            if tuple(b) in existing:continue
            assert 0<=b[0]<b[2]<=64 and 0<=b[1]<b[3]<=64
            self.serial+=1;self.tracks.append(dict(id=self.serial,box=list(b),source=source));existing.add(tuple(b))

    def advance(self,grid):
        import numpy as np
        current=np.asarray(grid,dtype=np.uint8)
        boxes,limited=program_boxes(grid)
        links=[];lost=[];ambiguous=[];old=self.tracks;self.tracks=[]
        changed=np.zeros((64,64),bool) if self.previous is None else current!=self.previous
        unchanged=self.previous is not None and not changed.any()
        if unchanged:
            self.tracks=[dict(t) for t in old]
            links=[dict(id=t['id'],before=t['box'],after=t['box']) for t in old]
        elif self.previous is not None:
            found={}
            for t in old:
                x1,y1,x2,y2=t['box'];w=x2-x1;h=y2-y1
                template=self.previous[y1:y2,x1:x2]
                left=max(0,x1-self.radius);top=max(0,y1-self.radius)
                right=min(64,x2+self.radius);bottom=min(64,y2+self.radius)
                windows=np.lib.stride_tricks.sliding_window_view(current[top:bottom,left:right],(h,w))
                matches=np.argwhere(np.all(windows==template,axis=(-2,-1)))
                if len(matches)!=1:
                    (lost if len(matches)==0 else ambiguous).append(t);continue
                y,x=matches[0];b=[int(x+left),int(y+top),int(x+left+w),int(y+top+h)]
                found.setdefault(tuple(b),[]).append(t)
            for b,tt in found.items():
                if len(tt)!=1:ambiguous.extend(tt);continue
                t=tt[0];self.tracks.append(dict(t,box=list(b)))
                links.append(dict(id=t['id'],before=t['box'],after=list(b)))
        self.add(boxes,'program')
        # Large scene/background rectangles must not hide unexplained changes.
        covered=np.zeros((64,64),bool)
        for t in old+self.tracks:
            x1,y1,x2,y2=t['box']
            if (x2-x1)*(y2-y1)<=256:covered[y1:y2,x1:x2]=True
        residual=int((changed&~covered).sum())
        local_unresolved=sum((t['box'][2]-t['box'][0])*(t['box'][3]-t['box'][1])<=256 for t in lost+ambiguous)
        self.previous=current.copy()
        return dict(links=links,lost=len(lost),ambiguous=len(ambiguous),changed_pixels=int(changed.sum()),
                    uncovered_changed_pixels=residual,local_unresolved=local_unresolved,
                    refresh_requested=bool(residual or local_unresolved or limited),unchanged=unchanged)

    def output(self):return [dict(t) for t in self.tracks]
