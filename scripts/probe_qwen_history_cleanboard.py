"""Exploratory follow-up: remove axis labels and margins from the same grid rendering."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from probe_qwen_history_coordinates import (cases_with_geometry, build_jobs, execute, analyze,
                                             digest, SCALE)
from agent.rendering import frame_image

REPS = ('normalized', 'marker', 'normalized_marker')


def clean_cases():
    cases = cases_with_geometry()
    for case in cases.values():
        board = frame_image(case['grid'])
        base = board.resize((64*SCALE,64*SCALE),Image.Resampling.NEAREST)
        marked = base.copy()
        x,y = case['gold']['game_xy']
        px,py = x*SCALE,y*SCALE
        draw = ImageDraw.Draw(marked)
        draw.rectangle((px-2,py-2,px+SCALE+1,py+SCALE+1),outline='black',width=1)
        draw.rectangle((px-1,py-1,px+SCALE,py+SCALE),outline='#ffff00',width=1)
        assert np.array_equal(np.asarray(base)[py:py+SCALE,px:px+SCALE],
                              np.asarray(marked)[py:py+SCALE,px:px+SCALE])
        center = [(x+.5)*SCALE,(y+.5)*SCALE]
        normalized = [round(1000*p/size) for p,size in zip(center,base.size)]
        case.update(base=base,marked=marked)
        case['gold'].update(image_size=list(base.size),origin=[0,0],image_xy=center,normalized_xy=normalized)
    return cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','run','analyze'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();cases=clean_cases()
    if args.phase=='prepare':
        out.mkdir(parents=True,exist_ok=False);(out/'assets').mkdir()
        for name,case in cases.items():
            for kind in ('base','marked'):case[kind].save(out/'assets'/f'{name}-{kind}.png')
        jobs=build_jobs(cases,3,REPS)
        design=dict(requests=len(jobs),repeats=3,representations=list(REPS),
                    source_experiment='../qwen-history-coordinates-20260929/design.json',
                    intervention='Same source grid and 6x scaling; no margins or numeric axis labels. '
                                 'Previous coordinate remapped consistently to this full-image viewport. '
                                 'Same instructions, evidence facts, output tools, and model.',
                    exploratory='Selected after seeing axis-like coordinates in primary planning outputs; '
                                'not an independently preregistered replication.',
                    limitations='Cannot separate axis text from margin removal. Natural language and bare/grid '
                                'conditions not repeated. Two histories on one board, no success-rate claim.',
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    job_hashes=[digest(j['payload']) for j in jobs])
        (out/'design.json').write_text(json.dumps(design,indent=2)+'\n')
        (out/'gold.json').write_text(json.dumps({n:c['gold'] for n,c in cases.items()},indent=2)+'\n')
        print('Prepared',len(jobs),'requests')
    elif args.phase=='run':
        design=json.loads((out/'design.json').read_text());jobs=build_jobs(cases,3,REPS)
        assert [digest(j['payload']) for j in jobs]==design['job_hashes']
        execute(jobs,out/'run',dict(phase='cleanboard',design_file='../design.json',
                                  probe_script_sha256=design['script_sha256']))
    else:analyze(out,cases)


if __name__=='__main__':main()
