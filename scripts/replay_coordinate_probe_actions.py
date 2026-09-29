"""Execute unique valid probe clicks in restored offline environments, never online.

Restore the recorded two-action prefix, assert the full grid matches the source,
then execute one proposed action. Outcomes are never fed into probe requests.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np


def final_grid(frame):
    assert frame is not None
    data=np.asarray(frame.frame)
    return data[-1] if data.ndim==3 else data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ.update(OPERATION_MODE='offline',ARC_API_KEY='local-evaluation')
    from arc_agi import Arcade,OperationMode
    from arcengine import GameAction
    jobs=defaultdict(list);gold={};rows=[]
    for directory in args.experiments:
        verification=json.loads((directory/'verification.json').read_text())
        assert verification['planned']==verification['completed']
        gold.update(json.loads((directory/'gold.json').read_text()))
        for row in json.loads((directory/'evidence.json').read_text()):
            if row['task']=='plan' and row['valid_click']:
                key=(row['case'],*row['game_xy'])
                jobs[key].append(dict(experiment=str(directory),case=row['case'],variant=row['variant'],
                                     repetition=row['repetition']))
    (out/'plan.json').write_text(json.dumps(dict(
        independent_resets=len(jobs),
        represented_valid_proposals=sum(map(len,jobs.values())),
        seed=0,operation_mode='OFFLINE',
        scope='One proposed click after the recorded prefix; no replanning and no complete-game success estimate.',
        inputs=[str(p) for p in args.experiments]),indent=2)+'\n')
    arcade=Arcade(operation_mode=OperationMode.OFFLINE,arc_api_key='local-evaluation',
                  environments_dir=str(ROOT/'environment_files'),recordings_dir=str(out/'recordings'))
    assert arcade.operation_mode==OperationMode.OFFLINE
    for (case,x,y),members in sorted(jobs.items()):
        obs_path=ROOT/gold[case]['observation_source']
        observation=next(r for r in map(json.loads,obs_path.read_text().splitlines()) if r['step']==2)
        decision_log=obs_path.with_name(obs_path.name.replace('.observations.jsonl','.jsonl'))
        prefix=[r['action'] for r in map(json.loads,decision_log.read_text().splitlines())
                if r['step']<2]
        assert len(prefix)==2 and all(a['action']=='ACTION6' for a in prefix)
        env=arcade.make(observation['game_id'],seed=0)
        assert env is not None
        before=env.reset()
        for action in prefix:
            before=env.step(GameAction.ACTION6,data={k:action[k] for k in ('x','y')})
        grid=final_grid(before).copy()
        assert np.array_equal(grid,np.asarray(observation['grid'])), 'Restored source board differs'
        before_levels=before.levels_completed
        after=env.step(GameAction.ACTION6,data={'x':x,'y':y})
        delta=int(np.count_nonzero(final_grid(after)!=grid))
        result=dict(case=case,game_xy=[x,y],prefix=[{k:a[k] for k in ('action','x','y')} for a in prefix],
                    restored_grid_matches=True,changed_cell_count=delta,
                    levels_delta=after.levels_completed-before_levels,
                    state=str(after.state),members=members)
        rows.append(result)
        (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(case,x,y,'changed',delta,'levels',result['levels_delta'],flush=True)
    arcade.close_scorecard()
    print('Completed',len(rows),'unique offline click replays')


if __name__=='__main__':main()
