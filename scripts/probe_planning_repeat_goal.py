"""Clarify the referent of repeat: same control versus same control AND target."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from probe_incremental_planning_inputs import execute
from probe_planning_minimal_choice import build as minimal_build
from probe_repeated_click_inputs import digest


def build():
    cases,source=minimal_build();jobs=[]
    for item in source:
        if item['variant']!='repeatability':continue
        for variant in ('original_repeat_goal','same_target_repeat_goal'):
            job=deepcopy(item);job['variant']=variant
            if variant=='same_target_repeat_goal':
                part=job['payload']['messages'][0]['content'][1]
                context=json.loads(part['text'])
                context['goal']='Check whether repeating the last CLICK on the same target gives the same observable result.'
                part['text']=json.dumps(context,ensure_ascii=False,separators=(',',':'))
            jobs.append(job)
    random.Random(2909202659).shuffle(jobs)
    return cases,jobs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cases,jobs=build()
    execute(jobs,args.output.resolve(),dict(
        design='Exploratory 8-request contrast after minimal-choice probe. Only goal wording changes: '
               'repeat last interaction versus repeat last CLICK on the same target. Same options, history and schema.',
        limitation='Tests explicit goal understanding, not spontaneous causal discovery or game progress. '
                   'No image, control-list addition, runtime changes or output rejection.',
        probe_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources={n:dict(path=c['source'],sha256=digest(c['original'])) for n,c in cases.items()}))


if __name__=='__main__':main()
