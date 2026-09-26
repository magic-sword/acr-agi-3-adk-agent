"""Render the current ADK graph and skill lifecycle without importing model dependencies."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def extract(root=ROOT):
    from scripts.runtime_structure import read_structure
    return read_structure(root)


def render_svg(data):
    from scripts.machine_graph import graph_svg
    html = graph_svg(data,{})
    return '<svg'+html.split('<svg',1)[1].split('</svg>',1)[0]+'</svg>'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/agent-visualization')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    data=extract();svg=render_svg(data)
    (args.output/'workflow.svg').write_text(svg)
    (args.output/'graph.json').write_text(json.dumps(data,indent=2))
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Agent workflow</title>'+svg)
    print(args.output/'index.html')

if __name__=='__main__':main()
