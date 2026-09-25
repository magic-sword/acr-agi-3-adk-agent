"""Render the current ADK graph and skill lifecycle without importing model dependencies."""
import argparse
import ast
import hashlib
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def extract(root=ROOT):
    source = (root/'agent/cognition/workflow.py').read_text()
    tree = ast.parse(source)
    build = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_build_graph')
    call = next(n for n in ast.walk(build) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Workflow')
    def read(n):
        if isinstance(n,ast.Constant):return n.value
        if isinstance(n,ast.Name):return n.id.upper()
        if isinstance(n,(ast.List,ast.Tuple)):return [read(x) for x in n.elts]
        if isinstance(n,ast.Dict):return {read(k):read(v) for k,v in zip(n.keys,n.values)}
        raise ValueError('unsupported graph expression')
    edges = read(next(k.value for k in call.keywords if k.arg=='edges'))
    skills = [str(p.relative_to(root)) for p in sorted((root/'agent/skills').glob('*/SKILL.md'))]
    return {'edges':edges,'method_skills':skills,'workflow_sha256':hashlib.sha256(source.encode()).hexdigest()}


def render_svg(data):
    detail = escape(json.dumps(data['edges'],ensure_ascii=False))
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 420" role="img" aria-label="Decision and execution with gated skill learning">
<style>text{{font-family:system-ui,sans-serif;fill:#172554}} .box{{fill:#eff6ff;stroke:#2563eb;stroke-width:2}}</style>
<defs><marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#2563eb"/></marker></defs>
<rect width="960" height="420" fill="white"/><text x="35" y="40" font-size="24">Autonomous skill learning</text>
<rect class="box" x="70" y="90" width="310" height="100" rx="12"/><text x="95" y="125" font-size="22">DECIDE</text><text x="95" y="160">Experiment design / review / skill builder</text>
<rect class="box" x="550" y="90" width="330" height="100" rx="12"/><text x="575" y="125" font-size="22">RUN</text><text x="575" y="160">Tools, actions, observed effect checks</text>
<path d="M380,115 H545" stroke="#2563eb" fill="none" marker-end="url(#a)"/><text x="407" y="102">next job</text>
<path d="M550,175 H387" stroke="#2563eb" fill="none" marker-end="url(#a)"/><text x="415" y="200">real result</text>
<text x="70" y="260" font-size="18">Skill lifecycle (host-owned)</text>
<text x="70" y="295">candidate → evaluate seed + fresh trials + regression → active → suspended on mismatch</text>
<text x="70" y="335">Methods: notebook · design-experiment · review-experiment · skill-creator</text><text x="70" y="375" font-size="12">Source edges: {detail}</text></svg>'''


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
