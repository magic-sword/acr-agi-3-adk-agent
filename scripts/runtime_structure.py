"""Read runtime declarations as data, without importing or executing agent code."""
import ast
import hashlib
from html import escape
from pathlib import Path

LABELS = {'interpret_world':'対象・関係・疑問を更新', 'select_goal':'小目標選択', 'assess_goal':'小目標評価', 'design_experiment':'実験設計',
          'inspect_target':'対象認識', 'judge_effect':'効果判定', 'choose_method':'方法選択',
          'resolve_arguments':'引数決定', 'skill_creation':'スキル作成'}


def read_structure(root):
    root = Path(root)
    paths = [root/'agent/cognition'/name for name in ('workflow.py','tasks.py','focused.py','routing.py','state.py','machine.py','world.py')]
    sources = {str(p.relative_to(root)):p.read_text() for p in paths if p.is_file()}
    if 'agent/cognition/workflow.py' not in sources:
        raise ValueError('この場所には実装ソースが保存されていません')
    required = ('agent/cognition/tasks.py','agent/cognition/machine.py')
    if any(name not in sources for name in required):
        raise ValueError('最新のステートマシン定義がありません')
    machine_tree=ast.parse(sources['agent/cognition/machine.py'])
    declarations={n.targets[0].id:ast.literal_eval(n.value) for n in machine_tree.body
                  if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
                  and n.targets[0].id in ('STATES','TRANSITIONS','GLOBAL_GATES')}
    if set(declarations) != {'STATES','TRANSITIONS','GLOBAL_GATES'}:
        raise ValueError('ステート・遷移・共通条件の宣言が不足しています')
    tasks = []
    source = sources.get('agent/cognition/tasks.py')
    if source:
        tree = ast.parse(source)
        assignments = {n.targets[0].id:n.value for n in tree.body if isinstance(n,ast.Assign)
                       and isinstance(n.targets[0],ast.Name)}
        instructions = ast.literal_eval(assignments['INSTRUCTIONS']) if 'INSTRUCTIONS' in assignments else {}
        classes = {n.name:n for n in ast.parse(sources.get('agent/cognition/state.py','')).body if isinstance(n,ast.ClassDef)}
        classes.update({n.name:n for n in tree.body if isinstance(n,ast.ClassDef)})
        registry = assignments.get('TASKS')
        if not isinstance(registry,ast.Dict):
            raise ValueError('TASKS宣言を静的に読み取れません')
        for key,value in zip(registry.keys,registry.values):
            name = ast.literal_eval(key)
            if not isinstance(value,ast.Tuple) or len(value.elts)!=2 or not isinstance(value.elts[1],ast.Name):
                raise ValueError('未対応のタスク宣言です: '+str(name))
            tool,contract=ast.literal_eval(value.elts[0]),value.elts[1].id
            def fields(class_name, seen=None):
                seen=set() if seen is None else seen
                if class_name in seen or class_name not in classes:return []
                seen.add(class_name);node=classes[class_name]
                inherited=[f for base in node.bases if isinstance(base,ast.Name) for f in fields(base.id,seen)]
                return inherited+[n.target.id for n in node.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)]
            tasks.append({'id':name,'tool':tool,'contract':contract,'fields':fields(contract),
                          'instruction':instructions.get(name,''),'line':getattr(value,'lineno',None)})
    tree=ast.parse(sources['agent/cognition/workflow.py'])
    def literal(node):
        if isinstance(node,ast.Constant):return node.value
        if isinstance(node,ast.Name):return node.id.upper()
        if isinstance(node,(ast.List,ast.Tuple)):return [literal(n) for n in node.elts]
        if isinstance(node,ast.Dict):return {literal(k):literal(v) for k,v in zip(node.keys,node.values)}
        raise ValueError('Workflowの動的な辺は静的抽出できません')
    edges=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='Workflow':
            edges=literal(next(k.value for k in node.keywords if k.arg=='edges'))
    states=declarations['STATES']; transitions=declarations['TRANSITIONS']
    if {t['id'] for t in tasks} != {key for key,value in states.items() if value[1]=='llm'}:
        raise ValueError('TASKSとステート定義が一致しません')
    if any(a not in states or b not in states for a,b,_,_ in transitions.values()):
        raise ValueError('遷移が未定義のステートを参照しています')
    digest=hashlib.sha256(''.join(k+'\n'+v for k,v in sorted(sources.items())).encode()).hexdigest()
    return {'root':str(root),'hash':digest,'tasks':tasks,'edges':edges,
            'files':list(sources),'states':states,'transitions':transitions,'gates':declarations['GLOBAL_GATES']}


def structure_html(data, machine=None):
    """An offline SVG of the executable transition declarations, including cycles."""
    from scripts.machine_graph import graph_svg
    details=''.join('<details><summary>'+escape(t['id'])+' · '+escape(t['contract'])+'</summary><p>'
        +escape(t['instruction'])+'</p><p>出力: '+escape(', '.join(t['fields']))+'</p></details>' for t in data['tasks'])
    return ('<div style="background:#f8fafc;padding:12px;border-radius:12px">'
        '<b>最新実装のステートマシン</b><p>青: LLMの判断　緑: ホスト処理　橙の破線: 差し戻し　赤: 停止</p>'
        + graph_svg(data,machine or {})
        + '<details><summary>各ステートの指示・出力契約</summary>'+details+'</details>'
        + '<small>構造ID: '+data['hash'][:12]+' · 実行側のmachine.pyから生成</small></div>')


def failure_points(events):
    points=[]
    for i,e in enumerate(events):
        reason=None
        if e.get('event')=='task_rejected':reason='差し戻し: '+str(e.get('reason'))
        elif e.get('event')=='model_decision' and (e.get('error') or e.get('schema_valid') is False):reason='モデル出力／契約エラー'
        elif e.get('event')=='tool_finished' and e.get('status') in ('error','rejected'):reason='ツール拒否: '+str(e.get('tool'))
        elif e.get('event')=='state_exited' and e.get('state')=='RUN':
            result=(e.get('output') or {}).get('result') or {}
            if result.get('status')=='stop':reason='停止: '+str(result.get('reason'))
        if reason:points.append((f"step {e.get('step','—')} · #{e.get('sequence')} · {reason}",i))
    return points


def trace_html(events, upto):
    """Only the visible prefix: no future failures or conclusions in playback."""
    prefix=events[:upto+1]; chain=[]
    for e in prefix:
        if e.get('event')=='state_entered' and e.get('state')=='DECIDE':
            task=e.get('input',{}).get('work') or e.get('work') or 'DECIDE'
            chain.append(task)
    failures=failure_points(prefix)
    actions=sum(e.get('event')=='action_dispatched' for e in prefix)
    calls=sum(e.get('event')=='model_decision' for e in prefix)
    step=prefix[-1].get('step') if prefix else None
    current_calls=sum(e.get('event')=='model_decision' and e.get('step')==step for e in prefix)
    path=' → '.join(escape(LABELS.get(t,t)) for t in chain[-18:]) or 'まだ判断の記録がありません'
    last=escape(failures[-1][0]) if failures else '記録なし'
    return f'<div style="padding:12px;background:#fff7ed"><b>表示時点までの実行経路</b><p>{path}</p><p>送信操作 {actions}回 · モデル呼出し記録 {calls}回（現在の観測: {current_calls}回） · 検出した問題イベント {len(failures)}件</p><p>直近の問題: {last}</p><small>問題イベントはログの拒否・例外・停止です。失敗の原因を自動で断定するものではありません。</small></div>'
