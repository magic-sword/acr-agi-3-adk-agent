"""Render the recorded plan, grounded skills and routing without future-state leakage."""
from html import escape
import json


def display(value):
    if value is None:return '記録なし'
    return escape(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2))


def cognition_html(snapshot):
    memory = snapshot.get('cognition') or {}
    context = snapshot.get('context') or {}
    plan = memory.get('plan') or {}
    response = snapshot.get('response') or {}
    rows = [('現在の目標',plan.get('goal') or context.get('goal')),
            ('熟考での解釈',plan.get('interpretation')),
            ('ゴールからの逆算',plan.get('backward_plan')),
            ('因果関係のメモ（仮説）',memory.get('causal_notes') or plan.get('causal_notes')),
            ('実行中のスキル・段階',memory.get('active_skill')),
            ('今回の期待効果と実測結果',context.get('last_result')),
            ('今回の選択肢',context.get('choices'))]
    if (response.get('step')==(snapshot.get('current') or {}).get('step') and
            response.get('work')==context.get('work') and response.get('schema_valid')):
        rows.append(('モデルの提出結果',response.get('response')))
    rows += [('熟考へ戻る理由',memory.get('replan_reason')),
             ('手順完了の判断履歴（モデル仮説）',memory.get('completion_history')),
             ('熟考への復帰履歴',memory.get('routing_history'))]
    table=''.join('<tr><th style="text-align:left;vertical-align:top">'+escape(k)+
        '</th><td style="white-space:pre-wrap">'+display(v)+'</td></tr>' for k,v in rows if v is not None)
    skills=''.join('<details><summary>'+escape(name)+'</summary><pre style="white-space:pre-wrap">'+
        display(skill)+'</pre></details>' for name,skill in memory.get('skills',{}).items())
    return '<h3>計画と高速実行</h3><table>'+table+'</table><h4>保持している手続き（モデル仮説）</h4>'+skills
