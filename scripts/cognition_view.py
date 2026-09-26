"""Render the visible prefix of goals, stage results and procedure evidence."""
from html import escape
import json


def display(value):
    if value is None:return '記録なし'
    return escape(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2))


def cognition_html(snapshot):
    memory=snapshot.get('cognition') or {}
    context=snapshot.get('context') or {}
    response=snapshot.get('response') or {}
    goal_id=memory.get('selected_goal_id')
    rows=[('熟考・実行の工程',memory.get('phase')),
          ('現在の小目標',memory.get('goals',{}).get(goal_id) or context.get('current_goal') or context.get('goal') or '未設定'),
          ('小目標の確認状態',memory.get('goal_status',{}).get(goal_id)),
          ('状況理解・注目対象・疑問',memory.get('understanding')),
          ('照準・対象の記述・観測との対応',context.get('cursor') or memory.get('cursor')),
          ('前提条件の逆算',memory.get('backchain')),
          ('目標の依存関係',memory.get('goals')),
          ('各目標の状態と根拠',memory.get('goal_status')),
          ('次に判断する問い',memory.get('handoff_question')),
          ('保存した記録数',memory.get('note_count')),
          ('記憶の読み出し',memory.get('reader')),
          ('次の熟考に渡す記憶',context.get('memory_brief') or memory.get('memory_brief')),
          ('開いている記録',context.get('opened_record')),
          ('具体化した計画・観測の基準点',memory.get('plan')),
          ('実行中のスキル・起動ID・段階',memory.get('active_skill')),
          ('今回の期待効果と実測結果',context.get('last_result')),
          ('照合待ちの結果・完了候補',memory.get('review')),
          ('今回の選択肢',context.get('choices'))]
    if (response.get('step')==(snapshot.get('current') or {}).get('step') and
            response.get('work')==context.get('work') and response.get('schema_valid')):
        rows.append(('モデルの提出結果',response.get('response')))
    rows += [('次の工程へ戻る理由',memory.get('replan_reason')),
             ('結果照合と仮説更新の履歴',memory.get('reconciliations'))]
    table=''.join('<tr><th style="text-align:left;vertical-align:top">'+escape(k)+
        '</th><td style="white-space:pre-wrap">'+display(v)+'</td></tr>' for k,v in rows if v is not None)
    skills=''.join('<details><summary>'+escape(name)+'</summary><pre style="white-space:pre-wrap">'+
        display(skill)+'</pre></details>' for name,skill in memory.get('skills',{}).items())
    return '<h3>計画と高速実行</h3><table>'+table+'</table><h4>保持している手続き（モデル仮説）</h4>'+skills
