"""Export auditable target-event reviews; positions and roles are not scored."""
import json,sys
from pathlib import Path

out=Path(sys.argv[1]);summary=json.loads((out/'summary.json').read_text())
rows=[]
for r in summary['outcomes']:
    expected={'block_1px':'moved','cross_1px':'moved','color_only':'color_changed','synthetic_static':'unchanged','synthetic_new':'appeared'}[r['case']]
    if r['arm'] in ['target_gray','target_black','restore_target','before_only','after_only']:
        expected='unchanged'
    obs=r['observations'];free=r['arm'].startswith('free_')
    selected=obs if free else [r['target']]
    assert all(x is not None for x in selected)
    kinds=[x['kind'] for x in selected] or ['unchanged']
    descriptions=[x['description'] for x in selected]
    notes=[]
    semantic=(kinds==[expected])
    # Explicit review of the observed color-only free answers: the description
    # says white -> orange, although the JSON kind says moved.
    if r['case']=='color_only' and free and obs:
        assert len(obs)==1 and obs[0]['kind']=='moved'
        assert 'orange' in obs[0]['description'].lower() and 'cross' in obs[0]['description'].lower()
        semantic=True;notes.append('色変化の記述は正しいが kind=moved と矛盾。')
    if r['arm'].startswith('target_'):
        notes.append('対象を両時点で隠した対照。unchanged は対象の存在確認に成功した意味ではない。')
    if r['arm']=='after_only':
        notes.append('両時点をAFTERに置換。元のBEFORE物体一覧は固定しているため画像と一覧が一致しない場合がある。')
    if r['backend']=='official' and r['case']=='synthetic_new' and r['arm'] in ['control_black','free_inventory']:
        notes.append('出現物体は正しい。upper left という位置記述は誤りだが、位置は本評価の採点対象外。')
    if r['backend']=='official' and r['case']=='color_only' and r['arm']=='control_black':
        notes.append('Bの移動誤認に加え、Xでオレンジ十字の新規出現を報告。同じ十字の色変化を別物体の出現として扱う誤り。')
    rows.append({k:r[k] for k in ['backend','id','case','arm']}|dict(expected_event=expected,reported_kinds=kinds,
        descriptions=descriptions,event_label_correct=kinds==[expected],description_event_correct=semantic,notes=notes,
        other_changes=[] if free else [x for x in obs if x!=r['target'] and x['kind']!='unchanged']))
result=dict(scope='固定した合成5例の対象イベント。精密座標・方向・役割を採点しない。マスク等で正解自体が変わる条件を含むため全110件を改善率として合算しない。',
    method='対象IDの分類を抽出し、非unchanged回答と自由記述を確認。色変化の分類・説明不整合と位置誤りを別記。全回答は元jsonlに保存。',rows=rows)
(out/'semantic-review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('Reviewed',len(rows),'target events')
