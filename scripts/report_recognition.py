"""Join auditable visual-review judgments with recognition replay measurements."""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_parallel import digest, read_lines

LABELS={'original_full':'現行形式・全体','original_crops':'現行形式・部分拡大',
        'grounded_full':'根拠付き形式・全体','grounded_crops':'根拠付き形式・部分拡大'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    output=parser.parse_args().output
    rows=[r for r in read_lines(output/'measurements.jsonl') if not r['warmup']]
    reviews={r['id']:r for r in json.loads((output/'visual-review.json').read_text())}
    cases=json.loads((output/'cases.json').read_text())
    groups=defaultdict(list)
    for r in rows:
        value=r.get('parsed',r.get('response',{}));review=reviews[digest(value)]
        r['review']=review
        names=[c['name'] for c in value.get('concepts',[])]
        r['stage_valid']=r['valid'] and len(set(names))==len(names) and all(
            t['concept'] in names for t in value.get('targets',[]))
        groups[(r['name'],r['arm'])].append(r)
    lines=['# 画面認識の4条件比較','',
           '保存フレームでの開発用予備実験。ゲーム操作は行っていない。',
           '初期3画面＋ls20の1操作前後。各条件を1回ウォームアップ後、3反復。',
           '意味の採点は事前に固定した6項目に対するCodexの画像照合であり、独立した人手評価ではない。',
           '同じ画像の反復は新しいゲーム事例を増やさない。未見ゲームへの一般化は未測定。','',
           '|画面|条件|事実への言及 /6|矛盾する記述数|Schema通過|実行時の概念参照も整合|秒 中央値|入力/出力トークン|',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    summary=[]
    for (name,arm),group in sorted(groups.items()):
        fact=statistics.mean(sum(r['review']['facts']) for r in group)
        errors=statistics.mean(len(r['review']['contradictions']) for r in group)
        sec=statistics.median(r['seconds'] for r in group)
        usage=[statistics.median((r.get('usage') or {}).get(k,0) for r in group) for k in ['prompt_tokens','completion_tokens']]
        schema=sum(r['valid'] for r in group);stage=sum(r['stage_valid'] for r in group)
        lines.append(f'|{name}|{LABELS[arm]}|{fact:.2f}|{errors:.2f}|{schema}/{len(group)}|{stage}/{len(group)}|{sec:.2f}|{usage[0]:g}/{usage[1]:g}|')
        summary.append({'case':name,'arm':arm,'fact_mean':fact,'contradiction_mean':errors,
                        'schema_valid':schema,'stage_valid':stage,'n':len(group),'seconds_median':sec,
                        'prompt_tokens_median':usage[0],'completion_tokens_median':usage[1]})
    lines+=['','「事実への言及」は関連する部品・関係を明示した数で、全物体の認識率ではない。',
            '「矛盾する記述」は明確な外観・位置・変化の誤記。未検証の役割名は別途review内で扱う。',
            '追加形式は指示文と出力Schemaの複合変更。両者の寄与は分離していない。',
            '部分拡大は各画像へ固定の重なり付き4分割画像を1枚追加。操作前後では全体2枚＋拡大2枚。',
            'temperature=0、seed=123、最大出力1000、HTTPタイムアウト45秒。再試行はしない。',
            'KVキャッシュあり。順序は反復ごとの固定乱数で変更。速度はこの再生条件の値で、冷えたキャッシュでの速度ではない。',
            '全体45秒のHTTP期限も満たせるかを記録するが、オンラインの記憶読み出し・計画・操作の時間は含まない。','',
            '入力全文：cases.json、全応答：measurements.jsonl、採点基準：rubric.json、',
            '個別の判断と引用：visual-review.json、画素差分：transition-evidence.json。',
            '比較ビュー：comparison.html。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    (output/'review-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    page=['<!doctype html><html lang="ja"><meta charset="utf-8"><title>画面認識比較</title>',
          '<style>body{font:16px system-ui;margin:24px;background:#f3f5f7;color:#18202b}section{margin:32px 0}article{background:white;padding:16px;margin:10px;border:1px solid #ccd}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px monospace}.arms{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr))}img{max-width:100%;image-rendering:pixelated}summary{cursor:pointer}</style>',
          '<h1>画面認識の比較</h1><p>同じ画面の反復を含む予備実験。採点はCodexによる暫定レビュー。</p>']
    for name in sorted({r['name'] for r in rows}):
        page.append('<section><h2>'+html.escape(name)+'</h2>')
        base=next(c for c in cases if c['name']==name and c['arm']=='original_full')
        for part in base['payload']['messages'][1]['content']:
            if part['type']=='image_url':page.append('<img width="428" src="'+part['image_url']['url']+'">')
        page.append('<div class="arms">')
        for arm in LABELS:
            group=groups[(name,arm)];page.append('<article><h3>'+LABELS[arm]+'</h3>')
            seen=set()
            for row in group:
                key=digest(row.get('parsed',row.get('response',{})))
                if key in seen:continue
                seen.add(key);review=row['review']
                page.append('<p>'+html.escape(f'事実 {sum(review["facts"])}/6・矛盾 {len(review["contradictions"])}件')+'</p>')
                page.append('<pre>'+html.escape(json.dumps(review,ensure_ascii=False,indent=2))+'</pre>')
                page.append('<details open><summary>応答全文</summary><pre>'+html.escape(json.dumps(row.get('parsed',row.get('response')),ensure_ascii=False,indent=2))+'</pre></details>')
            page.append('</article>')
        page.append('</div></section>')
    (output/'comparison.html').write_text('\n'.join(page)+'</html>')
    print('\n'.join(lines))


if __name__=='__main__':main()
