# 前景と記憶読み出しの並列推論：測定結果

2026-09-27。ローカルのRTX A2000 12GBで測定した範囲では、常時並列化による大幅な改善は確認できなかった。
現行1枠構成に対して、前景と背景の両方が終わるまでの時間は約4%悪化〜7%短縮だった。
推論サーバは測定後に元の1枠・文脈長16384へ戻した。ゲームの実行ロジックは変更していない。

## 測定条件

- GPU：NVIDIA RTX A2000 12GB、1枚。
- モデル：Qwen3-VL-4B-Instruct Q4_K_M＋Q8_0の画像プロジェクタ。
- llama.cpp：b11118-e6ab7c1a4。同じモデル重みを共有し、要求の処理枠を1→2へ変更。
- 枠あたりの文脈長：16384。総文脈長は1枠で16384、2枠で32768。
- 入力：outputs/history/evaluations/20260926T172310706477Zの実HTTP要求と画像。
- 前景：3ゲームのunderstandと、ls20のexecute_step。背景：各ゲームのread_memoryの8要求を順に再生。
- temperature=0、seed=123、キャッシュ再利用あり。各条件のウォームアップを除外し、3反復の中央値を比較。
- 2枠の直列／並列は順序を反転して実行。1枠と2枠の比較はサーバの切替を挟んでいる。

画像は保存されたSHA-256を検証して復元し、入力全体のハッシュも保存した。
要求・応答・生成トークン数・各処理の開始終了時刻・サーバ構成を記録している。
GPUメモリの測定開始時使用量は1枠5793MiB、2枠7987MiBだった。ピーク使用量の測定ではない。

## 結果

前景と背景の両方が完了するまでの秒数。背景はどの条件でも8回の読み出し。

|前景の入力|現行：1枠直列|2枠直列|2枠並列|現行からの変化|
|---|---:|---:|---:|---:|
|ft09の熟考|5.392|5.502|5.540|2.8%遅い|
|ls20の熟考|9.174|9.996|9.566|4.3%遅い|
|ls20の操作選択|0.733|0.701|0.682|7.0%短い|
|vc33の熟考|5.471|5.488|5.309|3.0%短い|

同じ2枠構成に限って直列と並列を比べても、全処理時間の変化は約0.7%悪化〜4.3%短縮。
一方、前景の応答自体は並列時に約2〜22%遅くなった。ls20の操作選択では0.074秒→0.090秒。
背景の一部を重ねても、前景の応答が遅くなるため、全体の短縮は小さかった。

同じ入力・temperature=0でも構成により熟考の応答が変わった。生成トークン数は次のとおり。

|入力|1枠直列|2枠直列|2枠並列|
|---|---:|---:|---:|
|ft09|308|312|322|
|ls20|520|551|535|
|vc33|303|297|297|

したがって、上の時間差は各構成で実際の応答が返るまでの差であり、完全に同じ生成計算量の速度差ではない。
少数ケース・3反復の測定で、数%の差を一般的な改善率として扱うこともできない。

## 測れたことと測れていないこと

この実験は推論基盤の比較であり、オンライン先読みの実装や攻略成績の評価ではない。
過去ログの背景要求は、今回のモデルの選択結果によって変更しない。
将来必要になる問いと資料が事前に分かっているという楽観的な条件で処理を重ねている。
実際には次の問いが変わる、先読み資料を使わないなどの無駄も発生する。

同一入力の再生でキャッシュがよく効いており、新しい観測・問いへの速度をそのまま表してはいない。
追加でcache_prompt=falseを試したが、サーバが実際にはKVを再利用し続けたため、この追加試験は除外した。
outputs/parallel-prefetch-20260927-no-cache/EXCLUDED.jsonに理由を保存し、測定スクリプトにも検出処理を追加した。
キャッシュなしの改善率は未測定。

現在の読み出しに残る再閲覧・終了判断の問題は、今回の並列化では解決していない。
KaggleのT4×2やRTX PRO 6000上の速度、複数GPUへのモデル分離も未測定。
このローカル測定を本番の改善率として外挿しない。

今回の結果から、ローカル環境で常時2枠に切り替える根拠は弱い。
本番機で同じ比較を行い、操作の応答遅延と有用な先読みの割合を含めて判断する必要がある。

## 保存物と再実行

- [集計レポート](../outputs/parallel-prefetch-20260927/report.md)
- outputs/parallel-prefetch-20260927/cases.json：実際に再生した入力。
- measurements-1.jsonl / measurements-2.jsonl：ウォームアップを含む全要求の応答と時刻。
- environment-1.json / environment-2.json：GPU、サーバ、入力ハッシュ、反復数。
- summary.json：中央値、最小最大、応答の変動、前景と背景それぞれの時間。
- [測定スクリプト](../scripts/benchmark_parallel.py)：ゲームへの操作やKaggleへの提出は行わない。

以下はリポジトリ直下で実行する例。対象ログを--sourceで指定する。
既存サーバを測定中だけ再起動するため、他のモデル実行と同時には行わない。

```bash
set -e
experiment_dir=outputs/parallel-prefetch-new
python3 scripts/benchmark_parallel.py prepare \
  --source outputs/history/evaluations/20260926T172310706477Z \
  --output "$experiment_dir" --reads 8
docker compose exec -T dev python scripts/benchmark_parallel.py run \
  --output "$experiment_dir" --slots 1 --repeats 3 --modes serial
trap 'docker compose up -d --no-deps vlm' EXIT
docker compose -f compose.yaml -f "$experiment_dir/server-2.json" up -d --no-deps vlm
docker compose exec -T dev python scripts/wait_model.py --url http://vlm:8080
docker compose exec -T dev python scripts/benchmark_parallel.py run \
  --output "$experiment_dir" --slots 2 --repeats 3 --modes serial overlap
python3 scripts/benchmark_parallel.py report --output "$experiment_dir"
```
