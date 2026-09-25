# ローカル評価

現行エージェントは[DECIDE/RUNとスキル学習](skill-learning-runtime-ja.md)。

画面・操作・状態の入出力を並べて調べる場合は、[JupyterLab観測ノートブック](../notebooks/agent_observatory.ipynb)を使う。[評価IDを選択して再生する手順](agent-monitor-ja.md)。

```bash
make benchmark-prepare
make model-up
make benchmark
make benchmark EVAL_GAMES=ls20 EVAL_STEPS=40 EVAL_SECONDS=180 EVAL_HARD_SECONDS=200
```

既定は公開3ゲーム、各12操作・1レベル・90秒、ワーカー上限110秒。公式ローカルHTTPゲートウェイとSDKのスコアを使い、ノートブックと同じソースをスナップショットして各ゲームを新しいプロセスで動かす。Kaggleへの提出は行わない。短縮IDが曖昧なら完全なゲーム版IDを指定する。

```bash
# 学習を止めた対照条件
 docker compose run --rm --no-deps dev python scripts/benchmark_local.py --games ls20 --no-learning
# 以前の学習成果を固定し、明示的に再利用
 docker compose run --rm --no-deps dev python scripts/benchmark_local.py \
   --games ls20 --no-learning --skill-library outputs/cognition/RUN/skills/library.json
# ドライバだけの確認（モデル性能を測る条件ではない）
 docker compose run --rm --no-deps dev python scripts/benchmark_local.py \
   --offline-policy --games ls20 --steps 2 --seconds 10 --hard-seconds 25
```

`RUN`は実際のログの実行IDに置き換える。学習・候補の失敗試験も時間・操作の費用に含む。ライブラリを渡さなければ毎ゲーム空から開始する。学習効果の評価では、学習に使わなかった課題・配置で、固定ライブラリの有無を比較する。追加試行予算を与える場合は明示し、通常条件へ混ぜない。

## 出力

`outputs/evaluations/<UTC日時>/`に保存する。既存結果は上書きしない。

|ファイル|内容|
|---|---|
|manifest.json / package/|予算、モデル・依存関係・ゲーム・ソース・入力ライブラリのハッシュと評価対象ソース|
|report.md / summary.json / summary.csv|公式SDKスコア、操作、時間、停止理由、診断|
|各ゲームのscorecard.json / gateway.jsonl|公式スコアと操作の実際の応答|
|cognition/*.jsonl / *.json|各手のDECIDE/RUN、作業記憶と停止状態|
|cognition/*.model.jsonl|判断入力、生応答、HTTP往復、時間・トークン|
|cognition/*.requests.jsonl / request-images/|実際に送信したテキスト・ツール定義・画像|
|cognition/*.tools.jsonl|ツール開始と終了、読み込んだメタスキル|
|cognition/*.artifacts.jsonl|受理した判断と選択した操作。実行受付とは別|
|cognition/*.execution.jsonl|実行送信、受付、結果不明|
|cognition/*.observations.jsonl / frames/|実画面・色IDと記録アニメーション|
|cognition/*.learning.jsonl|経験・候補作成・試行・評価・昇格・停止|
|cognition/*.states.jsonl|DECIDE/RUNの入場・退出、状態の入力／出力、実行終了。新しい走行には全ジャーナル共通のsequenceも付く|
|cognition/<run>/skills/|仕様、評価証拠、再利用可能なlibrary.json|
|cognition/decisions.html|自動生成した判断とスキル獲得のビューア|

`schema_valid_rate`は提出契約の受理率で、意味的な正答率ではない。`model_calls`とスキル参照等を含む`model_http_requests`を分ける。トークンはusageを取得した要求のみ。

スキルの作成数だけで改善とは評価しない。実際の再利用、効果の一致、誤適用、中断、旧版への回帰、最終課題の達成、総費用を併読する。

1レベルで止めてもSDKのスコア分母はゲーム全体。未取得のスコアはnull、全件揃わなければ全体平均もnull。ワーカー強制終了ではゲートウェイの受付済み操作から復元するが、実行中の結果不明が残り得る。

公開環境の短時間試験であり、非公開環境、KaggleのGPU・並行実行・推論サーバとの同一性は保証しない。旧方式の評価は[履歴](history/README.md)を参照。
