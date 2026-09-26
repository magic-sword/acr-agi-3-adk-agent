# ローカル評価

現行エージェントは[熟考と1トークン実行を分けるループ](fast-slow-runtime-ja.md)。

画面・操作・状態の入出力を並べて調べる場合は、[JupyterLab観測ノートブック](../notebooks/agent_observatory.ipynb)を使う。[評価IDを選択して再生する手順](agent-monitor-ja.md)。

```bash
make benchmark-prepare
make model-up
make benchmark
make benchmark EVAL_GAMES=ls20 EVAL_STEPS=40 EVAL_SECONDS=180 EVAL_HARD_SECONDS=200
```

既定は公開3ゲーム、各12操作・1レベル・90秒、ワーカー上限110秒。公式ローカルHTTPゲートウェイとSDKのスコアを使い、ノートブックと同じソースをスナップショットして各ゲームを新しいプロセスで動かす。Kaggleへの提出は行わない。短縮IDが曖昧なら完全なゲーム版IDを指定する。

```bash
# ドライバだけの確認（モデル性能を測る条件ではない）
 docker compose run --rm --no-deps dev python scripts/benchmark_local.py \
   --offline-policy --games ls20 --steps 2 --seconds 10 --hard-seconds 25
```

通常の継続は1操作1回の高速選択。初回・再計画・手順移行は複数回呼び出す。計画の不正出力だけ追加1要求で修正する。`COGNITION_REPAIR_ATTEMPTS=0`で修正を無効化できる。旧学習・ライブラリ指定のCLI引数は廃止した。

## 出力

`outputs/evaluations/<UTC日時>/`に保存する。既存結果は上書きしない。

|ファイル|内容|
|---|---|
|manifest.json / package/|予算、モデル・依存関係・ゲーム・ソースのハッシュと評価対象ソース|
|report.md / summary.json / summary.csv|公式SDKスコア、操作、時間、停止理由、診断|
|各ゲームのscorecard.json / gateway.jsonl|公式スコアと操作の実際の応答|
|cognition/*.jsonl / *.json|各手のDECIDE/RUN、作業記憶と停止状態|
|cognition/*.model.jsonl|判断入力、生応答、HTTP往復、時間・トークン|
|cognition/*.requests.jsonl / request-images/|実際に送信したテキスト・ツール定義・画像|
|cognition/*.tools.jsonl|操作提出ツールの開始と終了|
|cognition/*.artifacts.jsonl|計画・手順・高速選択・熟考への復帰と実測結果。選択と実行受付は別|
|cognition/*.execution.jsonl|実行送信、受付、結果不明|
|cognition/*.observations.jsonl / frames/|実画面・色IDと記録アニメーション|
|cognition/*.states.jsonl|DECIDE/RUNの入場・退出、状態の入力／出力、実行終了。新しい走行には全ジャーナル共通のsequenceも付く|
|cognition/decisions.html|自動生成した判断のビューア|

`schema_valid_rate`は提出契約の受理率で、意味的な正答率ではない。`model_calls`と実際の`model_http_requests`を分ける。トークンはusageを取得した要求のみ。

`fast_slow`診断は計画数、復帰数、完了スキル数、無変化、修正数と、熟考・スキル選択・手順実行それぞれの呼出し数・時間・トークン数を記録する。速度だけでなく、注意の切り替え、進展の誤認、レベル達成も確認する。

1レベルで止めてもSDKのスコア分母はゲーム全体。未取得のスコアはnull、全件揃わなければ全体平均もnull。ワーカー強制終了ではゲートウェイの受付済み操作から復元するが、実行中の結果不明が残り得る。

公開環境の短時間試験であり、非公開環境、KaggleのGPU・並行実行・推論サーバとの同一性は保証しない。旧方式の評価は[履歴](history/README.md)を参照。

工程別の呼出し数・中央値・出力トークンをreport.mdとsummary.jsonに記録する。熟考はunderstand/backchain/ground/reconcile、高速側はchoose_skill/execute_step。最新ログ形式はobservatory_schema=3。
