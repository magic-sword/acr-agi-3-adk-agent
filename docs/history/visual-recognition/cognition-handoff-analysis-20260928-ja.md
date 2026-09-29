認識から行動計画への引継ぎ分析（2026-09-28）

> 調査履歴：個別実験時点の記録です。現在の採用構成は[統合資料](../../visual-recognition-adopted-ja.md)、履歴の要点は[一覧](README.md)を参照してください。
================================================

**結論：変化の測定は改善しているが、測定を計画の制約にする接続が不足している。さらに、試行計画と遷移先の矛盾が受理され、操作前に再理解へ戻る経路がある。** 前者はls20の誤った結果説明、後者はvc33・ft09の操作0回として現れている。記憶参照はその反復を増幅し、時間と文脈を消費する。

対象は[引継ぎ資料](../../visual-recognition-handoff-ja.md)が指定した `outputs/measured-runtime-20260928/live-*-measured` の3実行。artifacts・model・requests・observationsの生ログ、保存画像、実行時ソースと現在の実装を照合した。各ゲーム1回・短時間の観測であり、改善案の効果はまだ検証していない。ゲーム内部の正解ルールは前提にしていない。分析開始時の `git status` はクリーンだった。

実行時と現在の `state.py`・`notebook.py`・`perception.py` は同一。`workflow.py`・`deliberation.py` の差は旧方式の分岐削除で、以下で指摘する処理は残っている。今回の変更はこの分析資料と[再集計JSON](../../../outputs/cognition-handoff-analysis-20260928/metrics.json)の追加のみ。

**1. 実行の概要**

| 指標 | ls20 | vc33 | ft09 |
|---|---:|---:|---:|
| 操作数 / クリア数 | 2 / 0 | 0 / 0 | 0 / 0 |
| 実行時間 | 120.54秒 | 60.55秒 | 60.55秒 |
| モデル呼び出し数 | 61 | 63 | 52 |
| うち記憶参照 | 43（70%） | 56（89%） | 46（88%） |
| 記憶参照区間の合計時間 | 26.21秒 | 15.85秒 | 13.85秒 |
| 任意記憶の参照セッション数 | 8 | 5 | 4 |
| 計画あり・再理解への矛盾した遷移の受理数 | 1 | 3 | 2 |
| 受理されたtarget記述のうち候補IDとの対応あり | 0 / 8 | 0 / 3 | 0 / 3 |

呼び出し数にはタイムアウトした要求を含む。target数は再理解ごとの記述を含み、物体のユニーク数ではない。記憶参照時間は `memory_prepared.seconds` の合計。**呼び出しの7〜9割が記憶参照でも、実行時間の7〜9割ではない**。時間比は約22〜26%で、長い理解・計画の生成も大きい。ls20の最初の測定ログから最初の操作選択までは48.44秒。

実行時間・成績は[比較集計](../../../outputs/measured-runtime-20260928/combined-summary.json)、その他は生ログから再集計した。以下のA・M・Rの数字はそれぞれartifacts・model・requests JSONLの物理行番号で、イベントの `sequence` とは異なる。末尾にログへのリンクをまとめた。

**2. ls20：期待した移動が、観測済みの事実へ変わっている**

保存された生グリッドを直接差分計算すると、変化は次のとおりだった。候補対応のアルゴリズムに依存しない確認である。

| 遷移 | 変化した全画素（x, y） | 色ID | 測定された候補の状況 |
|---|---|---|---|
| step 0 → 1 | (13,61), (13,62) | 11 → 3（黄 → 濃灰） | 不変9件、バーの前後2候補は対応未解決 |
| step 1 → 2 | (14,61), (14,62) | 11 → 3（黄 → 濃灰） | 不変9件、バーの内部色変化1件、位置差[0,0] |

「Player」と呼ばれた白い十字付近も「Portal」と呼ばれた上部の青い図形も画素は変わっていない。ただし、その役割名が正しいこと自体は未確認。

これに対して、次の連鎖がある。

1. step 0でDOWNを試す。「Playerが下へ移動する」は `prediction` に書かれた期待である（ls20 A:148）。
2. step 1の `reconcile` は “Player moved downward, aligning with the portal's position.” を `evidence` に記載する（A:176）。実測の2画素変化とは合わない。
3. 次の `ground` がその説明を `baseline` にほぼそのまま採用し、`probe` から `achieve` へ移る（A:191,193）。
4. 2回目のDOWN後、意味回答は `unknown`（A:212）。しかし `reconcile` は再び同じ移動説明を出す（A:230）。
5. 新しいattempt:3は操作0回のまま高速判断が完了候補を出し（A:255）、3回目の `reconcile` も同じ説明を繰り返す（A:268）。この照合では `current_invocation_result=null`、`attempt_results=[]` だった（M:59）。

3回の照合は観測IDを除く結果フィールドが同一で、2回目以降も `assessment=probe_result` のまま。その時点の計画は `achieve` であり、3回目は新規操作の結果さえない。試行の区別に必要なIDや結果欄は渡されているが、自由文に反映されていない。なお、`goal_status` はactiveのままで、環境の勝利や目標のconfirmedを誤確定したログではない。

原因となる受理経路は [`validate_stage()`](../../../agent/cognition/deliberation.py#L149)。goal ID・probeのconfirmed禁止・resume条件は検査する一方、`evidence` と測定の整合、`assessment` と試行種別は検査しない。[`Reconciliation`](../../../agent/cognition/state.py#L121) にも観測主張ごとの根拠参照がない。**測定台帳の不変性は守られていても、後段が別の説明を事実として使うことは防いでいない。**

**3. 照合へ届く情報が「変わったもの」に偏り、意味の不明さが画像要求に反映されない**

[`context_record()`](../../../agent/cognition/perception.py#L66) は不変候補を `unchanged_count` に圧縮し、候補一覧は `understand` にだけ渡す（[`_context()`](../../../agent/cognition/workflow.py#L129)）。ls20の後段には「不変9件」は届くが、「十字を構成する候補のどれが不変か」は届かない。rawの `changed_cells` は届いているので測定が全く失われたわけではないが、意図した対象との照合がモデルの補完に任される。

さらに、[`_visual_parts()`](../../../agent/cognition/workflow.py#L191) は `perception.requires_review=false` ならunderstand・aim以外の画像を省略する。このフラグが表すのは幾何対応・未被覆画素などの問題であり、意味回答の `unknown` ではない。

- step 1の照合要求は画像2枚あり（ls20 R:39）。それでも誤説明が起きたため、画像追加だけで解決する問題ではない。
- step 2では幾何対応が成立し `requires_review=false`。意味回答の `unknown` によりreconcileへ戻っているのに、その照合要求は画像0枚（R:50）。以降のground・高速実行も画像0枚（R:54〜56）。

つまり、**「幾何的には測定できた」と「対象の意味・対応まで説明できる」が、画像を渡す判断で混同されている**。曖昧な意味解釈を照合に戻す機構はあるが、解消に必要な追加情報が同時に増えない。

[`questions()`](../../../agent/cognition/perception.py#L92) も `changes` だけを質問対象にする。step 1は画素変化があるのに候補対応が未解決のため意味質問0件。step 2はバーの変化について1問。「操作対象が動くはずだったが、不変だったか」という期待効果側からの質問は作られない。一般にもchangesが空の場合、この層では期待効果の不成立を照会しない。ただしprobeは別経路でreconcileに戻る。

**4. vc33・ft09：試行を作った直後に、試行せず再理解へ戻る**

ft09 A:15では、クリックする1手の `intent=probe` 計画を書き、理由も「役割が不明なのでprobeが必要」としながら、遷移先は `next=understand`。A:129でも繰り返す。vc33ではA:15,136,245の3回で同じ構造になっている。その結果、どちらも `plan_created`・`choose_skill`・実操作に到達していない。

[`Grounding`](../../../agent/cognition/state.py#L110) のフィールド説明には「planありならexecute、再理解ならplan=null」とある。しかし [`validate_stage()`](../../../agent/cognition/deliberation.py#L122) はnextがexecute以外ならreasonの有無だけ見てreturnする。[`_accept_stage()`](../../../agent/cognition/deliberation.py#L184) はnextを優先し、計画を実行用に保存しない。これはプロンプトの弱さだけでなく、明示した契約と実装の不一致である。

単にnextをexecuteへ強制するだけでは不十分。計画の質にも問題がある。

- vc33の最初のクリック対象は離れた黒いバーと青いバーを同時に挙げ、後には「色つき図形とグリッドの関係」、最後は盤面全体になる。1か所のクリック対象を定めていない。
- vc33のexpected_effect・done_whenは「空間関係が理解できる」、ft09では「セルの役割が分かる」。盤面で判定できる結果より、エージェント内部の理解を完了条件にしている。
- ls20の2番目の計画ではcontinue_whenもdone_whenも「PlayerがPortalの真下」。同じ条件で継続と終了が成立し、次の選択を一意に制約しない（A:193）。

不確実な因果関係を試すことは妥当だが、**試す対象・操作・比較する観測結果が具体化されていない**。これは操作と効果の対応問題で、引継ぎ資料の固定問題でもbindingは1/4のまま改善していない。

**5. 対象の同定と質問の引継ぎが切れている**

受理された3ゲーム計14件のtarget記述すべてで `candidate_refs=[]`。ls20は意味名と測定候補が結び付かず、vc33・ft09は主に「グリッド全体」がtargetになっている。空欄は未解決を表せる正当な出力だが、後段で未解決のまま進む条件が厳密に扱われない。ID存在検査は空リストには何も保証しない。

加えて、vc33・ft09のunderstandは具体化した質問を一度受け取った後、再び初期の “What useful target relation or uncertainty should be investigated on this board?” を出している（vc33 A:196、ft09 A:189）。[`_accept_stage()`](../../../agent/cognition/deliberation.py#L165) はこれを次のhandoff_questionに採用する。質問を具体化した履歴が保持されても、進行中の課題は一般論へ戻る。

副次的な制約として、ft09の初回測定候補は43件だが、understandに渡るのは先頭24件で19件省略。省略数は明示され、画像もあるので情報が完全に消えたわけではない。ただし候補参照の検査も先頭24件限定で、後半候補をテキストの根拠として指定できない。今回の失敗への寄与は未分離だが、対象に基づく候補選択が必要な箇所である。

**6. 記憶参照は多いが、自発的に終了せず、操作後には情報を引き渡せない**

任意記憶を実際に参照した17セッションすべてが `read_time_budget` で終了し、finish_memoryによる終了は0回。毎回選択集合が空で再開され、一覧→本文→保持という操作を繰り返す（[`_read_memory()`](../runtime/retired-implementation-ja.md)）。新しい観測がないvc33・ft09でも、再理解・再計画のたびに同様の過去記述を集め直している。

ls20は操作後の6セッション中5セッションで選択結果が空。例えばM:47〜49では、分類を開く約1.56秒、項目を開く約1.63秒で4秒枠の多くを消費し、本文n10を保持する呼び出しが約0.82秒でタイムアウトする。開いた本文は、保持できなければ後続のmemory_briefに入らない。

これは情報伝達の実損だが、主因を記憶不足だけに帰すこともできない。最新のlast_result・last_review・understandingなどは別経路で常時渡される。実際、誤説明のreconcileには2画素の実測とunknownが届いていた。

入力の重複もある。ls20 step 2のreconcile（M:50）は、トップレベルのmeasured_objectsに加え、同じoutcomeをlast_result・current_invocation_result・attempt_results経由で受け取る。promptは7,390 tokens。重複が誤説明を発生させたと断定はできないが、短い判断ごとの入力負担を増やしている。記憶呼び出しも常時渡す測定・最新結果を含むため、1トークン出力でも軽い処理とは限らない。

**7. 改善の優先順と確認方法（未実装）**

| 優先 | 変更案 | 固定ログで確認する条件 |
|---|---|---|
| P0 | groundのplanとnextの相互制約を実装し、矛盾した出力は修正要求へ。再理解は具体的な不足情報と取得方法を要求する | vc33・ft09の「planあり・understand」を受理しない。曖昧なクリックを自動実行しない |
| P0 | reconcileの観測主張を構造化し、観測ID・試行ID・候補または画素領域・測定参照を結び付ける。未確認の移動をbaselineへ昇格させない | ls20の2画素変化から十字の移動を事実化しない。attempt:3で以前の操作結果を今回の成果にしない |
| P1 | 期待効果の対象について、変化・不変・対応不明を渡す。候補IDは観測IDと組にし、複数候補からなる対象や未解決を扱う | バーだけ変化／対象不変／対象対応不明を区別し、意味質問がchangesの有無だけに依存しない |
| P1 | 幾何的不確実性と意味的不確実性を分け、unknownの照合には必要な対象情報・前後画像を渡す | ls20 step 2でunknownなのに追加情報なし、という要求を解消する |
| P1 | probeを「仮説・具体的な1操作・観測可能な分岐」にする。continue/doneの矛盾を防ぎ、同一観測で同じ質問に戻る反復を検出する | 「役割が分かる」を完了条件にしない。未知の効果は残したまま、何を比べるか定められる |
| P2 | 記憶は関連する根拠を直接渡し、探索は追加情報が必要なときに限定する。選択の再利用・巡回停止・残り時間に応じた終了と入力重複削減を行う | 記憶参照時間・保持件数・本文を開いたまま失う件数・初回操作までの時間を比較する |

まず固定観測の再生で「矛盾した遷移の受理」「根拠のない事実化」「過去試行の流用」を検査し、その後に同じ操作・時間予算で実ゲームを比較する。操作数の増加だけでは改善判定にせず、対象の根拠、得られた新情報、進展も記録する。固定問題の移動16/16を維持する評価と、計画・照合の評価は分ける。

生ログの参照先：

- ls20：[A: artifacts](../../../outputs/measured-runtime-20260928/live-ls20-measured/ls20-9607627b/cognition/f93c11628b954bf2a4d4eb071d8dd71c.artifacts.jsonl)、[M: model](../../../outputs/measured-runtime-20260928/live-ls20-measured/ls20-9607627b/cognition/f93c11628b954bf2a4d4eb071d8dd71c.model.jsonl)、[R: requests](../../../outputs/measured-runtime-20260928/live-ls20-measured/ls20-9607627b/cognition/f93c11628b954bf2a4d4eb071d8dd71c.requests.jsonl)、[生グリッド](../../../outputs/measured-runtime-20260928/live-ls20-measured/ls20-9607627b/cognition/f93c11628b954bf2a4d4eb071d8dd71c.observations.jsonl)
- vc33：[A: artifacts](../../../outputs/measured-runtime-20260928/live-vc33-measured/vc33-5430563c/cognition/5a2cb44ebd0540beb130a82006d32de5.artifacts.jsonl)
- ft09：[A: artifacts](../../../outputs/measured-runtime-20260928/live-ft09-measured/ft09-0d8bbf25/cognition/3e0e19495a0f489191816ebe67a8796d.artifacts.jsonl)
