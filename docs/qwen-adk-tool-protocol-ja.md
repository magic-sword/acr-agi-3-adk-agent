# QwenとADKのツール呼び出し境界

Qwen3-VL-4B-Instructの[公式テンプレート](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/main/chat_template.json)は、`<tool_call>`で囲んだJSONをツール呼び出しとして出力する。これはHTMLではなく、モデルのチャット書式に含まれる区切りである。[Qwen公式資料](https://qwen.readthedocs.io/en/latest/framework/function_call.html)ではHermes形式と呼ばれており、Qwenだけの専用形式ではない。Qwen3-Coderの別の構文とは区別する。

```text
ADK FunctionDeclaration / FunctionResponse
  → LocalVisionLlm：APIのtools / role=toolへ変換
  → llama.cpp：チャットテンプレートを適用してQwenへ渡す
  → APIのtool_calls、またはcontent内のネイティブ呼び出し
  → QwenToolCallAdapter：検証してADK FunctionCallへ正規化
  → ADKが実行し、結果を次のモデル要求へ戻す
```

[llama.cppの説明](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)にあるとおり、通常はサーバーがテンプレート固有の出力をAPIの構造化された呼び出しへ変換する。このプロジェクトでも`--jinja`を使用する。既存のLocalVisionLlmには構造化された呼び出しの変換があり、今回その処理を独立した[QwenToolCallAdapter](../agent/qwen_protocol.py)へ移し、contentに残ったネイティブ形式も扱えるようにした。

## 変換契約

- 登録済みツールの名前、JSONオブジェクトの引数、完全な閉じタグを要求する。複数呼び出しにも対応する。
- 正規表現で切り出さずJSONを解析するため、引数の文字列に閉じタグが含まれていても壊れない。
- 呼び出しIDを保持し、ネイティブ形式にIDがなければ発行する。ADKからの結果は同じIDで戻す。
- 構造化形式とネイティブ形式が併記された場合、内容が同じなら一度だけ実行し、異なるなら拒否する。
- 説明文・コード例・最終JSON内に引用されたタグを探して実行しない。対象は単独の呼び出しブロックだけ。
- 打ち切られた生成、未登録ツール、不正なJSONは実行しない。ゲーム操作の`CLICK`などを勝手にツールへ読み替えない。
- `tool_choice=none`で返された呼び出しは、正しいネイティブ構文でも`ToolCallNotAllowedError`とする。変換器は予算やホストの実行制約を変更しない。

生応答に加えて`tool_choice`、`decoded_protocol`、`normalized_tool_calls`をログへ残す。集計はネイティブ形式で受理した要求も数え、構造化形式との二重計上を避ける。

## 検証と残る問題

プロトコル単体テストに加え、模擬ネイティブ出力を実際のADKへ渡し、観測一覧の取得結果が同じ呼び出しIDでモデルへ戻り、その後の計画が操作になる経路を検証した。

`make test`の全73件が通過した。[テストログ](../outputs/verification/qwen-protocol/tests.log)。提出用ノートブックと`make visualize`の可視化も再生成した。生成物と検証ログはGit管理対象外。

実際のローカルQwenでも`scripts/check_skills.py`が成功した。`list_skills`、`load_skill`、`load_skill_resource`を通し、リソース内だけにあるランダムな検証値を回答できた。[実行記録](../outputs/skill-checks/f55a5b37ada94acbace04bb99562c414/result.json)。これは実モデルとの接続確認であり、ネイティブ形式のフォールバックを必ず通す試験ではない。

直前の[公開ゲーム測定](local-evaluation-cognitive-refactor-ja.md)を再確認すると、構造化されたツール要求54件は既に存在し、タグ形式の出力6件はすべて最終HTTP枠の8回目だった。この枠では`tool_choice=none`を指定していた。したがって当時の失敗は「Qwenのツール呼び出し全般がADKに非対応」ではなく、未変換の形式と最終回答制約を区別する必要がある。今回の変換器だけで当該6件を実行可能にしたとは扱わず、ゲームスコアの改善も未検証である。
