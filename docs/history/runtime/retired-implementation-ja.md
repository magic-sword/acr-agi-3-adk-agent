# 廃止実装の削除記録

> 訂正：以下は整理時点の記録。計画まで削除したのは依頼の範囲を超えていたため、`focused_workflow.py`・`focused_state.py`・`candidates.py`・`fast_choice.py` と計画のテストは復元した。候補生成器注入も利用可能。カーソル・ノート巡回・旧比較ランタイムは削除のまま。[現行仕様](../../planning-restoration-ja.md)を参照。

2026-09-29。通常経路を単一操作判断に統一し、旧実装の復帰用スイッチと候補生成器注入APIも廃止した。元コードはGitの履歴（追跡済みファイル）または過去評価のソーススナップショットで参照する。現行ツリーに実行用の旧コピーは置かない。

## 削除した実装・専用テスト・実験コマンド

- `agent/cognition/focused_workflow.py`
- `agent/cognition/focused_state.py`
- `agent/cognition/candidates.py`
- `agent/cognition/planning_comparison.py`
- `agent/cognition/memory_comparison.py`
- `agent/cognition/action_update_comparison.py`
- `agent/cognition/cursor.py`
- `agent/cognition/notebook.py`
- `agent/cognition/tasks.py`
- `agent/fast_choice.py`
- `tests/test_focused_workflow.py`
- `tests/test_fast_slow.py`
- `tests/test_semantic_cursor.py`
- `tests/test_memory_reader.py`
- `tests/test_planning_comparison.py`
- `tests/test_memory_comparison.py`
- `tests/test_action_update_comparison.py`
- `scripts/benchmark_planning_comparison.py`
- `scripts/benchmark_memory_comparison.py`
- `scripts/benchmark_action_update.py`
- `scripts/audit_memory_comparison.py`
- `scripts/probe_action_update_frozen.py`
- `scripts/verify_hybrid_runtime.py`

## 残した共通機能と移行

- `workflow.py` は認識・観測・共通の実行グラフだけに縮小。SimpleRuntimeはFocusedRuntimeを継承しない。
- `deliberation.py` は型付きの単一モデル要求と最大1回の出力修正だけに縮小。
- `state.py` は現行メモリschema 15のみ。旧保存メモリを実行状態に読み込む移行機構はない。過去ログの閲覧は可能。
- 旧専用の80テストを削除。認識・受付・冪等性・ログ再生・パッケージ検証は維持し、期限・修正回数・境界・旧設定拒否の4テストを現行経路へ追加した。297件から221件になった理由は旧機能の廃止である。
- `scripts/recorded_proposer.py` にキャッシュ済みSAMの再生部品を移し、`benchmark_object_memory.py` は現行ランタイムを使う。
- 旧比較用環境変数が非空なら設定エラーにする。黙って別条件で測定しない。
- ノートブック生成から隠しディレクトリと `.ipynb_checkpoints` のコードを除外する。

実験ログ・認識の比較スクリプト・検証結果は保持する。旧設計書はこのディレクトリで現行仕様と区別する。
