# ドキュメント一覧

このリポジトリ内のすべての Markdown ファイルの概要。

---

## リポジトリルート

### [readme.md](../readme.md)
**使い方ガイド（メインドキュメント）**

学習・評価の実行コマンド、環境一覧、設定パラメータ、スレッド数ベンチマークをまとめた入口。まず読むべきファイル。

- 学習の起動方法（新規・チェックポイント再開）
- 評価コマンド（数値評価・mp4 生成・GUI ビューア）
- Hydra 設定パラメータ一覧
- MuJoCo → Choreonoid 移行の自動化コマンド

---

### [report_v2.md](../report_v2.md)
**移行作業レポート（最新・v2）**

Python 3.12 直接呼び出し構成への移行後の最終状態をまとめた技術レポート。バグ修正の詳細・並列サンプリング実装・学習進捗・評価結果を含む。

- Choreonoid バックエンドの設計とアーキテクチャ変遷（ZMQ → 直接呼び出し）
- バグ修正 10 件の詳細（Choreonoid 固有 5 件 + Python 3.12/NumPy 2.0 対応 5 件）
- 並列サンプリング実装（spawn + 永続ワーカー方式）
- 現在の学習進捗・評価結果（epoch 1607/2000、best.p で 80% 成功）

---

### [report.md](../report.md)
**移行作業レポート（旧・v1）**

ZMQ 構成時代（Python 3.8 二重プロセス方式）の記録。現在のアーキテクチャとは異なる。歴史的経緯の参照用。

---

## docs/

### [docs/system_overview.md](system_overview.md)
**システム概要（Stackelberg PPO の仕組み）**

Stackelberg PPO が「何をしているか」を大局的に説明。コードを読む前の背景知識として読む。

- リーダー（形態設計）とフォロワー（制御）の役割
- Stackelberg 戦略の意味（フォロワーの反応を先読みして設計する）
- 1 エピソードの 3 フェーズ（骨格変換・属性変換・実行）
- Transformer ネットワーク構造と可変長ボディへの対応
- 学習ログの指標説明（`exec_R_eps` vs `train_R_eps`）

---

### [docs/choreonoid_migration.md](choreonoid_migration.md)
**Choreonoid 移行の詳細技術メモ**

MuJoCo → Choreonoid 移行の全フェーズを時系列で記録。バグ調査の詳細・URDF 変換の制約・ベンチマーク結果を含む。

- Docker 環境構築（`akita_sp` イメージ）
- ZMQ → 直接呼び出しへのアーキテクチャ変遷
- バグ修正の根本原因調査（角速度誤取得・armature 欠落・capsule 慣性誤り等）
- MuJoCo XML → URDF の情報損失と API による補完
- スレッド数ベンチマーク（4 スレッドが最適）
- **起動・リセット時のログ解説**（`Loading Body`・`dynamics` 警告・ボディ名命名規則）

---

### [docs/choreonoid_gui_issue.md](choreonoid_gui_issue.md)
**Choreonoid GUI（3D 描画）の問題と現状**

コンテナ内で Choreonoid GUI の 3D シーンが正常描画できない問題の診断・解決策・現在の回避策。

- 原因：Mesa libGL vs NVIDIA GLX の不一致
- 解決策：`__GLX_VENDOR_LIBRARY_NAME=nvidia`（GLVND 経由）で起動すればクラッシュしなくなることを確認
- Python stdout のキャプチャ方法（`VIEWER_LOG` ファイルへのリダイレクト）
- 数値評価・matplotlib 可視化の代替手段（動作確認済み）
- `choreonoid --` が機能しない問題と環境変数方式への変更

---

### [docs/codebase_overview.md](codebase_overview.md)
**コードベースの構成と連携**

元リポジトリのファイル構成・役割・1 エポックの処理フローを解説。コードリーディングの入口として使う。

- ディレクトリ構成と各ファイルの役割
- `train.py` → `BodyGenAgent` → `PusherEnv` の呼び出し関係
- ネットワーク構造（Transformer ポリシー・バリュー関数）

---

### [docs/migration_changes.md](migration_changes.md)
**変更ファイル一覧（差分まとめ）**

元リポジトリから何をどう変えたかをファイル単位で列挙。コードレビューや差分確認の参照用。

- 削除ファイル（ZMQ サーバー等）
- 新規作成ファイル（`mujoco_env_choreonoid.py`・`worker_pool.py`・eval スクリプト群）
- 変更ファイルと変更内容の要約

---

### [docs/topology_fixed_optim.md](topology_fixed_optim.md)
**トポロジー固定・属性値最適化モード**

スケルトン変化を無効化し、リンク長・ギア比・関節可動域だけを Stackelberg 最適化する方法。スケッチ → 3D メッシュ → XML 変換パイプラインの構想も記載。

- 初期形状の指定方法（`xml_name` オプション、デフォルト形状の解説）
- topology 固定の設定方法（`max_nchild=0` + `enable_remove=false`）
- `fix_skeleton` フラグの追加実装案（コード 5 行）
- joint range 最適化の追加実装案（`Joint` クラスへの追記）
- スケッチ → Gemini → Tripo3D → MuJoCo XML 変換パイプライン構想
- ボディ命名規則（`str(ind+1) + parent_name`）と 4 脚 XML テンプレート

---

### [docs/mesh_segmentation.md](mesh_segmentation.md)
**メッシュ分割手法の比較と StackelbergPPO への統合方針**

3D 生成ツール出力の単一メッシュをロボットパーツに分割する手法の比較と、StackelbergPPO への接続設計。

- ヒューリスティック手法: スケルトン抽出（Reeb Graph / Medial Axis）、凹面ベース（VHACD）
- データドリブン手法: PointNet++ + K-Means（パーツ数 K を直接指定）、VLM プロンプト（言語主導の意味的分割）
- 手法比較表（制御しやすさ・計算コスト・StackelbergPPO との相性）
- Leader アクション空間の制限設計（fix_skeleton・リンク長±20%・ギア比範囲）
- 実装ロードマップ（Phase 1 手動 JSON → Phase 4 VLM 統合）

---

### [docs/mesh_to_xml_pipeline.md](mesh_to_xml_pipeline.md)
**メッシュ → MuJoCo XML 変換パイプライン設計方針**

Tripo3D 等の非定型メッシュをそのまま RL に渡すと発生する「爆発問題」の原因分析と、それを防ぐ堅牢なデータパイプラインの設計方針。

- 爆発原因の分類（初期干渉・スケール不一致・ジョイント位置ズレ・不自然な重心）
- FK による絶対座標解決（T-pose 整列で干渉を防ぐ核心ロジック）
- capsule フィッティング実装（OBB 主軸ベース）
- 干渉・COM・慣性の検証チェック（10 ステップシミュレーション自動検証含む）
- `scripts/mesh_to_xml/` の想定モジュール構成と使い方
- 既知の落とし穴と対策（凹凸メッシュ・max_nchild 超過・地面埋まり等）

---

### [docs/commit.md](commit.md)
**コミット履歴まとめ**

移行作業のコミットを時系列で記録。各コミットで何を実装・修正したかのサマリー。ZMQ 時代のコミットも含む歴史的記録。
