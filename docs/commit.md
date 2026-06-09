# コミット履歴まとめ

移行作業（MuJoCo+conda → Choreonoid+Docker）に関するコミットの概要。
元リポジトリのコミット（readme/visualization 等）は省略。

---

## 移行作業コミット一覧

### 2026-05-29

| ハッシュ | 内容 |
|---------|------|
| `1934b3a` | **ChoreonoidEnv を新規追加**。`mujoco_env_gym.py` と同一 API の差し替えクラス。ZMQ REQ ソケットで Choreonoid サーバーと通信する。 |
| `d29aaf1` | **cnoid_sim_server.py を新規追加**。Choreonoid 内で動く ZMQ REP サーバー。MuJoCo XML → URDF 変換器・armature 補完・gear 比スケーリング・AISTSimulator 手動ステップを実装。 |
| `1519e92` | **pusher.py に USE_CHOREONOID スイッチを追加**。`os.environ['USE_CHOREONOID']=='1'` で ChoreonoidEnv に切り替え。既存コードは無変更。 |
| `c06b793` | ドキュメント: Choreonoid 移行作業まとめ（フェーズ1〜3: Docker 環境・ZMQ 実装・バグ修正5件・残差調査）を追加。 |
| `e46c94a` | ドキュメント: MuJoCo XML → URDF の情報損失と補完方法（armature/gear/integrator/solimp 等）を追記。 |
| `2148c64` | **reset_epoch / reset_obs_norm フラグを config に追加**。MuJoCo → Choreonoid 移行時にエポックカウンタと観測正規化統計をリセットするオプション。 |
| `3e52d1f` | **train.py と genesis_agent.py を更新**。load_epoch（何を読むか）と start_epoch（どこから回すか）を分離。reset_obs_norm 条件を load_checkpoint に追加。 |
| `208817a` | **cnoid_transfer.py を新規追加**。MuJoCo → Choreonoid 移行ワークフロー自動化（形態引き継ぎ再学習 → 報酬比確認 → 必要ならスクラッチ再学習）。フェーズ4のドキュメントも追記。 |

### 2026-05-31

| ハッシュ | 内容 |
|---------|------|
| `2315a01` | **mujoco_py import を遅延化**。`envs/__init__.py` が全 env を一括 import するため、`USE_CHOREONOID=1` 時でも起動時に mujoco_py が呼ばれてクラッシュしていた。`mujoco_env_gym.py`・`mjviewer.py`・`mujoco_env.py`・全 env ファイル（7本）の import を `try/except` でラップし、インスタンス化時のみ失敗する形に変更。 |
| `658145b` | **Choreonoid サーバーを Jupyter カーネル方式で起動する `start_cnoid_server.py` を新規追加**。`xvfb-run choreonoid --python` の代わりに研究室標準の `jupyter_process.sh` を使用。主な問題と解決: ①`write_connection_file()` が空 HMAC キーを返す → UUID4 で手動生成、②`kc.wait_for_ready()` タイムアウト → ZMQ DEALER でハートビート直接ポーリング、③REQ タイムアウト後に再送不可 → DEALER ソケットに変更。`cnoid_transfer.py` も同方式に更新し、run_train() のログ追記機能を追加。 |
| `60403ca` | **ドキュメント2本を追加**。`docs/migration_changes.md`: 変更ファイル一覧と変更内容の技術的まとめ。`report.md`: 研究背景から現在の学習実行状況までのプロジェクトレポート。 |

---

## 現在の状態（2026-05-31 時点）

```
学習プロセス: PID 2213407（稼働中）
設定: USE_CHOREONOID=1, num_threads=1, min_batch_size=5000
ログ: single_run/pusher_cnoid/log/log_train.txt
エポック0 結果: exec_R_eps=6.81, ETA 5日14時間
```

~~num_threads=1 の制約~~: 下記 `9729990` で4スレッド化を実現。

### 2026-06-01

| ハッシュ | 内容 |
|---------|------|
| `9729990` | **4スレッド並列学習を実現**。ワーカーごとに専用の Choreonoid サーバーを用意し、ZMQ ソケット競合を解消。`start_cnoid_server.py` に `--num-servers N` オプション追加（N 個のサーバーを 5556〜5556+N-1 番ポートで起動）。`ChoreonoidEnv` に `reconnect(port)` メソッドと `_last_xml` キャッシュを追加（形態を維持したままワーカー専用サーバーに繋ぎ直す）。`genesis_agent.py` の `sample_worker()` で fork 後に `env.reconnect(5556+pid)` を呼ぶ。スレッド数比較: 1→4スレッドで T_sample が 246s→196s（約20%改善）。8スレッドは 210s と遅く、4スレッドが最適と判明。 |
| `8601b95` | **コードベース概要ドキュメントを追加**（`docs/codebase_overview.md`）。元リポジトリの全体フロー・ファイル関係・Stackelberg PPO の更新アルゴリズム・観測ベクトル構造・設定の流れ等をまとめた。 |

---

### 現在の学習状態（2026-06-01）

```
学習プロセス: PID 2311773（稼働中）
設定: USE_CHOREONOID=1, num_threads=4, min_batch_size=5000, eval_batch_size=2000
Choreonoid サーバー: port 5556〜5559 の 4 インスタンス
ログ: single_run/pusher_cnoid/log/log_train.txt
エポック24 結果: exec_R_eps=10.05（エポック0の6.79から上昇中）、ETA 約7日
```

---

### 2026-06-02〜06-08（ZMQ 廃止・直接呼び出しへの移行）

元の ZMQ + Jupyter カーネル方式を廃止し、`choreonoid --no-window --python` による直接呼び出し方式に全面移行した一連のコミット。

| ハッシュ | 内容 |
|---------|------|
| `0a33b60` | **アーキテクチャを ZMQ サーバー方式から直接呼び出しに全面移行**。`ChoreonoidEnv` を `ChoreonoidSimWorld` 内包方式に書き直し。`mujoco_xml_to_urdf()` を統合。`worker_pool.py`（`ChoreonoidWorkerPool`）・`choreonoid_train.py`・`worker_sampler.py` を新規追加。fork ベースのマルチプロセスを廃止し spawn + 永続ワーカー方式に変更。`os._exit(0)` での強制終了を採用。 |
| `b1c6ae5` | **eval スクリプト 3 本を追加**。`eval_cnoid_numerical.py`（数値評価・cube 変位計測）、`eval_cnoid_visual.py`（matplotlib 3D アニメーション mp4 生成）、`eval_cnoid_viewer.py`（Choreonoid GUI リアルタイムビューア、Python stdout の `_Tee` クラスによるログ保存を含む）。 |
| `5a3b2d1` | **ドキュメントを大幅更新**。`docs/system_overview.md` に学習指標（`exec_R_eps` vs `train_R_eps`）の説明を追加。`docs/choreonoid_migration.md` に起動ログ解説セクション追加。`docs/choreonoid_gui_issue.md` に GLVND/NVIDIA 解決策と `VIEWER_LOG` によるログキャプチャ方法を追記。`docs/index.md` を新規作成（全 md ファイルの索引）。`report_v2.md` を現状に合わせて更新（セクション 8〜11 を全面改訂）。 |
| `e2f9c44` | **`plot_rewards.py` を新規追加**。`log_train.txt` を解析し exec_R_eps / train_R_eps の推移グラフ（MA-30・best 保存マーカー・再開ライン付き）を PNG で保存。 |
| `7d4a8f3` | **`save_morphology_urdf.py` を新規追加**。学習済み best チェックポイントを読み込み、設計フェーズをポリシーに従って実行し、実行フェーズ開始時点の形態を URDF と MuJoCo XML で保存。10 エピソードで最も多くのボディを持つ形態を代表として選択。 |
| `c91d730` | **`eval_morphology.py` を新規追加**。ロボット形態を matplotlib 3D で 3 アングル可視化する PNG 生成スクリプト（後に URDF 保存方式に方針転換したため補助的な位置づけ）。 |
| `f3a1e09` | **`ant.py` に `USE_CHOREONOID` スイッチを追加**。crawler タスクが `env_name: ant` を使用するため、pusher と同様の Choreonoid/MuJoCo 切り替えを追加。 |

---

### 2026-06-09（出力ディレクトリ整理・設計ドキュメント追加）

| ハッシュ | 内容 |
|---------|------|
| `dd0fb45` | **eval スクリプトの出力先をサブディレクトリに整理**。`eval/`・`videos/`・`plots/`・`morphology/` の 4 サブディレクトリを設け、各スクリプトのデフォルト出力先を変更（`os.makedirs` も追加）。`report_v2.md` のパス記述を更新。`save_morphology_urdf.py` を新規追加。 |
| `ef2fcb2` | **crawler 用 `ant.py` に `USE_CHOREONOID` スイッチを追加**。crawler タスクが `env_name: ant` を使うため pusher と同様の切り替えを追加。`eval_morphology.py`（形態 3D 可視化スクリプト）も追加。 |
| `5589759` | **カスタム形状最適化・メッシュパイプライン設計ドキュメントを追加**。`docs/topology_fixed_optim.md`（初期形状指定・トポロジー固定・属性値最適化の設計方針）、`docs/mesh_to_xml_pipeline.md`（爆発問題の原因分析と FK による座標解決パイプライン設計）、`docs/mesh_segmentation.md`（スケルトン抽出・凹面・VLM 等の分割手法比較）の 3 ファイルを追加。`docs/index.md` を更新。 |
| `TBD` | **readme と commit.md を更新**。ドキュメント一覧を `docs/index.md` 参照形式に整理。commit.md に 2026-06-09 のコミット履歴を追記。 |
