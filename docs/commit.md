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

**num_threads=1 の制約**: ZMQ REP サーバーは1つのシミュレーション世界を持つため、
複数ワーカーが同時アクセスすると状態が混在する。フォーク後の REQ ソケット共有は
デッドロックを引き起こすため、現状はシングルスレッド。
→ 根本解決にはワーカーごとに Choreonoid サーバーを1つ用意する構成変更が必要。
