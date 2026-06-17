# 実装ロードマップ（タスク単位マイルストーン）

---

## 6月：基盤検証と限界突破（最優先ブロッカーの解消）

**目的:** 物理エンジンの整合性を証明し、生成モデルを強化学習に繋ぐ「爆発問題」の耐性とタスク適合性を最速でテストする。

### Task 1: MuJoCo版再現学習スクリプト 🔄 実行中
`run_mujoco_pusher.sh` → `single_run/pusher_resume`

epoch 200 チェックポイントから再開中（`single_run/pusher_resume_stdout.log`）。  
クラッシュ原因（`max_grad_norm=40` → 勾配爆発）を特定・修正済み。  
ETA: epoch 999 まで約1.5日。

### Task 2: 共通評価・レポート自動生成スクリプト ✅ 完成・実行待ち
`scripts/eval_cross_env.py`, `scripts/generate_comparison_report.py`

各ポリシーをネイティブ物理エンジン（MuJoCo学習→MuJoCo評価、Choreonoid→Choreonoid）で評価するサブプロセス分離方式。  
MuJoCo 学習完走後に以下で比較レポートを生成:
```bash
python3 scripts/generate_comparison_report.py \
  --runs single_run/pusher_resume single_run/pusher_cnoid_v3 \
  --labels MuJoCo Choreonoid --run_eval --n_episodes 20 \
  --output single_run/comparison/
```

### Task 3: rrbotを用いたレベル0テスト ⏳ XML完成・学習待ち
`assets/mujoco_envs/rrbot_arm.xml`, `data/rrbot_description/rrbot_topology.json`

2関節直列アームの XML 作成済み・xml_robot ロード確認済み。  
Task 1 完了後に実行:
```bash
python3 design_opt/train.py cfg=pusher xml_name=rrbot_arm num_threads=20
```

---

## 7月：スケッチパイプラインMVP（2軸アームの貫通）

**目的:** 手描きスケッチからPPO学習までのデータフローを、手動操作を許容しつつ1本の道として繋ぎ切る。

### Task 1: プロンプト固定化と品質QAチェックリストの作成

Geminiの出力を安定させるため、プロンプトを3パターン検証し最適なものを固定。Tripo出力メッシュの「穴あき・分離」を防ぐQAリストを整備する。

### Task 2: 手動セグメンテーションフローの確立（Blender等）

セグメンテーションの自動化（Medial Axis等）は一旦後回しにし、Blender等で確実に3パーツに手動分割する作業手順を固める。

### Task 3: メッシュパラメータ簡易抽出スクリプト
`mesh_to_params.py`

分割済みGLBメッシュからOBB（境界ボックス）や重心を計算し、`bone_offset` と `geom.size` を逆算して `topology.json` を出力する。

### Task 4: MVP学習実行スクリプト
`run_2axis_mvp.sh`

手描き由来の2軸アームでStackelberg PPOを回し、学習曲線のフラット化（学習停滞）が起きないか確認する。

---

## 8月：バリエーション生成と提案UI実装（Human-in-the-loop）

**目的:** ロボットを複雑化しつつ、ユーザーに専門知識を求めない「安全な修正ループ」の裏側を構築する。

### Task 1: トポロジーノイズ付与スクリプト
`generate_topology_variants.py`

抽出したベースのJSONの `bone_offset` 等に微小な乱数を加え、システム側から提示するための複数の構造バリエーション（A/B/C）を自動生成する。

### Task 2: 爆発チェッカー・安全性フィルタ
`check_explosion.py`

生成したXMLをChoreonoidにロードし、10ステップだけ事前シミュレーション。速度・位置のNaN（物理発散）を検知し、安全なバリエーションだけを通過させる。

---

## 9月：複数ロボット並列学習と汎化性の検証

**目的:** 3軸・4軸アームへと展開し、論文主張の根拠となる定量データを並列で収集する。

### Task 1: バッチ学習・自動再起動管理スクリプト
`batch_train_manager.sh`

複数ロボット×複数シードの学習プロセスを監視し、CUDAメモリエラー等で落ちた場合にチェックポイント（`+restore_dir`）から自動再開させる。

### Task 2: 形態変化の可視化スクリプト
`visualize_morph_changes.py`

初期エポックと最終エポックの `bone_offset` を比較し、形態最適化によるリンク長の伸び縮みをグラフやアニメーションで出力する。

---

## 10月：有効性の証明とコードフリーズ（バッファ月）

**目的:** 提案手法の優位性を証明するアブレーション実験を行い、すべての実装作業を完了させる。

### Task 1: ベースライン（通常PPO）比較実験の実施

`lamda=0` または `morph_prior=true` フラグを利用し、形態を固定して制御のみを最適化するベースライン学習を実行し、Stackelberg PPOとの報酬差を記録する。

### Task 2: 最終評価まとめスクリプト
`compile_final_results.py`

全実験ログをパースし、論文の表にそのまま使えるフォーマット（平均報酬±標準偏差、成功率）で一括出力する。
