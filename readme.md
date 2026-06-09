# 形態・制御共設計 via Stackelberg PPO（垣内研 Choreonoid 対応版）

元論文の公式実装をベースに、垣内研究室の **Docker + Choreonoid** 環境向けに移植・改造したリポジトリ。

**元論文**: "Efficient Morphology-Control Co-Design via Stackelberg Proximal Policy Optimization"  
Yanning Dai*, Yuhui Wang*, Dylan R. Ashley, Jürgen Schmidhuber — ICLR 2026  
[論文](https://openreview.net/pdf?id=sJ0vOOkclw) | [元リポジトリ](https://github.com/YanningDai/StackelbergPPO.git) | [プロジェクトページ](https://yanningdai.github.io/stackelberg-ppo-co-design/)

<img src="static/m.png" alt="description">

---

## 動作環境

| 項目 | 内容 |
|------|------|
| Docker イメージ | `akita_sp`（`irsl_system:noetic` ベース）|
| Python | 3.12 |
| シミュレータ | Choreonoid 2.3 |
| PyTorch | 2.7.0+cu128 |
| GPU | RTX 5060 Ti（sm_120）以上推奨 |

> MuJoCo・conda は**不要**。`akita_sp` コンテナで完結する。

---

## セットアップ

```bash
# 1. リポジトリをクローン
git clone https://github.com/YanningDai/StackelbergPPO.git
cd StackelbergPPO

# 2. akita_sp コンテナに入る（VS Code Dev Container 推奨）
# 以降の操作はすべてコンテナ内で実行する
```

依存ライブラリは `akita_sp` イメージに含まれているため、追加インストールは不要。

---

## 学習の実行

### 基本（4スレッド推奨）

```bash
USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
  choreonoid --no-window --python scripts/choreonoid_train.py \
  cfg=pusher num_threads=4 \
  hydra.run.dir=single_run/pusher_cnoid \
  enable_wandb=false
```

`choreonoid --no-window --python` が必要な理由: シミュレータ（AISTSimulatorItem）は Choreonoid の Qt アプリコンテキストが必要なため、通常の `python3` では動かない。

**利用可能な環境**: cheetah, crawler, glider-hard, glider-medium, glider-regular, pusher, stepper-hard, stepper, swimmer, terraincrosser, walker-hard, walker-medium, walker-regular

### チェックポイントから再開

```bash
USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
  choreonoid --no-window --python scripts/choreonoid_train.py \
  cfg=pusher num_threads=4 \
  hydra.run.dir=single_run/pusher_cnoid \
  +restore_dir=single_run/pusher_cnoid \
  enable_wandb=false
```

チェックポイントは `{hydra.run.dir}/models/` に保存される。
- `epoch_XXXX.p`：10エポックごと
- `best.p`：報酬最高値を更新するたび

### 形態のみ引き継いでスクラッチ再学習（MuJoCo → Choreonoid 移行時）

```bash
USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
  choreonoid --no-window --python scripts/choreonoid_train.py \
  cfg=pusher num_threads=4 \
  hydra.run.dir=single_run/pusher_cnoid_transfer \
  +restore_dir=single_run/pusher_mujoco \
  morph_prior=true reset_epoch=true \
  enable_wandb=false
```

---

## 評価

> **注意**: choreonoid は `--` を引数終端マーカーとして扱わない。
> eval スクリプトへのパラメータは**環境変数**で渡す。

```bash
# 数値評価（cube が押せているかを確認）
EVAL_RESTORE_DIR=single_run/pusher_cnoid EVAL_NUM_EPISODES=5 \
  USE_CHOREONOID=1 choreonoid --no-window --python scripts/eval_cnoid_numerical.py

# 可視化（mp4 保存・ウィンドウ不要）
EVAL_RESTORE_DIR=single_run/pusher_cnoid \
EVAL_OUTPUT=single_run/pusher_cnoid/videos/eval_visual.mp4 \
  USE_CHOREONOID=1 choreonoid --no-window --python scripts/eval_cnoid_visual.py

# Choreonoid GUI ビューアでリアルタイム再生（VirtualGL が必要・詳細は docs/choreonoid_gui_issue.md）
VIEWER_RESTORE_DIR=single_run/pusher_cnoid VIEWER_FPS=25 VIEWER_EPISODES=3 \
  vglrun choreonoid --python scripts/eval_cnoid_viewer.py

# 報酬推移グラフ生成
python3 scripts/plot_rewards.py single_run/pusher_cnoid

# 最終形態 URDF 保存（morphology/ に出力）
EVAL_RESTORE_DIR=single_run/pusher_cnoid \
  USE_CHOREONOID=1 choreonoid --no-window --python scripts/save_morphology_urdf.py
```

環境変数一覧:

| スクリプト | 変数 | デフォルト | 説明 |
|-----------|------|-----------|------|
| numerical | `EVAL_RESTORE_DIR` | （必須）| 学習ディレクトリ |
| numerical | `EVAL_EPOCH` | `best` | チェックポイント |
| numerical | `EVAL_NUM_EPISODES` | `5` | エピソード数 |
| visual | `EVAL_RESTORE_DIR` | （必須）| 学習ディレクトリ |
| visual | `EVAL_OUTPUT` | `{restore_dir}/videos/eval_visual.mp4` | 出力パス |
| visual | `EVAL_FPS` | `20` | フレームレート |
| viewer | `VIEWER_RESTORE_DIR` | （必須）| 学習ディレクトリ |
| viewer | `VIEWER_FPS` | `25` | 再生フレームレート |
| viewer | `VIEWER_EPISODES` | `3` | エピソード数（0=無限）|

### 出力ディレクトリ構成

評価・可視化スクリプトは `{restore_dir}` 以下のサブディレクトリに出力を整理する:

```
single_run/pusher_cnoid/
├── .hydra/                  ← Hydra 設定（自動生成）
├── log/                     ← 学習ログ（log_train.txt, log_eval.txt）
├── models/                  ← チェックポイント（best.p, epoch_XXXX.p）
├── tb/                      ← TensorBoard イベント
├── videos/                  ← 動画ファイル
│   ├── best_policy.mp4      ← Choreonoid GUI ビューア録画
│   └── eval_visual.mp4      ← eval_cnoid_visual.py 出力
├── eval/                    ← 数値評価グラフ
│   └── eval_numerical.png   ← eval_cnoid_numerical.py 出力
├── morphology/              ← 形態ファイル
│   ├── morphology_best.urdf ← save_morphology_urdf.py 出力（Choreonoid/RViz 用）
│   └── morphology_best.xml  ← save_morphology_urdf.py 出力（MuJoCo XML）
└── plots/                   ← 学習グラフ
    └── reward_plot.png      ← plot_rewards.py 出力
```

---

## MuJoCo → Choreonoid 移行の自動化

```bash
# 移行して報酬を比較（閾値 0.5 を下回ったら警告）
python3 scripts/cnoid_transfer.py --mujoco-dir single_run/pusher

# 移行品質が低ければ自動でスクラッチ再学習も実行
python3 scripts/cnoid_transfer.py --mujoco-dir single_run/pusher --auto-scratch
```

---

## 設定のカスタマイズ

設定は Hydra で管理。`design_opt/conf/` の YAML またはコマンドライン引数で上書き可能。

```bash
# 例: ラムダやバッチサイズを変更
choreonoid --no-window --python scripts/choreonoid_train.py \
  cfg=pusher num_threads=4 \
  lamda=5 min_batch_size=10000 \
  hydra.run.dir=single_run/pusher_custom
```

主な設定値:

| キー | デフォルト | 説明 |
|------|-----------|------|
| `num_threads` | 20 | ワーカー数（4推奨）|
| `min_batch_size` | 50000 | 1エポックのサンプル数 |
| `eval_batch_size` | 10000 | 評価サンプル数 |
| `max_epoch_num` | 1000 | 最大エポック数 |
| `save_model_interval` | 10 | チェックポイント保存間隔 |
| `reset_epoch` | false | 再学習時にエポックカウンタをリセット |

---

## スレッド数の目安

`min_batch_size=5000` でのベンチマーク（RTX 5060 Ti + 20コア CPU）:

| スレッド | T_sample | 1エポック wall時間 | sample 倍速 |
|---------|----------|-----------------|------------|
| 1 | 33.5s | 80s | 1.0x |
| 2 | 11.7s | 75s | 2.9x |
| **4** | **6.7s** | **69s（最速）** | **5.0x** |
| 8 | 6.4s | 91s | 5.2x |

8スレッドはワーカー起動オーバーヘッドで逆に遅くなる。

---

## ドキュメント

全ドキュメントの索引は [`docs/index.md`](docs/index.md) を参照。主要なものを以下に示す。

| ファイル | 内容 |
|---------|------|
| `docs/system_overview.md` | Stackelberg PPO の仕組み・学習指標の説明 |
| `docs/choreonoid_migration.md` | Choreonoid 移行の詳細（バグ修正・アーキテクチャ変遷）|
| `docs/choreonoid_gui_issue.md` | GUI（3D 描画）の問題と GLVND 解決策 |
| `docs/topology_fixed_optim.md` | 初期形状の指定方法・トポロジー固定で属性値のみ最適化する方法 |
| `docs/mesh_to_xml_pipeline.md` | 3D メッシュ → MuJoCo XML 変換パイプライン設計（爆発問題の対策）|
| `docs/mesh_segmentation.md` | メッシュ分割手法の比較（スケルトン抽出・凹面・VLM 等）|
| `report_v2.md` | 作業レポート（現状の最終構成・学習進捗・評価結果）|

---

## 謝辞

[BodyGen](https://github.com/Josh00-Lu/BodyGen) と [Transform2Act](https://github.com/Khrylx/Transform2Act) をベースとした元リポジトリの著者に感謝する。

## 引用

```bibtex
@inproceedings{dai2026stackelbergppo,
  title     = {Efficient Morphology--Control Co-Design via Stackelberg Proximal Policy Optimization},
  author    = {Dai, Yanning and Wang, Yuhui and Ashley, Dylan R. and Schmidhuber, Jürgen},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
