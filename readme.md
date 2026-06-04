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

```bash
# 数値評価（cube が押せているかを確認）
USE_CHOREONOID=1 \
  choreonoid --no-window --python scripts/eval_cnoid_numerical.py -- \
  --restore_dir single_run/pusher_cnoid --num_episodes 10

# Choreonoid GUI ビューアでリアルタイム再生（ウィンドウが開く）
vglrun choreonoid --python scripts/eval_cnoid_viewer.py -- \
  --restore_dir single_run/pusher_cnoid --fps 25 --episodes 3

# 可視化（mp4 保存・ウィンドウ不要）
USE_CHOREONOID=1 \
  choreonoid --no-window --python scripts/eval_cnoid_visual.py -- \
  --restore_dir single_run/pusher_cnoid \
  --output single_run/pusher_cnoid/eval.mp4
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

| ファイル | 内容 |
|---------|------|
| `docs/choreonoid_migration.md` | Choreonoid 移行の詳細（バグ修正・アーキテクチャ変遷・ベンチマーク）|
| `docs/migration_changes.md` | 変更ファイル一覧 |
| `docs/system_overview.md` | システム全体の概要（Stackelberg PPO の仕組み）|
| `report_v2.md` | 作業レポート（現状の最終構成）|

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
