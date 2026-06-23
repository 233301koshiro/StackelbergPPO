# 評価スクリプトの使い方

学習済み重みを使った評価・可視化スクリプトのまとめ。

> **注意**: choreonoid は `--` を引数終端マーカーとして扱わない。
> eval スクリプトへのパラメータは**環境変数**で渡す。

---

## eval の動作概要

eval でも毎エピソード最初から設計フェーズを走らせる。理由は「どんな形態を作るか」自体がポリシーネットワーク（Leader）の出力だからで、固定 URDF を読むのではなくエピソードのたびに形態を設計し直す。

```
eval の 1 エピソード:
  設計フェーズ → Leader が「どんな形にするか」を決める
    ├─ skeleton_transform（5 ステップ）: ボディの追加・削除
    └─ attribute_transform（1 ステップ）: リンク長・ギア比の調整
  実行フェーズ → Follower がその形でタスクを実行（最大 1000 ステップ）
```

### ターミナルログの流れ方

設計フェーズ（6 ステップ）は**スリープなしで全速**で走る。各ステップで `reload_sim_model()` が呼ばれ Choreonoid のロードログが一気に流れる。実行フェーズのみ指定 fps でスリープが入る。

```
[設計フェーズ] 6 回の reload → ログが一気に流れる
       ↓
[実行フェーズ] 最大 1000 ステップ、fps に合わせてゆっくり進む
       ↓
次のエピソードへ（VIEWER_EPISODES 回繰り返す）
```
choreonoidのmessageのばっーって出る間の時間は各エピソードの区切りの時間だよ

---

## ユースケース別スクリプト選択

| やりたいこと | スクリプト |
|-------------|-----------|
| 成功率・報酬を数値で確認したい | `eval_cnoid_numerical.py` |
| 動作を動画（mp4）で記録したい | `eval_cnoid_visual.py` |
| GUIでリアルタイムに動きを見たい | `eval_cnoid_viewer.py` |
| 学習曲線グラフを出したい | `plot_rewards.py` |
| 最終形態を URDF として保存したい | `save_morphology_urdf.py` |

---

## eval_cnoid_numerical.py — 数値で成功率を確認

**ユースケース**: 学習後に「本当にタスクが解けているか」を定量的に確かめたい。ヘッドレス（GUI不要）で動くので ssh 環境でも使える。

```bash
EVAL_RESTORE_DIR=single_run/pusher_cnoid EVAL_NUM_EPISODES=5 \
  USE_CHOREONOID=1 choreonoid --no-window --python scripts/eval_cnoid_numerical.py
```

**出力**: `{restore_dir}/eval/eval_numerical.png` に報酬の棒グラフを生成。

| 環境変数 | デフォルト | 説明 |
|---------|-----------|------|
| `EVAL_RESTORE_DIR` | （必須） | 学習ディレクトリ |
| `EVAL_EPOCH` | `best` | チェックポイント（`best` or エポック番号）|
| `EVAL_NUM_EPISODES` | `5` | 評価エピソード数 |

---

## eval_cnoid_visual.py — 動画として記録

**ユースケース**: 動作を mp4 で保存してあとから確認・共有したい。GUI不要で動く。

**注意**: Choreonoid の 3D 画面を録画するのではなく、ボディ座標を matplotlib でプロットした疑似可視化。ロボットは「青い球（関節）＋青い線（リンク）」、cube はオレンジの点として表示される。実際の capsule 形状・テクスチャは表示されない。

```bash
EVAL_RESTORE_DIR=single_run/pusher_cnoid \
EVAL_OUTPUT=single_run/pusher_cnoid/videos/eval_visual.mp4 \
  USE_CHOREONOID=1 choreonoid --no-window --python scripts/eval_cnoid_visual.py
```

**出力**: 指定パスに mp4 ファイルを生成（デフォルト: `{restore_dir}/videos/eval_visual.mp4`）。

| 環境変数 | デフォルト | 説明 |
|---------|-----------|------|
| `EVAL_RESTORE_DIR` | （必須） | 学習ディレクトリ |
| `EVAL_OUTPUT` | `{restore_dir}/videos/eval_visual.mp4` | 出力パス |
| `EVAL_FPS` | `20` | フレームレート |

---

## eval_cnoid_viewer.py — GUIでリアルタイム再生

**ユースケース**: Choreonoid の 3D ビューアで実際の動きをリアルタイムに確認したい。VirtualGL が必要（詳細は [choreonoid_gui_issue.md](../移行記録/choreonoid_gui_issue.md)）。

```bash
VIEWER_RESTORE_DIR=single_run/pusher_cnoid VIEWER_FPS=25 VIEWER_EPISODES=3 \
  vglrun choreonoid --python scripts/eval_cnoid_viewer.py
```

| 環境変数 | デフォルト | 説明 |
|---------|-----------|------|
| `VIEWER_RESTORE_DIR` | （必須） | 学習ディレクトリ |
| `VIEWER_FPS` | `25` | 再生フレームレート |
| `VIEWER_EPISODES` | `3` | エピソード数（0=無限ループ）|

---

## その他のスクリプト

```bash
# 学習曲線グラフ生成（plots/reward_plot.png に出力）
python3 scripts/plot_rewards.py single_run/pusher_cnoid

# 最終形態 URDF 保存（morphology/ に出力）
EVAL_RESTORE_DIR=single_run/pusher_cnoid \
  USE_CHOREONOID=1 choreonoid --no-window --python scripts/save_morphology_urdf.py
```

---

## 出力ディレクトリ構成

```
single_run/pusher_cnoid/
├── log/                     ← 学習ログ（log_train.txt）
├── models/                  ← チェックポイント（best.p, epoch_XXXX.p）
├── videos/                  ← 動画ファイル
│   └── eval_visual.mp4      ← eval_cnoid_visual.py 出力
├── eval/                    ← 数値評価グラフ
│   └── eval_numerical.png   ← eval_cnoid_numerical.py 出力
├── morphology/              ← 形態ファイル
│   ├── morphology_best.urdf ← save_morphology_urdf.py 出力
│   └── morphology_best.xml
└── plots/                   ← 学習グラフ
    └── reward_plot.png      ← plot_rewards.py 出力
```
