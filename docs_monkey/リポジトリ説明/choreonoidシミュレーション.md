# Choreonoid シミュレーション（猿でもわかる版）

---

## Choreonoid とは何か

物理シミュレーターの名前です。普通の AI 実験では MuJoCo（商用）を使いますが、
このプロジェクトでは **Choreonoid（無料・オープンソース）** に乗り換えています。

物理計算の中身は C++ で動いており、Python から直接呼ぶことができません。
だから学習スクリプトは Choreonoid 経由で起動する必要があります。

```bash
# NG: これだとシミュレーターが動かない
python3 scripts/choreonoid_train.py

# OK: Choreonoid 経由で起動する
choreonoid --no-window --python scripts/choreonoid_train.py
```

---

## 何が何をしているか

```
choreonoid プロセス（C++ アプリ）
└── Python スクリプトを内部で実行
      └── train.py
            └── BodyGenAgent（学習ループ）
                  └── PusherEnv（タスクルール・報酬計算）
                        └── ChoreonoidEnv（シミュレーター操作の窓口）
                              └── Choreonoid C++ 物理エンジン
```

**あなたがコードを読むときに触る層は「PusherEnv より上」です。**
ChoreonoidEnv より下は「物理の黒箱」として扱って大丈夫です。

---

## 1ステップで何が起きるか

```
① AI ネットワーク → 各関節に「この強さで動け」（トルク）を出力

② Choreonoid が物理計算を 4 回進める（1ステップ = 0.04秒の物理時間）

③ 関節の角度・速度・ロボット各部の位置を読み取る

④ 報酬を計算して AI に返す（次のステップへ）
```

---

## 形態が変わるときは何が起きるか

エピソード開始時に「形態変換フェーズ」があり、ロボットの形が変わります。
形が変わるたびにシミュレーターへ再ロードが走ります。

```
新しい形態の定義（XML 文字列）
    ↓
Choreonoid .body フォーマットに変換（一時ファイルに書き出し）
    ↓
Choreonoid がそのファイルを読み込む（シミュレーター再起動）
    ↓
新しい形態でシミュレーション続行
```

1エピソードで最大 6 回（骨格変換 5 回 + 属性変換 1 回）この再ロードが走ります。
そのため学習ログに「ばーっと」大量のメッセージが出る区切りがあります。
**あれが形態変換フェーズです。** 正常な動作なので驚かないでください。

---

## リセット（エピソードをやり直す）

```python
# Choreonoid 内部でやっていること
sim.stopSimulation()
ロボットを初期姿勢に戻す
sim.startSimulation()
```

ロボットを「最初の状態に戻して」新しいエピソードを始めます。

---

## 覚えておくべきこと

| ポイント | 内容 |
|---|---|
| 起動コマンド | 必ず `choreonoid --no-window --python ...` |
| `--no-window` | GUI ウィンドウを出さずにヘッドレスで動かすオプション |
| 1 RL ステップ | 0.04 秒の物理時間（4 サブステップ × 0.01 秒）|
| 形態変換時 | ログが一気に流れる → 正常 |
| 学習後の残留 | `pgrep -a choreonoid` で確認・`pkill choreonoid` で終了 |

---

## GUI で動作を確認したい場合

```bash
# VirtualGL が必要（ssh + X転送 or ローカルで実行）
VIEWER_RESTORE_DIR=single_run/pusher_cnoid VIEWER_FPS=25 VIEWER_EPISODES=3 \
  vglrun choreonoid --python scripts/eval_cnoid_viewer.py
```

GUI なしで動画を出したいときは `eval_cnoid_visual.py` を使ってください
（詳細: [評価スクリプト.md](評価スクリプト.md)）。
