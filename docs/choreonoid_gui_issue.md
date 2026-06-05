# Choreonoid GUI が akita_sp コンテナで起動できない問題

---

## 用語解説（前提知識）

この問題を理解するために必要な概念を先に説明する。

### 3D 描画の仕組み

Choreonoid のような 3D アプリケーションは、画面にロボットを描くために
**GPU（グラフィックカード）** を使う。CPU だけでは 3D 描画は遅すぎるため。

```
アプリ（Choreonoid）
  ↓ 「この頂点を描いて」「このテクスチャを貼って」などの命令
GPU（RTX 5060 Ti）
  ↓
モニターに 3D 映像が表示される
```

### OpenGL とは

GPU に 3D 描画命令を出すための**標準的な API（命令セット）**。
「この三角形を描け」「この色を塗れ」といった命令の規格。
ほぼすべての 3D アプリケーション（ゲーム・シミュレータ等）が使っている。

### libGL とは

OpenGL の命令を GPU に伝える**橋渡し役のライブラリファイル（.so ファイル）**。

```
アプリ
  ↓ OpenGL 命令（例: glDrawArrays()）
libGL.so ← ここが橋渡し役
  ↓
GPU ドライバ → GPU
```

libGL の実装には2種類ある：

| 実装 | 誰が作った？ | GPU を使う？ | 速度 |
|------|------------|------------|------|
| **NVIDIA libGL** | NVIDIA | ✅ RTX 5060 Ti を使う | 速い |
| **Mesa libGL** | オープンソース | ❌ CPU でソフトウェア計算 | 遅い |

今回のコンテナ内では Mesa libGL が入っている。

### X ディスプレイ・X サーバーとは

Linux でウィンドウを表示する仕組みを **X Window System（X11）** という。

```
アプリ（「ウィンドウを開きたい」）
  ↓  DISPLAY=:1 という環境変数で接続先を指定
X サーバー（ウィンドウを実際に画面に出す係）
  ↓
モニター
```

`:1` はポート番号のようなもので、「1番のディスプレイ」を意味する。
今回の環境では `:1` に HDMI 接続された実物の 4K モニターが繋がっている。

### GLX とは

X サーバーが OpenGL をサポートするための**拡張機能**。
「X 上でウィンドウに 3D を描く」ためのプロトコル。

```
アプリ
  ↓ GLX で「OpenGL の描画結果をこのウィンドウに表示して」
X サーバー（ GLX 対応）
  ↓
GPU で 3D 描画 → モニターに表示
```

今回の X サーバー（`:1`）には **NV-GLX**（NVIDIA 版 GLX）が入っており、
NVIDIA GPU の OpenGL 命令を受け付ける準備はできている。

### QOpenGLWidget とは

Choreonoid が使っている GUI フレームワーク **Qt** の部品のひとつ。
「ウィンドウの中に OpenGL で描画した 3D 映像を表示するための枠」。

```
Choreonoid ウィンドウ
  ┌──────────────────────────────────┐
  │ メニューバー                      │
  ├──────────────────────────────────┤
  │                                  │
  │  QOpenGLWidget ← ここで 3D 描画  │
  │  （ロボットのシーンビュー）        │
  │                                  │
  └──────────────────────────────────┘
```

`QOpenGLWidget` は内部で libGL を呼び出して GPU に描画命令を送る。

### Mesa とは

libGL のオープンソース実装。本来は NVIDIA GPU がない環境（仮想マシン等）用で、
GPU の代わりに CPU で OpenGL 命令を処理する。

```
Mesa（ソフトウェアレンダラー）
  CPU で 3D 計算（遅い）→ GPU は使わない
```

---

## 現象

`choreonoid --python script.py`（GUI あり）を実行すると、
プラグインの初期化途中でクラッシュし終了する（終了コード 1）。

`choreonoid --no-window --python script.py` は正常に動作する。

```
# 失敗するコマンド
DISPLAY=:1 choreonoid --python scripts/eval_cnoid_viewer.py

# 正常に動くコマンド（--no-window）
DISPLAY=:1 choreonoid --no-window --python scripts/choreonoid_train.py
```

---

## 環境

| 項目 | 内容 |
|------|------|
| Docker イメージ | `akita_sp`（`irsl_system:noetic` 継承）|
| ホスト OS | Ubuntu |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| X ディスプレイ | `:1`（HDMI 接続、4K モニター）|

---

## 原因の診断

### コンテナ内の libGL（確認コマンド: `ldconfig -p | grep libGL`）

```
/usr/lib/x86_64-linux-gnu/libGL.so.1 → libGL.so.1.7.0（Mesa ソフトウェアレンダラー）
```

コンテナ内の `libGL` は **Mesa**（CPU 計算版）を指している。
NVIDIA の高速な libGL ではない。

### X ディスプレイの GLX 対応状況（確認コマンド: `xdpyinfo`）

```
$ DISPLAY=:1 xdpyinfo | grep GL
    GLX        ← GLX 拡張あり
    NV-GLX     ← NVIDIA の GLX 拡張あり
```

X サーバー（ホスト側）は **NVIDIA GLX** をサポートしている。
つまりホスト側は「NVIDIA GPU で 3D 描画できる状態」になっている。

### 問題の構図

```
コンテナ内                              ホスト側 X サーバー
┌─────────────────────────┐           ┌──────────────────────┐
│  Choreonoid              │           │  X11 (:1)             │
│  ↓                       │           │  GLX: NVIDIA 対応     │
│  QOpenGLWidget           │──GLX──→  │  GPU: RTX 5060 Ti     │
│  ↓                       │           └──────────────────────┘
│  libGL.so を呼ぶ         │
│  ↓ Mesa（CPU版）         │
│  "NVIDIA GLX を使おうと  │
│   しているのに Mesa で   │
│   答えようとしている"→   │
│  不一致でクラッシュ      │
└─────────────────────────┘
```

**噛み合っていない部分:**
- ホスト X サーバー側：「NVIDIA GPU で描画するための GLX を持っている」
- コンテナ内 libGL 側：「Mesa（CPU）で描画しようとしている」

この2つのプロトコルが一致しないため `QOpenGLWidget` の初期化に失敗する。

**`--no-window` が動く理由:**
`--no-window` では Choreonoid がウィンドウを作らないため `QOpenGLWidget` も使わない。
OpenGL が不要になるのでクラッシュしない。

---

## 解決策

### 案1: VirtualGL を使う（推奨）

**VirtualGL とは**: コンテナ内の OpenGL 命令を「そのまま GPU には渡さず、
一度ホスト側に転送して NVIDIA GPU で処理してから結果を返す」という橋渡しツール。

```
コンテナ内 Choreonoid
  ↓ OpenGL 命令
VirtualGL（vglrun）← ここが橋渡し
  ↓ ホスト側の NVIDIA GPU に命令を転送
RTX 5060 Ti で 3D 描画
  ↓ 結果をコンテナに返す
ウィンドウに表示
```

ホスト側の設定（要管理者権限）:
```bash
apt install virtualgl
vglserver_config
```

コンテナ起動時:
```bash
docker run ... -e VGL_DISPLAY=:0 ...
```

コンテナ内で使う:
```bash
vglrun choreonoid --python scripts/eval_cnoid_viewer.py
```

`VGL_DISPLAY` が未設定（現状）なのが原因の可能性が高い。

### 案2: コンテナに NVIDIA OpenGL ライブラリを追加

Mesa の代わりに NVIDIA 版 libGL をコンテナ内に入れる。
ただしコンテナを再ビルドする必要があるため、現在の学習を止めてから実施する。

`Dockerfile.add_akita_sp` に追加:
```dockerfile
RUN apt-get install -y libnvidia-gl-<version>
```

### 案3: irsl_system イメージ側で対応

`irsl_system` が VirtualGL 対応済みなら、コンテナ起動時に `VGL_DISPLAY` を
渡すだけで解決する可能性がある。

```bash
# コンテナ起動時に追加
-e VGL_DISPLAY=:0
```

---

## 現在の回避策

GUI 表示の代わりに matplotlib で mp4 動画を生成する方法が動作確認済み。

```bash
USE_CHOREONOID=1 \
  choreonoid --no-window --python scripts/eval_cnoid_visual.py -- \
  --restore_dir single_run/pusher_cnoid \
  --output single_run/pusher_cnoid/videos/best_policy.mp4
```
