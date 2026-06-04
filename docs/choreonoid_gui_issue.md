# Choreonoid GUI が akita_sp コンテナで起動できない問題

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

## 環境

| 項目 | 内容 |
|------|------|
| Docker イメージ | `akita_sp`（`irsl_system:noetic` 継承）|
| ホスト OS | Ubuntu |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| X ディスプレイ | `:1`（HDMI 接続、4K モニター）|

## 原因の診断

### コンテナ内の libGL

```
/usr/lib/x86_64-linux-gnu/libGL.so.1 → libGL.so.1.7.0（Mesa ソフトウェアレンダラー）
```

コンテナ内の `libGL` は **Mesa**（ソフトウェア実装）を指している。

### X ディスプレイの GLX

```
$ DISPLAY=:1 xdpyinfo | grep GL
    GLX        ← GLX 拡張あり
    NV-GLX     ← NVIDIA の GLX 拡張あり
```

X サーバー（ホスト側）は **NVIDIA GLX** をサポートしている。

### 問題の構図

```
コンテナ内                         ホスト側 X サーバー
┌─────────────────────┐           ┌─────────────────────┐
│  Choreonoid          │           │  X11 (:1)            │
│  （QOpenGLWidget 使用）│──GLX──→  │  NVIDIA NV-GLX       │
│                      │           │  GPU: RTX 5060 Ti    │
│  libGL.so            │           └─────────────────────┘
│   ↓ Mesa (ソフトウェア)│
│  NVIDIA GL と不一致  │
└─────────────────────┘
```

コンテナ内の Mesa libGL と ホスト側の NVIDIA GLX の組み合わせが不一致のため、
`QOpenGLWidget` の初期化に失敗してクラッシュする。

`--no-window` では `QOpenGLWidget` を使わないため影響を受けない。

## 解決策

### 案1: VirtualGL を使う（推奨）

VirtualGL はコンテナ内の OpenGL 呼び出しをホストの GPU に転送する仕組み。

ホスト側の設定:
```bash
# ホスト側で VirtualGL サーバーを設定（要管理者権限）
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

### 案2: コンテナに NVIDIA OpenGL ライブラリを追加

`Dockerfile.add_akita_sp` に追加:
```dockerfile
# NVIDIA の libGL を優先させる
RUN apt-get install -y libnvidia-gl-<version>
```

または:
```dockerfile
# EGL（ディスプレイレス OpenGL）を使う設定
ENV __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
```

### 案3: irsl_system イメージ側で対応

`irsl_system` イメージ自体が VirtualGL 対応の場合、Choreonoid GUI が `VGL_DISPLAY` 経由で
正常動作するはず。現在の `akita_sp` で `VGL_DISPLAY` が未設定のため機能していない可能性がある。

```bash
# 設定確認
echo $VGL_DISPLAY  # → 未設定（これが原因の可能性）
```

## 現在の回避策

GUI 表示の代わりに matplotlib で mp4 動画を生成する方法が動作確認済み。

```bash
USE_CHOREONOID=1 \
  choreonoid --no-window --python scripts/eval_cnoid_visual.py -- \
  --restore_dir single_run/pusher_cnoid \
  --output single_run/pusher_cnoid/videos/best_policy.mp4
```
