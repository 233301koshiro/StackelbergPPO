#!/bin/bash
# ⭐ Choreonoid の画面でデモを再生する（大きな発表用。2026-10-05）
#
#   bash scripts/demo_cnoid.sh <タスク> [エピソード数]
#   タスク: reach | pusher | target_pusher | obstacle_avoid | obstacle_reach
#
# ⭐ 腕は**元のカラフルなメッシュ**で表示する（見た目だけ。物理はカプセルのまま。
#   khrylib/rl/envs/common/visual_mesh.py）。簡単なミーティングでは mp4（demo/color_*.mp4）を使う。
#
# ⚠️ 画面へ出す前に、**ホスト側（ubuntu ユーザー）で 1 回だけ** `xhost +SI:localuser:root` を打つ
#   （打っていないと `unable to open display`。評価スクリプト.md の eval_cnoid_viewer.py の節）。
# ⚠️ 学習と同時に走らせると重い（7/31 に GUI が落ちた記録がある）。発表の前に学習は止めておく。
set -eu
TASK=${1:?タスクを指定: reach | pusher | target_pusher | obstacle_avoid | obstacle_reach}
EPS=${2:-3}
case "$TASK" in
  reach)          RUN=e2e_a1v_reach ;;
  pusher)         RUN=e2e_a1v_gearonly_pusher_s2 ;;   # 台座が立ったまま・描いた形のまま
  target_pusher)  RUN=e2e_a1v_actuator_tp ;;
  obstacle_avoid) RUN=e2e_a1v_obspen_reach ;;         # 柱を避ける（目標の 210 mm 手前で止まる）
  obstacle_reach) RUN=e2e_a1v_obspen_reach_s2 ;;      # 目標に届く（柱の中を通る）
  *) echo "知らないタスク: $TASK"; exit 1 ;;
esac
cd "$(dirname "$0")/.."
env DISPLAY="${DISPLAY:-:1}" \
  VIEWER_RESTORE_DIR=single_run/$RUN VIEWER_EPOCH=best VIEWER_FPS=25 VIEWER_EPISODES=$EPS \
  CNOID_VISUAL_MESHES=data/test/A1/meshes CNOID_VISUAL_GLB=data/test/A1/3D/A1.glb \
  USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
  /choreonoid_ws/install/bin/choreonoid --python scripts/eval_cnoid_viewer.py
