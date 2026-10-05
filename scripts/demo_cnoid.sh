#!/bin/bash
# ⭐ Choreonoid の画面でデモを再生する（大きな発表用。2026-10-05）
#
#   bash scripts/demo_cnoid.sh <タスク> [エピソード数]
#   タスク: reach | pusher | target_pusher | obstacle_avoid | obstacle_reach
#
# ⭐ 腕は**元のカラフルなメッシュ**で表示する（見た目だけ。物理はカプセルのまま。
#   khrylib/rl/envs/common/visual_mesh.py）。簡単なミーティングでは mp4（demo/color_*.mp4）を使う。
# ⭐ Reach 系では使わない箱を**見えなくし**（物理には残す）、目標の位置に**見た目だけの印**（オレンジの球）を浮かべる。
#
# ⚠️ 画面へ出す前に 1 回だけ、コンテナの中で `su ubuntu -s /bin/bash -c "DISPLAY=:1 xhost +SI:localuser:root"` を打つ
#   （コンテナの ubuntu はホストのデスクトップのユーザーと同じ uid 1000 なので通る。2026-10-05 確認）。
#   打っていないと `Authorization required` → Qt が core dump する。下で先に確かめて案内を出す。
# ⚠️ 学習と同時に走らせると重い（7/31 に GUI が落ちた記録がある）。発表の前に学習は止めておく。
set -eu
TASK=${1:?タスクを指定: reach | pusher | target_pusher | obstacle_avoid | obstacle_reach}
EPS=${2:-3}
# HIDE: 見えなくする物体（物理には残す）／MARK: 目標の印を出すか
# CAM: 最初のカメラ（視点x,y,z,注視点x,y,z）。発表中はマウスで自由に動かせる
case "$TASK" in
  reach)          RUN=e2e_a1v_reach;              HIDE=cube; MARK=1; CAM=1.5,-2.2,1.3,0.45,0,0.45 ;;   # 形を伸ばしてタスクに合わせた（co-design の狙いどおり）。台座 1.1° で立っている
  pusher)         RUN=e2e_a1v_gearonly_pusher_s2; HIDE=;     MARK=0; CAM=1.6,-2.8,1.4,0.6,0,0.4 ;;   # 台座が立ったまま・描いた形のまま。目標は無い
  target_pusher)  RUN=e2e_a1v_actuator_tp;        HIDE=;     MARK=1; CAM=1.7,-3.4,1.7,1.0,0,0.35 ;;
  obstacle_avoid) RUN=e2e_a1v_obspen_reach;       HIDE=cube; MARK=1; CAM=1.5,-2.2,1.3,0.45,0,0.45 ;;   # 柱を避ける（目標の 210 mm 手前で止まる）
  obstacle_reach) RUN=e2e_a1v_obspen_reach_s2;    HIDE=cube; MARK=1; CAM=1.5,-2.2,1.3,0.45,0,0.45 ;;   # 目標に届く（柱の中を通る）
  *) echo "知らないタスク: $TASK"; exit 1 ;;
esac
cd "$(dirname "$0")/.."
# ⭐ 画面につながるかを先に確かめる（2026-10-05。つながらないと Qt が core dump して分かりにくい）
export DISPLAY="${DISPLAY:-:1}"
if ! timeout 5 xhost >/dev/null 2>&1; then
  echo "⛔ 画面 $DISPLAY に表示する許可がありません（Authorization required）。次の 1 行を打ってから、もう一度実行してください:"
  echo "    su ubuntu -s /bin/bash -c \"DISPLAY=$DISPLAY xhost +SI:localuser:root\""
  echo "   （コンテナの ubuntu はホストのデスクトップのユーザーと同じ uid 1000 なので、コンテナの中から許可を足せる。"
  echo "    再起動・ログアウトで消える。取り消しは + を - に）"
  exit 1
fi
TARGET=
if [ "$MARK" = 1 ]; then
  # ⭐ 目標は record_arm_trace.py（mp4 用）と同じ読み方。⚠️ TP の目標は cfg の雛形にあり、run の config.yaml には無い
  TARGET=$(python3 scripts/demo_target.py "single_run/$RUN")
  echo "目標の印: $TARGET"
fi
env \
  CNOID_HIDE_BODIES="$HIDE" CNOID_TARGET_MARK="$TARGET" VIEWER_CAMERA="$CAM" \
  VIEWER_RESTORE_DIR=single_run/$RUN VIEWER_EPOCH=best VIEWER_FPS=25 VIEWER_EPISODES=$EPS \
  CNOID_VISUAL_MESHES=data/test/A1/meshes CNOID_VISUAL_GLB=data/test/A1/3D/A1.glb \
  USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
  /choreonoid_ws/install/bin/choreonoid --python scripts/eval_cnoid_viewer.py
