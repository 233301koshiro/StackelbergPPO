#!/bin/bash
# ⭐ Choreonoid の画面でデモを再生する（大きな発表用。2026-10-05）
#
#   bash scripts/demo_cnoid.sh <タスク> [エピソード数]
#   ⭐ 録画用: 立ち上がってから START_DELAY 秒（既定 10）カウントダウンしてから再生する。
#      形態が変わるたびに DESIGN_PAUSE 秒（既定 1）止める。例: START_DELAY=20 bash scripts/demo_cnoid.sh reach
#   ⭐ ターミナルと Choreonoid のメッセージ欄に「第 N 話／形態変化 k 回目（リンク長）／実行に移ります
#      （描いた形 → 学習後）／実行中（先端と目標の距離・箱の移動）」を出す
#   タスク: reach | pusher | target_pusher | obstacle_avoid | obstacle_reach | hockey
#   ⭐ hockey（2026-10-09）: 制御コスト 0 の `hockey_bank9`（seed0、評価の動きで 3 話中 2 話ゴール。系譜 9-229）。
#     学習と同じ壁の反射・腕ブロックを有効にする。頭の話は当てにならない（Bug 47）ので既定で 3 話流す。ゴールした話を使うこと
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
TASK=${1:?タスクを指定: reach | pusher | target_pusher | obstacle_avoid | obstacle_reach | hockey}
EPS=${2:-1}   # ⭐ 既定は 1 話（録画用。2026-10-05。複数回は第 2 引数で）
# 既定は縦型 A1 のメッシュ。タスクごとに上書きする
VMESH=data/test/A1/meshes; VGLB=data/test/A1/3D/A1.glb; EXTRA=
# HIDE: 見えなくする物体（物理には残す）／MARK: 目標の印を出すか
# CAM: 最初のカメラ（視点x,y,z,注視点x,y,z）。発表中はマウスで自由に動かせる
case "$TASK" in
  reach)          RUN=e2e_a1v_reach;              HIDE=cube; MARK=1; CAM=1.5,-2.2,1.3,0.45,0,0.45 ;;   # 形を伸ばしてタスクに合わせた（co-design の狙いどおり）。台座 1.1° で立っている
  pusher)         RUN=e2e_a1v_gearonly_pusher_s2; HIDE=;     MARK=0; CAM=1.6,-2.8,1.4,0.6,0,0.4 ;;   # 台座が立ったまま・描いた形のまま。目標は無い
  target_pusher)  RUN=e2e_a1v_actuator_tp;        HIDE=;     MARK=1; CAM=1.7,-3.4,1.7,1.0,0,0.35 ;;
  obstacle_avoid) RUN=e2e_a1v_obspen_reach;       HIDE=cube; MARK=1; CAM=1.5,-2.2,1.3,0.45,0,0.45 ;;   # 柱を避ける（目標の 210 mm 手前で止まる）
  obstacle_reach) RUN=e2e_a1v_obspen_reach_s2;    HIDE=cube; MARK=1; CAM=1.5,-2.2,1.3,0.45,0,0.45 ;;   # 目標に届く（柱の中を通る）
  hockey)         RUN=hockey_bank9;               HIDE=;     MARK=1; CAM=0.9,-2.0,2.2,0.9,0,0.2      # 星＝ゴール口の中心（1.55, 0）
                  VMESH=data/test/hockey/meshes; VGLB=data/test/hockey/3D/hockey.glb
                  EXTRA="HOCKEY_WALL_RESTITUTION=0.75 HOCKEY_ARM_BLOCK=1"; EPS=${2:-3} ;;
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
env FONTCONFIG_FILE="$PWD/config/fontconfig_ja.conf" \
  CNOID_HIDE_BODIES="$HIDE" CNOID_TARGET_MARK="$TARGET" VIEWER_CAMERA="$CAM" \
  VIEWER_START_DELAY="${START_DELAY:-10}" VIEWER_DESIGN_PAUSE="${DESIGN_PAUSE:-1}" \
  VIEWER_RESTORE_DIR=single_run/$RUN VIEWER_EPOCH=best VIEWER_FPS=25 VIEWER_EPISODES=$EPS \
  CNOID_VISUAL_MESHES=$VMESH CNOID_VISUAL_GLB=$VGLB $EXTRA \
  USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
  /choreonoid_ws/install/bin/choreonoid --python scripts/eval_cnoid_viewer.py &
# ⭐ Ctrl+C で止まるようにする（2026-10-05）。⛔ Choreonoid は Ctrl+C（SIGINT）を受け付けず、ターミナルを握ったまま
#   動き続けた（ユーザーが ^C を何度押しても戻らなかった）。裏で起動し、Ctrl+C を受けたらこのスクリプトが強制終了させる
CPID=$!
trap 'echo; echo "[デモ] 停止します"; kill -9 $CPID 2>/dev/null' INT TERM
wait $CPID 2>/dev/null || true
