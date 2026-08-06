#!/usr/bin/env bash
# 2026-08-05: nsteps=1 実験（方針レビュー ⑤ の格上げ検証）。
#
#  baseline(nsteps=5) = single_run/tripo_arm_v2c_pusher     （cfg デフォルト 5、ep999）
#  ns2     (nsteps=2) = single_run/tripo_arm_v2c_pusher_ns2 （ep199 完走）
#  ns1     (nsteps=1) = ここで起動。論文の T^attr=1 相当（付録D Table 1）
#
# ns2 と「完全に同一設定」で skel_transform_nsteps だけ 2→1 に変える（単一変数）。
# 問い: 属性編集ステップを 1 まで落とすと「境界(±1.0)張り付き」が消えて内部解に
#       収束するか。→「境界収束」が nsteps 予算の産物か形態由来かを判定（第5章5.5.2）。
# 主指標: 収束した属性パラメータのうち ±1.0 に張り付いた個数（best と収束形態も記録）。
# 比較は全て ep200 時点（baseline は epoch_0200.p、ns2/ns1 は最終）で matched。
set -u
cd /userdir/StackelbergPPO

RUN=tripo_arm_v2c_pusher_ns1

# 二重起動ガード
if pgrep -f "hydra.run.dir=single_run/$RUN" >/dev/null 2>&1; then
  echo "[$(date '+%F %T')] $RUN は既に稼働中。起動をスキップ。"
  exit 0
fi

mkdir -p "single_run/$RUN"
nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
  --no-window --python scripts/choreonoid_train.py \
  cfg=pusher_tripo_v2c xml_name=tripo_arm_v2c fix_skeleton=true max_epoch_num=200 \
  num_threads=4 enable_wandb=false seed=0 skel_transform_nsteps=1 \
  hydra.run.dir="single_run/$RUN" \
  > "single_run/$RUN/stdout.log" 2>&1 &
echo "[$(date '+%F %T')] $RUN launched (PID $!, nsteps=1, 200ep)"
