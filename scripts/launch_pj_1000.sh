#!/usr/bin/env bash
# 2026-08-06: pj（PJ Reach）の 1000ep 延長。Pusher 版 launch_pjp_1000.sh の Reach 対応版。
#
# 目的（第4章 4.3.3.3 限界1 を閉じる）:
#   マトリクス判定の Reach 側は、末尾10epの伸びを実測すると
#     中間  : -3.43 → -3.42（+0.3%）   = 収束
#     長腕s0: -15.26 → -12.48（+18.2%） = **未収束**
#     長腕s1: -10.27 → -10.27（ep120以降不変） = 収束
#   と、**負けている長腕の seed0 だけが未収束**である。順位（中間 > 長腕）は seed1 の長腕が
#   -10.27 で収束していることに支えられているが、seed0 単独では収束を確認できていない。
#   1000ep へ延ばして長腕の到達水準を確定させ、順位が覆らないことを示す。
#
# ⚠️ **元の 200ep run の続きにはしない**（restore_dir を使わない）。
#   4.3.4.2 で確認した通り、学習率スケジュール等が max_epoch_num に比例するため、
#   「1000ep 設定で最初から回した run」と「200ep run の途中経過」は別物である。
#   Pusher 側の launch_pjp_1000.sh も同じ方針（ゼロスタート）なので揃える。
#
# ⚠️ 中間スケール側も同時に回す。長腕だけ 1000ep にすると打ち切り点が条件間で不公平になり、
#   M系 ablation で実際に起きた過小評価（第5章 5.4）と同じ罠を踏む。
#
# usage: launch_pj_1000.sh mid|long
set -u
cd /userdir/StackelbergPPO
FORM=${1:-}
case "$FORM" in
  mid)  XML=tripo_arm_v2c_pj_mid ;;
  long) XML=tripo_arm_v2c_pj_long ;;
  *) echo "usage: $0 mid|long"; exit 1 ;;
esac
RUN=tripo_pj_${FORM}_1000
if pgrep -f "hydra.run.dir=single_run/$RUN" >/dev/null 2>&1; then
  echo "[$(date '+%F %T')] $RUN は既に稼働中。スキップ。"; exit 0
fi
mkdir -p "single_run/$RUN"
# overrides は元の tripo_pj_{mid,long}（200ep）と max_epoch_num 以外すべて同一。
# Reach なので接触処理は check_init_contact=false のみ（Pusher 側とは異なる）。
nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
  --no-window --python scripts/choreonoid_train.py \
  cfg=pusher_gearonly xml_name=$XML num_threads=4 max_epoch_num=1000 \
  enable_wandb=false fix_skeleton=true seed=0 +robot_param_scale=1 \
  +reward_specs.use_reach=true +reward_specs.target_x=0.8 \
  +reward_specs.target_y=0.0 +reward_specs.target_z=0.15 \
  +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false \
  hydra.run.dir="single_run/$RUN" \
  > "single_run/$RUN/stdout.log" 2>&1 &
echo "[$(date '+%F %T')] $RUN launched (PID $!, 1000ep)"
