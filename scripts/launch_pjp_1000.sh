#!/usr/bin/env bash
# 2026-08-05: pjp（PJ Pusher）の 1000ep 延長（進捗.md 延長候補 ◎）。
# 元 pjp（200ep, seed0）と同一 overrides のまま max_epoch_num=1000 にするだけ。
# 目的: mid/long とも ep200 で未収束（勾配残存）だったので、best 値・倍率を 1000ep で固める。
# usage: launch_pjp_1000.sh mid|long
set -u
cd /userdir/StackelbergPPO
FORM=${1:-}
case "$FORM" in
  mid)  XML=tripo_arm_v2c_pj_mid ;;
  long) XML=tripo_arm_v2c_pj_long ;;
  *) echo "usage: $0 mid|long"; exit 1 ;;
esac
RUN=tripo_pjp_${FORM}_1000
if pgrep -f "hydra.run.dir=single_run/$RUN" >/dev/null 2>&1; then
  echo "[$(date '+%F %T')] $RUN は既に稼働中。スキップ。"; exit 0
fi
mkdir -p "single_run/$RUN"
nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
  --no-window --python scripts/choreonoid_train.py \
  cfg=pusher_gearonly xml_name=$XML num_threads=4 max_epoch_num=1000 \
  enable_wandb=false fix_skeleton=true seed=0 +robot_param_scale=1 \
  +reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 \
  +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true \
  hydra.run.dir="single_run/$RUN" \
  > "single_run/$RUN/stdout.log" 2>&1 &
echo "[$(date '+%F %T')] $RUN launched (PID $!, 1000ep)"
