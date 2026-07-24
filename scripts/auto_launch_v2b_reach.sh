#!/bin/bash
# 転用再実験（K1_v2/K2_v2/I1_v2）のいずれかが終了し次第、
# v2b_reach（cube damping のみ修正・関節可動域は旧のまま）を自動起動する。
#
# 背景: tripo_v2c_reach（可動域±180°化 + cube damping修正）の性能低下調査の一環。
# TA（旧tripo_arm_v2、可動域±60/90/45°、cube damping=2.0バグ）は best_R=-1.34 で
# 良好に収束したが、v2c は best_R=-8.27 で劣化。cube damping と可動域の2変数が
# TA→v2c で同時に変わっているため交絡がある。v2b（cube dampingのみ修正・可動域は
# 旧のまま）を1000epochまで走らせて一変数ずつ切り分ける（2026-07-24）。
#
# GPU/CPU 競合を避けるため同時実行数を3に保つ運用（2026-07-22 の判断）。
#
# 使い方:
#   nohup bash scripts/auto_launch_v2b_reach.sh > /dev/null 2>&1 &
#   disown
#   監視ログ: single_run/transfer_autostart_watcher.log

cd "$(dirname "$0")/.."

LOGF="single_run/transfer_autostart_watcher.log"
WATCH_RUNS=("rrbot_arm_pusher_K1_v2" "rrbot_arm_pusher_K2_v2" "rrbot_arm_pusher_I1_v2")

echo "$(date '+%F %T') [watcher-v2b] started. watching: ${WATCH_RUNS[*]}" >> "$LOGF"

while true; do
  for r in "${WATCH_RUNS[@]}"; do
    if ! pgrep -f "hydra.run.dir=single_run/${r}\$" > /dev/null; then
      echo "$(date '+%F %T') [watcher-v2b] ${r} has finished. launching v2b_reach..." >> "$LOGF"
      nohup bash scripts/run_cnoid_train.sh \
        cfg=pusher_tripo_v2b xml_name=tripo_arm_v2b fix_skeleton=true max_epoch_num=1000 \
        num_threads=4 enable_wandb=false seed=0 \
        +reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 \
        +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 \
        +env_specs.check_init_contact=false \
        hydra.run.dir=single_run/tripo_v2b_reach \
        >> "$LOGF" 2>&1 &
      disown
      echo "$(date '+%F %T') [watcher-v2b] v2b_reach launched (pid=$!). watcher exiting." >> "$LOGF"
      exit 0
    fi
  done
  sleep 60
done
