#!/bin/bash
# 現在実行中の3 run（K1_v2/K2_v2/I1_v2）のいずれかが終了するたびに、
# 次のキューを順番に自動起動し、常に3並列を維持するウォッチャー。
#
# キュー:
#   1. v2b_reach       — cube dampingのみ修正・可動域は旧のまま。TA/v2c間の交絡切り分け
#   2. M1_lenonly_1000 (seed0) — 長さのみablationの1000epoch再実験（200epでは判定に不十分）
#   3. M1_lenonly_1000 (seed1)
#   4. M2b_gearonly_1000 (seed0) — ギア比のみablationの1000epoch再実験（200ep時点で未収束だった）
#   5. M2b_gearonly_1000 (seed1)
#
# 使い方:
#   nohup bash scripts/auto_launch_queue2.sh > /dev/null 2>&1 &
#   disown
#   監視ログ: single_run/transfer_autostart_watcher.log

cd "$(dirname "$0")/.."

LOGF="single_run/transfer_autostart_watcher.log"
WATCH_RUNS=("rrbot_arm_pusher_K1_v2" "rrbot_arm_pusher_K2_v2" "rrbot_arm_pusher_I1_v2")
QUEUE_NAMES=("v2b_reach" "M1_1000_s0" "M1_1000_s1" "M2b_1000_s0" "M2b_1000_s1")

launch_cmd() {
  local name="$1"
  case "$name" in
    v2b_reach)
      nohup bash scripts/run_cnoid_train.sh \
        cfg=pusher_tripo_v2b xml_name=tripo_arm_v2b fix_skeleton=true max_epoch_num=1000 \
        num_threads=4 enable_wandb=false seed=0 \
        +reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 \
        +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 \
        +env_specs.check_init_contact=false \
        hydra.run.dir=single_run/tripo_arm_v2b_reach \
        >> "$LOGF" 2>&1 &
      disown
      echo "tripo_arm_v2b_reach"
      ;;
    M1_1000_s0)
      nohup bash scripts/run_cnoid_train.sh \
        cfg=pusher_lenonly xml_name=rrbot_arm num_threads=4 max_epoch_num=1000 \
        enable_wandb=false fix_skeleton=true seed=0 \
        +robot_param_scale=1 +reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 \
        +reward_specs.init_contact_penalty=50 \
        hydra.run.dir=single_run/rrbot_arm_pusher_M1_lenonly_1000 \
        >> "$LOGF" 2>&1 &
      disown
      echo "rrbot_arm_pusher_M1_lenonly_1000"
      ;;
    M1_1000_s1)
      nohup bash scripts/run_cnoid_train.sh \
        cfg=pusher_lenonly xml_name=rrbot_arm num_threads=4 max_epoch_num=1000 \
        enable_wandb=false fix_skeleton=true seed=1 \
        +robot_param_scale=1 +reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 \
        +reward_specs.init_contact_penalty=50 \
        hydra.run.dir=single_run/rrbot_arm_pusher_M1_lenonly_1000_s2 \
        >> "$LOGF" 2>&1 &
      disown
      echo "rrbot_arm_pusher_M1_lenonly_1000_s2"
      ;;
    M2b_1000_s0)
      nohup bash scripts/run_cnoid_train.sh \
        cfg=pusher_gearonly xml_name=rrbot_arm num_threads=4 max_epoch_num=1000 \
        enable_wandb=false fix_skeleton=true seed=0 \
        +robot_param_scale=1 +reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 \
        +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true \
        hydra.run.dir=single_run/rrbot_arm_pusher_M2b_gearonly_1000 \
        >> "$LOGF" 2>&1 &
      disown
      echo "rrbot_arm_pusher_M2b_gearonly_1000"
      ;;
    M2b_1000_s1)
      nohup bash scripts/run_cnoid_train.sh \
        cfg=pusher_gearonly xml_name=rrbot_arm num_threads=4 max_epoch_num=1000 \
        enable_wandb=false fix_skeleton=true seed=1 \
        +robot_param_scale=1 +reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 \
        +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true \
        hydra.run.dir=single_run/rrbot_arm_pusher_M2b_gearonly_1000_s2 \
        >> "$LOGF" 2>&1 &
      disown
      echo "rrbot_arm_pusher_M2b_gearonly_1000_s2"
      ;;
  esac
}

echo "$(date '+%F %T') [watcher-q2] started. watching: ${WATCH_RUNS[*]} / queue: ${QUEUE_NAMES[*]}" >> "$LOGF"

while [ "${#QUEUE_NAMES[@]}" -gt 0 ]; do
  finished_idx=-1
  for i in "${!WATCH_RUNS[@]}"; do
    r="${WATCH_RUNS[$i]}"
    if ! pgrep -f "hydra.run.dir=single_run/${r}\$" > /dev/null; then
      finished_idx=$i
      break
    fi
  done

  if [ "$finished_idx" -ge 0 ]; then
    finished_run="${WATCH_RUNS[$finished_idx]}"
    next_name="${QUEUE_NAMES[0]}"
    QUEUE_NAMES=("${QUEUE_NAMES[@]:1}")
    echo "$(date '+%F %T') [watcher-q2] ${finished_run} has finished. launching ${next_name}..." >> "$LOGF"
    new_run_dir=$(launch_cmd "$next_name")
    WATCH_RUNS[$finished_idx]="$new_run_dir"
    echo "$(date '+%F %T') [watcher-q2] ${next_name} launched as ${new_run_dir}. now watching: ${WATCH_RUNS[*]} / remaining queue: ${QUEUE_NAMES[*]}" >> "$LOGF"
  fi

  sleep 60
done

echo "$(date '+%F %T') [watcher-q2] queue empty. exiting." >> "$LOGF"
