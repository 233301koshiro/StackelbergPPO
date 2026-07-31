#!/bin/bash
# 空いた実行枠に転用再実験（K1_v2 → K2_v2 → I1_v2）を順番に連鎖起動するウォッチャー。
#
# 背景: GPU/CPU 競合を避けるため同時実行数を3に保つ運用（2026-07-22 の判断）。
# 現在実行中の3 run のうちどれか1つが終了するたびに、キューの先頭を起動して
# 常に3並列を維持する。キューが尽きたら監視を終了する。
#
# 使い方:
#   nohup bash scripts/auto_launch_next_transfer.sh > /dev/null 2>&1 &
#   disown
#   監視ログ: single_run/transfer_autostart_watcher.log

cd "$(dirname "$0")/.."

LOGF="single_run/transfer_autostart_watcher.log"

# 監視対象（hydra.run.dir の basename）。実行中に動的に入れ替える。
WATCH_RUNS=("rrbot_arm_pusher_K1_v2" "tripo_arm_v2c_pusher" "tripo_arm_v2c_tp")

# 起動待ちキュー（run 名と、対応する hydra オーバーライド）
QUEUE_NAMES=("K2_v2" "I1_v2")

launch_cmd() {
  local name="$1"
  case "$name" in
    K2_v2)
      nohup bash scripts/run_cnoid_train.sh \
        cfg=pusher xml_name=rrbot_arm fix_skeleton=true max_epoch_num=1000 \
        num_threads=4 enable_wandb=false seed=0 \
        +robot_param_scale=1 \
        +restore_dir=single_run/rrbot_arm_pusher_H1 \
        epoch=best morph_prior=true reset_epoch=true \
        hydra.run.dir=single_run/rrbot_arm_pusher_K2_v2 \
        >> "$LOGF" 2>&1 &
      disown
      echo "rrbot_arm_pusher_K2_v2"
      ;;
    I1_v2)
      nohup bash scripts/run_cnoid_train.sh \
        cfg=pusher xml_name=rrbot_arm fix_skeleton=true max_epoch_num=1000 \
        num_threads=4 enable_wandb=false seed=0 \
        +robot_param_scale=1 \
        +restore_dir=single_run/rrbot_arm_pusher_H1 \
        epoch=best reset_epoch=true \
        hydra.run.dir=single_run/rrbot_arm_pusher_I1_v2 \
        >> "$LOGF" 2>&1 &
      disown
      echo "rrbot_arm_pusher_I1_v2"
      ;;
  esac
}

echo "$(date '+%F %T') [watcher] started. watching: ${WATCH_RUNS[*]} / queue: ${QUEUE_NAMES[*]}" >> "$LOGF"

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
    echo "$(date '+%F %T') [watcher] ${finished_run} has finished. launching ${next_name}..." >> "$LOGF"
    new_run_dir=$(launch_cmd "$next_name")
    WATCH_RUNS[$finished_idx]="$new_run_dir"
    echo "$(date '+%F %T') [watcher] ${next_name} launched as ${new_run_dir}. now watching: ${WATCH_RUNS[*]} / remaining queue: ${QUEUE_NAMES[*]}" >> "$LOGF"
  fi

  sleep 60
done

echo "$(date '+%F %T') [watcher] queue empty. exiting." >> "$LOGF"
