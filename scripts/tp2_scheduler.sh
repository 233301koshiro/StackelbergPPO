#!/usr/bin/env bash
# TP2 スケジューラ（2026-07-13 仕込み）
# L1/L1_s2 の完走（ep999）を待ち、確定分のみ自動実行する:
#   1. L1/L1_s2 の機械的完走処理: 形態推移ダンプ（gear 込み）+ hover 実測 → ファイル保存（md は触らない）
#   2. メモリ確認 → TP2 / TP2_s2 起動（target_pusher 1000ep × 2 seed、I1 から control_prior = TP1 と同一方式）
# 判定（G1 の Reach 側）は人間/エージェントがダンプを見て行う。
set -u
cd /userdir/StackelbergPPO
LOG=single_run/tp2_scheduler.log
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

last_ep() {
  grep "T_sample" "single_run/$1/log/log_train.txt" 2>/dev/null \
    | tail -1 | grep -oE "\] [0-9]+" | tr -dc '0-9'
}

say "scheduler start: waiting for L1/L1_s2 (epoch 999)"
while :; do
  e1=$(last_ep rrbot_arm_reach_L1); e2=$(last_ep rrbot_arm_reach_L1_s2)
  [ "${e1:--1}" -ge 999 ] && [ "${e2:--1}" -ge 999 ] && break
  if [ -n "$(find single_run/rrbot_arm_reach_L1/log/log_train.txt -mmin -180 2>/dev/null)" ] \
     || [ -n "$(find single_run/rrbot_arm_reach_L1_s2/log/log_train.txt -mmin -180 2>/dev/null)" ]; then
    sleep 600
  else
    say "WARNING: L1/L1_s2 logs stale >3h (L1=ep${e1:-?} L1_s2=ep${e2:-?}). proceeding anyway"
    break
  fi
done
say "L1/L1_s2 done or stale (L1=ep$(last_ep rrbot_arm_reach_L1) L1_s2=ep$(last_ep rrbot_arm_reach_L1_s2))"

# ── 1. 機械的完走処理（G1 Reach 側の判定材料）─────────────────────────
for run in rrbot_arm_reach_L1 rrbot_arm_reach_L1_s2; do
  timeout -s KILL 2400 env EVAL_RESTORE_DIR="single_run/$run" EVAL_SAMPLE_EVERY=50 \
    USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/visualize_morph_changes.py \
    > "single_run/comparison/${run}_morph_final.txt" 2>&1
  say "morph dump saved: ${run}_morph_final.txt"
  timeout -s KILL 1800 env EVAL_RESTORE_DIR="single_run/$run" EVAL_NUM_EPISODES=1 \
    USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/eval_reach_hover.py \
    > "single_run/comparison/${run}_hover_final.txt" 2>&1
  say "hover probe saved: ${run}_hover_final.txt"
done

# ── 2. メモリ確認 → TP2 / TP2_s2 起動 ─────────────────────────────────
for i in $(seq 1 36); do
  avail_gb=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
  [ "$avail_gb" -ge 14 ] && break
  say "waiting for memory (available=${avail_gb}GB < 14GB) [$i/36]"
  sleep 600
done
avail_gb=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
if [ "$avail_gb" -lt 14 ]; then
  say "ABORT TP2 launch: memory still ${avail_gb}GB after 6h. launch manually"
  exit 1
fi

launch_tp() {  # $1=run名 $2=追加フラグ（seed 等、空可）
  mkdir -p "single_run/$1"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=target_pusher xml_name=rrbot_arm num_threads=4 max_epoch_num=1000 \
    enable_wandb=false fix_skeleton=true $2 +robot_param_scale=1 \
    +reward_specs.init_contact_penalty=50 \
    +restore_dir=single_run/rrbot_arm_pusher_I1 control_prior=true reset_epoch=true \
    hydra.run.dir="single_run/$1" \
    > "single_run/$1/stdout.log" 2>&1 &
  say "$1 launched (PID $!)"
}

launch_tp rrbot_arm_targetpusher_TP2 ""
sleep 120   # 起動ピークをずらす
launch_tp rrbot_arm_targetpusher_TP2_s2 "seed=1"

sleep 600
for run in rrbot_arm_targetpusher_TP2 rrbot_arm_targetpusher_TP2_s2; do
  if [ -s "single_run/$run/log/log_train.txt" ] || grep -q "epoch" "single_run/$run/stdout.log" 2>/dev/null; then
    say "$run startup confirmed"
  else
    say "WARNING: $run may have failed to start. check stdout.log"
  fi
done
say "scheduler done"
