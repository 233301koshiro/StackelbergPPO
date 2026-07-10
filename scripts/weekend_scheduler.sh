#!/usr/bin/env bash
# 週末スケジューラ（2026-07-10 金曜仕込み）
# 「仮説の結果に依存しない確定分」だけを自動実行する:
#   1. I1/I2 の完走を待つ（log の epoch 999 到達で判定。pgrep 自己マッチ = Bug 8 を避け log-grep 方式）
#   2. 機械的な完走処理: audit_runs.sh と compare_morphology / visualize_morph_changes の
#      結果をファイルに保存（md への解釈の書き込みは人間/エージェントが月曜に行う）
#   3. メモリが空いたのを確認して L2_s2（Pusher, seed=1, L2 と同一設定）を起動
# 条件付きの実験（TP1 成功時の 1000ep 本番など）は積まない。
set -u
cd /userdir/StackelbergPPO
LOG=single_run/weekend_scheduler.log
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

last_ep() {  # run 名 → log_train.txt の最終 epoch 番号（なければ -1）
  grep "T_sample" "single_run/$1/log/log_train.txt" 2>/dev/null \
    | tail -1 | grep -oE "\] [0-9]+" | tr -dc '0-9'
}

say "scheduler start: waiting for I1/I2 (epoch 999)"

# ── 1. I1/I2 完走待ち ────────────────────────────────────────────────
while :; do
  e1=$(last_ep rrbot_arm_pusher_I1); e2=$(last_ep rrbot_arm_pusher_I2)
  [ "${e1:--1}" -ge 999 ] && [ "${e2:--1}" -ge 999 ] && break
  # フォールバック: 両 log が3時間更新なし（クラッシュ等）なら待ち続けず先へ進む
  if [ -n "$(find single_run/rrbot_arm_pusher_I1/log/log_train.txt -mmin -180 2>/dev/null)" ] \
     || [ -n "$(find single_run/rrbot_arm_pusher_I2/log/log_train.txt -mmin -180 2>/dev/null)" ]; then
    sleep 600
  else
    say "WARNING: I1/I2 logs stale >3h (I1=ep${e1:-?} I2=ep${e2:-?}). proceeding anyway"
    break
  fi
done
say "I1/I2 done or stale (I1=ep$(last_ep rrbot_arm_pusher_I1) I2=ep$(last_ep rrbot_arm_pusher_I2))"

# ── 2. 機械的な完走処理（結果はファイルへ。md は触らない）──────────────
bash scripts/audit_runs.sh > single_run/comparison/audit_after_I_final.txt 2>&1
say "audit saved: single_run/comparison/audit_after_I_final.txt"

for run in rrbot_arm_pusher_I1 rrbot_arm_pusher_I2; do
  timeout -s KILL 2400 env EVAL_RESTORE_DIR="single_run/$run" EVAL_SAMPLE_EVERY=50 \
    USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/visualize_morph_changes.py \
    > "single_run/comparison/${run}_morph_final.txt" 2>&1
  say "morph dump saved: single_run/comparison/${run}_morph_final.txt"
done

timeout -s KILL 2400 env \
  COMPARE_RUNS="single_run/rrbot_arm_pusher_I1:single_run/rrbot_arm_pusher_I2:single_run/rrbot_arm_reach_G3:single_run/rrbot_arm_reach_G4" \
  COMPARE_LABELS="I1_pusher_warm:I2_pusher_scratch:G3_reach_warm:G4_reach_scratch" \
  USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
  --no-window --python scripts/compare_morphology.py \
  > single_run/comparison/final_2x2_comparison.txt 2>&1
say "2x2 comparison saved: single_run/comparison/final_2x2_comparison.txt"

# ── 3. メモリ確認 → L2_s2 起動 ─────────────────────────────────────────
for i in $(seq 1 36); do   # 最大6時間待つ
  avail_gb=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
  [ "$avail_gb" -ge 14 ] && break
  say "waiting for memory (available=${avail_gb}GB < 14GB) [$i/36]"
  sleep 600
done
avail_gb=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
if [ "$avail_gb" -lt 14 ]; then
  say "ABORT L2_s2 launch: memory still ${avail_gb}GB after 6h. launch manually on Monday"
  exit 1
fi

mkdir -p single_run/rrbot_arm_pusher_L2_s2
nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
  --no-window --python scripts/choreonoid_train.py \
  cfg=pusher xml_name=rrbot_arm num_threads=4 max_epoch_num=1000 \
  enable_wandb=false fix_skeleton=true seed=1 +robot_param_scale=1 \
  +reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 \
  +reward_specs.init_contact_penalty=50 \
  hydra.run.dir=single_run/rrbot_arm_pusher_L2_s2 \
  > single_run/rrbot_arm_pusher_L2_s2/stdout.log 2>&1 &
say "L2_s2 launched (PID $!)"

# 起動確認（10分後に log が生まれているか）
sleep 600
if [ -s single_run/rrbot_arm_pusher_L2_s2/log/log_train.txt ] \
   || grep -q "epoch" single_run/rrbot_arm_pusher_L2_s2/stdout.log 2>/dev/null; then
  say "L2_s2 startup confirmed"
else
  say "WARNING: L2_s2 may have failed to start. check stdout.log"
fi
say "scheduler done"
