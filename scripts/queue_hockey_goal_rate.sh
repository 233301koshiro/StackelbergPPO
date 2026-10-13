#!/bin/bash
# ⭐ ホッケーの取り直し（hockey_bank9_phys / _s2）のゴール率を 20 話で測る（系譜 9-237）。学習はしない。
#
# ⛔ 9-229 の「3 話中 2 話ゴール」は、種が固定された初期状態のくじの 3〜5 本目でしかなかった（ゴール率ではない）。
# ⭐ 判定の方法を結果を見る前に書き直す（9-230 の読みの「5 話・頭 2 話を捨てる」を置き換える）:
#   best を 20 話（全話）再生し、置いた瞬間の隙間 ≥ 10 cm のパックがゴールに入った話の数を数える。
#   ① どちらかの seed で 1 話以上ゴール → 自己干渉ありでもゴールは成立（修論は「20 話中 k 話」で書く）
#   ② ゴールは無いが離れたパックを 0.1 m 以上動かす話がある → 打撃は成立、ゴールは主張しない
#   ③ どの話でも動かない → 原因を調べる
# 再生の環境は launch_env.json から自動（9-236）。出力は trace/arm_trace_20.npz（自動再生キューの arm_trace.npz とは別）
set -u
cd /userdir/StackelbergPPO
OUT=single_run/hockey_goal_rate.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $OUT; }
log "=== ホッケーの取り直しのゴール率（20 話）。完走を待つ"
for R in hockey_bank9_phys hockey_bank9_phys_s2; do
  until grep -q "training done" single_run/$R/log/log_train.txt 2>/dev/null; do sleep 600; done
  mkdir -p single_run/$R/trace
  EVAL_RESTORE_DIR=single_run/$R EVAL_CHECKPOINT=best TRACE_EPISODES=20 TRACE_OUT=single_run/$R/trace/arm_trace_20.npz \
    USE_CHOREONOID=1 OMP_NUM_THREADS=1 timeout -k 30 3600 /choreonoid_ws/install/bin/choreonoid --no-window \
    --python scripts/record_arm_trace.py > single_run/$R/trace_20.log 2>&1
  log "$R 再生 exit=$? $(grep -h '話ごとの' single_run/$R/trace_20.log | tail -1)"
done
log "=== キュー終了（ゴールの判定は軌跡のパックの位置とゴール口から。人が見る）"
