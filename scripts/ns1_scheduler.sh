#!/usr/bin/env bash
# 2026-08-05: ns1 実験を「tripo_arm_v3_pusher（中核実験）完走後」に自動起動するスケジューラ。
# コンテナ内に detached 常駐し、v3 のプロセスが消えたら launch_nsteps_ns1.sh を1回だけ呼ぶ。
# 進捗は single_run/ns1_scheduler.log に追記。reach_s2 は残っていてもよい（1本＝余力あり）。
cd /userdir/StackelbergPPO
LOG=single_run/ns1_scheduler.log
{
  echo "[$(date '+%F %T')] scheduler start (pid $$): waiting for tripo_arm_v3_pusher to finish (10min poll)"
  while pgrep -f 'hydra.run.dir=single_run/tripo_arm_v3_pusher' >/dev/null 2>&1; do
    sleep 600
  done
  echo "[$(date '+%F %T')] tripo_arm_v3_pusher finished -> launching ns1"
  bash /userdir/StackelbergPPO/scripts/launch_nsteps_ns1.sh
  echo "[$(date '+%F %T')] scheduler done"
} >> "$LOG" 2>&1
