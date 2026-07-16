#!/usr/bin/env bash
# M 系 seed=1 完走を検知 → audit ダンプ → RESTART_READY.txt 作成
LOG=single_run/m_s2_watcher.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

RUNS=(rrbot_arm_pusher_M1_lenonly_s2 rrbot_arm_pusher_M2b_gearonly_s2)
say "watcher start: monitoring ${RUNS[*]}"

until_done() {
  for run in "${RUNS[@]}"; do
    ep=$(grep "T_sample" "single_run/$run/log/log_train.txt" 2>/dev/null | tail -1 | grep -oE "\] [0-9]+" | grep -oE "[0-9]+")
    [ "${ep:-0}" -ge 199 ] || return 1
  done
  return 0
}

for i in $(seq 1 144); do   # 最大24時間
  until_done && break
  sleep 600
done

say "all M-s2 runs done — running audit"
bash scripts/audit_runs.sh > single_run/comparison/audit_after_M_s2_final.txt 2>/dev/null
say "audit saved"

# M 系 s2 の best を記録
for run in "${RUNS[@]}"; do
  best=$(grep "exec_R_eps" "single_run/$run/log/log_train.txt" | grep -oE "exec_R_eps -?[0-9.]+" | awk '{print $2}' | sort -n | tail -1)
  say "$run best=$best"
done

touch single_run/RESTART_READY.txt
say "RESTART_READY.txt created — docker restart 可能"
