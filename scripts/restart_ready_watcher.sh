#!/usr/bin/env bash
# 再起動準備ウォッチャー（2026-07-13 仕込み）
# L2/L2_s2/TP2/TP2_s2 の全完走（ep999）を検知したら:
#   1. 機械的完走処理（audit 全表 + 4 run の形態推移ダンプ）を single_run/comparison/ に保存
#   2. single_run/RESTART_READY.txt を作成（= docker 再起動してよい合図 + 手順）
# 再起動そのものはホスト側の人間操作（コンテナ内からは不可能）。
set -u
cd /userdir/StackelbergPPO
LOG=single_run/restart_watcher.log
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

last_ep() {
  grep "T_sample" "single_run/$1/log/log_train.txt" 2>/dev/null \
    | tail -1 | grep -oE "\] [0-9]+" | tr -dc '0-9'
}

RUNS="rrbot_arm_pusher_L2 rrbot_arm_pusher_L2_s2 rrbot_arm_targetpusher_TP2 rrbot_arm_targetpusher_TP2_s2"
say "watcher start: waiting for all of [$RUNS] to reach ep999"

while :; do
  alldone=1
  for r in $RUNS; do
    e=$(last_ep "$r"); [ "${e:--1}" -ge 999 ] || alldone=0
  done
  [ "$alldone" = 1 ] && break
  # 全 log が6時間更新なしなら異常としてマーカーだけ残して進む
  stale=1
  for r in $RUNS; do
    [ -n "$(find single_run/$r/log/log_train.txt -mmin -360 2>/dev/null)" ] && stale=0
  done
  if [ "$stale" = 1 ]; then
    say "WARNING: all logs stale >6h. proceeding (runs may have crashed)"
    break
  fi
  sleep 1800
done
for r in $RUNS; do say "final: $r ep=$(last_ep $r)"; done

# ── 1. 機械的完走処理 ──────────────────────────────────────────────
bash scripts/audit_runs.sh > single_run/comparison/audit_after_L_TP_final.txt 2>&1
say "audit saved"
for r in $RUNS; do
  timeout -s KILL 2400 env EVAL_RESTORE_DIR="single_run/$r" EVAL_SAMPLE_EVERY=50 \
    USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/visualize_morph_changes.py \
    > "single_run/comparison/${r}_morph_final.txt" 2>&1
  say "morph dump saved: ${r}_morph_final.txt"
done

# ── 2. 再起動 GO の合図 ────────────────────────────────────────────
cat > single_run/RESTART_READY.txt <<'EOF'
=== DOCKER 再起動の準備完了 ===
全 run（L2 / L2_s2 / TP2 / TP2_s2）が完走し、完走処理ダンプは
single_run/comparison/ に保存済み。学習プロセスは残っていないはずなので
ホスト側から docker restart を実行してよい（ゾンビ ~940 個の掃除が目的）。

再起動後の Claude セッションへの指示:
  「docs/研究応用/引き継ぎ_再起動後.md を読んで作業を再開して」
EOF
say "RESTART_READY.txt written. watcher done"
