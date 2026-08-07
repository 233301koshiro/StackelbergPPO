#!/usr/bin/env bash
# 2026-08-07: 診断機能 M7（第3章 3.12）の三層すべてを、正解が分かっている run で検証する。
#
# 動機: M7 は要旨の第三の貢献に含まれるが、検証されているのは**第1層だけ**だった
#   （3.12.3、Reach 3形態）。第2層は「反例で訂正した」記録しかなく、第3層は検証がない。
#
# 方法: マトリクス判定（第4章 4.4.5）の6条件は**正解の順位が確定している**ので、
#   そこへ三層診断をかけ、出力と実際の結果を突き合わせる。
#   新規学習は不要（既存 checkpoint のみ）。
#
# ⚠️ **逐次実行する**。診断1本あたり数 GB 使い、学習 run が6本並走している状況では
#   同時に走らせるとメモリが尽きる（1 run 約 8.7 GB、残り 16 GB）。
#
# 使い方: nohup bash scripts/validate_diagnosis.sh > /dev/null 2>&1 & disown
#   結果は single_run/diag_validation/<run>.txt に1本ずつ出る。
set -u
cd /userdir/StackelbergPPO
OUT=single_run/diag_validation
mkdir -p "$OUT"
LOG="$OUT/_progress.log"

RUNS="tripo_pj_short tripo_pj_mid tripo_pj_long tripo_pjp_short tripo_pjp_mid tripo_pjp_long"

echo "[$(date '+%F %T')] 診断検証を開始（逐次・6本）" | tee -a "$LOG"
for r in $RUNS; do
  if [ ! -d "single_run/$r/models" ]; then
    echo "[$(date '+%F %T')] $r: checkpoint 無し、スキップ" | tee -a "$LOG"; continue
  fi
  echo "[$(date '+%F %T')] $r 開始" | tee -a "$LOG"
  EVAL_RESTORE_DIR="single_run/$r" USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
    timeout -s KILL 1800 /choreonoid_ws/install/bin/choreonoid --no-window \
    --python scripts/diagnose_morphology.py > "$OUT/$r.txt" 2>&1
  rc=$?
  if grep -q "総合判定" "$OUT/$r.txt" 2>/dev/null; then
    echo "[$(date '+%F %T')] $r 完了" | tee -a "$LOG"
  else
    echo "[$(date '+%F %T')] ⚠️ $r 失敗（rc=$rc）— 出力に総合判定が無い" | tee -a "$LOG"
  fi
done
echo "[$(date '+%F %T')] 全件終了" | tee -a "$LOG"
