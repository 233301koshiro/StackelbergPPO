#!/usr/bin/env bash
# 2026-08-05: 実験キュー（ns1_scheduler.sh を置き換える統合スケジューラ）。
# 順次自動起動: v3(中核)完走 → ns1(200ep) → [境界比較(matched ep200)] → pjp_long_1000 → pjp_mid_1000。
# 各段は「起動を待ち→完走を待つ」。他の run は独立に並走してよい。
# 実測では1 run あたりの CPU は約92%（≒1コア）で、20コアに対し4本同時でも load 5.8 程度。
# 進捗は single_run/experiment_queue.log。
cd /userdir/StackelbergPPO
LOG=single_run/experiment_queue.log

# 完走判定は **stdout.log の完走マーカー** を第一とし、プロセス存在は補助にしか使わない。
# 理由（Bug 19、2026-08-06）: `pgrep -f "hydra.run.dir=single_run/<run>"` は、
# **その文字列をコマンドラインに含む第三者のプロセス**（AI が流した監視コマンド等）にも
# マッチするため、学習が終わっていても「まだ稼働中」と誤判定してキューが止まる。
# 実際に v3 完走後、約6分キューが空転した。マーカー判定にはこの弱点がない。
finished() {  # $1=run名: 完走していれば 0
  grep -q "All workers terminated" "single_run/$1/stdout.log" 2>/dev/null
}

running() {  # $1=run名: 学習プロセスが生きていれば 0。
  # pgrep のパターンに run 名を直接置かず、choreonoid_train を実行しているプロセスの
  # コマンドラインを取り出してから run 名で絞る。こうすると呼び出し側のコマンドライン
  # （このスクリプト自身を含む）はパターンに現れないので Bug 19 の自己汚染を避けられる。
  ps -eo cmd --no-headers 2>/dev/null \
    | grep -F 'choreonoid_train.py' \
    | grep -qF "hydra.run.dir=single_run/$1"
}

wait_finish() {  # $1=run名: 完走マーカーが出るまで待つ（プロセス消滅は補助条件）
  local r=$1
  while true; do
    finished "$r" && { echo "[$(date '+%F %T')] $r 完走マーカー検出"; return 0; }
    if ! running "$r"; then
      # マーカーが無いのにプロセスも居ない = 異常終了。無限待機を避けて先へ進む
      echo "[$(date '+%F %T')] ⚠️ $r はプロセスが消えたが完走マーカーが無い（異常終了の疑い）。次段へ進む"
      return 1
    fi
    sleep 300
  done
}

wait_run() {  # $1=run名: 起動を最大30分待ってから、完走を待つ
  local r=$1 i
  for i in $(seq 1 30); do
    running "$r" && break
    finished "$r" && return 0      # 待っている間に完走していた場合
    sleep 60
  done
  wait_finish "$r"
}

{
  echo "[$(date '+%F %T')] queue start (pid $$)"
  echo "[$(date '+%F %T')] 待機: tripo_arm_v3_pusher（中核）の完走"
  wait_finish tripo_arm_v3_pusher || true

  echo "[$(date '+%F %T')] v3 完走 → ns1 起動"
  bash scripts/launch_nsteps_ns1.sh
  wait_run tripo_arm_v2c_pusher_ns1

  # --- ⑤ 境界張り付き比較（matched ep200）: baseline(5)/ns2(2)/ns1(1) ---
  echo "[$(date '+%F %T')] ns1 完走 → 境界比較(matched ep200) を実行"
  BC_RUNS="single_run/tripo_arm_v2c_pusher:single_run/tripo_arm_v2c_pusher_ns2:single_run/tripo_arm_v2c_pusher_ns1" \
  BC_LABELS="nsteps5:nsteps2:nsteps1" BC_CKPT="epoch_0200" \
  timeout -s KILL 600 env USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/boundary_compare.py \
    > single_run/boundary_compare_ep200.txt 2>&1
  echo "[$(date '+%F %T')] 境界比較 完了 → single_run/boundary_compare_ep200.txt"

  echo "[$(date '+%F %T')] → pjp_long_1000 起動"
  bash scripts/launch_pjp_1000.sh long
  wait_run tripo_pjp_long_1000

  echo "[$(date '+%F %T')] pjp_long_1000 完走 → pjp_mid_1000 起動"
  bash scripts/launch_pjp_1000.sh mid
  wait_run tripo_pjp_mid_1000

  echo "[$(date '+%F %T')] queue done（全段完走）"
} >> "$LOG" 2>&1
