#!/usr/bin/env bash
# 2026-09-20: ⭐⭐ **リンク長の探索範囲に根拠を与えた条件**（実験系譜 9-150）。
#
# ⛔⛔ 旧 `lb:[-0.5,-0.5] ub:[0.5,0.5]` は cfg 28 本に同じ値で入っており（cheetah・walker と同値）
#   元論文の移動ロボットからの引き継ぎだった（9-148・9-149）。幅が絶対値なので
#   0.2195 m のリンクは [-0.280, +0.720] ＝ **負になれ、実際に 18 run 中 16 run で反転していた。**
#
# ⭐ 新 `rel_frac: [0.5, 2.93, 0.5]`（設計長に対する倍率）:
#   下限 0.5 = リンクが反転も消失もしない（幾何の要請）
#   上限 2.93 = 自重＋押しトルクが **150 N·m（UR5e。ギア比 9-129 と同じ出どころ）**を超えない限界
#   横方向 0.5 = 絶対値だと短いリンクだけ自由になるので長さに比例させる
#
# ⚠️ **対照は `e2e_a1v_real_pusher` / `_s2`（同じ物理・旧範囲）。変数は探索範囲だけ。**
#   対照の帯: [266.74, 299.49]
#
# 読み方（**結果を見る前に決める**。§5-2 ⑤-2）:
#   ⭐ ① 成績が対照と重なる → ⭐⭐ **最良。反転は性能に不要だった＝根拠をつけても損をしない**
#   ⚠️ ② 落ちる → **反転や極端な長さが性能に寄与していた**
#        ＝ **従来の成績は物理的に不自然な形態に依存していたと言える**
#   ⭐ ③ 上がる → 探索空間が狭まって学習が楽になった
#
# ⚠️ **結論前に check_before_conclusion.py と収束形態の反転チェックを回す。**
#
# 起動: nohup bash scripts/queue_reallen.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_reallen.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
njobs(){ ps -eo comm,args 2>/dev/null | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
launch(){ local RUN=$1 S=$2
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_tripo_v3_reallen xml_name=e2e_a1v_real num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +env_specs.arm_safe_init=true \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)"; sleep 60; }

log "=== キュー開始（リンク長の探索範囲に根拠・2 seed）"
launch e2e_a1v_reallen_pusher 0
launch e2e_a1v_reallen_pusher_s2 1
log "=== キュー終了"
