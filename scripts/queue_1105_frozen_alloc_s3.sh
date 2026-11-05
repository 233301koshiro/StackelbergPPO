#!/bin/bash
# 凍結の配分（9-334 ㊵）に seed 2 を足す 2 本（ユーザー判断 2026-11-05 夜「やろう」、系譜 9-335、想定問答 Q59 の B②）。
#   目的: 修論 4.4.3.3 の「形を凍結すると配分を区別できない」を 2 seed から 3 seed へそろえる（5.3 の 4「R1〜R3 の検証は 3」）。
#   ⚠️ 帯（best の最小〜最大）は seed を足すと広がるだけなので、重なっている帯は 3 seed でも重なる。結論（㊵-b）は変わりえない。
#
# ⭐ 動いている queue_1104_frozen_alloc_455.sh は書き換えない。そのキューの bash（PID 2332237・2332263）が終わるまで待ち、
#   その後は学習が 2 本未満になるたびに起動する（2 つのキューが同時に空きを取り合わない）。
# ⭐ 起動行・凍結・種の確かめは queue_1104_frozen_alloc_455.sh の launch() をそのまま使う（元の run の overrides.yaml と launch_env.json を読む）。
#   変えるのは seed=2 と +robot_param_scale=1 → 0.0001 だけ。先端重は門を直した tripo_pjdp_dist_phys_cg から作る（9-333・9-334 と同じ）。
# ⭐ 見張り: 完走後の自動再生はこの 2 本の RUNS で起動する。学習中の見張り watch_midrun.sh は、起動時に動いていなければ起動する（二重にしない）。
#
# 事前登録した読み（系譜 9-335 と同じ文。結果を見てから変えない）:
#   前提: 完走・check_before_conclusion.py・目視・ワーカーの種 2001〜2004・凍結（robot_param_scale = 0.0001、best の形が初期のまま）。
#   ㊷ 3 seed の帯で、凍結の根元重 対 凍結の先端重は必ず重なる（2 seed で重なっているため）。→ 修論 4.4.3.3・5.2.2 と Q58 の「2 シード」を「3 シード」に直す。
#   判定に使わない観察: seed 2 の best が凍結の他の腕と同じ 6.0〜6.9 に収まるか（9-334 の「既定の値で押しの上限が決まっている可能性」）。
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_1105_frozen_alloc_s3.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
JOBS="tripo_pjdp_prox_frozen_phys_fx_s3|tripo_pjdp_prox_phys|2|1
tripo_pjdp_dist_frozen_phys_cg_fx_s3|tripo_pjdp_dist_phys_cg|2|1"
RUNS=$(echo "$JOBS" | cut -d'|' -f1 | tr '\n' ' ')
launch(){ local RUN=$1 SRC=$2 SEED=$3 FRZ=$4
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return 0; fi
  if [ ! -f "single_run/$SRC/.hydra/overrides.yaml" ] || [ ! -f "single_run/$SRC/launch_env.json" ]; then log "⛔ $SRC の overrides か launch_env が無い。飛ばす"; return 0; fi
  local ARGS ENVS
  ARGS=$(sed -n 's/^- //p' single_run/$SRC/.hydra/overrides.yaml | sed "s/^seed=.*/seed=$SEED/" | tr '\n' ' ')
  if [ "$FRZ" = 1 ]; then
    ARGS=$(echo "$ARGS" | sed 's/+robot_param_scale=1 /+robot_param_scale=0.0001 /')
    case "$ARGS" in *"+robot_param_scale=0.0001 "*) ;; *) log "⛔⛔ $RUN: 凍結の置き換えが効かない（$ARGS）。飛ばす"; return 0;; esac
  fi
  ENVS=$(python3 -c "import json;print(' '.join(f'{k}={v}' for k,v in json.load(open('single_run/$SRC/launch_env.json')).items()))")
  while [ "$(njobs)" -ge 2 ]; do sleep 15; done
  local AV; AV=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "$AV" -lt 3 ]; then log "⛔⛔ ディスクの空き ${AV} GB（3 GB 未満）。$RUN 以降を起動せずに止まる"; return 1; fi
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 $ENVS \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    $ARGS hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)  元 $SRC  env: $ENVS  args: $ARGS"; sleep 60
  local want="[$((1000*SEED+1)), $((1000*SEED+2)), $((1000*SEED+3)), $((1000*SEED+4))]" s
  s=$(grep -ao 'ワーカーの乱数の種 \[[0-9, ]*\]' single_run/$RUN/stdout.log | head -1)
  case "$s" in *"$want"*) log "✅ $RUN: $s";; *) log "⛔⛔ $RUN: 種の行が期待（$want）と違う／無い: '$s'";; esac
  if [ "$FRZ" = 1 ]; then
    grep -q "robot_param_scale = 0.0001" single_run/$RUN/stdout.log && log "✅ $RUN: 凍結 robot_param_scale = 0.0001" || log "⛔⛔ $RUN: 凍結の行が無い（Bug 74）"
  fi
  return 0; }

log "=== 11-05 夜のキュー（系譜 9-335、凍結の配分の seed 2 の 2 本）HEAD=$(git rev-parse --short HEAD)。前のキュー（PID 2332237・2332263）が終わるまで待つ"
while kill -0 2332237 2>/dev/null || kill -0 2332263 2>/dev/null; do sleep 60; done
log "前のキューの bash が終わった。空きを待って起動する"
RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh を起動（PID $!、RUNS=$RUNS）"
echo "$JOBS" | while IFS='|' read -r RUN SRC SEED FRZ; do
  launch "$RUN" "$SRC" "$SEED" "$FRZ" || break
  if ! ps -eo args | grep -q "[w]atch_midrun.sh"; then
    nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
    log "学習中の見張り watch_midrun.sh を起動（PID $!）"
  fi
done
log "=== 2 本の起動が済んだ（または止まった）"
