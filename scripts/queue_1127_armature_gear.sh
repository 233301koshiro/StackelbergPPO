#!/bin/bash
# ギア比に動力学の費用を付けた学習（系譜 9-361、2026-11-27、ユーザー「修論に書いて実験をやろう」「回転子の慣性 ∝ ギア比²」「縦型 R2 ＋ 水平の押し 中腕」）。
#
# ⭐ なぜ: 9-360 で、ギア比の上限への張り付きは不足ではなく、ギア比に費用が無い（制御コストは制御入力にかかるので、高いギア比ほど同じトルクが安い）
#   ことの結果と読めた。費用を付けると、張り付きが消えるか、タスクごとの形が分かれるか（R2）を見る。
# ⭐ 変えるのは 1 つ: cfg を `*_ag` に替える（反射慣性 ＝ 1.71 ×（ギア比 / 150）²、下限 0.05。元の cfg との差は 3 行。ギア比 150 の形は元と同じ物理）。
#   起動行は元 run の `.hydra/overrides.yaml` と `launch_env.json`（Bug 106・107 の直しを含む `_vp` の元と同じ）をそのまま読み、cfg 名だけを差し替える。
#   seed も元のまま。run 名は元の名前の末尾に `_vp_ag`。
# ⭐ 起動から 60 秒後に種の行と PRETICK の行を確かめる（9-351 の取り直しと同じ）。回転子の慣性が効いたかは、完走後の再生の
#   「実行時のギア比と回転子の慣性」の行で確かめる（record_arm_trace.py、9-361）。
# ⭐ 事前登録した読み 61 は系譜 9-361 にある（結果を見てから変えない）。
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_1127_armature_gear.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo comm,args | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
FIX_ENV="CNOID_JOINT_MAX_VEL_PRETICK=1 CNOID_CAPSULE_INERTIA_ROT=1"
SRCS="e2e_a1v_reach_phys e2e_a1v_pusher_phys
e2e_a1v_reach_phys_s2_ws e2e_a1v_pusher_phys_s2_ws
e2e_a1v_reach_phys_s3 e2e_a1v_pusher_phys_s3
tripo_pjp_mid_phys tripo_pjp_mid_phys_s2_ws
tripo_pjp_mid_phys_s3"
SRCS=$(echo $SRCS)
RUNS=$(for s in $SRCS; do printf '%s_vp_ag ' "$s"; done)

launch(){ local SRC=$1 RUN=${1}_vp_ag
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return 0; fi
  if [ ! -f "single_run/$SRC/.hydra/overrides.yaml" ] || [ ! -f "single_run/$SRC/launch_env.json" ]; then log "⛔ $SRC の overrides か launch_env が無い。飛ばす"; return 0; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 15; done
  local ARGS ENVS SEED CFG
  CFG=$(sed -n 's/^- cfg=//p' single_run/$SRC/.hydra/overrides.yaml | head -1)
  case "$CFG" in pusher_tripo_v3_reallen|pusher_gearonly_real) ;; *) log "⛔⛔ $SRC の cfg が想定外（$CFG）。飛ばす"; return 0;; esac
  ARGS=$(sed -n 's/^- //p' single_run/$SRC/.hydra/overrides.yaml | sed "s/^cfg=$CFG\$/cfg=${CFG}_ag/" | tr '\n' ' ')
  SEED=$(sed -n 's/^- seed=//p' single_run/$SRC/.hydra/overrides.yaml | head -1)
  ENVS=$(python3 -c "import json;print(' '.join(f'{k}={v}' for k,v in json.load(open('single_run/$SRC/launch_env.json')).items()))")
  local AV; AV=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "$AV" -lt 3 ]; then log "⛔⛔ ディスクの空き ${AV} GB（3 GB 未満）。$RUN 以降を起動せずに止まる"; return 1; fi
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 $ENVS $FIX_ENV \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    $ARGS hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)  元 $SRC  seed $SEED  env: $ENVS $FIX_ENV  args: $ARGS"; sleep 60
  local want="[$((1000*SEED+1)), $((1000*SEED+2)), $((1000*SEED+3)), $((1000*SEED+4))]" s p
  s=$(grep -ao 'ワーカーの乱数の種 \[[0-9, ]*\]' single_run/$RUN/stdout.log | head -1)
  case "$s" in *"$want"*) log "✅ $RUN: $s";; *) log "⛔⛔ $RUN: 種の行が期待（$want）と違う／無い: '$s'";; esac
  p=$(grep -ao 'tick の前にトルクを絞って速度の上限を守る: [0-9]* 関節' single_run/$RUN/stdout.log | head -1)
  if [ -n "$p" ]; then log "✅ $RUN: $p"; else log "⛔⛔ $RUN: PRETICK の行が無い（Bug 106 の直しが効いていない疑い）"; fi
  grep -q "cfg=${CFG}_ag" single_run/$RUN/.hydra/overrides.yaml 2>/dev/null && log "✅ $RUN: cfg=${CFG}_ag" || log "⛔⛔ $RUN: overrides に cfg=${CFG}_ag が無い"
  return 0; }

log "=== 11-27 ギア比に回転子の慣性の費用を付けた学習（系譜 9-361、読み 61）HEAD=$(git rev-parse --short HEAD)。$(echo $RUNS | wc -w) 本"
git diff HEAD -- khrylib design_opt scripts/choreonoid_train.py > single_run/queue_1127_armature_gear.code.diff
log "コードの未コミットの差分 $(wc -l < single_run/queue_1127_armature_gear.code.diff) 行を single_run/queue_1127_armature_gear.code.diff に残した（空ならコミット済み）"
if ! pgrep -f "^bash scripts/watch_midrun.sh" > /dev/null; then
  nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
  log "学習中の見張り watch_midrun.sh を起動（PID $!）"
else
  log "学習中の見張り watch_midrun.sh は既に動いている"
fi
RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh を起動（PID $!、$(echo $RUNS | wc -w) 本）"
for SRC in $SRCS; do
  launch "$SRC" || break
done
log "=== $(echo $RUNS | wc -w) 本の起動が済んだ（または止まった）"
