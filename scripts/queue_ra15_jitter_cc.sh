#!/bin/bash
# 軸6 の打った後の揺れ（系譜 9-330 の追記）を消せるか — 回 1: β の制御コスト 0.04 → 0.2 を 2 seed（系譜 9-332、探索）。
# ⚠️ ブランチ feat/layer4-robotassembler にだけある。worktree /userdir/StackelbergPPO_ra から起動する。
# ⭐ 起動行は queue_ra15_l4.sh の β（robot_param_scale 1）と同じで、変えるのは +reward_specs.ctrl_cost_coeff 0.04 → 0.2 だけ。
#   修論 4.8 の 4 本（ra15_pusher_*）は置き換えない。
# ⭐ 事前登録した読み（系譜 9-332 に同じ文。結果を見てから変えない）。揺れは scripts/measure_jitter.py（main）の 5 話の中央値:
#   ㉟-a 2 seed とも揺れ 15 回以下・箱の進み 1.0 m 以上 → 「0.2 では 2 seed とも揺れなかった」まで（0.04 でも seed 0 は揺れていない）
#   ㉟-b どちらかが 30 回超 → 制御コストでは消えない
#   ㉟-c どちらかの箱の進みが 1.0 m 未満 → 押しが崩れた
#   どれでもない → 言わない
set -u
cd /userdir/StackelbergPPO_ra
Q=single_run/queue_ra15_jitter_cc.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
PHYS="CNOID_SELF_COLLISION=1 CNOID_CULLING_DEPTH=1.0 CNOID_CTRL_CLIP=1 CNOID_HINGE_DAMP_TORQUE=1 CNOID_JOINT_MAX_VEL=5.97 CNOID_JOINT_MAX_VEL_STRICT=1 CNOID_TICK_SYNC=1 CNOID_SUBSTEPS=4 CNOID_JOINT_LIMIT_TORQUE=1"
COMMON="cfg=pusher_ra15 xml_name=e2e_a1v_ra15 max_epoch_num=200 num_threads=4 enable_wandb=false fix_skeleton=true +reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true +fix_base=true +env_specs.init_self_contact=true +env_specs.init_self_contact_caps=true +env_specs.cube_init_noise=0.02 +env_specs.init_contact_margin=0.006"
# run 名|seed|robot_param_scale
JOBS="ra15_pusher_beta_cc02|0|1
ra15_pusher_beta_cc02_s2|1|1"
RUNS=$(echo "$JOBS" | cut -d'|' -f1 | tr '\n' ' ')

launch(){ local RUN=$1 SEED=$2 SCALE=$3
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return 0; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 30; done
  local AV; AV=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "$AV" -lt 3 ]; then log "⛔⛔ ディスクの空き ${AV} GB（3 GB 未満）。$RUN 以降を起動せずに止まる"; return 1; fi
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 $PHYS \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    $COMMON seed=$SEED +robot_param_scale=$SCALE hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)  seed $SEED  robot_param_scale $SCALE  env: $PHYS"; sleep 90
  local want="[$((1000*SEED+1)), $((1000*SEED+2)), $((1000*SEED+3)), $((1000*SEED+4))]" s
  s=$(grep -ao 'ワーカーの乱数の種 \[[0-9, ]*\]' single_run/$RUN/stdout.log | head -1)
  case "$s" in *"$want"*) log "✅ $RUN: $s";; *) log "⛔⛔ $RUN: 種の行が期待（$want）と違う／無い: '$s'";; esac
  grep -q "可動域をトルクで掛ける: 4 関節" single_run/$RUN/stdout.log && log "✅ $RUN: 可動域のトルク 4 関節" || log "⛔⛔ $RUN: 可動域のトルクの行が無い"
  grep -q "4 分割" single_run/$RUN/stdout.log && log "✅ $RUN: 刻み 4 分割" || log "⛔⛔ $RUN: 刻みの分割の行が無い"
  return 0; }

log "=== 軸6 の揺れ 回 1（系譜 9-332）HEAD=$(git rev-parse --short HEAD)（ブランチ $(git rev-parse --abbrev-ref HEAD)）。$(echo $RUNS | wc -w) 本"
PROJ_DIR=/userdir/StackelbergPPO_ra RUN_FILTER='^ra15_pusher_beta_cc02' nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
log "学習中の見張り watch_midrun.sh（ブランチ、ra15_ だけ）を起動（PID $!）"
PROJ_DIR=/userdir/StackelbergPPO_ra RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh（ブランチ）を起動（PID $!、RUNS=$RUNS）"
echo "$JOBS" | while IFS='|' read -r RUN SEED SCALE; do
  launch "$RUN" "$SEED" "$SCALE" || break
done
log "=== 2 本の起動が済んだ（または止まった）"
