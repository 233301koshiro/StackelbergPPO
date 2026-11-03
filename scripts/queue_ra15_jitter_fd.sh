#!/bin/bash
# 軸6 の打った後の揺れ（系譜 9-330 の追記）— 回 2: 関節の摩擦 0.08 N·m と指令の遅れ 1 step を入れて β を 2 seed 学習する（系譜 9-332、探索）。
# ⚠️ ブランチ feat/layer4-robotassembler にだけある。worktree /userdir/StackelbergPPO_ra から起動する。
# ⭐ 起動行は queue_ra15_l4.sh の β と同じ（制御コスト 0.04）で、物理の環境変数に CNOID_ACT_DELAY=1 CNOID_JOINT_FRICTION=0.08 を足すだけ。
#   2 つは「実物に近い物理」の束として同時に入れる（どちらが効いたかは言わない）。修論 4.8 の 4 本は置き換えない。
# ⭐ 回 1（queue_ra15_jitter_cc.sh）の 2 本が走っている間は、launch の njobs が 2 未満になるまで待つ。
# ⭐ 事前登録した読み（系譜 9-332 の ㊲。結果を見てから変えない）。比べる相手は同じ物理で今の β を再生したもの（13 回・1.45 m / 63 回・1.69 m）:
#   ㊲-a 2 seed とも揺れ 15 回以下・箱の進み 1.0 m 以上 → 「この物理で学習した 2 seed は揺れなかった」まで
#   ㊲-b どちらかが 30 回超 → 抑えない
#   ㊲-c どちらかの箱の進みが 1.0 m 未満 → 押しが崩れた
#   どれでもない → 言わない
set -u
cd /userdir/StackelbergPPO_ra
Q=single_run/queue_ra15_jitter_fd.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
PHYS="CNOID_SELF_COLLISION=1 CNOID_CULLING_DEPTH=1.0 CNOID_CTRL_CLIP=1 CNOID_HINGE_DAMP_TORQUE=1 CNOID_JOINT_MAX_VEL=5.97 CNOID_JOINT_MAX_VEL_STRICT=1 CNOID_TICK_SYNC=1 CNOID_SUBSTEPS=4 CNOID_JOINT_LIMIT_TORQUE=1 CNOID_ACT_DELAY=1 CNOID_JOINT_FRICTION=0.08"
COMMON="cfg=pusher_ra15 xml_name=e2e_a1v_ra15 max_epoch_num=200 num_threads=4 enable_wandb=false fix_skeleton=true +reward_specs.ctrl_cost_coeff=0.04 +reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true +fix_base=true +env_specs.init_self_contact=true +env_specs.init_self_contact_caps=true +env_specs.cube_init_noise=0.02 +env_specs.init_contact_margin=0.006"
# run 名|seed|robot_param_scale
JOBS="ra15_pusher_beta_fd|0|1
ra15_pusher_beta_fd_s2|1|1"
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
  local PID=$!
  log "$RUN launched (PID $PID)  seed $SEED  robot_param_scale $SCALE  env: $PHYS"; sleep 90
  local want="[$((1000*SEED+1)), $((1000*SEED+2)), $((1000*SEED+3)), $((1000*SEED+4))]" s
  s=$(grep -ao 'ワーカーの乱数の種 \[[0-9, ]*\]' single_run/$RUN/stdout.log | head -1)
  case "$s" in *"$want"*) log "✅ $RUN: $s";; *) log "⛔⛔ $RUN: 種の行が期待（$want）と違う／無い: '$s'";; esac
  grep -q "可動域をトルクで掛ける: 4 関節" single_run/$RUN/stdout.log && log "✅ $RUN: 可動域のトルク 4 関節" || log "⛔⛔ $RUN: 可動域のトルクの行が無い"
  grep -q "4 分割" single_run/$RUN/stdout.log && log "✅ $RUN: 刻み 4 分割" || log "⛔⛔ $RUN: 刻みの分割の行が無い"
  grep -q "1 制御 step 遅らせる" single_run/$RUN/stdout.log && log "✅ $RUN: 指令の遅れ 1 step" || log "⛔⛔ $RUN: 指令の遅れの行が無い"
  # 摩擦の行は最初の tick で出るので、tick しない本体の stdout.log ではなくワーカーのログに出る（11-03 に誤報を出した、系譜 9-332）
  local nf; nf=$(grep -l "摩擦 0.08 N·m" /tmp/cnoid_worker_${PID}_*.log 2>/dev/null | wc -l)
  [ "$nf" -ge 1 ] && log "✅ $RUN: 関節の摩擦 0.08（ワーカーのログ ${nf} 個）" || log "⛔⛔ $RUN: 関節の摩擦の行がワーカーのログに無い"
  grep -q '"CNOID_ACT_DELAY": "1"' single_run/$RUN/launch_env.json && grep -q '"CNOID_JOINT_FRICTION": "0.08"' single_run/$RUN/launch_env.json && log "✅ $RUN: launch_env に遅れと摩擦" || log "⛔⛔ $RUN: launch_env に遅れか摩擦が無い"
  return 0; }

log "=== 軸6 の揺れ 回 2（系譜 9-332）HEAD=$(git rev-parse --short HEAD)（ブランチ $(git rev-parse --abbrev-ref HEAD)）。$(echo $RUNS | wc -w) 本"
PROJ_DIR=/userdir/StackelbergPPO_ra RUN_FILTER='^ra15_pusher_beta_fd' nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
log "学習中の見張り watch_midrun.sh（ブランチ、ra15_ だけ）を起動（PID $!）"
PROJ_DIR=/userdir/StackelbergPPO_ra RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh（ブランチ）を起動（PID $!、RUNS=$RUNS）"
echo "$JOBS" | while IFS='|' read -r RUN SEED SCALE; do
  launch "$RUN" "$SEED" "$SCALE" || break
done
log "=== 2 本の起動が済んだ（または止まった）"
