#!/bin/bash
# 軸6 L-4: 部品で組んだ 1/5 の A1v（e2e_a1v_ra15）の押しで、β（長さだけ学習）と α（形を固定）を 2 seed ずつ回す（系譜 9-329）。
# ⚠️ ブランチ feat/layer4-robotassembler にだけある。worktree /userdir/StackelbergPPO_ra から起動する（cfg pusher_ra15 と XML はブランチにだけある）。
#   worktree の single_run は main の single_run へのシンボリックリンクなので、run のフォルダは main の single_run にできる。
# ⭐ 起動行（9-326〜9-328 で決めた値）: cfg pusher_ra15・xml e2e_a1v_ra15・ctrl_cost × k（0.04）・init_self_contact_caps・cube_init_noise 0.02・
#   init_contact_margin 0.006・CNOID_JOINT_MAX_VEL 5.97・CNOID_SUBSTEPS 4・CNOID_JOINT_LIMIT_TORQUE 1。それ以外は e2e_a1v_pusher_phys と同じ。
#   β と α の差は +robot_param_scale（1 / 0.0001、9-320 と同じ凍結）だけ。
# ⭐ 順番: main のキュー（queue_1029_fix1_seed2.sh・queue_1029_long_frozen.sh）がどちらも終わってから（研究方針「キューへの投入の優先度」）。
#   2 本ずつ（GPU が律速）: seed 0 の β・α → seed 1 の β・α。
# ⭐ 見張りと自動再生はブランチの側で起動する（PROJ_DIR=worktree、見張りは RUN_FILTER=^ra15_）。
#   ⚠️ main の watch_midrun.sh がまだ動いていると、ra15_ の run を main の cfg で再生しようとして失敗した行を midrun_check.log に出す（害はない）。
# ⭐ 事前登録した読み（系譜 9-329 に同じ文。結果を見てから変えない）:
#   ㉜ 前提: 4 本とも完走・check_before_conclusion.py・目視。β の上腕と前腕の長さが初期（80.4 / 65.9 mm）から動いた（best の XML）、α は 1 mm 以内。
#     不変条件（床・めり込み）と、可動域の端を越えた角が 1° 以内（9-327 の 0.3°）。
#   ㉜-a 共設計が足した: β の 2 seed の best の帯の下端 > α の帯の上端。
#   ㉜-b 差が言えない: 帯が重なる。→ 「部品の制約の中では形を変える余地が小さかった」と書き、順位は言わない。
#   ㉜-c 共設計が妨げた: β の帯の上端 < α の帯の下端。
#   ㉝（L-4b）: β の best の長さを、組み方の系統（9-325、隙間 ≤ 1.7 mm）で作れる最も近い長さへ丸めて再生し、50 話の中央値が β の帯の中に入る。
#   予想（判定に使わない）: ㉜-a。R2 の A1v の押しは骨の長さを伸ばした（9-314 の H1）ので、β は前腕を伸ばす。
set -u
cd /userdir/StackelbergPPO_ra
Q=single_run/queue_ra15_l4.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
PHYS="CNOID_SELF_COLLISION=1 CNOID_CULLING_DEPTH=1.0 CNOID_CTRL_CLIP=1 CNOID_HINGE_DAMP_TORQUE=1 CNOID_JOINT_MAX_VEL=5.97 CNOID_JOINT_MAX_VEL_STRICT=1 CNOID_TICK_SYNC=1 CNOID_SUBSTEPS=4 CNOID_JOINT_LIMIT_TORQUE=1"
COMMON="cfg=pusher_ra15 xml_name=e2e_a1v_ra15 max_epoch_num=200 num_threads=4 enable_wandb=false fix_skeleton=true +reward_specs.ctrl_cost_coeff=0.04 +reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true +fix_base=true +env_specs.init_self_contact=true +env_specs.init_self_contact_caps=true +env_specs.cube_init_noise=0.02 +env_specs.init_contact_margin=0.006"
# run 名|seed|robot_param_scale
JOBS="ra15_pusher_beta|0|1
ra15_pusher_alpha|0|0.0001
ra15_pusher_beta_s2|1|1
ra15_pusher_alpha_s2|1|0.0001"
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

log "=== 軸6 L-4（系譜 9-329）HEAD=$(git rev-parse --short HEAD)（ブランチ $(git rev-parse --abbrev-ref HEAD)）。$(echo $RUNS | wc -w) 本。main のキューの終わりを待つ"
while ps -eo args | grep -qE "[q]ueue_1029_fix1_seed2.sh|[q]ueue_1029_long_frozen.sh"; do sleep 300; done
log "main のキューが終わった。起動を始める"
PROJ_DIR=/userdir/StackelbergPPO_ra RUN_FILTER='^ra15_' nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
log "学習中の見張り watch_midrun.sh（ブランチ、ra15_ だけ）を起動（PID $!）"
PROJ_DIR=/userdir/StackelbergPPO_ra RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh（ブランチ）を起動（PID $!、RUNS=$RUNS）"
echo "$JOBS" | while IFS='|' read -r RUN SEED SCALE; do
  launch "$RUN" "$SEED" "$SCALE" || break
done
log "=== 4 本の起動が済んだ（または止まった）"
