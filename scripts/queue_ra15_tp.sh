#!/bin/bash
# 軸6 の Target-Pusher: 部品の腕（β、長さだけ学習）で、箱を目標まで押して止める（系譜 9-341、事前登録 ㊽、修論 4.7）。
#   ユーザー判断 2026-11-06「robot assemblerでtarget pusherがいいな」「修論にも入れよう。印は映えを意識して星マークで」。
# ⚠️ ブランチ feat/layer4-robotassembler にだけある。worktree /userdir/StackelbergPPO_ra から起動する。
# ⭐ 起動行は queue_ra15_l4.sh の β（ra15_pusher_beta）と同じで、変えるのは次の 2 つ（押し → 目標まで押すへタスクを替える）:
#   +reward_specs.use_target_reward=true +reward_specs.target_x=0.40 +reward_specs.target_y=0.0（箱の 0.2 m 先。全寸の Target-Pusher の 1/5）
#   +reward_specs.ctrl_cost_coeff 0.04 → 0.067（距離の報酬の上限は 1/5 で 0.5 → 0.167。上限と制御コストの比を全寸 0.5 : 0.2 と同じに保つ）
# ⭐ B・C-a（main の queue_1106_advice.sh）を先にする: その bash が 4 本を起動し終えて消えるまで待ってから起動する。
#
# ⭐ 事前登録した読み（系譜 9-341 に同じ文。結果を見てから変えない）:
#   前提: 完走・check_before_conclusion.py（比べる相手は無い）・目視（目視記録に 1 行）・ワーカーの種。
#   量: best を 50 話再生し、最後の step の箱の中心と目標（0.40, 0）の水平の距離 d_end。0.03 m（箱の辺の半分）以内を「止めた」と数える。
#   ㊽-a 2 seed とも 40 / 50 話以上で止めた → 修論 4.7 に「目標の位置へ押して止めるタスクでも、部品の設計空間で学習から形の出力まで機能した」
#   ㊽-b ㊽-a でなく、2 seed とも d_end の中央値が 0.10 m 未満（初期の距離の半分） → 「目標へ近づけたが、0.03 m 以内に止める精度には届かない」と数値つきで
#   ㊽-c どちらかの seed で d_end の中央値が 0.10 m 以上 → 修論 5.3 の課題に 1 文（200 エポックでは目標で止める動きを学習できなかった）。原因の推測は再生して見るまで書かない
#   判定に使わない観察: best の exec_R_eps と上限 0.167 の比（9-88 の頭打ち）・揺れ（measure_jitter.py の 5 話の中央値）・best の長さ
set -u
cd /userdir/StackelbergPPO_ra
Q=single_run/queue_ra15_tp.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
PHYS="CNOID_SELF_COLLISION=1 CNOID_CULLING_DEPTH=1.0 CNOID_CTRL_CLIP=1 CNOID_HINGE_DAMP_TORQUE=1 CNOID_JOINT_MAX_VEL=5.97 CNOID_JOINT_MAX_VEL_STRICT=1 CNOID_TICK_SYNC=1 CNOID_SUBSTEPS=4 CNOID_JOINT_LIMIT_TORQUE=1"
COMMON="cfg=pusher_ra15 xml_name=e2e_a1v_ra15 max_epoch_num=200 num_threads=4 enable_wandb=false fix_skeleton=true +reward_specs.ctrl_cost_coeff=0.067 +reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true +fix_base=true +env_specs.init_self_contact=true +env_specs.init_self_contact_caps=true +env_specs.cube_init_noise=0.02 +env_specs.init_contact_margin=0.006 +reward_specs.use_target_reward=true +reward_specs.target_x=0.40 +reward_specs.target_y=0.0"
# run 名|seed|robot_param_scale
JOBS="ra15_tp_beta|0|1
ra15_tp_beta_s2|1|1"
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
  grep -q "use_target_reward: true" single_run/$RUN/.hydra/config.yaml && grep -q "target_x: 0.4" single_run/$RUN/.hydra/config.yaml \
    && log "✅ $RUN: 目標の報酬（x = 0.40）" || log "⛔⛔ $RUN: config.yaml に目標の報酬か目標の位置が無い"
  return 0; }

log "=== 軸6 の Target-Pusher（系譜 9-341 ㊽）HEAD=$(git rev-parse --short HEAD)（ブランチ $(git rev-parse --abbrev-ref HEAD)）。$(echo $RUNS | wc -w) 本"
while ps -eo args | grep -q "[q]ueue_1106_advice.sh"; do sleep 60; done
log "main の queue_1106_advice.sh（B・C-a）が 4 本を起動し終えたので始める"
PROJ_DIR=/userdir/StackelbergPPO_ra RUN_FILTER='^ra15_tp_beta' nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
log "学習中の見張り watch_midrun.sh（ブランチ、ra15_tp_ だけ）を起動（PID $!）"
PROJ_DIR=/userdir/StackelbergPPO_ra RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh（ブランチ）を起動（PID $!、RUNS=$RUNS）"
echo "$JOBS" | while IFS='|' read -r RUN SEED SCALE; do
  launch "$RUN" "$SEED" "$SCALE" || break
done
log "=== 2 本の起動が済んだ（または止まった）"
