#!/bin/bash
# 軸6 L-4 の取り直し: 制御コスト 0.04 → 0.16（系譜 9-343、事前登録 ㊾・㊿、修論 4.7 の 4 本を置き換える）。
#   ユーザー判断 2026-11-06「ｃで」（Bug 94 の直し方 C: 比を保つ制御コストで 4.7 の 4 本を取り直す）。
# ⚠️ ブランチ feat/layer4-robotassembler にだけある。worktree /userdir/StackelbergPPO_ra から起動する。
# ⭐ なぜ 0.16: CNOID_SUBSTEPS=4 で env.dt が 1/4 になり（Bug 94）、押しの報酬は箱の速さの 4 倍で積まれる。全寸（4.1.2、SUBSTEPS なし）
#   の報酬と制御コストの比を保つ係数は 0.2 × (1/5) × 4 = 0.16（旧 0.04 は 1/4 の重さだった）。env.dt は直さない（比べ方を元の 4 本とそろえる）。
# ⭐ 起動行は queue_ra15_l4.sh と同じで、変えるのは +reward_specs.ctrl_cost_coeff 0.04 → 0.16 だけ。β と α の差は +robot_param_scale（1 / 0.0001）だけ。
# ⭐ 順番: Target-Pusher（queue_ra15_tp.sh）の bash が 2 本を起動し終えて消えるまで待つ（そのキューを書き換えずに後ろへ付ける）。
#
# ⭐ 事前登録した読み（系譜 9-343 に同じ文。結果を見てから変えない）:
#   前提: 完走・check_before_conclusion.py（比べる相手 ra15_pusher_* との起動行の差が ctrl_cost_coeff だけ）・目視・ワーカーの種。帯は best で読む。
#   ㊾-a 共設計が足した: β の 2 seed の best の帯の下端 > α の帯の上端
#   ㊾-b 差が言えない: 帯が重なる → 「部品の制約の中では形を変える余地が小さかった」と書き、順位は言わない
#   ㊾-c 共設計が妨げた: β の帯の上端 < α の帯の下端
#   ㊿ 丸め（replay_rounded_lengths.py、50 話、ROUND=1 と対照の ROUND=0）: 丸めた長さの 50 話の中央値が β の 2 seed の帯に入れば
#      「出力した形は性能を保ったまま部品で組める」。入らないときは、丸めない対照の中央値と比べて丸めの効き（%）を書く（9-329 の ㉝ と同じ扱い）
#   判定に使わない観察: 揺れ（measure_jitter.py の 5 話の中央値。9-332 の基準 β seed 1 が 60 回）・best の長さ
set -u
cd /userdir/StackelbergPPO_ra
Q=single_run/queue_ra15_cc016.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
PHYS="CNOID_SELF_COLLISION=1 CNOID_CULLING_DEPTH=1.0 CNOID_CTRL_CLIP=1 CNOID_HINGE_DAMP_TORQUE=1 CNOID_JOINT_MAX_VEL=5.97 CNOID_JOINT_MAX_VEL_STRICT=1 CNOID_TICK_SYNC=1 CNOID_SUBSTEPS=4 CNOID_JOINT_LIMIT_TORQUE=1"
COMMON="cfg=pusher_ra15 xml_name=e2e_a1v_ra15 max_epoch_num=200 num_threads=4 enable_wandb=false fix_skeleton=true +reward_specs.ctrl_cost_coeff=0.16 +reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true +fix_base=true +env_specs.init_self_contact=true +env_specs.init_self_contact_caps=true +env_specs.cube_init_noise=0.02 +env_specs.init_contact_margin=0.006"
# run 名|seed|robot_param_scale
JOBS="ra15_pusher_beta_cc016|0|1
ra15_pusher_alpha_cc016|0|0.0001
ra15_pusher_beta_cc016_s2|1|1
ra15_pusher_alpha_cc016_s2|1|0.0001"
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
  grep -q "ctrl_cost_coeff: 0.16" single_run/$RUN/.hydra/config.yaml && log "✅ $RUN: 制御コスト 0.16" || log "⛔⛔ $RUN: config.yaml の制御コストが 0.16 でない"
  return 0; }

log "=== 軸6 L-4 の取り直し・制御コスト 0.16（系譜 9-343 ㊾・㊿）HEAD=$(git rev-parse --short HEAD)（ブランチ $(git rev-parse --abbrev-ref HEAD)）。$(echo $RUNS | wc -w) 本"
while ps -eo args | grep -q "[q]ueue_ra15_tp.sh"; do sleep 60; done
log "queue_ra15_tp.sh（Target-Pusher）が 2 本を起動し終えたので始める"
PROJ_DIR=/userdir/StackelbergPPO_ra RUN_FILTER='^ra15_pusher_(beta|alpha)_cc016' nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
log "学習中の見張り watch_midrun.sh（ブランチ、cc016 だけ）を起動（PID $!）"
PROJ_DIR=/userdir/StackelbergPPO_ra RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh（ブランチ）を起動（PID $!、RUNS=$RUNS）"
echo "$JOBS" | while IFS='|' read -r RUN SEED SCALE; do
  launch "$RUN" "$SEED" "$SCALE" || break
done
log "=== 4 本の起動が済んだ（または止まった）"
