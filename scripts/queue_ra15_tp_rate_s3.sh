#!/bin/bash
# 軸6 の Target-Pusher・Δφ/dt（系譜 9-344）に seed 2 を 1 本足す（系譜 9-345、事前登録 読み 52）。
#   ユーザー判断 2026-11-10「せめて別のseedでもう1本やれれば分かるかな」→「どちらもやる」。
# ⚠️ ブランチ feat/layer4-robotassembler にだけある。worktree /userdir/StackelbergPPO_ra から起動する。
# ⭐ 起動行は queue_ra15_tp_rate.sh（9-344）と同じで、変えるのは seed（2、ワーカーの種 2001〜2004）と run 名だけ。
#   9-344 の 2 本と同じコードで回すため、worktree へ main を取り込む前に起動する。
#
# ⭐ 事前登録した読み 52（系譜 9-345 に同じ文。結果を見てから変えない）:
#   前提: 完走・check_before_conclusion.py・目視（目視記録に 1 行）・ワーカーの種・起動の確認（target_rate・制御コスト 0.16・dt 0.0100 の行）。
#   量: best を 50 話再生。箱の x の移動の中央値（0.01 m 以上を「触れた」）。
#   52-a 中央値 ≥ 0.01 m → 「触れるかどうかは seed で分かれた（3 seed 中 2 seed が触れた）」と書く。
#   52-b 中央値 < 0.01 m → 「触れるかどうかは seed で分かれた（3 seed 中 1 seed しか触れなかった）」と書く。
#   どちらでも: 触れない原因は書かない（seed は初期の重み・探索の乱数・箱の初期位置をまとめた番号で、理由ではない）。
#   打ち切り: seed はこれ以上足さない。報酬・制御コスト・エポックも変えない（9-344 の打ち切りのまま）。
#   判定に使わない観察: d_end の中央値・0.03 m 以内の話の数・評価が初めて正になった epoch・腕を −y 側へ倒して止まる姿勢か。
set -u
cd /userdir/StackelbergPPO_ra
Q=single_run/queue_ra15_tp_rate_s3.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo comm,args 2>/dev/null | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
PHYS="CNOID_SELF_COLLISION=1 CNOID_CULLING_DEPTH=1.0 CNOID_CTRL_CLIP=1 CNOID_HINGE_DAMP_TORQUE=1 CNOID_JOINT_MAX_VEL=5.97 CNOID_JOINT_MAX_VEL_STRICT=1 CNOID_TICK_SYNC=1 CNOID_SUBSTEPS=4 CNOID_JOINT_LIMIT_TORQUE=1"
COMMON="cfg=pusher_ra15 xml_name=e2e_a1v_ra15 max_epoch_num=200 num_threads=4 enable_wandb=false fix_skeleton=true +reward_specs.ctrl_cost_coeff=0.16 +reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true +fix_base=true +env_specs.init_self_contact=true +env_specs.init_self_contact_caps=true +env_specs.cube_init_noise=0.02 +env_specs.init_contact_margin=0.006 +reward_specs.use_target_reward=true +reward_specs.target_x=0.40 +reward_specs.target_y=0.0 +reward_specs.target_rate=true"
# run 名|seed|robot_param_scale
JOBS="ra15_tp_rate_beta_s3|2|1"
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
  local C=single_run/$RUN/.hydra/config.yaml
  grep -q "use_target_reward: true" $C && grep -q "target_x: 0.4" $C && grep -q "target_rate: true" $C && grep -q "ctrl_cost_coeff: 0.16" $C \
    && log "✅ $RUN: 目標の報酬（x = 0.40、Δφ/dt）・制御コスト 0.16" || log "⛔⛔ $RUN: config.yaml の目標の報酬・target_rate・制御コストのどれかが違う"
  # Δφ/dt の行は env が最初に箱の報酬を計算したときに出る（期待: dt = 0.0100 s、× 100.0）。最初の epoch の評価まで最大 10 分待つ
  local k=0; while [ $k -lt 20 ] && ! grep -q "\[target\] 目標の報酬を Δφ/dt にする" single_run/$RUN/stdout.log; do sleep 30; k=$((k+1)); done
  s=$(grep -ao "\[target\] 目標の報酬を Δφ/dt にする: dt = [0-9.]* s（Δφ の × [0-9.]*）" single_run/$RUN/stdout.log | head -1)
  case "$s" in *"dt = 0.0100 s（Δφ の × 100.0）"*) log "✅ $RUN: $s";; *) log "⛔⛔ $RUN: Δφ/dt の行が期待（dt = 0.0100 s、× 100.0）と違う／無い: '$s'";; esac
  return 0; }

log "=== 軸6 の Target-Pusher・Δφ/dt の seed 2（系譜 9-345 読み 52）HEAD=$(git rev-parse --short HEAD)（ブランチ $(git rev-parse --abbrev-ref HEAD)）。$(echo $RUNS | wc -w) 本"
if ! grep -q "target_rate" design_opt/envs/pusher.py; then
  log "⛔⛔ worktree の pusher.py に target_rate が無い（main を git merge していない）。起動せずに止まる"; exit 1
fi
PROJ_DIR=/userdir/StackelbergPPO_ra RUN_FILTER='^ra15_tp_rate_beta_s3' nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
log "学習中の見張り watch_midrun.sh（ブランチ、ra15_tp_rate_beta_s3 だけ）を起動（PID $!）"
PROJ_DIR=/userdir/StackelbergPPO_ra RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh（ブランチ）を起動（PID $!、RUNS=$RUNS）"
echo "$JOBS" | while IFS='|' read -r RUN SEED SCALE; do
  launch "$RUN" "$SEED" "$SCALE" || break
done
log "=== 1 本の起動が済んだ（または止まった）"
