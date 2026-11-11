#!/bin/bash
# 軸6 の Target-Pusher・Δφ/dt（系譜 9-344・9-345）の 3 seed を、200 エポックの続きから 400 エポックまで回す（系譜 9-346、事前登録 読み 53）。
#   ユーザー判断 2026-11-11「200EPOCHから続けて400ぐらいまでやってみる？」→「3 seed すべて」。
# ⚠️ ブランチ feat/layer4-robotassembler にだけある。worktree /userdir/StackelbergPPO_ra から起動する。
# ⭐ 起動行は queue_ra15_tp_rate.sh（9-344）・queue_ra15_tp_rate_s3.sh（9-345）と同じで、変えるのは
#   max_epoch_num 200 → 400・+restore_dir=single_run/<元の run> epoch=200（epoch_0200.p を読み ep200 から回す）・run 名だけ。
#   元の run のフォルダには書かない。
# ⚠️ 続きとして同じでない所: Adam の内部の状態は checkpoint に無く再開時にリセット。乱数は起動時に seed へ戻る。影響は seed 1（対照）で見る。
#
# ⭐ 事前登録した読み 53（系譜 9-346 に同じ文。結果を見てから変えない）:
#   前提: 完走（ep399 まで）・check_before_conclusion.py・目視（目視記録に 1 行）・ワーカーの種・起動の確認
#         （epoch_0200.p を読んだ・最初の行が ep200・目標の報酬と制御コスト 0.16・dt 0.0100 の行）。
#   量: 各 run の best を 50 話再生。① 箱の x の移動の中央値（0.01 m 以上を「触れた」）② d_end（最後の箱と目標 (0.40, 0) の水平の距離）の中央値。
#       best は新しいフォルダの best.p（epoch_0200.p の best_rewards を超えたときだけ保存）。無ければ 200 エポックの best のまま＝判定もそのまま。
#   seed 0: 53-0a ① ≥ 0.01 m →「200 エポックで触れなかったのは学習の量の不足でありうる（400 では触れた）」。53-0b ① < 0.01 m →「400 エポックでも触れない」。
#   seed 2: 53-2a ② < 0.03 m →「200 エポックでは学び切っていなかった（seed 1 の水準に来る）」。53-2b ② ≥ 0.03 m →「倍にしても seed 1 の精度に届かない」。
#   seed 1（対照）: 53-1a ② < 0.03 m を保つ →「続きから回すことは解を崩さない」。53-1b ② ≥ 0.03 m →「崩しうる」（53-0・53-2 にも添える）。
#   打ち切り: 400 で止める。報酬・制御コスト・seed は変えない。触れない・届かない原因は書かない。
#   判定に使わない観察: 評価の報酬の 20 エポック平均の推移・ep400 でまだ上がっているか・押し方（当てて離した後の滑り）・制御コスト。
set -u
cd /userdir/StackelbergPPO_ra
Q=single_run/queue_ra15_tp_rate_ext400.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo comm,args 2>/dev/null | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
PHYS="CNOID_SELF_COLLISION=1 CNOID_CULLING_DEPTH=1.0 CNOID_CTRL_CLIP=1 CNOID_HINGE_DAMP_TORQUE=1 CNOID_JOINT_MAX_VEL=5.97 CNOID_JOINT_MAX_VEL_STRICT=1 CNOID_TICK_SYNC=1 CNOID_SUBSTEPS=4 CNOID_JOINT_LIMIT_TORQUE=1"
COMMON="cfg=pusher_ra15 xml_name=e2e_a1v_ra15 max_epoch_num=400 num_threads=4 enable_wandb=false fix_skeleton=true +reward_specs.ctrl_cost_coeff=0.16 +reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true +fix_base=true +env_specs.init_self_contact=true +env_specs.init_self_contact_caps=true +env_specs.cube_init_noise=0.02 +env_specs.init_contact_margin=0.006 +reward_specs.use_target_reward=true +reward_specs.target_x=0.40 +reward_specs.target_y=0.0 +reward_specs.target_rate=true"
# run 名|seed|robot_param_scale|続きを読む元の run（seed 0 と seed 2 を先に並列、seed 1 は空いたら）
JOBS="ra15_tp_rate_beta_ext400|0|1|ra15_tp_rate_beta
ra15_tp_rate_beta_s3_ext400|2|1|ra15_tp_rate_beta_s3
ra15_tp_rate_beta_s2_ext400|1|1|ra15_tp_rate_beta_s2"
RUNS=$(echo "$JOBS" | cut -d'|' -f1 | tr '\n' ' ')

launch(){ local RUN=$1 SEED=$2 SCALE=$3 SRC=$4
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return 0; fi
  if [ ! -f "single_run/$SRC/models/epoch_0200.p" ]; then log "⛔⛔ single_run/$SRC/models/epoch_0200.p が無い。$RUN を起動しない"; return 0; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 30; done
  local AV; AV=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "$AV" -lt 3 ]; then log "⛔⛔ ディスクの空き ${AV} GB（3 GB 未満）。$RUN 以降を起動せずに止まる"; return 1; fi
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 $PHYS \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    $COMMON seed=$SEED +robot_param_scale=$SCALE +restore_dir=single_run/$SRC epoch=200 \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)  seed $SEED  robot_param_scale $SCALE  ⭐ $SRC の epoch_0200.p から 400 まで  env: $PHYS"; sleep 90
  local want="[$((1000*SEED+1)), $((1000*SEED+2)), $((1000*SEED+3)), $((1000*SEED+4))]" s
  s=$(grep -ao 'ワーカーの乱数の種 \[[0-9, ]*\]' single_run/$RUN/stdout.log | head -1)
  case "$s" in *"$want"*) log "✅ $RUN: $s";; *) log "⛔⛔ $RUN: 種の行が期待（$want）と違う／無い: '$s'";; esac
  grep -q "可動域をトルクで掛ける: 4 関節" single_run/$RUN/stdout.log && log "✅ $RUN: 可動域のトルク 4 関節" || log "⛔⛔ $RUN: 可動域のトルクの行が無い"
  grep -q "4 分割" single_run/$RUN/stdout.log && log "✅ $RUN: 刻み 4 分割" || log "⛔⛔ $RUN: 刻みの分割の行が無い"
  local C=single_run/$RUN/.hydra/config.yaml
  grep -q "use_target_reward: true" $C && grep -q "target_x: 0.4" $C && grep -q "target_rate: true" $C && grep -q "ctrl_cost_coeff: 0.16" $C && grep -q "max_epoch_num: 400" $C \
    && log "✅ $RUN: 目標の報酬（x = 0.40、Δφ/dt）・制御コスト 0.16・max_epoch_num 400" || log "⛔⛔ $RUN: config.yaml の目標の報酬・target_rate・制御コスト・max_epoch_num のどれかが違う"
  # 続きから読んだか（genesis_agent.load_checkpoint の行）と、最初のエポックの行が 200 か。最初の epoch の行まで最大 10 分待つ
  local L=single_run/$RUN/log/log_train.txt
  grep -aq "loading model from checkpoint: .*/single_run/$SRC/models/epoch_0200.p" $L 2>/dev/null \
    && log "✅ $RUN: single_run/$SRC/models/epoch_0200.p を読んだ" || log "⛔⛔ $RUN: epoch_0200.p を読んだ行が log_train.txt に無い（続きになっていない）"
  local k=0; while [ $k -lt 20 ] && ! grep -aqP '\] \d+\tT_sample' $L 2>/dev/null; do sleep 30; k=$((k+1)); done
  s=$(grep -aoP '\] \d+\tT_sample' $L 2>/dev/null | head -1 | grep -oP '\d+')
  [ "$s" = "200" ] && log "✅ $RUN: 最初のエポックの行が 200" || log "⛔⛔ $RUN: 最初のエポックの行が 200 でない／無い: '$s'"
  # Δφ/dt の行は env の中で出るので、ワーカーのログ（/tmp/cnoid_worker_<学習の PID>_*.log）にも探す（9-344 の誤報の型）
  local P; P=$(ps -eo pid,args | awk -v r="hydra.run.dir=single_run/$RUN" '/[c]horeonoid_train\.py/ && index($0, r) {print $1; exit}')
  s=$(cat single_run/$RUN/stdout.log /tmp/cnoid_worker_${P}_*.log 2>/dev/null | grep -ao "\[target\] 目標の報酬を Δφ/dt にする: dt = [0-9.]* s（Δφ の × [0-9.]*）" | head -1)
  case "$s" in *"dt = 0.0100 s（Δφ の × 100.0）"*) log "✅ $RUN: $s";; *) log "⛔⛔ $RUN: Δφ/dt の行が期待（dt = 0.0100 s、× 100.0）と違う／無い（学習の PID $P）: '$s'";; esac
  return 0; }

log "=== 軸6 の Target-Pusher・Δφ/dt の 3 seed を 400 エポックへ（系譜 9-346 読み 53）HEAD=$(git rev-parse --short HEAD)（ブランチ $(git rev-parse --abbrev-ref HEAD)）。$(echo $RUNS | wc -w) 本"
if ! grep -q "target_rate" design_opt/envs/pusher.py; then
  log "⛔⛔ worktree の pusher.py に target_rate が無い。起動せずに止まる"; exit 1
fi
PROJ_DIR=/userdir/StackelbergPPO_ra RUN_FILTER='^ra15_tp_rate_beta.*_ext400' nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
log "学習中の見張り watch_midrun.sh（ブランチ、*_ext400 だけ）を起動（PID $!）"
PROJ_DIR=/userdir/StackelbergPPO_ra RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh（ブランチ）を起動（PID $!、RUNS=$RUNS）"
echo "$JOBS" | while IFS='|' read -r RUN SEED SCALE SRC; do
  launch "$RUN" "$SEED" "$SCALE" "$SRC" || break
done
log "=== 3 本の起動が済んだ（または止まった）"
