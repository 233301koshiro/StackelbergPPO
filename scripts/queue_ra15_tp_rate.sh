#!/bin/bash
# 軸6 の Target-Pusher を、目標の報酬を Δφ/dt にして取り直す（系譜 9-344、事前登録 読み 51。9-341 ㊽ の取り直し、1 回で打ち切る）。
#   ユーザー判断 2026-11-08「やろう」（「原因がわかりきってそうなものを試さずに限界に書くのも嫌だ」「報酬設計に沼るのは嫌」）。
# ⚠️ ブランチ feat/layer4-robotassembler にだけある。worktree /userdir/StackelbergPPO_ra から起動する。
# ⭐ なぜ: 9-341 の seed 0 は 200 エポックで一度も箱を動かさなかった。形は箱に届き（最後の体が 0.246 m、箱 0.203 m）、
#   同じ種の押し ra15_pusher_beta は ep0 の試行で箱を平均約 0.18 m 動かして ep1 で箱を狙った。Δφ は 1 cm で約 0.007 で、
#   1 話の制御コスト約 −17 に埋もれていた（9-341 の追記）。
# ⭐ 起動行は queue_ra15_tp.sh と同じで、変えるのは 2 つ:
#   +reward_specs.target_rate=true（Δφ/dt。押しの Δx/dt と同じ「進む速さ」の形。dt = 0.01 s なので Δφ の 100 倍。main 2d3a253 の pusher.py）
#   +reward_specs.ctrl_cost_coeff 0.067 → 0.16（押しの取り直し queue_ra15_cc016.sh と同じ）。⇒ 押しの cc016 との差はタスク（何に近づくか）だけ
# ⭐ 起動の条件: worktree の pusher.py に target_rate が入っていること（main からの git merge の後）。入っていなければ起動せずに止まる。
# ⭐ 本数は実行ファイル名で数える（Bug 101: ps のコマンド行の文字列で数えると、それを含むシェルを 1 本と数えて待ち続ける）。
#
# ⭐ 事前登録した読み（系譜 9-344 に同じ文。結果を見てから変えない）:
#   前提: 完走・check_before_conclusion.py・目視（目視記録に 1 行）・ワーカーの種・起動の確認（target_rate・制御コスト 0.16・dt 0.0100 の行）。
#   量: best を 50 話再生。① 箱の x の移動の中央値（0.01 m 以上を「触れた」）② 最後の step の箱と目標 (0.40, 0) の水平の距離 d_end。
#   51-a 2 seed とも箱の移動の中央値 ≥ 0.01 m → 9-341 の seed 0 が触れなかったのは、箱の報酬の信号が制御コストに埋もれていたため、と書く。
#        そのうえで止める精度を ㊽ と同じ基準で a / b / c と付ける（2 seed とも 40/50 話が 0.03 m 以内 / 2 seed とも中央値 < 0.10 m / それ以外）。
#   51-b どちらかの seed で箱の移動の中央値 < 0.01 m → 信号の大きさだけでは説明できない。報酬はこれ以上変えず、限界として書く。
#   打ち切り: 結果によらず、報酬・制御コスト・エポックをこれ以上変えた取り直しはしない。
#   修論: 判定の後に、研究方針「増やした実験を修論に書くかの基準」（11-05）で置き場を決める。9-341 の 2 本は台帳に残す。
#   判定に使わない観察: 評価が初めて正になった epoch・best の exec_R_eps と上限（0.167 / 0.01 = 16.7）の比・揺れ（measure_jitter.py）・best の長さ。
set -u
cd /userdir/StackelbergPPO_ra
Q=single_run/queue_ra15_tp_rate.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo comm,args 2>/dev/null | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
PHYS="CNOID_SELF_COLLISION=1 CNOID_CULLING_DEPTH=1.0 CNOID_CTRL_CLIP=1 CNOID_HINGE_DAMP_TORQUE=1 CNOID_JOINT_MAX_VEL=5.97 CNOID_JOINT_MAX_VEL_STRICT=1 CNOID_TICK_SYNC=1 CNOID_SUBSTEPS=4 CNOID_JOINT_LIMIT_TORQUE=1"
COMMON="cfg=pusher_ra15 xml_name=e2e_a1v_ra15 max_epoch_num=200 num_threads=4 enable_wandb=false fix_skeleton=true +reward_specs.ctrl_cost_coeff=0.16 +reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true +fix_base=true +env_specs.init_self_contact=true +env_specs.init_self_contact_caps=true +env_specs.cube_init_noise=0.02 +env_specs.init_contact_margin=0.006 +reward_specs.use_target_reward=true +reward_specs.target_x=0.40 +reward_specs.target_y=0.0 +reward_specs.target_rate=true"
# run 名|seed|robot_param_scale
JOBS="ra15_tp_rate_beta|0|1
ra15_tp_rate_beta_s2|1|1"
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

log "=== 軸6 の Target-Pusher の取り直し・Δφ/dt（系譜 9-344 読み 51）HEAD=$(git rev-parse --short HEAD)（ブランチ $(git rev-parse --abbrev-ref HEAD)）。$(echo $RUNS | wc -w) 本"
if ! grep -q "target_rate" design_opt/envs/pusher.py; then
  log "⛔⛔ worktree の pusher.py に target_rate が無い（main を git merge していない）。起動せずに止まる"; exit 1
fi
PROJ_DIR=/userdir/StackelbergPPO_ra RUN_FILTER='^ra15_tp_rate_beta' nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
log "学習中の見張り watch_midrun.sh（ブランチ、ra15_tp_rate_ だけ）を起動（PID $!）"
PROJ_DIR=/userdir/StackelbergPPO_ra RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh（ブランチ）を起動（PID $!、RUNS=$RUNS）"
echo "$JOBS" | while IFS='|' read -r RUN SEED SCALE; do
  launch "$RUN" "$SEED" "$SCALE" || break
done
log "=== 2 本の起動が済んだ（または止まった）"
