#!/bin/bash
# ばねで引き戻される箱を目標まで押して保つ課題（系譜 9-354、事前登録 読み 58）の段 1、6 本。
# 第2層（ギア比の上限への張り付き → 出力不足）に、学習の外の正解（静力学）を持たせて確かめる（11-25 ユーザー「aにするか」「1,2をいれる」「制御コストを 0 に」「いいよ」）。
#
# ⭐ 起動行は縦型 A1v の押し e2e_a1v_pusher_phys_s3_vp（Bug 106・107 を直した物理）の `.hydra/overrides.yaml` と `launch_env.json` を読み、次だけを変える:
#   seed／+reward_specs.ctrl_cost_coeff 0.2 → 0／+reward_specs.use_target_reward=true・target_x=1.0・target_y=0.30 を足す／
#   環境変数 CNOID_CUBE_SPRING_K（350・900）と CNOID_CUBE_SPRING_AXIS=y を足す。cfg（ギア比 25〜150）・XML・他の物理は同じ。
# ⭐ 学習の本数は実行ファイル名が choreonoid のものだけを数える（Bug 101）。見張り 2 つもこのキューから起動する。ディスクの空きが 3 GB を切ったら止まる。
# ⭐ 段 2（上限 500 の閉ループ e2e_a1v_spring900_ub500・_s2_ws・_s3）はこのキューに入れない。読み 58-2 が a か b のときだけ、ユーザーに確かめて別に投入する。
#
# ⭐ 事前登録した読み 58（結果を見てから変えない。系譜 9-354 に同じ文）:
#   前提: 完走・check_before_conclusion.py・目視（目視記録に 1 行）・ワーカーの種・起動の確認（launch_env.json の CNOID_CUBE_SPRING_K、第 10 エポックの再生ログ trace_mid/ep0010.log の「⭐ 箱にばねを掛ける … k=350（900） N/m・関節 ['cube_slide2']」の行、
#         overrides の ctrl_cost_coeff=0 と target_y=0.30、launch_env.json の CNOID_CUBE_SPRING_K）。
#   量: best を 50 話再生。d_end（最後の step の箱と目標 (1.0, 0.30) の水平の距離）の中央値。届いた＝中央値 ≤ 0.05 m。第2層の出力（diagnose_morphology.py）。旋回関節（1_joint）のギア比。
#   静力学の検査（学習なし）: ① k 350 は要るトルク 81〜129 N·m ≤ 150 で保てる。② k 900 は 209〜331 N·m > 150 で保てない。
#   58-1（①、e2e_a1v_spring350・_s2_ws・_s3）: a 3 seed とも届き、第2層が出力不足と出さない／b 3 seed とも届くが、1 seed 以上で第2層が出力不足と出す／c 届かない seed がある（判定に使わない）。
#   58-2（②、e2e_a1v_spring900・_s2_ws・_s3）: ⛔ 1 seed でも届いたら抜け穴を疑い、結論を書かずに調べる。届かないとき: a 3 seed とも第2層が出力不足と出す／
#         b 3 seed とも旋回関節のギア比 ≥ 125 N·m だが、第2層が出力不足と出さない seed がある／c 旋回関節が 125 N·m に届かない seed がある。
#   58-3（段 2）: a 3 seed とも届く／b 届かない seed がある。
#   書き方: どの結果でも、制御コスト 0 の課題での結果で、第2層を当てている押しの run（0.2）とは条件が違うと書く。学習が要ることの証拠にはしない。
#   記録のみ: 旋回関節のギア比と静力学の必要値（① 81〜129、② 209〜331 N·m）、ほかの関節のギア比、best、d_end の分布。
#   予想（判定に使わない）: 58-1 は b 寄り、58-2 は b 寄り。
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_1125_spring_box.log
SRC=e2e_a1v_pusher_phys_s3_vp
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo comm,args | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
# run 名|seed|k [N/m]
JOBS="e2e_a1v_spring350|0|350
e2e_a1v_spring900|0|900
e2e_a1v_spring350_s2_ws|1|350
e2e_a1v_spring900_s2_ws|1|900
e2e_a1v_spring350_s3|2|350
e2e_a1v_spring900_s3|2|900"
RUNS=$(echo "$JOBS" | cut -d'|' -f1 | tr '\n' ' ')
launch(){ local RUN=$1 SEED=$2 K=$3
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return 0; fi
  if [ ! -f "single_run/$SRC/.hydra/overrides.yaml" ] || [ ! -f "single_run/$SRC/launch_env.json" ]; then log "⛔⛔ $SRC の overrides か launch_env が無い。止まる"; return 1; fi
  local ARGS ENVS
  ARGS=$(sed -n 's/^- //p' single_run/$SRC/.hydra/overrides.yaml | sed "s/^seed=.*/seed=$SEED/; s/^+reward_specs.ctrl_cost_coeff=.*/+reward_specs.ctrl_cost_coeff=0/" | tr '\n' ' ')
  ARGS="$ARGS +reward_specs.use_target_reward=true +reward_specs.target_x=1.0 +reward_specs.target_y=0.30"
  case "$ARGS" in *"ctrl_cost_coeff=0 "*) ;; *) log "⛔⛔ $RUN: 起動行に ctrl_cost_coeff=0 が無い。止まる: $ARGS"; return 1;; esac
  ENVS=$(python3 -c "import json;print(' '.join(f'{k}={v}' for k,v in json.load(open('single_run/$SRC/launch_env.json')).items()))")
  ENVS="$ENVS CNOID_CUBE_SPRING_K=$K CNOID_CUBE_SPRING_AXIS=y"
  while [ "$(njobs)" -ge 2 ]; do sleep 15; done
  local AV; AV=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "$AV" -lt 3 ]; then log "⛔⛔ ディスクの空き ${AV} GB（3 GB 未満）。$RUN 以降を起動せずに止まる"; return 1; fi
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 $ENVS \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    $ARGS hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)  元 $SRC  env: $ENVS  args: $ARGS"; sleep 90
  local want="[$((1000*SEED+1)), $((1000*SEED+2)), $((1000*SEED+3)), $((1000*SEED+4))]" s b
  s=$(grep -ao 'ワーカーの乱数の種 \[[0-9, ]*\]' single_run/$RUN/stdout.log | head -1)
  case "$s" in *"$want"*) log "✅ $RUN: $s";; *) log "⛔⛔ $RUN: 種の行が期待（$want）と違う／無い: '$s'";; esac
  # ばねの行（「⭐ 箱にばねを掛ける … k=… 関節 ['cube_slide2']」）はワーカーにだけ出て run のフォルダに残らない（PRETICK と同じ、9-351）。
  #   起動の直後は launch_env.json に k が入ったかを見る。行そのものは見張りの第 10 エポックの再生（trace_mid/ep0010.log。launch_env.json を当てて再生する）で見る。
  b=$(python3 -c "import json;print(json.load(open('single_run/$RUN/launch_env.json')).get('CNOID_CUBE_SPRING_K',''))" 2>/dev/null)
  if [ "$b" = "$K" ]; then log "✅ $RUN: launch_env.json の CNOID_CUBE_SPRING_K=$b"; else log "⛔⛔ $RUN: launch_env.json の CNOID_CUBE_SPRING_K が '$b'（期待 $K）"; fi
  return 0; }
log "=== 11-25 ばねの箱 段 1（系譜 9-354、読み 58）HEAD=$(git rev-parse --short HEAD)。$(echo $RUNS | wc -w) 本"
nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
log "学習中の見張り watch_midrun.sh を起動（PID $!）"
RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh を起動（PID $!、RUNS=$RUNS）"
echo "$JOBS" | while IFS='|' read -r RUN SEED K; do
  launch "$RUN" "$SEED" "$K" || break
done
log "=== $(echo $RUNS | wc -w) 本の起動が済んだ（または止まった）"
