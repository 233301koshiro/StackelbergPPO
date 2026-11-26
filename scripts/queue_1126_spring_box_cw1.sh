#!/bin/bash
# ばねで引き戻される箱（系譜 9-354）の取り直し、6 本。事前登録 読み 59（11-26）。
# 11-25 の段 1（読み 58、queue_1125_spring_box.sh）は seed 0・1 の 4 本が 200 エポックで箱に一度も触れなかった。
# 触れるまで報酬が 0（contact_weight 0・制御コスト 0）で学習が立ち上がらない報酬の失敗と見て、直して取り直す
# （11-26 ユーザー「ロボットのせいじゃなく、報酬設計で失敗しているのならこれは直すべき失敗」「いいよ」）。
#
# ⭐ 起動行は 11-25 の段 1 と同じ作り方（e2e_a1v_pusher_phys_s3_vp の overrides と launch_env.json を読む）で、
#   段 1 から変えるのは +reward_specs.contact_weight 0 → 1.0 の 1 つだけ（腕の先が箱に近づく Δφ の報酬。pusher.py の既定値）。
#   段 1 と同じ: seed／ctrl_cost_coeff 0／use_target_reward=true・target_x=1.0・target_y=0.30／CNOID_CUBE_SPRING_K（350・900）・CNOID_CUBE_SPRING_AXIS=y。
# ⭐ 学習の本数は実行ファイル名が choreonoid のものだけを数える（Bug 101）。2 本ずつ。ディスクの空きが 3 GB を切ったら止まる。
# ⭐ 見張り watch_midrun.sh は動いている学習を全部拾うので、既に動いていれば起動しない（二重に再生しない）。完走後の自動再生はこの 6 本だけを RUNS で渡して別に起動する。
#
# ⭐ 事前登録した読み 59（11-26 ユーザー「いいよ」で確定。結果を見てから変えない。系譜 9-354 に同じ文）:
#   前提: 読み 58 と同じ（完走・check_before_conclusion.py・目視・ワーカーの種・ばねの行・launch_env.json の k）＋ overrides の contact_weight=1.0。
#   量: 触れた＝見張りの再生（第 10・50・100・150 エポック、各 5 話）か best の 50 話の再生のどこかで、箱が 1 mm 以上動いた話が 1 話以上ある（9-258）。
#       届いた＝d_end（最後の step の箱と目標 (1.0, 0.30) の水平の距離）の 50 話の中央値 ≤ 0.05 m。第2層の出力（diagnose_morphology.py）。旋回関節（1_joint）のギア比。
#   59-0（報酬の切り分け）: a 6 本とも触れる → 段 1 が触れなかったのは報酬の失敗。段 1 の 6 本は読み 58 の判定に使わない／
#         b 触れない run がある → 近づく報酬を足しても立ち上がりが seed に依る／c 1 本も触れない → 報酬以外を疑い、ユーザーに諮る。
#   59-1（k 350: e2e_a1v_spring350_cw1・_s2_ws_cw1・_s3_cw1）・59-2（k 900: e2e_a1v_spring900_cw1・_s2_ws_cw1・_s3_cw1）:
#         その k の 3 seed がすべて触れたときだけ判定する。a〜c は 58-1・58-2 と同じ。触れない seed がある k は判定に使わずユーザーに諮る。
#         ⛔ k 900 で 1 seed でも届いたら抜け穴を疑い、結論を書かずに調べる。
#   59-3（段 2、上限 500、_cw1 付き）: 59-2 が a か b のときだけユーザーに確かめて投入。読みは 58-3 と同じ。
#   書き方: 制御コスト 0 の課題で、contact_weight が押しの主線（0）と違い、比べる相手は静力学の正解。学習が要ることの証拠にはしない。
#   記録のみ: 最初に触れたエポック、収束した形の長さと腕の向き（第0段の発見 4）、読み 58 の記録のみの項目。
#   予想（判定に使わない）: 59-0 は a 寄り。触れても +y へ押す向きを見つけず「触れるが届かない」が k 350 で出うる。
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_1126_spring_box_cw1.log
SRC=e2e_a1v_pusher_phys_s3_vp
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo comm,args | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
# run 名|seed|k [N/m]
JOBS="e2e_a1v_spring350_cw1|0|350
e2e_a1v_spring900_cw1|0|900
e2e_a1v_spring350_s2_ws_cw1|1|350
e2e_a1v_spring900_s2_ws_cw1|1|900
e2e_a1v_spring350_s3_cw1|2|350
e2e_a1v_spring900_s3_cw1|2|900"
RUNS=$(echo "$JOBS" | cut -d'|' -f1 | tr '\n' ' ')
launch(){ local RUN=$1 SEED=$2 K=$3
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return 0; fi
  if [ ! -f "single_run/$SRC/.hydra/overrides.yaml" ] || [ ! -f "single_run/$SRC/launch_env.json" ]; then log "⛔⛔ $SRC の overrides か launch_env が無い。止まる"; return 1; fi
  local ARGS ENVS
  ARGS=$(sed -n 's/^- //p' single_run/$SRC/.hydra/overrides.yaml | sed "s/^seed=.*/seed=$SEED/; s/^+reward_specs.ctrl_cost_coeff=.*/+reward_specs.ctrl_cost_coeff=0/; s/^+reward_specs.contact_weight=.*/+reward_specs.contact_weight=1.0/" | tr '\n' ' ')
  ARGS="$ARGS +reward_specs.use_target_reward=true +reward_specs.target_x=1.0 +reward_specs.target_y=0.30"
  case "$ARGS" in *"ctrl_cost_coeff=0 "*) ;; *) log "⛔⛔ $RUN: 起動行に ctrl_cost_coeff=0 が無い。止まる: $ARGS"; return 1;; esac
  case "$ARGS" in *"contact_weight=1.0 "*) ;; *) log "⛔⛔ $RUN: 起動行に contact_weight=1.0 が無い。止まる: $ARGS"; return 1;; esac
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
  local want="[$((1000*SEED+1)), $((1000*SEED+2)), $((1000*SEED+3)), $((1000*SEED+4))]" s b c
  s=$(grep -ao 'ワーカーの乱数の種 \[[0-9, ]*\]' single_run/$RUN/stdout.log | head -1)
  case "$s" in *"$want"*) log "✅ $RUN: $s";; *) log "⛔⛔ $RUN: 種の行が期待（$want）と違う／無い: '$s'";; esac
  # ばねの行はワーカーにだけ出る（9-351）。起動の直後は launch_env.json の k を見て、行そのものは見張りの第 10 エポックの再生ログ trace_mid/ep0010.log で見る。
  b=$(python3 -c "import json;print(json.load(open('single_run/$RUN/launch_env.json')).get('CNOID_CUBE_SPRING_K',''))" 2>/dev/null)
  if [ "$b" = "$K" ]; then log "✅ $RUN: launch_env.json の CNOID_CUBE_SPRING_K=$b"; else log "⛔⛔ $RUN: launch_env.json の CNOID_CUBE_SPRING_K が '$b'（期待 $K）"; fi
  c=$(grep -o 'contact_weight=[0-9.]*' single_run/$RUN/.hydra/overrides.yaml 2>/dev/null | head -1)
  if [ "$c" = "contact_weight=1.0" ]; then log "✅ $RUN: overrides の $c"; else log "⛔⛔ $RUN: overrides の contact_weight が '$c'（期待 1.0）"; fi
  return 0; }
log "=== 11-26 ばねの箱の取り直し contact_weight 1.0（系譜 9-354、読み 59）HEAD=$(git rev-parse --short HEAD)。$(echo $RUNS | wc -w) 本"
W=$(ps -eo pid,comm,args | awk '$2=="bash" && /scripts\/watch_midrun\.sh/ {print $1}' | head -1)
if [ -n "$W" ]; then log "学習中の見張りは既に動いている（PID $W）。起動しない"
else nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
  log "学習中の見張り watch_midrun.sh を起動（PID $!）"; fi
RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh を起動（PID $!、RUNS=$RUNS）"
echo "$JOBS" | while IFS='|' read -r RUN SEED K; do
  launch "$RUN" "$SEED" "$K" || break
done
log "=== $(echo $RUNS | wc -w) 本の起動が済んだ（または止まった）"
