#!/bin/bash
# R2（修論 4.4.2、タスクによる収束形態の分離）の縦型 A1v アームの到達・押しに seed 3・4（_s4・_s5）を 2 本ずつ、計 4 本足す（ユーザー判断 2026-11-15「優先度の高いものをキューに入れて」、系譜 9-349）。
# 修論の核（表 4.3 の R2 の欄）。
#
# ⭐ なぜ: 3 seed で分かれるのは関節 4 のギア比（到達 [93.5, 142.1] 対 押し [148.7, 150.0]、隙間 6.6 N·m）とリンク長の合計（[1.103, 1.78] 対 [1.957, 3.211] m、隙間 0.177 m）だけ（系譜 9-318 ㉙）。
#   2 seed → 3 seed で関節 3 が重なり、リンク長の隙間は 0.827 → 0.177 m に縮んだ。修論 4.4.2 の限界は「シードを増やした場合には重なる可能性がある」と書いている。
#   押しの配分（9-347）で教授が突いた点と同じ形なので、5 seed で答える。
# ⭐ 起動行は seed 0 の run（e2e_a1v_reach_phys・e2e_a1v_pusher_phys）の `.hydra/overrides.yaml` と `launch_env.json` をそのまま読む。変えるのは seed だけ
#   （seed 2 の _s3 とも seed 以外の差が無いことを diff で確かめた。11-15）。10-29（seed 2 の起動）以降に main へ入った学習側のコードの変更
#   （CNOID_ACT_DELAY・CNOID_JOINT_FRICTION・CNOID_TORQUE_SPEED・target_rate・CNOID_FLOOR_BODY・目標の印の形・四脚の cfg と XML）は、どれも指定しなければ既存と同じで、起動行はどれも指定しない。
# ⭐ 学習の本数は実行ファイル名が choreonoid のものだけを数える（Bug 101 の教訓）。見張り 2 つもこのキューから起動する。ディスクの空きが 3 GB を切ったら止まる。
#
# ⭐ 事前登録した読み 56（結果を見てから変えない。系譜 9-349 に同じ文）:
#   前提: 4 本とも完走・check_before_conclusion.py・目視（目視記録に 1 行）・ワーカーの種（seed 3 は 3001〜3004、seed 4 は 4001〜4004）・不変条件。
#   値は 9-318 と同じ: d6_bands.py の trace_metrics（best の checkpoint を再生した軌跡の ep_gear・ep_bone）。5 seed ＝ seed 0・新しい seed 1（_s2_ws）・seed 2（_s3）・seed 3（_s4）・seed 4（_s5）。
#   判定する量は 2 つ: 関節 4 のギア比、リンク長の合計。
#   56-a 両方保たれる: 2 つとも 5 seed の帯が重ならない → 修論 4.4.2 は「5 シードの範囲が重ならない」と書き、限界の但し書きは隙間の大きさで書き直す。
#   56-b 片方だけ保たれる: 重ならない方だけを「分かれる」と書き、重なった方は関節 1〜3 と同じ「シードによって重なる」へ移す（4.4.2・表 4.5・5.2.2・要旨・第1章の該当の文、主張の木）。
#   56-c 両方重なる: 「タスクのみを変えたとき、5 シードの範囲で分かれる設計変数は無かった」と書き、R2 は満たされなかったとする（同じ箇所）。
#   記録のみ（判定に使わない）: 関節 1〜3 のギア比・先端の太さ・best・押しの距離。
#   予想（判定に使わない）: 56-b（到達の関節 4 の帯が 93.5〜142.1 と広く上限 150 に近い seed がある一方、リンク長の合計も到達の seed 2 の 1.78 m で隙間が縮んでいる。どちらが重なるかは予想しない）。
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_1115_r2_a1v_s45.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo comm,args | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
# run 名|元の run（seed 0）|seed
JOBS="e2e_a1v_reach_phys_s4|e2e_a1v_reach_phys|3
e2e_a1v_pusher_phys_s4|e2e_a1v_pusher_phys|3
e2e_a1v_reach_phys_s5|e2e_a1v_reach_phys|4
e2e_a1v_pusher_phys_s5|e2e_a1v_pusher_phys|4"
RUNS=$(echo "$JOBS" | cut -d'|' -f1 | tr '\n' ' ')

launch(){ local RUN=$1 SRC=$2 SEED=$3
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return 0; fi
  if [ ! -f "single_run/$SRC/.hydra/overrides.yaml" ] || [ ! -f "single_run/$SRC/launch_env.json" ]; then log "⛔ $SRC の overrides か launch_env が無い。飛ばす"; return 0; fi
  local ARGS ENVS
  ARGS=$(sed -n 's/^- //p' single_run/$SRC/.hydra/overrides.yaml | sed "s/^seed=.*/seed=$SEED/" | tr '\n' ' ')
  ENVS=$(python3 -c "import json;print(' '.join(f'{k}={v}' for k,v in json.load(open('single_run/$SRC/launch_env.json')).items()))")
  while [ "$(njobs)" -ge 2 ]; do sleep 15; done
  local AV; AV=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "$AV" -lt 3 ]; then log "⛔⛔ ディスクの空き ${AV} GB（3 GB 未満）。$RUN 以降を起動せずに止まる"; return 1; fi
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 $ENVS \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    $ARGS hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)  元 $SRC  env: $ENVS  args: $ARGS"; sleep 60
  local want="[$((1000*SEED+1)), $((1000*SEED+2)), $((1000*SEED+3)), $((1000*SEED+4))]" s
  s=$(grep -ao 'ワーカーの乱数の種 \[[0-9, ]*\]' single_run/$RUN/stdout.log | head -1)
  case "$s" in *"$want"*) log "✅ $RUN: $s";; *) log "⛔⛔ $RUN: 種の行が期待（$want）と違う／無い: '$s'";; esac
  return 0; }

log "=== 11-15 の R2 縦型 A1v seed 3・4 のキュー（系譜 9-349、修論 4.4.2）HEAD=$(git rev-parse --short HEAD)。$(echo $RUNS | wc -w) 本"
nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
log "学習中の見張り watch_midrun.sh を起動（PID $!）"
RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh を起動（PID $!、RUNS=$RUNS）"
echo "$JOBS" | while IFS='|' read -r RUN SRC SEED; do
  launch "$RUN" "$SRC" "$SEED" || break
done
log "=== $(echo $RUNS | wc -w) 本の起動が済んだ（または止まった）"
