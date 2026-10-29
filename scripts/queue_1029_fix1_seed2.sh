#!/bin/bash
# GPU の空き（11-02 の軸6 L-4 まで）に 10 本を回す（ユーザー判断 2026-10-29「1〜4 すべて」、系譜 9-317・9-318）。
#
# ⭐ 起動行は元の run の `.hydra/overrides.yaml` と `launch_env.json` をそのまま読む（手で書き写さない。queue_seed1_ws.sh と同じ）。
#   変えるのは seed（と ① の xml_name）だけ。ワーカーの乱数の種は 1000×seed + 番号 + 1（Bug 76 の後の式）で、起動 60 秒後に stdout.log の行を確かめる。
# ⭐ 順番（研究方針「キューへの投入の優先度」、修論の核から）:
#   ① 第3層の提案どおり、到達の中間の関節 1 を固定（xml tripo_arm_v2c_pj_mid_phys_arm171_fix1、seed 0 / 1）
#   ② 到達・押しのマトリクスの中間・長腕に seed 2（_s3）
#   ③ R2 の縦型 A1v の到達・押しに seed 2（_s3）
#   ④ ホッケー `_ax` に seed 2・3（_s3・_s4）
# ⭐ 見張り 2 つ（学習中の watch_midrun.sh・完走後の queue_postrun_replay.sh）もこのキューから起動する（10-28 の落とし穴: 起動するキューが無かった）。
# ⭐ ディスクの空きが 3 GB を切ったら、次の run を起動せずに止まる（1 本約 240 MB、10-29 の空き 14 GB）。
#
# ⭐ 事前登録した読み（結果を見てから変えない。系譜 9-318 に同じ文）:
#   ㉗ ①（比べる相手: 可動の tripo_pj_mid_phys / _s2_ws、帯 [−1.85, −1.64]）
#     前提: 再生の軌跡で関節 1 の角度が動かない（固定が効いた。⑤-3-2）・exec_R_eps の曲線が可動と一致しない（Bug 74）・不変条件。
#     ㉗-a 提案が効く（落ちない）: 固定の 2 seed の best の帯の上端 ≥ −1.85 で、2 seed とも再生の全 5 話で 10 mm 以内に入る。
#     ㉗-b 提案が外れる（落ちる）: 固定の帯が可動の帯より完全に下（上端 < −1.85）、または 10 mm に入らない話がある。
#     予想は ㉗-a（使用率 5 / 8 / 7 %、関節 1 を 0° にしても関節 2・3 が ±26° / ∓66.5° で目標に届く）。50 話の中央値は添えるだけで判定に使わない。
#   ㉘ ②: 到達は中間 ＞ 長腕、押しは長腕 ＞ 中間が seed 2 でも保たれ、3 seed の帯が重ならない（今の帯: 到達 中間 [−1.85, −1.64] 対 長腕 [−9.68, −9.67]、押し 長腕 [21.51, 24.92] 対 中間 [10.42, 10.44]）。
#     重なれば、9-268・9-301 の「2 seed で帯が重ならない」を「3 seed では重なる」に狭める（修論 4.4.3.2）。
#   ㉙ ③: 肘から先のギア比（関節 3・4）とリンク長の合計は、到達と押しで 3 seed の帯が重ならない（9-307 の言い方が保たれる）。重なればさらに狭める。
#     根元側（関節 1・2）は 3 seed で重なるかを記録するだけ（9-307 で既に「seed によって重なる」）。
#   ㉚ ④: 読み ⑧〜⑩（queue_phys.sh）を seed ごとに判定し、4 seed のゴールの数を並べる（judge_hockey_goal.py、隙間 ≥ 10 cm の 20 話）。
#     修論 4.7.3 の「シード間の差が大きい」は 4 seed の数で書き直す（主張の向きは変えない）。
#   共通: check_before_conclusion.py・目視（目視記録.md）・②③ は best の帯で読む（9-297 と同じく 50 話の中央値は添えるだけ）。
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_1029_fix1_seed2.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
# run 名|元の run|seed|xml_name（空なら元のまま）
JOBS="tripo_pj_mid_phys_fix1|tripo_pj_mid_phys|0|tripo_arm_v2c_pj_mid_phys_arm171_fix1
tripo_pj_mid_phys_fix1_s2|tripo_pj_mid_phys_s2_ws|1|tripo_arm_v2c_pj_mid_phys_arm171_fix1
tripo_pj_mid_phys_s3|tripo_pj_mid_phys|2|
tripo_pj_long_phys_s3|tripo_pj_long_phys|2|
tripo_pjp_long_phys_s3|tripo_pjp_long_phys|2|
tripo_pjp_mid_phys_s3|tripo_pjp_mid_phys|2|
e2e_a1v_reach_phys_s3|e2e_a1v_reach_phys|2|
e2e_a1v_pusher_phys_s3|e2e_a1v_pusher_phys|2|
hockey_bank9_phys_g10_blk3d_fs_ax_s3|hockey_bank9_phys_g10_blk3d_fs_ax|2|
hockey_bank9_phys_g10_blk3d_fs_ax_s4|hockey_bank9_phys_g10_blk3d_fs_ax|3|"
RUNS=$(echo "$JOBS" | cut -d'|' -f1 | tr '\n' ' ')

launch(){ local RUN=$1 SRC=$2 SEED=$3 XML=$4
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return 0; fi
  if [ ! -f "single_run/$SRC/.hydra/overrides.yaml" ] || [ ! -f "single_run/$SRC/launch_env.json" ]; then log "⛔ $SRC の overrides か launch_env が無い。飛ばす"; return 0; fi
  local ARGS ENVS
  ARGS=$(sed -n 's/^- //p' single_run/$SRC/.hydra/overrides.yaml | sed "s/^seed=.*/seed=$SEED/" | tr '\n' ' ')
  [ -n "$XML" ] && ARGS=$(echo "$ARGS" | sed "s/xml_name=[^ ]*/xml_name=$XML/")
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

log "=== 10-29 のキュー（系譜 9-317・9-318）HEAD=$(git rev-parse --short HEAD)。$(echo $RUNS | wc -w) 本"
nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
log "学習中の見張り watch_midrun.sh を起動（PID $!）"
RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh を起動（PID $!、RUNS=$RUNS）"
echo "$JOBS" | while IFS='|' read -r RUN SRC SEED XML; do
  launch "$RUN" "$SRC" "$SEED" "$XML" || break
done
log "=== 10 本の起動が済んだ（または止まった）"
