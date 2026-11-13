#!/bin/bash
# 押しの配分（根元重・先端重、修論 4.4.3.2）に seed 3・4（_s4・_s5）を 2 本ずつ、計 4 本足す（ユーザー判断 2026-11-13「足す」、系譜 9-347）。
#
# ⭐ なぜ: 3 seed の帯 [9.72, 11.62] 対 [5.38, 9.52] は重ならないが、間は 0.20 しかなく、各形態の seed 間の幅（1.90・4.14）より小さい（系譜 9-333 ㊳-a）。
#   教授の指摘「シードを足すと重なるのでは」に、5 seed で答える。
# ⭐ 起動行は seed 0 の run の `.hydra/overrides.yaml` と `launch_env.json` をそのまま読む（queue_1103_seed2_core.sh と同じ。手で書き写さない）。
#   変えるのは seed だけ。ワーカーの乱数の種は 1000×seed + 番号 + 1 で、起動 60 秒後に stdout.log の行を確かめる。
#   先端重は seed 0 の `tripo_pjdp_dist_phys_cg`（門を直した、init_self_contact_caps あり）から作る（seed 2 と同じ）。
#   11-03 以降に main へ入ったコードの変更（CNOID_TORQUE_SPEED・target_rate・CNOID_FLOOR_BODY・init_root_z_noise）は、どれも指定しなければ既存と同じ（コミットの文で確かめた）。起動行はどれも指定しない。
# ⭐ 学習の本数は実行ファイル名が choreonoid のものだけを数える（Bug 101 の教訓。args の grep はシェルの行に当たる）。
# ⭐ 見張り 2 つ（学習中の watch_midrun.sh・完走後の queue_postrun_replay.sh）もこのキューから起動する。
# ⭐ ディスクの空きが 3 GB を切ったら、次の run を起動せずに止まる。
#
# ⭐ 事前登録した読み 54（結果を見てから変えない。系譜 9-347 に同じ文）:
#   前提: 4 本とも完走・check_before_conclusion.py・目視（目視記録に 1 行）・ワーカーの種（seed 3 は 3001〜3004、seed 4 は 4001〜4004）・不変条件。
#   帯は best（exec_R_eps の最大）で読む（9-333 と同じ）。5 seed ＝ seed 0・新しい seed 1（_s2_ws）・seed 2（_s3）・seed 3（_s4）・seed 4（_s5）。
#   54-a 保たれる: 5 seed の帯が重ならず、根元重が上 → 修論 4.4.3.2 は「5 シードの範囲が重ならない」と書き、暫定の但し書きは隙間の大きさで書き直す。
#   54-b 重なる: 5 seed の帯が重なる → 「3 シードでは重ならず、5 シードでは重なる」と書き、押しの配分の順位を主張しない。
#        表 4.30 の「形態の最適化を含む判定のみが区別した」組は無くなる側で書き直す（4.4.3.2・5.2.2・要旨・第1章の該当の文、主張の木）。
#   54-c 逆転: 先端重の帯が根元重の帯より完全に上 → 54-b と同じく書き直し、逆転したことも書く。
#   添えるだけ（判定に使わない）: 5 seed の平均と、seed ごとの best の並び。
#   予想（判定に使わない）: 54-b（隙間 0.20 に対し、先端重の幅が 4.14 と広い）。
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_1113_push_alloc_s45.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo comm,args | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
# run 名|元の run（seed 0）|seed
JOBS="tripo_pjdp_prox_phys_s4|tripo_pjdp_prox_phys|3
tripo_pjdp_dist_phys_cg_s4|tripo_pjdp_dist_phys_cg|3
tripo_pjdp_prox_phys_s5|tripo_pjdp_prox_phys|4
tripo_pjdp_dist_phys_cg_s5|tripo_pjdp_dist_phys_cg|4"
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

log "=== 11-13 の押しの配分 seed 3・4 のキュー（系譜 9-347、修論 4.4.3.2）HEAD=$(git rev-parse --short HEAD)。$(echo $RUNS | wc -w) 本"
nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
log "学習中の見張り watch_midrun.sh を起動（PID $!）"
RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh を起動（PID $!、RUNS=$RUNS）"
echo "$JOBS" | while IFS='|' read -r RUN SRC SEED; do
  launch "$RUN" "$SRC" "$SEED" || break
done
log "=== $(echo $RUNS | wc -w) 本の起動が済んだ（または止まった）"
