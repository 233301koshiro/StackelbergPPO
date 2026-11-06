#!/bin/bash
# 助言を強める閉ループの 4 本（ユーザー判断 2026-11-06「全部やる」「c-aで」、系譜 9-339・9-340、想定問答 Q64、研究方針「助言を強める」）。
#   B（9-339 ㊻）: 押しの倍率に、箱を 0.1 m 押し込むぶんを足した 2.36 倍（旧 2.11 倍は最も奥の箱に触れるだけ。修論 4.5.4 の 42〜45 / 50）。
#     元 tripo_pjr211p_phys の起動行から xml_name だけを tripo_arm_v2c_pj_rec236_phys_arm171 に置き換える。seed 0・1。
#   C-a（9-340 ㊼）: 第2層の「ギア比が上限に張り付いた＝出力不足」に従い、ギア比の上限を 150 → 300 N·m に上げる。
#     元 tripo_pjp_mid_phys の起動行から cfg だけを pusher_gearonly_real_ub300 に置き換える。seed 0・1（根元のギア比が上限に張り付いた 2 シード）。
#
# ⭐ 起動行・種の確かめは queue_1105_frozen_alloc_s3.sh の launch() と同じ（元の run の overrides.yaml と launch_env.json を読む）。
#   置き換えは 1 か所だけで、置き換えが効いたことを起動前に確かめる（効かなければ飛ばす）。
# ⭐ 学習は 2 本まで（GPU が律速。CLAUDE.md §6）。動いている凍結の配分の seed 2 と空きを取り合う。
# ⭐ 見張り: 完走後の自動再生はこの 4 本の RUNS で起動する。学習中の見張り watch_midrun.sh は、起動時に動いていなければ起動する。
#
# 事前登録した読み（系譜 9-339・9-340 と同じ文。結果を見てから変えない）:
#   前提（共通）: 完走・check_before_conclusion.py（比べる相手と起動行の差が置き換えた 1 キーだけ）・目視（目視記録に 1 行）・ワーカーの種。帯は best で読む。
#   ㊻ B: 2.36 倍の 2 シードとも、最良方策の 50 話の再生で箱を 0.1 m 以上動かした話が 48 / 50 以上。
#        ㊻-a 成立 → 修論 4.5.4 の押しを「閉ループが倍率の欠陥（押す距離を含まない）を見つけ、直した倍率で取りこぼしが消えた」に。
#        ㊻-b 2.11 倍（42〜45 / 50）より増えるが 48 に届かない → 押す距離は一因で、残りは学習の側。言えるのは「減った」まで。
#        ㊻-c 増えない → 取りこぼしは倍率では説明できない。4.5.4 の「余裕を零として計算されるため」を書き直す。
#   ㊼ C-a: 上限 300 の 2 シードの best の帯が、上限 150 の 3 シードの帯 [9.37, 10.44] と重ならず上にある。
#        ㊼-a 上にある → 第2層の「出力不足」の助言に従うと成績が上がった（学習から出る助言の閉ループの初めての例）。
#        ㊼-b 重なる → 上限を上げても成績は変わらない。第2層の「上限への張り付き＝出力不足」は、この形とタスクでは成績の不足を意味しなかった。
#        ㊼-c 下にある → 上限を上げると悪くなった（探索範囲が広がった影響を含む）。
#        判定に使わない観察: 根元のギア比が 300 の内側に収まるか、また 300 に張り付くか（張り付くなら第2層は同じ報告を繰り返す）。
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_1106_advice.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
# RUN|元|seed|置き換える前|置き換えた後
JOBS="tripo_pjr236p_phys|tripo_pjr211p_phys|0|xml_name=tripo_arm_v2c_pj_rec211_phys_arm171|xml_name=tripo_arm_v2c_pj_rec236_phys_arm171
tripo_pjp_mid_phys_ub300|tripo_pjp_mid_phys|0|cfg=pusher_gearonly_real|cfg=pusher_gearonly_real_ub300
tripo_pjr236p_phys_s2|tripo_pjr211p_phys|1|xml_name=tripo_arm_v2c_pj_rec211_phys_arm171|xml_name=tripo_arm_v2c_pj_rec236_phys_arm171
tripo_pjp_mid_phys_ub300_s2|tripo_pjp_mid_phys|1|cfg=pusher_gearonly_real|cfg=pusher_gearonly_real_ub300"
RUNS=$(echo "$JOBS" | cut -d'|' -f1 | tr '\n' ' ')
launch(){ local RUN=$1 SRC=$2 SEED=$3 FROM=$4 TO=$5
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return 0; fi
  if [ ! -f "single_run/$SRC/.hydra/overrides.yaml" ] || [ ! -f "single_run/$SRC/launch_env.json" ]; then log "⛔ $SRC の overrides か launch_env が無い。飛ばす"; return 0; fi
  local ARGS ENVS
  ARGS=$(sed -n 's/^- //p' single_run/$SRC/.hydra/overrides.yaml | sed "s/^seed=.*/seed=$SEED/" | sed "s/^$FROM\$/$TO/" | tr '\n' ' ')
  case " $ARGS" in *" $TO "*) ;; *) log "⛔⛔ $RUN: 置き換え（$FROM → $TO）が効かない（$ARGS）。飛ばす"; return 0;; esac
  ENVS=$(python3 -c "import json;print(' '.join(f'{k}={v}' for k,v in json.load(open('single_run/$SRC/launch_env.json')).items()))")
  while [ "$(njobs)" -ge 2 ]; do sleep 15; done
  local AV; AV=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "$AV" -lt 3 ]; then log "⛔⛔ ディスクの空き ${AV} GB（3 GB 未満）。$RUN 以降を起動せずに止まる"; return 1; fi
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 $ENVS \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    $ARGS hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)  元 $SRC  置き換え $FROM → $TO  env: $ENVS  args: $ARGS"; sleep 60
  local want="[$((1000*SEED+1)), $((1000*SEED+2)), $((1000*SEED+3)), $((1000*SEED+4))]" s
  s=$(grep -ao 'ワーカーの乱数の種 \[[0-9, ]*\]' single_run/$RUN/stdout.log | head -1)
  case "$s" in *"$want"*) log "✅ $RUN: $s";; *) log "⛔⛔ $RUN: 種の行が期待（$want）と違う／無い: '$s'";; esac
  return 0; }

log "=== 11-06 のキュー（系譜 9-339 ㊻・9-340 ㊼、助言を強める 4 本）HEAD=$(git rev-parse --short HEAD)"
RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh を起動（PID $!、RUNS=$RUNS）"
echo "$JOBS" | while IFS='|' read -r RUN SRC SEED FROM TO; do
  launch "$RUN" "$SRC" "$SEED" "$FROM" "$TO" || break
  if ! ps -eo args | grep -q "[w]atch_midrun.sh"; then
    nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
    log "学習中の見張り watch_midrun.sh を起動（PID $!）"
  fi
done
log "=== 4 本の起動が済んだ（または止まった）"
