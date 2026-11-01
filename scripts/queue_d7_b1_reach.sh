#!/bin/bash
# D-7 結合テストの本番（ユーザー判断 2026-11-01 夜「D7はやる」「既存の B1 で到達」、系譜 9-331）。
#
# なぜ: 修論 4.5.4 は助言の倍率を短腕（マトリクスの形）に当てた。手描きをパイプラインに通した形が棄却され、
#   助言どおりに直した形が学習で届くところまでを 1 本の通しで回した記録は無い（結合テスト_見込み違いの形態.md）。
#   旧い試み e2e_b1（押し）は Bug 50 で有効な棄却例ではなかった（9-19 の B1 は修正後の判定器では押しを通過する）。
# ⭐ 入力: 手描き B1 → M1〜M5 の出力 e2e_b1.xml に、e2e_a1_phys_arm171 と同じ手順で実在の物理の値を当てた e2e_b1_phys_arm171
#   （make_realistic_density_xml.py --armature 1.71 --object-armature-zero。同じ手順で A1 を作ると既存の XML と一致することを確かめた）。
#   close_loop.py --task reach: ① 棄却（届く範囲 0.789 m・目標 0.800 m）→ ② 1.02 倍 → e2e_b1_phys_arm171_fix（0.8049 m、公称と実効が一致）→ ③ 通過。
# ⭐ 起動行は e2e_a1_reach_phys の `.hydra/overrides.yaml`・`launch_env.json` をそのまま読み、変えるのは xml_name と seed だけ
#   （e2e_a1_reach_phys と _s2_ws の違いは seed だけなのを diff で確かめた。cfg=pusher_gearonly_real でリンク長は凍結＝判定モード）。
# ⭐ 起動 90 秒後に、ワーカーの種（stdout.log）を確かめる。完走後の自動再生も起動する。
#
# ⭐ 事前登録した読み（結果を見てから変えない。系譜 9-331 に同じ文）:
#   前提: 再生の骨が XML のまま（直す前 0.7891 m・直した形 0.8049 m）・不変条件（「有限」を含む）・話の長さ。
#   判定は best の再生 50 話（TRACE_EPISODES=50、9-304 と同じ道具）の最後の誤差と、best exec_R_eps の帯（2 seed）。
#   ㉞-a 閉じた: 帯が離れ（直した形の下限 ＞ 直す前の上限）、かつ直した形が 2 seed とも 50 話中 45 話以上で最後の誤差 5 mm 未満
#         → 手描きをパイプラインに通した不適合な形が、助言の倍率だけで（描き直さずに）学習で届く形になった。
#   ㉞-b 割れる: 帯は離れるが、5 mm の条件を満たすのが 1 seed だけ → 閉じたとは言わない。seed で割れたと書く。
#   ㉞-c 閉じない: 帯が重なる、または 2 seed とも 5 mm の条件を満たさない → 結合の穴として書く（4.5.4 の 1.99 倍は余裕 2 mm で届いたので、形の違いを調べる）。
#   前提が崩れた扱い: 直す前の形が 5 mm 未満で終える話を持つ（幾何的な最小は 10.9 mm）→ 測り方か設定の誤りを疑い、判定しない。
#   予想（判定に使わない）: ㉞-a。4.5.4 の 1.99 倍（余裕 2 mm）が 3 本とも 50 / 50 話で 1 mm 未満だった（9-304）。
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_d7_b1_reach.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
SRC=e2e_a1_reach_phys
JOBS="e2e_b1_fix_reach_phys|e2e_b1_phys_arm171_fix|0
e2e_b1_reach_phys|e2e_b1_phys_arm171|0
e2e_b1_fix_reach_phys_s2|e2e_b1_phys_arm171_fix|1
e2e_b1_reach_phys_s2|e2e_b1_phys_arm171|1"
RUNS=$(echo "$JOBS" | cut -d'|' -f1 | tr '\n' ' ')

launch(){ local RUN=$1 XML=$2 SEED=$3
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return 0; fi
  local ARGS ENVS
  ARGS=$(sed -n 's/^- //p' single_run/$SRC/.hydra/overrides.yaml | sed "s/^xml_name=.*/xml_name=$XML/; s/^seed=.*/seed=$SEED/" | tr '\n' ' ')
  case "$ARGS" in *"xml_name=$XML seed=$SEED "*) ;; *) log "⛔⛔ $RUN: 起動行に xml_name=$XML seed=$SEED が入らなかった。止まる"; return 1;; esac
  ENVS=$(python3 -c "import json;print(' '.join(f'{k}={v}' for k,v in json.load(open('single_run/$SRC/launch_env.json')).items()))")
  while [ "$(njobs)" -ge 2 ]; do sleep 30; done
  local AV; AV=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "$AV" -lt 3 ]; then log "⛔⛔ ディスクの空き ${AV} GB（3 GB 未満）。$RUN 以降を起動せずに止まる"; return 1; fi
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 $ENVS \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    $ARGS hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)  元 $SRC  env: $ENVS  args: $ARGS"; sleep 90
  local want="[$((1000*SEED+1)), $((1000*SEED+2)), $((1000*SEED+3)), $((1000*SEED+4))]" s
  s=$(grep -ao 'ワーカーの乱数の種 \[[0-9, ]*\]' single_run/$RUN/stdout.log | head -1)
  case "$s" in *"$want"*) log "✅ $RUN: $s";; *) log "⛔⛔ $RUN: 種の行が期待（$want）と違う／無い: '$s'";; esac
  return 0; }

log "=== D-7 結合テスト（B1 の到達、系譜 9-331）HEAD=$(git rev-parse --short HEAD)"
RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh を起動（PID $!、RUNS=$RUNS）"
nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
log "学習中の見張り watch_midrun.sh を起動（PID $!）"
echo "$JOBS" | while IFS='|' read -r RUN XML SEED; do
  launch "$RUN" "$XML" "$SEED" || break
done
log "=== 4 本の起動が済んだ（または止まった）"
