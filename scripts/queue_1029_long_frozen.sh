#!/bin/bash
# 障害物 Reach（D-5）の長い腕で形を固定した対照（ユーザー判断 2026-10-29「修論に入れるか入れないかは別として設計固定はたしかにやるべき」、系譜 9-320）。
#
# なぜ: 長い腕から共設計した `_long` / `_long_s2` は、学習の初めに形を下限まで縮めて届かなかった（9-313）。
#   「形の利点が、方策がまだ獲得していない動作（柱の迂回）を通してしか現れないので、共設計が形を過小評価した」のか、
#   「迂回そのものを今の学習で覚えられない」のかは、形を固定した対照が無いので分けられなかった（9-314 の「言わないこと」）。
# ⭐ 起動行は `_long` / `_long_s2` の `.hydra/overrides.yaml`・`launch_env.json` をそのまま読み、変えるのは +robot_param_scale=1 → 0.0001 だけ
#   （R1 の凍結 `_fx` と同じ値。10-28 のスモーク `_smoke_obs_long` で、この値なら再生の骨・半径・ギアが XML のままと確かめてある、9-313。
#   0 は試していないので使わない。Bug 74 を直した後の config.py は FLAG から読む）。
# ⭐ 起動 90 秒後に、ワーカーの種（stdout.log）と、効いた robot_param_scale（log_train.txt の ⭐ 行）を確かめる。
# ⭐ 順番（研究方針「キューへの投入の優先度」: D-5 は核の外）: queue_1029_fix1_seed2.sh の 10 本の起動がすべて済んでから、枠が空き次第入る。
# ⭐ 完走後の自動再生は、このキューから RUNS を 2 本にして別に起動する。学習中の見張り（watch_midrun.sh）は 10 本のキューが起動したものが続けて見る。
#
# ⭐ 事前登録した読み（結果を見てから変えない。系譜 9-320 に同じ文）:
#   前提: 形が固定された（log_train.txt の robot_param_scale 0.0001・再生の骨が初期の形と同じ）・exec_R_eps の曲線が `_long` と一致しない（Bug 74）・
#         不変条件（「有限」を含む、Bug 82）・話の長さが約 1000 step。
#   ㉛-a 届く: 2 seed とも、best の再生の全 5 話で先端が目標の 10 mm 以内に入る → 長い腕は迂回して届く力を持つ。共設計が縮めたのは、
#         迂回をまだ獲得していない方策で形を評価したため（形の過小評価、Stackelberg PPO p.2 の具体例）と言える。
#   ㉛-b 届かない: 2 seed とも、どの話も 10 mm に入らない → 迂回そのものを今の学習で覚えられない。形の評価の問題とは言えない。
#   ㉛-c 割れる: seed で a と b に分かれる → どちらも言わない（1 seed ずつの差を条件の効果と呼ばない）。
#   予想は持たない（幾何の余裕は最大 61 mm と細い、9-313）。50 話の再生は添えるだけで判定に使わない。
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_1029_long_frozen.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
JOBS="e2e_a1v_obspen_reach_phys_long_frozen|e2e_a1v_obspen_reach_phys_long|0
e2e_a1v_obspen_reach_phys_long_frozen_s2|e2e_a1v_obspen_reach_phys_long_s2|1"
RUNS=$(echo "$JOBS" | cut -d'|' -f1 | tr '\n' ' ')

launch(){ local RUN=$1 SRC=$2 SEED=$3
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return 0; fi
  local ARGS ENVS
  ARGS=$(sed -n 's/^- //p' single_run/$SRC/.hydra/overrides.yaml | sed 's/^+robot_param_scale=.*/+robot_param_scale=0.0001/' | tr '\n' ' ')
  case "$ARGS" in *"+robot_param_scale=0.0001 "*) ;; *) log "⛔⛔ $RUN: 起動行に +robot_param_scale=0.0001 が入らなかった。止まる"; return 1;; esac
  ENVS=$(python3 -c "import json;print(' '.join(f'{k}={v}' for k,v in json.load(open('single_run/$SRC/launch_env.json')).items()))")
  while [ "$(njobs)" -ge 2 ]; do sleep 30; done
  local AV; AV=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "$AV" -lt 3 ]; then log "⛔⛔ ディスクの空き ${AV} GB（3 GB 未満）。$RUN 以降を起動せずに止まる"; return 1; fi
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 $ENVS \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    $ARGS hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)  元 $SRC  env: $ENVS  args: $ARGS"; sleep 90
  local want="[$((1000*SEED+1)), $((1000*SEED+2)), $((1000*SEED+3)), $((1000*SEED+4))]" s p
  s=$(grep -ao 'ワーカーの乱数の種 \[[0-9, ]*\]' single_run/$RUN/stdout.log | head -1)
  case "$s" in *"$want"*) log "✅ $RUN: $s";; *) log "⛔⛔ $RUN: 種の行が期待（$want）と違う／無い: '$s'";; esac
  p=$(grep -ao '⭐ robot_param_scale = [0-9.]*（flag から）' single_run/$RUN/log/log_train.txt 2>/dev/null | head -1)
  case "$p" in *"= 0.0001（flag から）"*) log "✅ $RUN: 効いた値 '$p'";; *) log "⛔⛔ $RUN: robot_param_scale が 0.0001（flag から）と読めない: '$p'（形が固定されていない恐れ、Bug 74）";; esac
  return 0; }

log "=== D-5 の長い腕の形の固定（系譜 9-320）HEAD=$(git rev-parse --short HEAD)。queue_1029_fix1_seed2.sh の 10 本の起動を待つ"
until grep -q "=== 10 本の起動が済んだ" single_run/queue_1029_fix1_seed2.log 2>/dev/null; do sleep 120; done
log "10 本の起動が済んだ。枠が空き次第入る"
RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh を起動（PID $!、RUNS=$RUNS）"
echo "$JOBS" | while IFS='|' read -r RUN SRC SEED; do
  launch "$RUN" "$SRC" "$SEED" || break
done
log "=== 2 本の起動が済んだ（または止まった）"
