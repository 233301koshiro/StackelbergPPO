#!/bin/bash
# GPU の空き（11-03〜）に、修論の核のうち 2 seed しか無い欄へ seed 2（_s3）を 5 本足す（ユーザー判断 2026-11-03「お願いします」、系譜 9-333）。
#
# ⭐ 埋める欄（研究方針「キューへの投入の優先度」）: 表 4.4 の R3 配分（4.4.3.3、到達・押し、2 seed → 3 seed）と R1 凍結（4.4.1、凍結 2 → 3 seed）。
# ⭐ 起動行は seed 0 の run の `.hydra/overrides.yaml` と `launch_env.json` をそのまま読む（queue_1029_fix1_seed2.sh と同じ。手で書き写さない）。
#   変えるのは seed だけ。ワーカーの乱数の種は 1000×seed + 番号 + 1 で、起動 60 秒後に stdout.log の行を確かめる。
#   ⚠️ 押しの先端重は seed 0 の `tripo_pjdp_dist_phys_cg`（門を直した、init_self_contact_caps あり）から作る。
#      元の seed 1 `tripo_pjdp_dist_phys_s2_ws` は門なしの起動行（9-279: 食い込みが無いので形の比較として読める）。seed 2 は今の正しい門にそろえる。
#   凍結（robot_param_scale 0.0001）の run は、起動 60 秒後に stdout.log の「robot_param_scale = 0.0001」を確かめる（Bug 74）。
# ⭐ 見張り 2 つ（学習中の watch_midrun.sh・完走後の queue_postrun_replay.sh）もこのキューから起動する。
# ⭐ ディスクの空きが 3 GB を切ったら、次の run を起動せずに止まる。
#
# ⭐ 事前登録した読み（結果を見てから変えない。系譜 9-333 に同じ文）:
#   前提（各読み共通）: 完走・check_before_conclusion.py・目視（目視記録に 1 行）・ワーカーの種 2001〜2004・不変条件。
#   ㊳ R3 配分（4.4.3.3）: 3 seed の帯（seed 0 ＋ 新しい seed 1〔_s2_ws〕＋ seed 2）で、根元重 ＞ 先端重 が到達と押しのそれぞれで保たれる（帯が重ならない）。
#     今の帯: 到達 根元重 [−1.91, −1.81] 対 先端重 [−2.28, −2.18]、押し 根元重 [11.39, 11.62] 対 先端重 [5.38, 9.52]。
#     ㊳-a 保たれる: 到達・押しとも 3 seed の帯が重ならず、根元重が上。
#     ㊳-b 狭める: どちらかのタスクで 3 seed の帯が重なる → そのタスクは「2 seed では離れ、3 seed では重なる」と書き、順位を主張しない（9-318 ㉘ と同じ扱い。修論 4.4.3.3）。
#     ㊳-c 逆転: どちらかのタスクで先端重の帯が根元重の帯より完全に上 → 4.4.3.3 の主張を取り下げる側で書き直す。
#     予想（判定に使わない）: 到達は ㊳-a（2 seed の帯が狭い）。押しは先端重の帯が広い（5.38〜9.52）ので重なりうる。
#   ㊴ R1 凍結（4.4.1）: 凍結の 3 seed の帯が、最適化の 3 seed の帯 [9.37, 10.44] より完全に下のまま（今の凍結の帯 [6.57, 6.77]）。
#     前提に足す: 凍結が効いた（log の robot_param_scale 0.0001・best の形が凍結の形〔ギア比 150 / 100 / 80・先端の太さ 0.032〕のまま）。
#     ㊴-a 保たれる: 凍結の seed 2 の best ＜ 9.37。
#     ㊴-b 狭める: 凍結の seed 2 の best ≥ 9.37 → 「2 seed では離れ、3 seed では重なる」と書く（修論 4.4.1）。
#     予想（判定に使わない）: ㊴-a（凍結の 2 seed は 6.57・6.77 と近い）。
#   帯は best（exec_R_eps の最大）で読む。50 話の中央値・箱の移動は添えるだけ（9-297 と同じ）。
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_1103_seed2_core.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
# run 名|元の run（seed 0）|seed
JOBS="tripo_pjd_prox_phys_s3|tripo_pjd_prox_phys|2
tripo_pjd_dist_phys_cg_s3|tripo_pjd_dist_phys_cg|2
tripo_pjdp_prox_phys_s3|tripo_pjdp_prox_phys|2
tripo_pjdp_dist_phys_cg_s3|tripo_pjdp_dist_phys_cg|2
tripo_pjp_mid_frozen_phys_fx_s3|tripo_pjp_mid_frozen_phys_fx|2"
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
  case "$ARGS" in *robot_param_scale=0.0001*)
    grep -q "robot_param_scale = 0.0001" single_run/$RUN/stdout.log && log "✅ $RUN: 凍結 robot_param_scale = 0.0001" || log "⛔⛔ $RUN: 凍結の行が無い（Bug 74）";;
  esac
  return 0; }

log "=== 11-03 の seed 2 のキュー（系譜 9-333、表 4.4 の R3 配分・R1 凍結）HEAD=$(git rev-parse --short HEAD)。$(echo $RUNS | wc -w) 本"
nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
log "学習中の見張り watch_midrun.sh を起動（PID $!）"
RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh を起動（PID $!、RUNS=$RUNS）"
echo "$JOBS" | while IFS='|' read -r RUN SRC SEED; do
  launch "$RUN" "$SRC" "$SEED" || break
done
log "=== $(echo $RUNS | wc -w) 本の起動が済んだ（または止まった）"
