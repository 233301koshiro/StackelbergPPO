#!/bin/bash
# 11-15 までの GPU に 9 本（ユーザー判断 2026-11-04「まわす」、系譜 9-334）。
#   ① 形を凍結した押しの配分（根元重・先端重 × 2 seed）: 判定の中の形の最適化は順位を変えるのか（想定問答 Q58）。
#   ② 4.5.5（関節の固定）の seed 2: A1 の関節 3 の可動・固定 × 到達・押し、中間の関節 1 固定の到達（第3層の提案）。
#
# ⭐ 起動行は元の run の `.hydra/overrides.yaml` と `launch_env.json` をそのまま読む（queue_1103_seed2_core.sh と同じ。手で書き写さない）。
#   変えるのは seed と、①だけ `+robot_param_scale=1` → `0.0001`（R1 の `_fx` と同じ凍結）。置き換わらなければ起動しない。
#   ⚠️ 先端重は門を直した `tripo_pjdp_dist_phys_cg`（init_self_contact_caps あり）から作る（9-333 と同じ）。
#   ワーカーの乱数の種は 1000×seed + 番号 + 1 で、起動 60 秒後に stdout.log の行を確かめる。凍結の run は「robot_param_scale = 0.0001」も（Bug 74）。
# ⭐ 見張り 2 つ（学習中の watch_midrun.sh・完走後の queue_postrun_replay.sh）もこのキューから起動する。
# ⭐ ディスクの空きが 3 GB を切ったら、次の run を起動せずに止まる。
#
# ⭐ 事前登録した読み（結果を見てから変えない。系譜 9-334 に同じ文）:
#   前提（各読み共通）: 完走・check_before_conclusion.py・目視（目視記録に 1 行）・ワーカーの種・不変条件。帯は best（exec_R_eps の最大）で読む。50 話の中央値・箱の移動は添えるだけ。
#   ㊵ 形を凍結した押しの配分（2 seed の帯、凍結の根元重 対 凍結の先端重）。前提に足す: 凍結が効いた（log の robot_param_scale 0.0001・best の形が初期の形〔ギア比 150 / 100 / 80・先端の太さ 0.032〕のまま）。
#     ㊵-a 同じ順位: 凍結の根元重の帯が凍結の先端重の帯より完全に上 → 「形を凍結しても配分の順位は同じ」。Q58 は「判定に要るのは学習で、形の最適化が効くのは性能（R1）」と狭める側で答える。
#     ㊵-b 重なる: → 「形を凍結すると配分を区別できない」。順位を付けるのに形の最適化が要った実測として Q58 に答える。
#     ㊵-c 逆転: 凍結の先端重の帯が完全に上 → 「既定値のまま比べると順位が逆になる」。判定が既定値に左右される実測として Q58 に答える。
#     予想（判定に使わない）: ㊵-a（最適化した帯の間が広く、R1 の中間は凍結で 3 割ほど下がった。同じ割合で下がれば順位は残る）。
#   ㊶ 4.5.5 の 3 seed の帯（seed 0 ＋ seed 1 ＋ seed 2）。今の表は 2 seed。
#     ㊶-1 A1 の到達、関節 3 固定 対 可動（今: 固定が上、重ならない、隙間 0.20）:
#       a 固定が上で重ならない → 表はそのまま（seed 数だけ直す）／ b 重なる → 表の「重ならない」を直す。「低下しない」は保つ／ c 可動の帯が完全に上 → 「固定しても性能が低下しない」と、見落とし（3.12.2）の実例を取り下げる。
#     ㊶-2 A1 の押し、関節 3 可動 対 固定（今: 可動が上、重ならない、隙間 17.6）:
#       a 可動が上で重ならない → そのまま／ b 重なる → 「固定すると性能が低下」を言わない／ c 固定の帯が完全に上 → 取り下げる。
#     ㊶-3 中間の到達、関節 1 固定 対 可動（第3層の提案。今: 固定が上、重ならない、隙間 0.12）:
#       a 固定が上で重ならない → そのまま／ b 重なる → 表の「重ならない」を直す。「提案に従って減らしても低下しなかった」は保つ／ c 可動の帯が完全に上 → それを取り下げる。
#       可動の seed 2 は既にある（`tripo_pj_mid_phys_s3`、best −1.68）。
#     予想（判定に使わない）: ㊶-2 は a（隙間が広い）。㊶-1・㊶-3 は隙間が狭いので b もありうる。
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_1104_frozen_alloc_455.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
# run 名|元の run（seed 0）|seed|凍結するか
JOBS="tripo_pjdp_prox_frozen_phys_fx|tripo_pjdp_prox_phys|0|1
tripo_pjdp_dist_frozen_phys_cg_fx|tripo_pjdp_dist_phys_cg|0|1
tripo_pjdp_prox_frozen_phys_fx_s2|tripo_pjdp_prox_phys|1|1
tripo_pjdp_dist_frozen_phys_cg_fx_s2|tripo_pjdp_dist_phys_cg|1|1
e2e_a1_reach_phys_s3|e2e_a1_reach_phys|2|0
e2e_a1_fix3_reach_phys_s3|e2e_a1_fix3_reach_phys|2|0
e2e_a1_pusher_phys_s3|e2e_a1_pusher_phys|2|0
e2e_a1_fix3_pusher_phys_s3|e2e_a1_fix3_pusher_phys|2|0
tripo_pj_mid_phys_fix1_s3|tripo_pj_mid_phys_fix1|2|0"
RUNS=$(echo "$JOBS" | cut -d'|' -f1 | tr '\n' ' ')

launch(){ local RUN=$1 SRC=$2 SEED=$3 FRZ=$4
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return 0; fi
  if [ ! -f "single_run/$SRC/.hydra/overrides.yaml" ] || [ ! -f "single_run/$SRC/launch_env.json" ]; then log "⛔ $SRC の overrides か launch_env が無い。飛ばす"; return 0; fi
  local ARGS ENVS
  ARGS=$(sed -n 's/^- //p' single_run/$SRC/.hydra/overrides.yaml | sed "s/^seed=.*/seed=$SEED/" | tr '\n' ' ')
  if [ "$FRZ" = 1 ]; then
    ARGS=$(echo "$ARGS" | sed 's/+robot_param_scale=1 /+robot_param_scale=0.0001 /')
    case "$ARGS" in *"+robot_param_scale=0.0001 "*) ;; *) log "⛔⛔ $RUN: 凍結の置き換えが効かない（$ARGS）。飛ばす"; return 0;; esac
  fi
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
  if [ "$FRZ" = 1 ]; then
    grep -q "robot_param_scale = 0.0001" single_run/$RUN/stdout.log && log "✅ $RUN: 凍結 robot_param_scale = 0.0001" || log "⛔⛔ $RUN: 凍結の行が無い（Bug 74）"
  fi
  return 0; }

log "=== 11-04 のキュー（系譜 9-334、凍結の配分 4 本・4.5.5 の seed 2 の 5 本）HEAD=$(git rev-parse --short HEAD)。$(echo $RUNS | wc -w) 本"
nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
log "学習中の見張り watch_midrun.sh を起動（PID $!）"
RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh を起動（PID $!、RUNS=$RUNS）"
echo "$JOBS" | while IFS='|' read -r RUN SRC SEED FRZ; do
  launch "$RUN" "$SRC" "$SEED" "$FRZ" || break
done
log "=== $(echo $RUNS | wc -w) 本の起動が済んだ（または止まった）"
