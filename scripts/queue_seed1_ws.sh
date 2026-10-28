#!/bin/bash
# ⛔ Bug 76（系譜 9-283 A）: seed 1 の run はワーカーの乱数 4 本中 3 本を seed 0 と共有していた（種 seed + 番号 + 1）。
#   ユーザー判断（10-22 16 時 40 分「おすすめで」）: 核（修論 表 4.3 の R2・R3・軸3）の seed 1 の 14 本を新しい種（1000×seed + 番号 + 1）で取り直す。
#
# ⭐ 起動行は元の run の `.hydra/overrides.yaml` と `launch_env.json` をそのまま読む（手で書き写さない）。違うのは run 名（末尾 `_ws`）とコード（種の式）だけ。
#   元の run は消さない（Bug 76 の前後の比較に使う）。名前を変えるのは launch() が既にあるフォルダを飛ばすため。
# ⭐ 起動の確認: stdout.log の `[WorkerPool] ⭐ ワーカーの乱数の種 [1001, 1002, 1003, 1004]` をこのキューのログに出す。違えば ⛔⛔ を出す（止めるのは人が判断する）。
# ⭐ 読み（結果を見てから変えない）: 各条件の読みは元のキューの事前登録のまま（queue_phys.sh・queue_dist_capgate.sh・queue_rec211p_s2.sh）。
#   帯は seed 0 と新しい seed 1（`_ws`）で作り直し、元の seed 1 の帯と並べる。帯の向きが変われば、その節の主張を取り下げる。
# ⭐ 順番（修論の核を埋めるものを先に、研究方針「キューへの投入の優先度」）: 空欄を埋める A1 の押し 4 本（queue_phys.sh の最後）の起動を待ってから、
#   15 秒ごとに空いた枠へ入る（ホッケー seed1 のキューは 180 秒ごとなので、こちらが先に取る）。並びは節の順: R3 マトリクス → R3 配分 → 軸3 → R2。
# ⚠️ queue_postrun_replay.sh の一覧には無い。投入と一緒に、RUNS を渡した postrun を別に起動する（10-21 と同じやり方）。
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_seed1_ws.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
RUNS="tripo_pj_mid_phys_s2 tripo_pj_long_phys_s2 tripo_pjp_mid_phys_s2 tripo_pjp_long_phys_s2
tripo_pjd_prox_phys_s2 tripo_pjd_dist_phys_cg_s2 tripo_pjdp_prox_phys_s2 tripo_pjdp_dist_phys_s2
tripo_pjr199_phys_s2 e2e_a1_reach_phys_s2 e2e_a1_fix3_reach_phys_s2 tripo_pjr211p_phys_s2
e2e_a1v_reach_phys_s2 e2e_a1v_pusher_phys_s2"
launch(){ local SRC=$1 RUN=$1_ws
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  if [ ! -f "single_run/$SRC/.hydra/overrides.yaml" ] || [ ! -f "single_run/$SRC/launch_env.json" ]; then log "⛔ $SRC の overrides か launch_env が無い。飛ばす"; return; fi
  local ARGS ENVS
  ARGS=$(sed -n 's/^- //p' single_run/$SRC/.hydra/overrides.yaml | tr '\n' ' ')
  ENVS=$(python3 -c "import json;print(' '.join(f'{k}={v}' for k,v in json.load(open('single_run/$SRC/launch_env.json')).items()))")
  while [ "$(njobs)" -ge 2 ]; do sleep 15; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 $ENVS \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    $ARGS hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)  env: $ENVS  args: $ARGS"; sleep 60
  local s; s=$(grep -o 'ワーカーの乱数の種 \[[0-9, ]*\]' single_run/$RUN/stdout.log | head -1)
  case "$s" in *"[1001, 1002, 1003, 1004]"*) log "✅ $RUN: $s";; *) log "⛔⛔ $RUN: 種の行が期待（[1001, 1002, 1003, 1004]）と違う／無い: '$s'";; esac; }

log "=== seed 1 の取り直し（Bug 76・系譜 9-283 A）HEAD=$(git rev-parse --short HEAD)。e2e_a1_fix3_pusher_phys_s2 の起動を待つ"
until [ -d single_run/e2e_a1_fix3_pusher_phys_s2/log ]; do sleep 60; done
log "A1 の押しの最後が起動した。空き次第入る"
for R in $RUNS; do launch $R; done
log "=== 14 本の起動が済んだ"
