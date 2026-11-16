#!/bin/bash
# 修論 4.5.4（助言の倍率の押し）と 4.5.5（A1 の押し、関節 3 の可動・固定）の 10 本を、Bug 106・107 を直した物理で取り直す
# （ユーザー判断 2026-11-16「たしていいよ」、系譜 9-351 の追記）。queue_1116_bug106_retake.sh（37 本）の後ろに続ける。
#
# ⭐ なぜ: 37 本に入れなかった押しを直した物理で再生すると、4.5.4 は箱の移動が 0.72〜0.93 倍、4.5.5 は 0.83〜1.07 倍で、
#   核の押しと同じ程度に Bug 106 が効いていた。第 4 章の押しを全部同じ物理にそろえる。4.7（軸6）は直しが一度も効かず（絞った回数 0）、
#   到達は報酬が完全に一致したので取り直さない。
# ⭐ 1 本目のキュー（queue_1116_bug106_retake.sh）の bash が 37 本の起動を終えて抜けるまで待ち、その後は同じく 2 本ずつ起動する。
#   起動行・直しの環境変数・確認の仕方・run 名（元＋_vp）は 1 本目と同じ。
#
# ⭐ 事前登録した読み 57 の続き（結果を見てから変えない。系譜 9-351 の追記に同じ文）:
#   前提は 57 と同じ。
#   57-6 4.5.4 の 2.36 倍の押し（2 seed）: 読み ㊻ と同じ物差し（best の 50 話の再生で箱を 0.1 m 以上動かした話）。a 2 seed とも 48 / 50 以上（修論 4.5.4 の押しの結論を保つ）／b どちらかが 48 未満（4.5.4 の押しの結論を書き直す）。
#        2.11 倍（2 seed）は記録のみ（修論の「旧 2.11 倍は 42〜45 / 50」の比較相手として同じ物差しで数える）。
#   57-7 4.5.5 の A1 の押し（可動・関節 3 固定、3 seed）: 読み ㊶-2 と同じ物差し（best の帯）。a 可動の帯が固定より上で重ならない（「固定すると押しの性能が低下する」を言える）／b 重なる（今の修論の書き方のまま）／c 固定の帯が上で重ならない。
#   予想（判定に使わない）: 57-6 は a、57-7 は b。
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_1116b_bug106_extra.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo comm,args | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
FIX_ENV="CNOID_JOINT_MAX_VEL_PRETICK=1 CNOID_CAPSULE_INERTIA_ROT=1"
# 元の run（2 本ずつ、比べる組がそろって終わる順）
SRCS="tripo_pjr236p_phys tripo_pjr236p_phys_s2
tripo_pjr211p_phys tripo_pjr211p_phys_s2_ws
e2e_a1_pusher_phys e2e_a1_fix3_pusher_phys
e2e_a1_pusher_phys_s2 e2e_a1_fix3_pusher_phys_s2
e2e_a1_pusher_phys_s3 e2e_a1_fix3_pusher_phys_s3"
SRCS=$(echo $SRCS)
RUNS=$(for s in $SRCS; do printf '%s_vp ' "$s"; done)

launch(){ local SRC=$1 RUN=${1}_vp
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return 0; fi
  if [ ! -f "single_run/$SRC/.hydra/overrides.yaml" ] || [ ! -f "single_run/$SRC/launch_env.json" ]; then log "⛔ $SRC の overrides か launch_env が無い。飛ばす"; return 0; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 15; done
  # 元の run が走っている最中（9-349 の _s5）なら、完走して log_train.txt が 200 epoch になるまで待つ必要は無い（overrides は起動時にできる）
  local ARGS ENVS SEED
  ARGS=$(sed -n 's/^- //p' single_run/$SRC/.hydra/overrides.yaml | tr '\n' ' ')
  SEED=$(sed -n 's/^- seed=//p' single_run/$SRC/.hydra/overrides.yaml | head -1)
  ENVS=$(python3 -c "import json;print(' '.join(f'{k}={v}' for k,v in json.load(open('single_run/$SRC/launch_env.json')).items()))")
  local AV; AV=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "$AV" -lt 3 ]; then log "⛔⛔ ディスクの空き ${AV} GB（3 GB 未満）。$RUN 以降を起動せずに止まる"; return 1; fi
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 $ENVS $FIX_ENV \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    $ARGS hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)  元 $SRC  seed $SEED  env: $ENVS $FIX_ENV  args: $ARGS"; sleep 60
  local want="[$((1000*SEED+1)), $((1000*SEED+2)), $((1000*SEED+3)), $((1000*SEED+4))]" s p
  s=$(grep -ao 'ワーカーの乱数の種 \[[0-9, ]*\]' single_run/$RUN/stdout.log | head -1)
  case "$s" in *"$want"*) log "✅ $RUN: $s";; *) log "⛔⛔ $RUN: 種の行が期待（$want）と違う／無い: '$s'";; esac
  p=$(grep -ao 'tick の前にトルクを絞って速度の上限を守る: [0-9]* 関節' single_run/$RUN/stdout.log | head -1)
  if [ -n "$p" ]; then log "✅ $RUN: $p"; else log "⛔⛔ $RUN: PRETICK の行が無い（Bug 106 の直しが効いていない疑い）"; fi
  if python3 -c "import json,sys;d=json.load(open('single_run/$RUN/launch_env.json'));sys.exit(0 if d.get('CNOID_JOINT_MAX_VEL_PRETICK')=='1' and d.get('CNOID_CAPSULE_INERTIA_ROT')=='1' else 1)" 2>/dev/null; then
    log "✅ $RUN: launch_env.json に 2 つの直しが残っている"; else log "⛔⛔ $RUN: launch_env.json に直しの変数が無い（再生が旧い物理になる）"; fi
  return 0; }

log "=== 11-16 の Bug 106・107 の取り直しの続き 10 本（系譜 9-351 の追記、読み 57-6・57-7）。1 本目のキューの bash が抜けるのを待つ"
while pgrep -f "bash scripts/queue_1116_bug106_retake.sh" > /dev/null; do sleep 60; done
log "1 本目のキューが抜けた。HEAD=$(git rev-parse --short HEAD)。$(echo $RUNS | wc -w) 本を起動する"
git diff HEAD -- khrylib/rl/envs/common/mujoco_env_choreonoid.py scripts/choreonoid_train.py > single_run/queue_1116b_bug106_extra.code.diff
log "コードの未コミットの差分 $(wc -l < single_run/queue_1116b_bug106_extra.code.diff) 行（空ならコミット済み。1 本目の起動時のコードは 96144c5e）"
if ! pgrep -f "^bash scripts/watch_midrun.sh" > /dev/null; then
  nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
  log "学習中の見張り watch_midrun.sh を起動（PID $!）"
else
  log "学習中の見張り watch_midrun.sh は既に動いている（走っている学習を ps から拾うので、新しい run も見る）"
fi
RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh を起動（PID $!、$(echo $RUNS | wc -w) 本）"
for SRC in $SRCS; do
  launch "$SRC" || break
done
log "=== $(echo $RUNS | wc -w) 本の起動が済んだ（または止まった）"
