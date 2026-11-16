#!/bin/bash
# 修論の核の 37 本を、Bug 106・107 を直した物理で取り直す（ユーザー判断 2026-11-16「全然予備審査前に終わるなら回してもいいよ」、系譜 9-350・9-351）。
#
# ⭐ なぜ: Fable 5.1 の見直し（系譜 9-350）で、速度の上限 3.14 rad/s を tick の後で切るため、押しの打つ瞬間の接触が上限を超えた腕の速さで
#   解かれていたと分かった（Bug 106。箱に触れた tick の超え幅は中央 0.35〜0.82・最大 1.27 rad/s）。ギア比の高い条件ほど得をするので、
#   修論 4.5.4（上限 300 N·m で向上）・4.4.3.2（押しの配分、best の帯の間 0.08）が揺れる。あわせて縦型のカプセルの慣性の向き（Bug 107）を直す。
#   箱が床から浮いたスライドである点（Bug 108）は直さず、修論の限界として書く（ユーザー判断 11-16）。
# ⭐ 直し方（ユーザー判断 11-16）: CNOID_JOINT_MAX_VEL_PRETICK=1（tick の前に、そのトルクだけでは上限を超えない大きさまでモータのトルクを絞る。
#   後で切る処理は安全弁として残す）と CNOID_CAPSULE_INERTIA_ROT=1（カプセルの慣性を軸の向きに回す）。両方とも全 37 本に掛ける
#   （平面アームはヨー軸まわりの慣性が変わらないので Bug 107 の直しは効かないが、物理を全 run でそろえるため）。
#   再生での確認（11-16）: オフでは既存の再生と 1 mm 単位で同じ。オンで箱に触れた tick の超え幅は中央 0.01〜0.06・最大 0.08〜0.13 rad/s。
# ⭐ 範囲（ユーザー判断 11-16「修論の核 37 本」）: R1（中間の凍結・最適化 各 3）・マトリクスの長腕 3・上限 300 N·m 2・押しの配分（根元重・先端重 各 5、
#   凍結 各 3）・R2（縦型 A1v の到達・押し 各 5）。R2 は到達と押しを比べるので、到達も同じ物理で取り直す。
#   到達のほかの核（マトリクス・配分の到達）は対象に触れないので Bug 106 が効かず、取り直さない（Fable が到達の核 17 本で確認）。
# ⭐ 起動行は各元 run の `.hydra/overrides.yaml` と `launch_env.json` をそのまま読み（seed も元のまま）、上の 2 つの環境変数だけを足す。
#   run 名は元の名前の末尾に `_vp`。投入時の HEAD と、コードの未コミットの差分を single_run/queue_1116_bug106_retake.code.diff に残す。
# ⭐ 起動から 60 秒後に、種の行と「tick の前にトルクを絞って」の行（掴んだ関節の数）を確かめてログに書く（CLAUDE.md ⑤-3-2）。
#
# ⭐ 事前登録した読み 57（結果を見てから変えない。系譜 9-351 に同じ文）:
#   前提: 37 本とも完走・check_before_conclusion.py・目視（目視記録に 1 行）・ワーカーの種・不変条件・stdout に PRETICK の行。
#   判定の規則は修論 4.2.2 と同じ（各条件の best exec_R_eps のシードにわたる範囲＝帯。重ならないときだけ順位を言う）。R2 は 9-349 と同じ d6_bands.py の trace_metrics。
#   57-1 R1（中間、3 seed）: a 最適化の帯が凍結の帯より上で重ならない（R1 を保つ）／b 重なる（R1 は押しで満たされない）。
#   57-2 マトリクスの押し（3 seed）: a 長腕 ＞ 中間で重ならない／b 重なる／c 中間 ＞ 長腕。
#   57-3 押しの配分（5 seed）: a 根元重 ＞ 先端重で重ならない／b 重なる／c 逆。あわせて凍結（3 seed）: 重なるなら修論 4.4.3.2「形態の最適化を含む判定のみが区別した」を保つ候補、離れるなら取り下げる。
#        （57-3 が b か c なら、「のみが区別した」は凍結の結果によらず取り下げる）
#   57-4 4.5.4（上限 300 N·m 2 seed 対 150 N·m 3 seed、どちらも中間）: a 300 の帯が上で重ならない（「向上」を保つ）／b 重なる・下（「向上」を取り下げる）。
#   57-5 R2（5 seed）: 9-349 の読み 56 と同じ 2 つの量（関節 4 のギア比・リンク長の合計）。a 両方とも帯が重ならない／b 片方だけ／c 両方重なる。
#        記録のみ: 関節 1〜3 のギア比・先端の太さ（判定に使う量を結果を見てから足さない）。
#   記録のみ（判定に使わない）: 各条件の帯の値、旧い物理（_vp なし）との比。
#   予想（判定に使わない）: 押しの best は全体に 2 割ほど下がる（再生で 0.77〜0.80 倍）。57-4 は b。ほかは予想しない。
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_1116_bug106_retake.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo comm,args | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
FIX_ENV="CNOID_JOINT_MAX_VEL_PRETICK=1 CNOID_CAPSULE_INERTIA_ROT=1"
# 元の run（2 本ずつ、比べる組がそろって終わる順）
SRCS="tripo_pjp_mid_phys tripo_pjp_mid_frozen_phys_fx
tripo_pjp_mid_phys_s2_ws tripo_pjp_mid_frozen_phys_fx_s2
tripo_pjp_mid_phys_s3 tripo_pjp_mid_frozen_phys_fx_s3
tripo_pjp_long_phys tripo_pjp_long_phys_s2_ws
tripo_pjp_long_phys_s3 tripo_pjp_mid_phys_ub300
tripo_pjp_mid_phys_ub300_s2 tripo_pjdp_prox_phys
tripo_pjdp_dist_phys_cg tripo_pjdp_prox_phys_s2_ws
tripo_pjdp_dist_phys_s2_ws tripo_pjdp_prox_phys_s3
tripo_pjdp_dist_phys_cg_s3 tripo_pjdp_prox_phys_s4
tripo_pjdp_dist_phys_cg_s4 tripo_pjdp_prox_phys_s5
tripo_pjdp_dist_phys_cg_s5 tripo_pjdp_prox_frozen_phys_fx
tripo_pjdp_dist_frozen_phys_cg_fx tripo_pjdp_prox_frozen_phys_fx_s2
tripo_pjdp_dist_frozen_phys_cg_fx_s2 tripo_pjdp_prox_frozen_phys_fx_s3
tripo_pjdp_dist_frozen_phys_cg_fx_s3 e2e_a1v_reach_phys
e2e_a1v_pusher_phys e2e_a1v_reach_phys_s2_ws
e2e_a1v_pusher_phys_s2_ws e2e_a1v_reach_phys_s3
e2e_a1v_pusher_phys_s3 e2e_a1v_reach_phys_s4
e2e_a1v_pusher_phys_s4 e2e_a1v_reach_phys_s5
e2e_a1v_pusher_phys_s5"
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

log "=== 11-16 の Bug 106・107 の取り直し（系譜 9-351、読み 57）HEAD=$(git rev-parse --short HEAD)。$(echo $RUNS | wc -w) 本"
git diff HEAD -- khrylib/rl/envs/common/mujoco_env_choreonoid.py scripts/choreonoid_train.py > single_run/queue_1116_bug106_retake.code.diff
log "コードの未コミットの差分 $(wc -l < single_run/queue_1116_bug106_retake.code.diff) 行を single_run/queue_1116_bug106_retake.code.diff に残した（空ならコミット済み）"
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
