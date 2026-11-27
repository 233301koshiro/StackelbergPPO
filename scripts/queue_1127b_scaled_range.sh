#!/bin/bash
# 太さと後方延長の探索範囲をリンクの設計長に比例させた確かめの学習 4 本（系譜 9-362、2026-11-27、ユーザー「じぶんの研究に明確なミスがあるまま進めたくない」「おすすめで」）。
#
# ⭐ なぜ: 太さ size [0.03, 0.10] m と後方延長 ext_start [0, 0.2] m は、原論文の公開実装の Pusher（リンク 0.566 m）の絶対値のまま。
#   リンク長とギア比の範囲は UR5e に合わせて変えたのに、この 2 つは変えていない。マトリクスは同じ形を 0.5・1.1・1.8 倍した比較なので、
#   上限 ÷ リンク長が短腕 196 %・中腕 89 %・長腕 55 %（後方延長）とそろわない。全面の取り直し（約 70 本・12 日）の前に、
#   判定（中腕と長腕の順位）が範囲の取り方で動くかを 4 本で確かめる。
# ⭐ 変えるのは 1 つ: cfg を `pusher_gearonly_real_sc` に替える（元の cfg との差は bone_frac の 2 行。size と ext_start の範囲を
#   設計長 × [0.0530, 0.1768]・× [0, 0.3536] にする＝原論文の Pusher の比を縮尺。xml_robot.py の Geom._apply_bone_frac）。
#   起動行は元 run の `.hydra/overrides.yaml` と `launch_env.json` をそのまま読み、cfg 名だけを差し替える。seed も元のまま（seed 0）。
#   押しの元は取り直し（9-351）の `_vp`（Bug 106・107 の直しの入った物理）、到達の元は修論の到達マトリクスの run（直しの前の物理）。
#   どちらも元と同じ物理で回すので、違うのは範囲だけ。
# ⭐ 起動から 60 秒後に、種の行・[bone_frac] の行（6 行＝3 リンク × 2 変数）・PRETICK の行（元が持つときだけ）を確かめる。
# ⭐ 9-361 のキュー（queue_1127_armature_gear.sh）が全部を起動し終えてから始める（ユーザー「おすすめで」: 9-361 を続け、その後ろ）。
#
# ⭐⭐ 事前登録した読み 62（結果を見てから変えない。全文は系譜 9-362）:
#   元の帯（3 seed の最良エピソード報酬）: 押し 中腕 [8.38, 8.63]・長腕 [25.15, 40.31]、到達 中腕 [-1.85, -1.64]・長腕 [-9.68, -7.71]。
#   62-1 押し: a 新しい 2 本がどちらも元の帯の中 → 範囲の取り方で判定は動かない（1 seed）。b 長腕 ＞ 中腕は保つが、どちらかが帯の外
#        → 値は動く。seed 1・2 を足して帯で見る。c 長腕 ≦ 中腕 → 判定が範囲に依る。全面の取り直しをユーザーと決める。
#   62-2 到達: 同じ a・b・c（順位は 中腕 ＞ 長腕、報酬は負の距離の和なので大きいほど良い）。あわせて両方とも目標に届くか。
#   62-3 形（記録だけ、判定に使わない）: 元の押しの中腕 seed 0 は先端のリンクの半径が上限 0.10 m に張り付いていた。新しい上限 0.040 m に
#        張り付くかと、後方延長を上限の何 % 使うか。
#   予測: 到達は a（届くかは運動学で決まり、太さと後方延長は届く範囲を変えない）。押しは b か a（中腕の先端が軽くなる）。
#   停止条件: 床・台座・触れる前の ⛔⛔ が出たら、判定の前に再生して見る。
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_1127b_scaled_range.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo comm,args | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
# 元 → 新しい run（check_before_conclusion.py の事前登録の判定が run 名を拾えるように、名前は省かずに書く）
PAIRS="tripo_pjp_mid_phys_vp:tripo_pjp_mid_phys_vp_sc
tripo_pjp_long_phys_vp:tripo_pjp_long_phys_vp_sc
tripo_pj_mid_phys:tripo_pj_mid_phys_sc
tripo_pj_long_phys:tripo_pj_long_phys_sc"
RUNS="tripo_pjp_mid_phys_vp_sc tripo_pjp_long_phys_vp_sc tripo_pj_mid_phys_sc tripo_pj_long_phys_sc"

launch(){ local SRC=$1 RUN=$2
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return 0; fi
  if [ ! -f "single_run/$SRC/.hydra/overrides.yaml" ] || [ ! -f "single_run/$SRC/launch_env.json" ]; then log "⛔ $SRC の overrides か launch_env が無い。飛ばす"; return 0; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 15; done
  local ARGS ENVS SEED CFG
  CFG=$(sed -n 's/^- cfg=//p' single_run/$SRC/.hydra/overrides.yaml | head -1)
  [ "$CFG" = pusher_gearonly_real ] || { log "⛔⛔ $SRC の cfg が想定外（$CFG）。飛ばす"; return 0; }
  ARGS=$(sed -n 's/^- //p' single_run/$SRC/.hydra/overrides.yaml | sed "s/^cfg=$CFG\$/cfg=${CFG}_sc/" | tr '\n' ' ')
  SEED=$(sed -n 's/^- seed=//p' single_run/$SRC/.hydra/overrides.yaml | head -1)
  ENVS=$(python3 -c "import json;print(' '.join(f'{k}={v}' for k,v in json.load(open('single_run/$SRC/launch_env.json')).items()))")
  local AV; AV=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "$AV" -lt 3 ]; then log "⛔⛔ ディスクの空き ${AV} GB（3 GB 未満）。$RUN 以降を起動せずに止まる"; return 1; fi
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 $ENVS \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    $ARGS hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)  元 $SRC  seed $SEED  env: $ENVS  args: $ARGS"; sleep 60
  local want="[$((1000*SEED+1)), $((1000*SEED+2)), $((1000*SEED+3)), $((1000*SEED+4))]" s p nb
  s=$(grep -ao 'ワーカーの乱数の種 \[[0-9, ]*\]' single_run/$RUN/stdout.log | head -1)
  case "$s" in *"$want"*) log "✅ $RUN: $s";; *) log "⛔⛔ $RUN: 種の行が期待（$want）と違う／無い: '$s'";; esac
  nb=$(grep -ac '^\[bone_frac\]' single_run/$RUN/stdout.log)
  if [ "$nb" -eq 6 ]; then log "✅ $RUN: [bone_frac] の行 6 本（3 リンク × size・ext_start）"; grep -a '^\[bone_frac\]' single_run/$RUN/stdout.log | sed 's/^/    /' | tee -a $Q > /dev/null
  else log "⛔⛔ $RUN: [bone_frac] の行が ${nb} 本（期待 6）。範囲の比例が効いていない疑い"; fi
  if grep -q CNOID_JOINT_MAX_VEL_PRETICK single_run/$SRC/launch_env.json; then
    p=$(grep -ao 'tick の前にトルクを絞って速度の上限を守る: [0-9]* 関節' single_run/$RUN/stdout.log | head -1)
    if [ -n "$p" ]; then log "✅ $RUN: $p"; else log "⛔⛔ $RUN: PRETICK の行が無い（元は持つのに）"; fi
  fi
  grep -q "cfg=${CFG}_sc" single_run/$RUN/.hydra/overrides.yaml 2>/dev/null && log "✅ $RUN: cfg=${CFG}_sc" || log "⛔⛔ $RUN: overrides に cfg=${CFG}_sc が無い"
  return 0; }

log "=== 11-27 太さと後方延長の範囲を比例させた確かめ（系譜 9-362、読み 62）HEAD=$(git rev-parse --short HEAD)。4 本。9-361 のキューの起動が終わるのを待つ"
git diff HEAD -- khrylib design_opt scripts/choreonoid_train.py > single_run/queue_1127b_scaled_range.code.diff
log "コードの未コミットの差分 $(wc -l < single_run/queue_1127b_scaled_range.code.diff) 行を single_run/queue_1127b_scaled_range.code.diff に残した（空ならコミット済み）"
while pgrep -f "^bash scripts/queue_1127_armature_gear.sh" > /dev/null; do sleep 60; done
log "9-361 のキューは起動を終えた。始める"
if ! pgrep -f "^bash scripts/watch_midrun.sh" > /dev/null; then
  nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
  log "学習中の見張り watch_midrun.sh を起動（PID $!）"
else
  log "学習中の見張り watch_midrun.sh は既に動いている"
fi
RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh を起動（PID $!、4 本）"
for P in $PAIRS; do
  launch "${P%%:*}" "${P##*:}" || break
done
log "=== 4 本の起動が済んだ（または止まった）"
