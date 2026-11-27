#!/bin/bash
# 縦型の到達で腕の先端に荷物を持たせ、難しさの段（運動学 → 静力学）に判定が追従するかを確かめる学習 8 本（系譜 9-366、読み 65）。
#   2026-11-27 夜、ユーザー「境界をしるための難しくて実装が簡単なタスクはなかったってこと？」→「いいよ」→「やろう。…判定機の核なのではやめにやらない？」。
#
# ⭐ なぜ: 比べた 4 組はどれも運動学で順位が決まる条件だった（9-363）。押しの中では物理を変えても動力学の段が無かった（9-364・9-365）。
#   縦型の到達に先端の荷物を足すと、荷物の質量 1 つで段を振れる（9-366、`probe_vertical_payload.py`）: 0 kg は運動学、10 kg は静力学で順位が出て、
#   運動学の物差しは外れる。根元重 対 先端重は 0 kg で先端重が上（計算 15 %）、10 kg で根元重が上（計算 24 %）。
# ⭐ 変えるもの: 形（XML `e2e_a1v_phys_arm171_vprox`・`_vdist`。リンク 0.5/0.3/0.2102・0.2102/0.3/0.5。基準は 0.4031/0.3308/0.2763）と、
#   荷物（10 kg は cfg `pusher_tripo_v3_reallen_pay10`。元の cfg との差は robot の payload_mass の 1 行。起動行の `+robot.payload_mass` は読まれない）。
#   それ以外の起動行は元 run（`e2e_a1v_reach_phys_vp`〔seed 0〕・`e2e_a1v_reach_phys_s2_ws_vp`〔seed 1〕）の overrides と launch_env をそのまま読む。
#   比べる相手はこの 8 本の中だけ（同じ荷物の 2 形、同じ形の 2 荷物）。
# ⭐ 起動から 60 秒後に確かめる: 種の行・PRETICK の行（ワーカー 0）・cfg と xml_name・「robot=… は読まれない」の警告が無いこと。
#   荷物が物理に入ったことは、完走後の自動再生（record_arm_trace.py）の「実行時のリンクの質量」の行で見る（先端のリンク = カプセル + 10 kg）。
# ⭐ 9-361 のキュー（queue_1127_armature_gear.sh）が全部を起動し終えてから始める。9-362（queue_1127b）・E1（queue_1127d）はこのキューの後ろ（ユーザー判断）。
#
# ⭐⭐ 事前登録した読み 65（投入前の 11-27 夜に書いた。結果を見てから変えない。全文は系譜 9-366）:
#   量: 各 run の最良のエピソード報酬（exec_R_eps の最大）。2 seed の範囲（帯）で比べる。帯が重なれば順位を述べない。
#   65-1（10 kg）: a 根元重の帯が先端重の帯より上で重ならない（静力学・計算の予測どおり。運動学の物差しは逆を予測）／b 重なる／c 先端重が上で重ならない。
#   65-2（0 kg）: 計算の差（1.0）は、基準の形の学習の seed の幅（−18〜−27）より小さいので、b（重なる）を予測。a′ 先端重が上で重ならない（計算の向き）も矛盾しない。
#        根元重が上で重ならない（c′）なら、計算の 0 kg の予測と食い違う。
#   65-3（荷物の効き、各形）: 10 kg の帯が 0 kg の帯より下（荷物を止め続ける費用）。先端重の方が下がり幅が大きい（計算 0→10 kg: 根元重 −6.7→−38.6、先端重 −5.7→−50.9）。
#   65-4（記録だけ）: 最終の誤差、学習したギア比（10 kg では止める費用を下げるため上限 150 側に寄ると予測）、20 話の再生で目標の 10 mm 以内に入る割合。
#   判定器として言えること: 65-1 が a なら「荷物で正しい物差しが運動学から静力学へ替わる場面で、学習の判定は替わった後の物差しに追従した」。
#   停止条件: 床・台座・触れる前の ⛔⛔ が出たら、判定の前に再生して見る。完走ごとに目視（目視記録.md）。
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_1127e_vertical_payload.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo comm,args | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
# 元:新しい run:XML:cfg（10 kg の組を先に。同じ荷物の 2 形を続けて起動するので、2 本ずつ同時に終わる）
PLAN="e2e_a1v_reach_phys_vp:e2e_a1v_reach_vprox_p10_vp:e2e_a1v_phys_arm171_vprox:pusher_tripo_v3_reallen_pay10
e2e_a1v_reach_phys_vp:e2e_a1v_reach_vdist_p10_vp:e2e_a1v_phys_arm171_vdist:pusher_tripo_v3_reallen_pay10
e2e_a1v_reach_phys_s2_ws_vp:e2e_a1v_reach_vprox_p10_s2_vp:e2e_a1v_phys_arm171_vprox:pusher_tripo_v3_reallen_pay10
e2e_a1v_reach_phys_s2_ws_vp:e2e_a1v_reach_vdist_p10_s2_vp:e2e_a1v_phys_arm171_vdist:pusher_tripo_v3_reallen_pay10
e2e_a1v_reach_phys_vp:e2e_a1v_reach_vprox_p00_vp:e2e_a1v_phys_arm171_vprox:pusher_tripo_v3_reallen
e2e_a1v_reach_phys_vp:e2e_a1v_reach_vdist_p00_vp:e2e_a1v_phys_arm171_vdist:pusher_tripo_v3_reallen
e2e_a1v_reach_phys_s2_ws_vp:e2e_a1v_reach_vprox_p00_s2_vp:e2e_a1v_phys_arm171_vprox:pusher_tripo_v3_reallen
e2e_a1v_reach_phys_s2_ws_vp:e2e_a1v_reach_vdist_p00_s2_vp:e2e_a1v_phys_arm171_vdist:pusher_tripo_v3_reallen"
RUNS=$(echo "$PLAN" | cut -d: -f2 | tr '\n' ' ')

launch(){ local SRC=$1 RUN=$2 XML=$3 NCFG=$4
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return 0; fi
  if [ ! -f "single_run/$SRC/.hydra/overrides.yaml" ] || [ ! -f "single_run/$SRC/launch_env.json" ]; then log "⛔ $SRC の overrides か launch_env が無い。飛ばす"; return 0; fi
  if [ ! -f "assets/mujoco_envs/$XML.xml" ] || [ ! -f "design_opt/cfg/$NCFG.yml" ]; then log "⛔⛔ $XML.xml か $NCFG.yml が無い。飛ばす"; return 0; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 15; done
  local ARGS ENVS SEED CFG PID W
  CFG=$(sed -n 's/^- cfg=//p' single_run/$SRC/.hydra/overrides.yaml | head -1)
  [ "$CFG" = pusher_tripo_v3_reallen ] || { log "⛔⛔ $SRC の cfg が想定外（$CFG）。飛ばす"; return 0; }
  ARGS=$(sed -n 's/^- //p' single_run/$SRC/.hydra/overrides.yaml | sed "s/^cfg=$CFG\$/cfg=$NCFG/; s/^xml_name=e2e_a1v_phys_arm171\$/xml_name=$XML/" | tr '\n' ' ')
  SEED=$(sed -n 's/^- seed=//p' single_run/$SRC/.hydra/overrides.yaml | head -1)
  ENVS=$(python3 -c "import json;print(' '.join(f'{k}={v}' for k,v in json.load(open('single_run/$SRC/launch_env.json')).items()))")
  local AV; AV=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "$AV" -lt 3 ]; then log "⛔⛔ ディスクの空き ${AV} GB（3 GB 未満）。$RUN 以降を起動せずに止まる"; return 1; fi
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 $ENVS \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    $ARGS hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  PID=$!
  log "$RUN launched (PID $PID)  元 $SRC  seed $SEED  env: $ENVS  args: $ARGS"; sleep 60
  W=/tmp/cnoid_worker_${PID}_0.log
  local want="[$((1000*SEED+1)), $((1000*SEED+2)), $((1000*SEED+3)), $((1000*SEED+4))]" s p
  s=$(grep -ao 'ワーカーの乱数の種 \[[0-9, ]*\]' single_run/$RUN/stdout.log | head -1)
  case "$s" in *"$want"*) log "✅ $RUN: $s";; *) log "⛔⛔ $RUN: 種の行が期待（$want）と違う／無い: '$s'";; esac
  p=$(grep -ao 'tick の前にトルクを絞って速度の上限を守る: [0-9]* 関節' "$W" single_run/$RUN/stdout.log 2>/dev/null | head -1)
  if [ -n "$p" ]; then log "✅ $RUN: $p"; else log "⛔⛔ $RUN: PRETICK の行が無い（ワーカー 0 のログ $W と stdout のどちらにも）"; fi
  grep -q "^- cfg=$NCFG\$" single_run/$RUN/.hydra/overrides.yaml 2>/dev/null && log "✅ $RUN: cfg=$NCFG" || log "⛔⛔ $RUN: overrides に cfg=$NCFG が無い"
  grep -q "^- xml_name=$XML\$" single_run/$RUN/.hydra/overrides.yaml 2>/dev/null && log "✅ $RUN: xml_name=$XML" || log "⛔⛔ $RUN: overrides に xml_name=$XML が無い"
  if grep -aq 'は読まれない' single_run/$RUN/stdout.log; then log "⛔⛔ $RUN: 起動行の robot の指定が読まれていない警告がある"; fi
  return 0; }

log "=== 11-27 縦型の到達＋先端の荷物（系譜 9-366、読み 65）HEAD=$(git rev-parse --short HEAD)。8 本。9-361 のキューの起動が終わるのを待つ"
git diff HEAD -- khrylib design_opt scripts/choreonoid_train.py > single_run/queue_1127e_vertical_payload.code.diff
log "コードの未コミットの差分 $(wc -l < single_run/queue_1127e_vertical_payload.code.diff) 行を single_run/queue_1127e_vertical_payload.code.diff に残した（空ならコミット済み）"
while pgrep -f "^bash scripts/queue_1127_armature_gear.sh" > /dev/null; do sleep 60; done
log "9-361 のキューは起動を終えた。始める"
if ! pgrep -f "^bash scripts/watch_midrun.sh" > /dev/null; then
  nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
  log "学習中の見張り watch_midrun.sh を起動（PID $!）"
else
  log "学習中の見張り watch_midrun.sh は既に動いている"
fi
RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh を起動（PID $!、8 本）"
while IFS=: read -r SRC RUN XML NCFG; do
  [ -n "$SRC" ] || continue
  launch "$SRC" "$RUN" "$XML" "$NCFG" || break
done <<< "$PLAN"
log "=== 8 本の起動が済んだ（または止まった）"
