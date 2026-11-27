#!/bin/bash
# ばねで引き戻される箱（系譜 9-354）を、直した設計空間で取り直す 6 本。事前登録 読み 63（11-27 夕方。全文は系譜 9-354「読み 63」）。
#   ユーザー「つまり直した設計空間でばねの箱をやるってことね」「確かめの四本が終わったらすぐやるようにキューに割り込める？」
#   → 「（ギア比の費用も）両方入れる」。
#
# ⭐ なぜ: 読み 59 の 6 本（*_cw1）は 9-362 で直す前の設計空間だった（後方延長あり・太さは絶対値 [0.03, 0.10] m・密度は固定・ギア比に費用なし）。
#   ばねの箱は「ギア比の上限への張り付き（第2層）」を静力学の正解と比べる課題なので、費用が無いと要らなくても上限へ流れる（9-360）。
# ⭐ 変えるもの（読み 59 の起動行から。比べる相手は読み 59 ではなく静力学の正解なので、まとめて「最終の設計空間」にする）:
#   ① cfg を pusher_tripo_v3_reallen → pusher_tripo_v3_reallen_nxag（後方延長なし・太さは設計長 × [0.053, 0.177]・回転子の慣性 ∝ ギア比²）
#   ② 起動行に CNOID_PIPE_WALL=0.002（質量をいまの半径から薄肉パイプで）
#   それ以外（seed・ばね k・向き y・目標 (1.0, 0.30)・contact_weight 1.0・制御コスト 0・Bug 106・107 の直し）は元 run の
#   `.hydra/overrides.yaml` と `launch_env.json` をそのまま読む。
# ⭐ 起動から 60 秒後に確かめる: 種の行、[bone_frac] の行（size 4 本・ext_start 0 本。リンク 1 の柱は fix_base で形を固定するので範囲は使わない）、
#   ワーカー 0 のログの [pipe] の行と PRETICK の行（回転子の慣性 J がギア比で関節ごとに違う）、overrides の cfg、launch_env.json の PIPE_WALL とばねの k。
# ⭐ 9-362 のキュー（queue_1127b_scaled_range.sh）が 4 本を起動し終えて抜けてから、学習の枠（2 本）が空くたびに起動する
#   （＝ 9-362 の 4 本が走り終わる順に入る）。動いているキューは書き換えない。
# ⭐ 学習の本数は実行ファイル名が choreonoid のものだけを数える（Bug 101）。ディスクの空きが 3 GB を切ったら止まる。
#
# ⭐⭐ 事前登録した読み 63（投入前に書いた。結果を見てから変えない。全文は系譜 9-354）:
#   量は読み 59 と同じ（触れた・d_end の 50 話の中央値・旋回関節 1_joint のギア比・第2層。scripts/judge_reading59.py の定義。
#   上限 150 の 20 % 以内＝ 125 以上）。静力学の正解は読み 58 と同じ（k 350: 要るトルク 81〜129 ≤ 150 → 保てる、k 900: 209〜331 > 150 → 保てない）。
#   63-0: a 6 本とも触れる／b 触れない run がある／c 1 本も触れない（ユーザーに諮る）。
#   ⭐ 読み 59 の弱さを塞ぐ: 63-1・63-2 は、その k の 3 seed がどれも best の 50 話の半分（25 話）以上で箱に当たったときだけ判定する。
#      満たさない k は判定に使わずユーザーに諮る。
#   63-1（k 350: e2e_a1v_spring350_cw1_nxag・e2e_a1v_spring350_s2_ws_cw1_nxag・e2e_a1v_spring350_s3_cw1_nxag）: a〜c は 58-1 と同じ。
#   63-2（k 900: e2e_a1v_spring900_cw1_nxag・e2e_a1v_spring900_s2_ws_cw1_nxag・e2e_a1v_spring900_s3_cw1_nxag）: a〜c は 58-2 と同じ。
#      ⛔ k 900 で 1 seed でも届いたら抜け穴を疑い、結論を書かずに調べる。
#   63-3（上限 500）: 63-2 が a か b のときだけユーザーに確かめて投入。
#   記録のみ: 読み 59 の同じ seed との差（触れた話・d_end・旋回のギア比）、関節ごとの回転子の慣性、形の長さ、最初に触れたエポック。
#   予想（判定に使わない）: 63-0 は a。63-1・63-2 は読み 59 と同じく届かない seed が出やすい（設計空間は学習の届きにくさを直さない）。
#   停止条件: 床・台座・触れる前の ⛔⛔ が出たら、判定の前に再生して見る（ばねの箱の「触れる前の加速」はばねの式と照らす。9-354）。
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_1127c_spring_box_nxag.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo comm,args | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
NEW_CFG=pusher_tripo_v3_reallen_nxag
NEW_ENV="CNOID_PIPE_WALL=0.002"
# 元 → 新しい run（check_before_conclusion.py の事前登録の判定が run 名を拾えるように、名前は省かずに書く）
PAIRS="e2e_a1v_spring350_cw1:e2e_a1v_spring350_cw1_nxag
e2e_a1v_spring900_cw1:e2e_a1v_spring900_cw1_nxag
e2e_a1v_spring350_s2_ws_cw1:e2e_a1v_spring350_s2_ws_cw1_nxag
e2e_a1v_spring900_s2_ws_cw1:e2e_a1v_spring900_s2_ws_cw1_nxag
e2e_a1v_spring350_s3_cw1:e2e_a1v_spring350_s3_cw1_nxag
e2e_a1v_spring900_s3_cw1:e2e_a1v_spring900_s3_cw1_nxag"
RUNS="e2e_a1v_spring350_cw1_nxag e2e_a1v_spring900_cw1_nxag e2e_a1v_spring350_s2_ws_cw1_nxag e2e_a1v_spring900_s2_ws_cw1_nxag e2e_a1v_spring350_s3_cw1_nxag e2e_a1v_spring900_s3_cw1_nxag"

launch(){ local SRC=$1 RUN=$2
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return 0; fi
  if [ ! -f "single_run/$SRC/.hydra/overrides.yaml" ] || [ ! -f "single_run/$SRC/launch_env.json" ]; then log "⛔ $SRC の overrides か launch_env が無い。飛ばす"; return 0; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 15; done
  local ARGS ENVS SEED CFG K PID W
  CFG=$(sed -n 's/^- cfg=//p' single_run/$SRC/.hydra/overrides.yaml | head -1)
  [ "$CFG" = pusher_tripo_v3_reallen ] || { log "⛔⛔ $SRC の cfg が想定外（$CFG）。飛ばす"; return 0; }
  ARGS=$(sed -n 's/^- //p' single_run/$SRC/.hydra/overrides.yaml | sed "s/^cfg=$CFG\$/cfg=$NEW_CFG/" | tr '\n' ' ')
  SEED=$(sed -n 's/^- seed=//p' single_run/$SRC/.hydra/overrides.yaml | head -1)
  K=$(python3 -c "import json;print(json.load(open('single_run/$SRC/launch_env.json')).get('CNOID_CUBE_SPRING_K',''))")
  ENVS=$(python3 -c "import json;print(' '.join(f'{k}={v}' for k,v in json.load(open('single_run/$SRC/launch_env.json')).items()))")
  local AV; AV=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "$AV" -lt 3 ]; then log "⛔⛔ ディスクの空き ${AV} GB（3 GB 未満）。$RUN 以降を起動せずに止まる"; return 1; fi
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 $ENVS $NEW_ENV \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    $ARGS hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  PID=$!
  log "$RUN launched (PID $PID)  元 $SRC  seed $SEED  k $K  env: $ENVS $NEW_ENV  args: $ARGS"; sleep 60
  W=/tmp/cnoid_worker_${PID}_0.log
  local want="[$((1000*SEED+1)), $((1000*SEED+2)), $((1000*SEED+3)), $((1000*SEED+4))]" s p nb ne np
  s=$(grep -ao 'ワーカーの乱数の種 \[[0-9, ]*\]' single_run/$RUN/stdout.log | head -1)
  case "$s" in *"$want"*) log "✅ $RUN: $s";; *) log "⛔⛔ $RUN: 種の行が期待（$want）と違う／無い: '$s'";; esac
  nb=$(grep -ac '^\[bone_frac\].* の size:' single_run/$RUN/stdout.log); ne=$(grep -ac '^\[bone_frac\].* の ext_start:' single_run/$RUN/stdout.log)
  if [ "$nb" -eq 4 ] && [ "$ne" -eq 0 ]; then log "✅ $RUN: [bone_frac] は size の 4 本だけ（後方延長は設計変数に無い）"
  else log "⛔⛔ $RUN: [bone_frac] の行が size ${nb} 本・ext_start ${ne} 本（期待 4・0）"; fi
  np=$(grep -ac '^\[pipe\]' "$W" 2>/dev/null)
  if [ "${np:-0}" -ge 3 ]; then log "✅ $RUN: [pipe] の行 ${np} 本（ワーカー 0）"; grep -a '^\[pipe\]' "$W" | head -4 | sed 's/^/    /' | tee -a $Q > /dev/null
  else log "⛔⛔ $RUN: ワーカー 0 のログ（$W）に [pipe] の行が無い（質量の式が効いていない疑い）"; fi
  p=$(grep -ao 'tick の前にトルクを絞って速度の上限を守る: [^（]*' "$W" 2>/dev/null | head -1)
  if [ -n "$p" ]; then log "✅ $RUN: $p（ワーカー 0。最後の数が関節ごとの回転子の慣性 J ＝ ギア比の費用）"; else log "⛔⛔ $RUN: ワーカー 0 のログに PRETICK の行が無い"; fi
  grep -q "cfg=$NEW_CFG" single_run/$RUN/.hydra/overrides.yaml 2>/dev/null && log "✅ $RUN: cfg=$NEW_CFG" || log "⛔⛔ $RUN: overrides に cfg=$NEW_CFG が無い"
  grep -q '"CNOID_PIPE_WALL": "0.002"' single_run/$RUN/launch_env.json 2>/dev/null && log "✅ $RUN: launch_env.json に CNOID_PIPE_WALL=0.002" || log "⛔⛔ $RUN: launch_env.json に CNOID_PIPE_WALL が無い"
  grep -q "\"CNOID_CUBE_SPRING_K\": \"$K\"" single_run/$RUN/launch_env.json 2>/dev/null && log "✅ $RUN: launch_env.json の CNOID_CUBE_SPRING_K=$K" || log "⛔⛔ $RUN: launch_env.json のばねの k が $K でない"
  grep -q 'contact_weight=1.0' single_run/$RUN/.hydra/overrides.yaml 2>/dev/null && log "✅ $RUN: overrides の contact_weight=1.0" || log "⛔⛔ $RUN: overrides に contact_weight=1.0 が無い"
  return 0; }

log "=== 11-27 ばねの箱を直した設計空間で（系譜 9-354、読み 63）HEAD=$(git rev-parse --short HEAD)。6 本。9-362 のキューの起動が終わるのを待つ"
git diff HEAD -- khrylib design_opt scripts/choreonoid_train.py > single_run/queue_1127c_spring_box_nxag.code.diff
log "コードの未コミットの差分 $(wc -l < single_run/queue_1127c_spring_box_nxag.code.diff) 行を single_run/queue_1127c_spring_box_nxag.code.diff に残した（空ならコミット済み）"
while pgrep -f "^bash scripts/queue_1127_armature_gear.sh" > /dev/null || pgrep -f "^bash scripts/queue_1127b_scaled_range.sh" > /dev/null; do sleep 60; done
log "9-361・9-362 のキューは起動を終えた。枠が空くたびに始める"
if ! pgrep -f "^bash scripts/watch_midrun.sh" > /dev/null; then
  nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
  log "学習中の見張り watch_midrun.sh を起動（PID $!）"
else
  log "学習中の見張り watch_midrun.sh は既に動いている"
fi
RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh を起動（PID $!、6 本）"
for P in $PAIRS; do
  launch "${P%%:*}" "${P##*:}" || break
done
log "=== 6 本の起動が済んだ（または止まった）"
