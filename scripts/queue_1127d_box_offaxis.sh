#!/bin/bash
# 押しの根元重 対 先端重を、箱を根元から延びる軸の外へ置いて学習する 8 本（系譜 9-363、事前登録 読み 64）。
#   ユーザー「差が出そうな条件で確かめるべきではないか？」→「E1 をばねの箱より前に」（11-27 夜）。
#
# ⭐ なぜ: 修論 4.4.3.2 の押しで 2 形態に差が出なかった条件（箱の手前の面 (0.85, 0)）では、2 形態は伸ばすと同じ棒で、
#   先端の最大の速さも点対称で一致する（9-363）。箱の手前の面を (0.75, 0.3) に置くと、同じ指標で先端重が 1.39 倍
#   （根元重 2.228・先端重 3.098 m/s）。対照の (0.75, 0) では一致（1.054・1.054）。2 つの位置は y だけが違う。
# ⭐ 変えるもの（元 run の起動行から）: +env_specs.cube_x_offset=-0.1 と +env_specs.cube_y_offset={0, 0.3}（design_opt/envs/pusher.py）。
#   ⚠️ 元 run の門の旗は形ごとにそろっていなかった（先端重の seed 0 だけ init_self_contact_caps=true。9-279）。
#   この 8 本は全部に init_self_contact_caps=true を付けてそろえる（Bug 72 の直した門）。比べる相手はこの 8 本の中だけ。
#   それ以外（cfg pusher_gearonly_real・xml・ctrl_cost 0.2・contact_weight 0・init_contact_penalty 50・arm_safe_init・
#   launch_env の物理〔速度の上限 3.1416・PRETICK ほか〕）は元 run の `.hydra/overrides.yaml` と `launch_env.json` をそのまま読む。
# ⭐ 11-27 夜（ユーザー「いいよ」）: 縦型の荷物（queue_1127e、系譜 9-366）を 9-361 の直後へ入れた。順は 9-361 → 9-366 → 9-362 → この E1 → ばねの箱。
#   待っていた bash（何も起動していない）を PID で止め、待ちの条件に queue_1127e を足して起動し直した。
# ⭐ 9-362 のキュー（queue_1127b_scaled_range.sh）が起動を終えて抜けてから、学習の枠（2 本）が空くたびに起動する。
#   8 本を起動し終えたら、ばねの箱のキュー（queue_1127c_spring_box_nxag.sh）を起動し直す（ユーザーの指定でばねの箱より前に入れるため、
#   待っていただけのばねの箱のキューはユーザーの承認で止めてある。中身は変えていない）。
# ⭐ 起動から 60 秒後に確かめる: 種の行、overrides の cube_x_offset・cube_y_offset・init_self_contact_caps、launch_env.json の速度の上限。
#
# ⭐⭐ 事前登録した読み 64（投入前に書いた。結果を見てから変えない。全文は系譜 9-363「読み 64」）:
#   量: best のエピソード報酬（2 seed の帯）、best を 50 話再生して箱を 0.1 m 以上動かした話の数。
#   64-0: 8 本とも、best の 50 話のうち 25 話以上で箱を 0.1 m 以上動かす（満たさない run があれば判定せずユーザーに諮る）。
#   64-1（軸上 y 0）: a 2 形の帯が重なる／b 分かれる（運動学は同点なので、動力学か学習の差。仕組みを調べる）。
#   64-2（軸外 y 0.3）: a 先端重の帯が根元重より完全に上（運動学の予測と同じ向き）／b 重なる／c 根元重が上（予測と逆）。
#   予想（判定に使わない）: 64-1 a、64-2 a。
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_1127d_box_offaxis.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo comm,args | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
# 元 run : 新しい run : y のずれ
PLAN="tripo_pjdp_prox_phys_vp:tripo_pjdp_prox_phys_bx_y30:0.3
tripo_pjdp_dist_phys_cg_vp:tripo_pjdp_dist_phys_bx_y30:0.3
tripo_pjdp_prox_phys_vp:tripo_pjdp_prox_phys_bx_y00:0
tripo_pjdp_dist_phys_cg_vp:tripo_pjdp_dist_phys_bx_y00:0
tripo_pjdp_prox_phys_s2_ws_vp:tripo_pjdp_prox_phys_bx_y30_s2:0.3
tripo_pjdp_dist_phys_s2_ws_vp:tripo_pjdp_dist_phys_bx_y30_s2:0.3
tripo_pjdp_prox_phys_s2_ws_vp:tripo_pjdp_prox_phys_bx_y00_s2:0
tripo_pjdp_dist_phys_s2_ws_vp:tripo_pjdp_dist_phys_bx_y00_s2:0"
RUNS="tripo_pjdp_prox_phys_bx_y30 tripo_pjdp_dist_phys_bx_y30 tripo_pjdp_prox_phys_bx_y00 tripo_pjdp_dist_phys_bx_y00 tripo_pjdp_prox_phys_bx_y30_s2 tripo_pjdp_dist_phys_bx_y30_s2 tripo_pjdp_prox_phys_bx_y00_s2 tripo_pjdp_dist_phys_bx_y00_s2"

launch(){ local SRC=$1 RUN=$2 DY=$3
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return 0; fi
  if [ ! -f "single_run/$SRC/.hydra/overrides.yaml" ] || [ ! -f "single_run/$SRC/launch_env.json" ]; then log "⛔ $SRC の overrides か launch_env が無い。飛ばす"; return 0; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 15; done
  local ARGS ENVS SEED CFG PID
  CFG=$(sed -n 's/^- cfg=//p' single_run/$SRC/.hydra/overrides.yaml | head -1)
  [ "$CFG" = pusher_gearonly_real ] || { log "⛔⛔ $SRC の cfg が想定外（$CFG）。飛ばす"; return 0; }
  ARGS=$(sed -n 's/^- //p' single_run/$SRC/.hydra/overrides.yaml | grep -v '^+env_specs.init_self_contact_caps=' | tr '\n' ' ')
  ARGS="$ARGS +env_specs.init_self_contact_caps=true +env_specs.cube_x_offset=-0.1 +env_specs.cube_y_offset=$DY"
  SEED=$(sed -n 's/^- seed=//p' single_run/$SRC/.hydra/overrides.yaml | head -1)
  ENVS=$(python3 -c "import json;print(' '.join(f'{k}={v}' for k,v in json.load(open('single_run/$SRC/launch_env.json')).items()))")
  local AV; AV=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "$AV" -lt 3 ]; then log "⛔⛔ ディスクの空き ${AV} GB（3 GB 未満）。$RUN 以降を起動せずに止まる"; return 1; fi
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 $ENVS \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    $ARGS hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  PID=$!
  log "$RUN launched (PID $PID)  元 $SRC  seed $SEED  y のずれ $DY  env: $ENVS  args: $ARGS"; sleep 60
  local want="[$((1000*SEED+1)), $((1000*SEED+2)), $((1000*SEED+3)), $((1000*SEED+4))]" s
  s=$(grep -ao 'ワーカーの乱数の種 \[[0-9, ]*\]' single_run/$RUN/stdout.log | head -1)
  case "$s" in *"$want"*) log "✅ $RUN: $s";; *) log "⛔⛔ $RUN: 種の行が期待（$want）と違う／無い: '$s'";; esac
  local O=single_run/$RUN/.hydra/overrides.yaml
  grep -q -- "+env_specs.cube_y_offset=$DY\$" $O 2>/dev/null && log "✅ $RUN: overrides の cube_y_offset=$DY" || log "⛔⛔ $RUN: overrides に cube_y_offset=$DY が無い"
  grep -q -- '+env_specs.cube_x_offset=-0.1$' $O 2>/dev/null && log "✅ $RUN: overrides の cube_x_offset=-0.1" || log "⛔⛔ $RUN: overrides に cube_x_offset=-0.1 が無い"
  [ "$(grep -c -- '+env_specs.init_self_contact_caps=true' $O 2>/dev/null)" = 1 ] && log "✅ $RUN: init_self_contact_caps=true が 1 つ" || log "⛔⛔ $RUN: init_self_contact_caps の行が 1 つでない"
  grep -q '"CNOID_JOINT_MAX_VEL": "3.1416"' single_run/$RUN/launch_env.json 2>/dev/null && log "✅ $RUN: launch_env.json の速度の上限 3.1416" || log "⛔⛔ $RUN: launch_env.json に速度の上限 3.1416 が無い"
  return 0; }

log "=== 11-27 夜 箱を軸の外へ（系譜 9-363、読み 64）HEAD=$(git rev-parse --short HEAD)。8 本。9-361・9-362 のキューの起動が終わるのを待つ"
git diff HEAD -- khrylib design_opt scripts/choreonoid_train.py > single_run/queue_1127d_box_offaxis.code.diff
log "コードの未コミットの差分 $(wc -l < single_run/queue_1127d_box_offaxis.code.diff) 行を single_run/queue_1127d_box_offaxis.code.diff に残した（空ならコミット済み）"
while pgrep -f "^bash scripts/queue_1127_armature_gear.sh" > /dev/null || pgrep -f "^bash scripts/queue_1127e_vertical_payload.sh" > /dev/null || pgrep -f "^bash scripts/queue_1127b_scaled_range.sh" > /dev/null; do sleep 60; done
log "9-361・9-362 のキューは起動を終えた。枠が空くたびに始める"
if ! pgrep -f "^bash scripts/watch_midrun.sh" > /dev/null; then
  nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
  log "学習中の見張り watch_midrun.sh を起動（PID $!）"
fi
RUNS="$RUNS" nohup setsid bash scripts/queue_postrun_replay.sh > /dev/null 2>&1 < /dev/null &
log "完走後の自動再生 queue_postrun_replay.sh を起動（PID $!、8 本）"
for P in $PLAN; do
  IFS=: read -r SRC RUN DY <<< "$P"
  launch "$SRC" "$RUN" "$DY" || break
done
log "=== 8 本の起動が済んだ（または止まった）。ばねの箱のキューを起動し直す"
nohup setsid bash scripts/queue_1127c_spring_box_nxag.sh > /dev/null 2>&1 < /dev/null &
log "queue_1127c_spring_box_nxag.sh を起動（PID $!）"
