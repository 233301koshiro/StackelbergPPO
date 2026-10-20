#!/bin/bash
# ⭐ Bug 72 の取り直し・短腕（ユーザー判断 10-21「直して取り直す」、系譜 9-270）:
#   到達の短腕 tripo_pj_short_phys（10-21 06:46 完走）も、`11` のカプセルが関節の後ろへ 0.2 m はみ出して（設計変数 ext_start）
#   台座の球に 114 mm 食い込んだ形で学習していた（全 step）。再生 5 話のうち 3 話は NaN で 3〜5 step で切れ、
#   best −120.66 は切れた話が混ざった値（完走した話は −400 前後＝誤差約 400 mm）。短腕は 1 本目 0.15 m ＜ 0.33 m で食い込みうる形（9-264）。
#   押しの短腕 tripo_pjp_short_phys は同じ XML でまだ走っていないので、最初から門を直して走らせる。
#
# ⭐ 変えるのは 1 つ: `+env_specs.init_self_contact_caps=true`（queue_dist_capgate.sh と同じ）。起動行はそれ以外 queue_phys.sh の元の run と同じ。
#
# ⭐ queue_phys.sh が門なしの tripo_pjp_short_phys を後で走らせないよう、single_run/tripo_pjp_short_phys/log/ を目印として先に作る
#   （queue_phys.sh の launch() は log/ があると「既にある。飛ばす」）。目印には SKIPPED.txt で理由を書く。学習のログは無い。
#
# ⭐ 事前登録した読み（queue_phys.sh の ①・③・⑯ を変えずに読む。結果を見てから変えない）:
#   ⑳ 門が効く: 2 本とも最終形態の `0`×`11` の食い込みが 20 mm 未満、再生で NaN の話が 0。効かなければ以下は読まない（門の実装を疑う）
#   ① の後半: 到達の短腕が中間・長腕より下（帯が重ならない）／③ 1.99 倍が短腕より桁で良い／⑯ 押しの短腕は対象を 0.1 m 以上動かさない
#
# ⭐ 順番: queue_dist_capgate.sh が最後の 1 本（tripo_pjdp_dist_phys_cg）を起動した後、R1 より先に到達 → 押しの順で入る。
#   ⚠️ queue_phys.sh と 2 本の枠を取り合う。こちらは 15 秒ごとに見る（向こうは 180 秒）ので先に取る
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_short_capgate.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
PHYS="CNOID_SELF_COLLISION=1 CNOID_CULLING_DEPTH=1.0 CNOID_CTRL_CLIP=1 CNOID_HINGE_DAMP_TORQUE=1 CNOID_JOINT_MAX_VEL=3.1416 CNOID_JOINT_MAX_VEL_STRICT=1 CNOID_TICK_SYNC=1"
EXTRA=""
launch(){ local RUN=$1; shift
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 15; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 $PHYS $EXTRA \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    "$@" +env_specs.init_self_contact=true \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)"; sleep 60; }
C="num_threads=4 enable_wandb=false fix_skeleton=true +robot_param_scale=1"
PUSH="+reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true"
REACH8="+reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false +env_specs.cube_x_offset=100"
P="cfg=pusher_gearonly_real max_epoch_num=200 $C $PUSH"
R="cfg=pusher_gearonly_real max_epoch_num=200 $C $REACH8 +reward_specs.init_contact_penalty=1900"
CG="+env_specs.init_self_contact_caps=true"

log "=== 短腕の取り直し（門を本当のカプセルで、Bug 72・系譜 9-270）HEAD=$(git rev-parse --short HEAD)"
if [ ! -d single_run/tripo_pjp_short_phys/log ]; then
  mkdir -p single_run/tripo_pjp_short_phys/log
  echo "学習していない。queue_phys.sh が門なしで走らせないための目印（10-21、系譜 9-270）。門を直した版は tripo_pjp_short_phys_cg（queue_short_capgate.sh）" \
    > single_run/tripo_pjp_short_phys/SKIPPED.txt
  log "目印を作った: single_run/tripo_pjp_short_phys/log/（queue_phys.sh はこの run を飛ばす）"
fi
log "tripo_pjdp_dist_phys_cg の起動を待つ"
until [ -d single_run/tripo_pjdp_dist_phys_cg/log ]; do sleep 60; done
log "tripo_pjdp_dist_phys_cg が起動した。枠が空き次第入る"
launch tripo_pj_short_phys_cg   xml_name=tripo_arm_v2c_pj_short_phys_arm171 seed=0 $R $CG
launch tripo_pjp_short_phys_cg  xml_name=tripo_arm_v2c_pj_short_phys_arm171 seed=0 $P $CG
log "=== 2 本とも起動した"
