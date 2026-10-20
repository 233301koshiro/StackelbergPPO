#!/bin/bash
# ⭐ 系譜 9-267（ユーザー判断 10-20「A で」）: 閉ループの押し、2.11 倍の腕に seed1 を足す（tripo_pjr211p_phys_s2）。
#
# ⭐ なぜ: queue_phys.sh は 2.11 倍の押しを seed0 だけで組んでおり、1 本で足りる理由は記録に無かった（進捗.md、10-19 に気づいた）。
#   読み ⑰「2.11 倍は対象を 0.1 m 以上動かす」は学習がうまくいくかで決まり、幾何では保証されないので 2 seed にする。
#   短腕（tripo_pj_short_phys・tripo_pjp_short_phys）は 1 seed のまま: 腕の届く範囲 0.403 m に対し、到達の目標は 0.8 m 先・押しの箱は手前の面が 0.85 m 先（中心 x = 1.0、半辺 0.15。XML）で、
#   どの方策でも届かないことが幾何で決まる（同じ判断で、理由を修論 4.1.2 に書く）。
#
# ⭐ 起動行は queue_phys.sh の tripo_pjr211p_phys と seed だけが違う（launch() も同じ）。
# ⭐ 読みは queue_phys.sh の ⑰ をそのまま 2 seed で読む（どちらの seed でも対象が 0.1 m 以上動くか）。結果を見てから変えない。
#
# ⭐ 順番: queue_phys.sh が tripo_pjr211p_phys（seed0）を起動した後、次の e2e_a1_pusher_phys より先に入る（⑰ を 2 seed で同じ頃に読むため）。
#   ⚠️ queue_phys.sh と 2 本の枠を取り合う。こちらは 15 秒ごとに見る（向こうは 180 秒）ので先に取る（queue_dist_capgate.sh と同じ方式）
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_rec211p_s2.log
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
P="cfg=pusher_gearonly_real max_epoch_num=200 $C $PUSH"

log "=== 2.11 倍の押しの seed1（系譜 9-267）HEAD=$(git rev-parse --short HEAD)。tripo_pjr211p_phys の起動を待つ"
until [ -d single_run/tripo_pjr211p_phys/log ]; do sleep 60; done
log "tripo_pjr211p_phys が起動した。枠が空き次第入る"
launch tripo_pjr211p_phys_s2 xml_name=tripo_arm_v2c_pj_rec211_phys_arm171 seed=1 $P
log "=== 起動した"
