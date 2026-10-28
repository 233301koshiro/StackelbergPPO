#!/bin/bash
# ホッケー（実在の物理）の取り直し 4 回目 — 床をカプセルの表面で測る（Bug 78、系譜 9-288、ユーザー判断 2026-10-22「修正して取り直す」「全部この案で」）
# → その後に障害物 Reach（D-5）の 2 本（queue_obspen_phys.sh から引き継ぐ。あちらは古い条件の blk3d_s2 を起動する作りだったので止めた）
#
# ⭐ ホッケーの起動行は queue_hockey_phys_g10_blk3d.sh と**同じ**で、変えたのは +env_specs.floor_by_surface=true の 1 つだけ:
#   開始の姿勢の探索（arm_safe_init）と開始時点の罰（check_floor_penetration）が床をリンクの軸で測っていて、
#   先端を太くした形（_wfix、半径 0.0997 m）で表面が床の 50〜69 mm 下から始まった。表面（軸 − 半径）で測る。
#   確かめ（9-288）: _wfix の best の 20 話の再生で、設定なしは 8 話が床の下から始まるのを再現、設定ありは 20 話すべて 22 mm 以上上。
#   ⚠️ seed0 も取り直す（両 seed を同じコード・新しいワーカーの種〔Bug 76〕でそろえる）。blk3d seed0 は記録として残す
#   ⚠️ プレー中に先端が 29〜47 mm 沈む型（9-288）はこの設定では直らない。床下の step が 1 % を超えれば check_invariants.py が ⛔
#
# ⭐ 事前登録した読みは queue_phys.sh の ⑧〜⑩ を**そのまま**引き継ぐ（結果を見てから変えない）:
#   ⑧ どちらかの seed で隙間 ≥ 10 cm のパックがゴール ／ ⑨ ゴールは無いが 0.1 m 以上動く ／ ⑩ 動かない
#   判定: python3 scripts/judge_hockey_goal.py hockey_bank9_phys_g10_blk3d_fs:arm_trace_20.npz hockey_bank9_phys_g10_blk3d_fs_s2:arm_trace_20.npz
#   共通: 不変条件（check_invariants.py）・目視。跳ね返りの比（9-263 ⑥ の測り方）が 0.75 付近にそろうことも見る
#   ⚠️ 完走後の自動再生（queue_postrun_replay.sh）の一覧に無いので、完走後に手で回す（20 話の軌跡も）
# ⭐ D-5 の起動行と読み（㉑・㉒）は queue_obspen_phys.sh と同じ（そちらの冒頭が正）
#
# ⭐ 順番（研究方針「キューへの投入の優先度」: ホッケー・D-5 は核の外）: 核の seed1 の取り直し（queue_seed1_ws.sh）の
#   最後の起動（e2e_a1v_pusher_phys_s2_ws）を待つ → ホッケー 2 本 → D-5 の 2 本。✅ 投入はユーザーの確認済み（10-22「全部この案で」）
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_hockey_fs_obspen.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
PHYS="CNOID_SELF_COLLISION=1 CNOID_CULLING_DEPTH=1.0 CNOID_CTRL_CLIP=1 CNOID_HINGE_DAMP_TORQUE=1 CNOID_JOINT_MAX_VEL=3.1416 CNOID_JOINT_MAX_VEL_STRICT=1 CNOID_TICK_SYNC=1"
launch(){ local RUN=$1 EXTRA=$2; shift 2
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 15; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 $PHYS $EXTRA \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    "$@" +env_specs.init_self_contact=true \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)"; sleep 60
  # ⭐ 起動後に、設定が届いたこととワーカーの種を出す（Bug 74・76 の型）
  sleep 120
  log "$RUN: $(grep -a -m1 'ワーカーの乱数の種' single_run/$RUN/stdout.log | cut -c1-80)"
  log "$RUN: 床の測り方 $(grep -a -m1 -o '床は表面で測る[^:]*' single_run/$RUN/stdout.log /tmp/cnoid_worker_*.log 2>/dev/null | head -1 | cut -c1-80)"; }

log "=== ホッケーの取り直し 4 回目（床を表面で測る、Bug 78・9-288）→ D-5 の 2 本。HEAD=$(git rev-parse --short HEAD)。e2e_a1v_pusher_phys_s2_ws の起動を待つ"
until [ -d single_run/e2e_a1v_pusher_phys_s2_ws/log ]; do sleep 60; done
log "核の seed1 の取り直しの最後が起動した。枠が空き次第入る"
HX="HOCKEY_WALL_RESTITUTION=0.75 HOCKEY_ARM_BLOCK=3d"
for S in 0 1; do X=$([ $S = 1 ] && echo _s2)
  launch hockey_bank9_phys_g10_blk3d_fs$X "$HX" cfg=pusher_tripo_v3_reallen xml_name=e2e_hockey_easy_phys_arm171 num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +env_specs.cube_y_noise=0.25 +env_specs.arm_safe_init=true \
    +env_specs.arm_init_clear_y=0.45 +env_specs.arm_init_pitch_search=true \
    +env_specs.arm_init_after_noise=true +env_specs.init_contact_all_links=true \
    +env_specs.arm_init_box_gap=true +env_specs.arm_init_margin=0.10 \
    +env_specs.floor_by_surface=true \
    +reward_specs.use_target_reward=true +reward_specs.target_x=1.55 \
    +reward_specs.target_y=0.0 +reward_specs.contact_weight=0 \
    +reward_specs.ctrl_cost_coeff=0 +reward_specs.init_contact_penalty=1.0 +fix_base=true
done
C="num_threads=4 enable_wandb=false fix_skeleton=true +robot_param_scale=1"
FB="+fix_base=true"
REACH8="+reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false +env_specs.cube_x_offset=100"
OBS="+reward_specs.obstacle_penalty_scale=1.0"
for S in 0 1; do X=$([ $S = 1 ] && echo _s2)
  launch e2e_a1v_obspen_reach_phys$X "" cfg=pusher_tripo_v3_reallen xml_name=e2e_a1v_obs_tall2_phys_arm171 max_epoch_num=200 seed=$S $C $REACH8 +reward_specs.init_contact_penalty=1900 $FB $OBS
done
log "=== 4 本とも起動した"
