#!/bin/bash
# ホッケー（実在の物理）の取り直し 5 回目 — 腕の壁ブロックのカプセルの向きを軸と角度で読む（Bug 81、系譜 9-309、
# ユーザー判断 2026-10-26「バグを直す」「取り直して」）
#
# ⭐ 起動行は queue_hockey_fs_obspen.sh のホッケー（`_blk3d_fs`）と**同じ**で、変えたのは
#   HOCKEY_ARM_BLOCK=3d → 3d_axis の 1 つだけ。
#   旧い読み方は `.body` の回転が 90° なら真上向きとし、傾いた骨のカプセルを端で 4〜7 cm ずれた線分で判定していた。
#   ブロックは tick の 2〜3 割で腕を差し戻す（`_blk3d_fs` の best の 20 話で 19437 回）。同じ checkpoint を直したブロックで
#   再生するとゴール 15 → 13（9-309 追記）。学習し直したときの方策は再生からは分からないので取り直す。
#   ⚠️ `_blk3d_fs` / `_fs_s2` は旧いブロックの記録として残す（上書きしない。名前は `_ax`）
#
# ⭐ 事前登録した読みは queue_phys.sh の ⑧〜⑩ を**そのまま**引き継ぐ（結果を見てから変えない）:
#   ⑧ どちらかの seed で隙間 ≥ 10 cm のパックがゴール ／ ⑨ ゴールは無いが 0.1 m 以上動く ／ ⑩ 動かない
#   判定: python3 scripts/judge_hockey_goal.py hockey_bank9_phys_g10_blk3d_fs_ax:arm_trace_20.npz hockey_bank9_phys_g10_blk3d_fs_ax_s2:arm_trace_20.npz
#   共通: 不変条件（check_invariants.py）・目視・跳ね返りの比（scripts/hockey_bounce_ratio.py、Bug 70 型が 0 か）
#   ⚠️ 完走後の自動再生（queue_postrun_replay.sh）の一覧に無いので、完走後に手で回す（20 話の軌跡も。9-309 と同じ手順）
#
# ⭐ 順番（研究方針「キューへの投入の優先度」: ホッケーは核の外）: queue_hockey_fs_obspen.sh が D-5 の最後の 1 本
#   （e2e_a1v_obspen_reach_phys_s2）を起動するのを待ってから、枠が空き次第入る（D-5 と枠を取り合わない）。
#   修論で埋める欄: 4.7（ホッケー）と第5章のホッケーの文。核（表 4.4 の R1〜R4・軸3）ではない。
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_hockey_axis.log
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
  local PID=$!
  log "$RUN launched (PID $PID)"; sleep 180
  # ⭐ 起動後に、設定が届いたこととワーカーの種を出す（Bug 74・76 の型）。
  #   ⚠️ ワーカーのログはこの run の PID のものだけを見る（queue_hockey_fs_obspen.sh は /tmp/cnoid_worker_*.log 全部の
  #   先頭 1 件を拾い、前の run の行を出していた。9-309）
  log "$RUN: $(grep -a -m1 'ワーカーの乱数の種' single_run/$RUN/stdout.log | cut -c1-80)"
  log "$RUN: 床の測り方 $(grep -a -h -m1 -o '床は表面で測る[^:]*' /tmp/cnoid_worker_${PID}_*.log 2>/dev/null | head -1 | cut -c1-60)"
  log "$RUN: ブロック $(grep -a -h -m1 '腕のカプセル本体' /tmp/cnoid_worker_${PID}_*.log 2>/dev/null | head -1 | cut -c1-90)"; }

log "=== ホッケーの取り直し 5 回目（腕の壁ブロックの向き、Bug 81・9-309）。HEAD=$(git rev-parse --short HEAD)。D-5 の最後（e2e_a1v_obspen_reach_phys_s2）の起動を待つ"
until [ -d single_run/e2e_a1v_obspen_reach_phys_s2/log ]; do sleep 60; done
log "D-5 の最後が起動した。枠が空き次第入る"
HX="HOCKEY_WALL_RESTITUTION=0.75 HOCKEY_ARM_BLOCK=3d_axis"
for S in 0 1; do X=$([ $S = 1 ] && echo _s2)
  launch hockey_bank9_phys_g10_blk3d_fs_ax$X "$HX" cfg=pusher_tripo_v3_reallen xml_name=e2e_hockey_easy_phys_arm171 num_threads=4 max_epoch_num=200 \
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
log "=== 2 本とも起動した"
