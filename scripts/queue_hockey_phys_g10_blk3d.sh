#!/bin/bash
# ホッケー（実在の物理）の取り直し 3 回目 — 腕の壁ブロックを 3 次元にした（Bug 71、系譜 9-263 ⑧⑨、ユーザー判断 2026-10-19「取り直しで」）
#
# ⭐ 起動行は queue_hockey_phys_g10_wfix.sh と**同じ**で、変えたのは環境変数 1 つ（HOCKEY_ARM_BLOCK=1 → 3d）だけ:
#   腕の壁ブロックが壁と板を xy の柱として持ち、板や壁の 18〜30 cm 上を通る腕を凍らせていた（Bug 71。seed1 の動かない 16 話のうち 14 話）。
#   3d は箱の上面までの高さを見る（上を越えるのは許す。下端は床まで＝宙に浮いた壁の下をくぐるのは塞ぐ）。
#   確かめ（9-263 ⑨）: _g10_wfix_s2 の再生で凍った話 14 → 0、台本で壁の下へ突っ込ませると 108 回発火して止まる（ブロックなしはくぐる）
# ⭐ 投入時のコミットを下の HEAD に書く（run はブランチ・コミットに紐づかない）
#
# ⭐ 事前登録した読みは queue_phys.sh の ⑧〜⑩ を**そのまま**引き継ぐ（結果を見てから変えない）:
#   ⑧ どちらかの seed で隙間 ≥ 10 cm のパックがゴール ／ ⑨ ゴールは無いが 0.1 m 以上動く ／ ⑩ 動かない
#   判定: python3 scripts/judge_hockey_goal.py hockey_bank9_phys_g10_blk3d:arm_trace_20.npz hockey_bank9_phys_g10_blk3d_s2:arm_trace_20.npz
#   共通: 不変条件（check_invariants.py）・目視。⭐ 跳ね返りの比（9-263 ⑥ の測り方）が 0.75 付近にそろうことも見る
# ⚠️ 完走後の自動再生（queue_postrun_replay.sh）の一覧に無いので、完走後に手で回す（9-263 ⑦ と同じ手順。凍った話の数も数える）
# ⚠️ queue_phys.sh と同時に回すと、2 本の上限を両方が見て空いた枠を取り合う
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_hockey_phys_g10_blk3d.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
PHYS="CNOID_SELF_COLLISION=1 CNOID_CULLING_DEPTH=1.0 CNOID_CTRL_CLIP=1 CNOID_HINGE_DAMP_TORQUE=1 CNOID_JOINT_MAX_VEL=3.1416 CNOID_JOINT_MAX_VEL_STRICT=1 CNOID_TICK_SYNC=1"
EXTRA="HOCKEY_WALL_RESTITUTION=0.75 HOCKEY_ARM_BLOCK=3d"
launch(){ local RUN=$1; shift
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 $PHYS $EXTRA \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    "$@" +env_specs.init_self_contact=true \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)"; sleep 60; }
log "=== ホッケーの取り直し 3 回目（腕の壁ブロックを 3 次元に、Bug 71・9-263 ⑨）HEAD=$(git rev-parse --short HEAD)"
for S in 0 1; do X=$([ $S = 1 ] && echo _s2)
  launch hockey_bank9_phys_g10_blk3d$X cfg=pusher_tripo_v3_reallen xml_name=e2e_hockey_easy_phys_arm171 num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +env_specs.cube_y_noise=0.25 +env_specs.arm_safe_init=true \
    +env_specs.arm_init_clear_y=0.45 +env_specs.arm_init_pitch_search=true \
    +env_specs.arm_init_after_noise=true +env_specs.init_contact_all_links=true \
    +env_specs.arm_init_box_gap=true +env_specs.arm_init_margin=0.10 \
    +reward_specs.use_target_reward=true +reward_specs.target_x=1.55 \
    +reward_specs.target_y=0.0 +reward_specs.contact_weight=0 \
    +reward_specs.ctrl_cost_coeff=0 +reward_specs.init_contact_penalty=1.0 +fix_base=true
done
log "=== キュー終了"
