#!/bin/bash
# 障害物 Reach（D-5）の取り直し — 物理の刻みを 0.01 s → 0.002 s（CNOID_SUBSTEPS=5）にして、柱に当たった後の発散を止める
# （Bug 82、系譜 9-310。ユーザー判断 2026-10-27「bug は致命的なものは直していこう」）
#
# ⭐ 起動行は queue_hockey_fs_obspen.sh の D-5（`e2e_a1v_obspen_reach_phys` / `_s2`）と**同じ**で、足したのは CNOID_SUBSTEPS=5 の 1 つだけ。
#   旧い 2 本は、太い末端リンク（半径 100 mm）が柱の側面に 20〜80 mm 食い込んだ 1〜2 step 後に状態が NaN になり、罰なしで話が切れた。
#   到達の報酬は毎 step 負なので、話の長さが約 1000 → 30〜70 step に縮んだ（9-310）。
#   投入前の確かめ（10-27、seed 0 の best を再生）: 物理のつまみ 7 つを 1 つずつ外しても NaN は残り、刻みを 5・10 分割すると
#   5 話とも 1000 step まで有限、柱への食い込みは最深 3 mm（5 分割）。柱なしの `e2e_a1v_reach_phys` の best は 5 分割でも
#   到達の距離（0.1〜0.5 mm）と 10 mm に入る step（32〜35）が変わらない。再生の時間は約 2.5 倍。
#   ⚠️ 比べる相手（`e2e_a1v_reach_phys` / `_s2`）とは、柱の XML・obstacle_penalty_scale・刻みの 3 つが違う。刻みは接触の無い到達では
#   結果を変えないことを上で確かめた（柱の効果と罰の効果は、もともと分けない）
#   ⚠️ 旧い 2 本（`e2e_a1v_obspen_reach_phys` / `_s2`。`_s2` は ep165 で止めた）は記録として残す（上書きしない。名前は `_sub5`）
#
# ⭐ 事前登録した読みは queue_obspen_phys.sh の ㉑・㉒ を**そのまま**引き継ぐ（結果を見てから変えない）。判定は best の再生 5 話:
#   ㉑ 柱が効く（前提）: 全話・全 step で中心線の柱への食い込みが半幅 70 mm 未満、かつ不変条件が通る（「有限」を含む）。
#      ⭐ あわせて check_before_conclusion.py の「話の長さ」（Bug 82 で足した）が ✅
#   ㉒-a 2 seed とも全話で届かない → 「両立する姿勢は幾何的にあるのに、学習が届かない」（9-161）を今の物理の結果として書く
#   ㉒-b どちらかの seed で 5 話中 3 話以上が届く（㉑ を満たしたまま）→ 「避けて届く」が出た。seed で割れたら割れたと書く
#   ㉒-c 届くが ㉑ を満たさない → 通り抜けで届いた。㉑ に戻る
#   判定: python3 scripts/check_before_conclusion.py e2e_a1v_obspen_reach_phys_sub5 ・ check_obstacle_clearance.py
#   ⚠️ 完走後の自動再生（queue_postrun_replay.sh）の一覧に無いので、完走後に手で再生する
#
# ⭐ 順番（研究方針「キューへの投入の優先度」: D-5 は核の外）: 枠が空き次第（ホッケーの `_ax` / `_ax_s2` の後）
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_obspen_sub5.log
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
  # ⭐ 起動後に、設定が届いたこととワーカーの種を出す（Bug 74・76 の型）。ワーカーのログはこの run の PID のものだけを見る
  log "$RUN: $(grep -a -m1 'ワーカーの乱数の種' single_run/$RUN/stdout.log | cut -c1-80)"
  log "$RUN: 刻み $(grep -a -h -m1 -o '物理の刻み[^（]*' single_run/$RUN/stdout.log /tmp/cnoid_worker_${PID}_*.log 2>/dev/null | head -1 | cut -c1-60)"
  # ⭐ 再生が同じ刻みを使えるように、launch_env.json に CNOID_SUBSTEPS が残ったか（10-27 に記録の一覧へ足した）
  log "$RUN: launch_env.json の刻み $(grep -o '"CNOID_SUBSTEPS": "[0-9]*"' single_run/$RUN/launch_env.json)"; }

log "=== 障害物 Reach（D-5）の取り直し（物理の刻み 0.002 s、Bug 82・9-310）。HEAD=$(git rev-parse --short HEAD)。枠が空き次第入る"
C="num_threads=4 enable_wandb=false fix_skeleton=true +robot_param_scale=1"
FB="+fix_base=true"
REACH8="+reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false +env_specs.cube_x_offset=100"
OBS="+reward_specs.obstacle_penalty_scale=1.0"
for S in 0 1; do X=$([ $S = 1 ] && echo _s2)
  launch e2e_a1v_obspen_reach_phys_sub5$X "CNOID_SUBSTEPS=5" cfg=pusher_tripo_v3_reallen xml_name=e2e_a1v_obs_tall2_phys_arm171 max_epoch_num=200 seed=$S $C $REACH8 +reward_specs.init_contact_penalty=1900 $FB $OBS
done
log "=== 2 本とも起動した"
