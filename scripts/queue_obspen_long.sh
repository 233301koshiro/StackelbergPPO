#!/bin/bash
# 障害物 Reach（D-5）の追加の実験 — 柱を越えて届く形（長い腕）から共設計を始めて、届くか（系譜 9-313。ユーザー判断 2026-10-28「Cで。」「共設計でいいよ」）
#
# ⭐ なぜ: 取り直し `_sub5` / `_sub5_s2` は 2 seed とも全話で届かなかったが、初期の形でも学習した形でも、柱と床を避けて
#   目標に届く姿勢はもともと無かった（9-312。「避けて届く姿勢は幾何的にある」は判定器の欠陥 Bug 83）。
#   直した判定器は初期の形に「見つからない」と答え、学習も届かない。届く形で学習が届くかを確かめないと、
#   D-5 は判定器の一致・不一致のどちらの例にもならない。
# ⭐ 形: `e2e_a1v_obs_tall2_phys_arm171_long`（11・111・1111 の長さを 0.4031/0.3308/0.2763 → 0.802/0.475/0.800 m。
#   半径・ギア 100・可動域 ±90°・柱・目標は同じ）。設計の範囲（9-150 の rel_frac）でばらまいた 1500 個のうち、柱と床を避けて
#   届いた 59 個の 1 つ（9-312）。一様に伸ばすだけでは届く形にならない（1.0〜3.0 倍のどれも 0 個）。
#   直した判定器: 「避けて目標に届く姿勢があります（57 個のうち 57 個、最大の余裕 61 mm）」。
#   自重による肩のトルクは水平に伸ばして約 37 N·m（ギア 100 の内側）。
# ⭐ 共設計のまま（`+robot_param_scale=1`）。⭐ `_sub5` / `_sub5_s2` との起動行の違いは**初期の形（XML）だけ**。
#   ユーザー判断（10-28）: 設計の固定も考えたが、共設計にする。「形のせいか制御のせいか」の切り分けは先行研究からの
#   ずっとのテーマで、この実験で決着させるものではない。共設計の系として、届く形から始めたら届くかを見る。
#   ⚠️ 設計の範囲（9-150 の rel_frac）は初期の長さに比例するので、範囲も長い腕に合わせて広がる（tripo_pjr199 と同じ扱い）。
#
# ⭐ 事前登録した読み（結果を見てから変えない）。判定は best の再生 5 話（check_before_conclusion.py・check_obstacle_clearance.py）:
#   ㉑ 柱が効く（前提）: 全話・全 step でリンクの中心線が柱に入らない、不変条件（「有限」を含む）と「話の長さ」が ✅
#   ㉓ 到達（閾値は check_before_conclusion の 10 mm）:
#      ㉓-a どちらかの seed で 5 話中 3 話以上が届く（㉑ を満たしたまま）→ 届く形から始めれば、共設計は届く。
#            初期の形だけが違う `_sub5`（届かない、9-312）と並べて「始める形で結果が分かれた」と書く
#      ㉓-b 2 seed とも届かない（㉑ を満たしたまま）→ 届く形から始めても、共設計は届かない
#      ㉓-c 届くが ㉑ を満たさない → 通り抜けで届いた。㉑ に戻る
#   あわせて必ず出す（読みの補助。結果で読みを変えない）: 学習後の形（best の再生の骨）に、直した判定器の柱の目撃が
#   「ある」と言うか。㉓-b で「ない」なら、形が届く範囲の外へ動いた（共設計が形を縮めた）、「ある」なら形は届くのに制御が届かない。
#   seed で割れたら割れたと書く。
#   予想（確かめていない）: ㉓-b 寄り。余裕 61 mm の細い姿勢の族で、柱を越えて先端を真下へ下ろす必要があり、
#   `_sub5` では共設計が肩のリンクを最短まで縮めた。
#   判定: python3 scripts/check_before_conclusion.py e2e_a1v_obspen_reach_phys_long ・ check_obstacle_clearance.py
#   ⚠️ 完走後の自動再生（queue_postrun_replay.sh）の一覧に無いので、完走後に手で再生する
#
# ⭐ 順番（研究方針「キューへの投入の優先度」: D-5 は核の外）: いま走っている学習は無い（10-28）
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_obspen_long.log
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
  # ⭐ 起動後に、設定が届いたこと（種・刻み・robot_param_scale）を出す（Bug 74・76 の型）。ワーカーのログはこの run の PID のものだけを見る
  log "$RUN: $(grep -a -m1 'ワーカーの乱数の種' single_run/$RUN/stdout.log | cut -c1-80)"
  log "$RUN: 刻み $(grep -a -h -m1 -o '物理の刻み[^（]*' single_run/$RUN/stdout.log /tmp/cnoid_worker_${PID}_*.log 2>/dev/null | head -1 | cut -c1-60)"
  log "$RUN: launch_env.json の刻み $(grep -o '"CNOID_SUBSTEPS": "[0-9]*"' single_run/$RUN/launch_env.json)"
  log "$RUN: $(grep -a -h -m1 -o 'robot_param_scale = [^（]*' single_run/$RUN/log/log_train.txt single_run/$RUN/stdout.log 2>/dev/null | head -1)"; }

log "=== 障害物 Reach（D-5）の追加の実験: 柱を越えて届く長い腕から共設計（9-313）。HEAD=$(git rev-parse --short HEAD)"
C="num_threads=4 enable_wandb=false fix_skeleton=true +robot_param_scale=1"
REACH8="+reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false +env_specs.cube_x_offset=100"
OBS="+reward_specs.obstacle_penalty_scale=1.0"
for S in 0 1; do X=$([ $S = 1 ] && echo _s2)
  launch e2e_a1v_obspen_reach_phys_long$X "CNOID_SUBSTEPS=5" cfg=pusher_tripo_v3_reallen xml_name=e2e_a1v_obs_tall2_phys_arm171_long max_epoch_num=200 seed=$S $C $REACH8 +reward_specs.init_contact_penalty=1900 +fix_base=true $OBS
done
log "=== 2 本とも起動した"
