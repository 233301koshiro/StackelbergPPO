#!/usr/bin/env bash
# 2026-09-18: GPU を空けないためのキュー（実験系譜 9-111）。
#
#  ① e2e_a1_fix3_auto13_s2  ⭐ **fix 良化の seed=1**
#       9-111 で「閾値 10 % が良い助言を取り逃している」と分かったが **1 seed**。
#       ⚠️ 9-105 の教訓（1 seed で断定しない）に従い seed を足す。
#
#  ② e2e_a1v_obs_tall_reach  ⭐ **柱を高くした障害物 Reach**
#       9-110 で「腕が柱を 4 mm の余裕でまたいでいた」と判明。
#       ⭐ **またげない高さにして、本当に迂回が要るかを見る。**
#       ⚠️ 到達可能性が消えないことを第1層で先に確認してから投入する。
#
# 起動: nohup bash scripts/queue_next3.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_next3.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
# ⚠️ **args だけで絞ると自分のシェルを数える**（Bug 19 と同型。9-111 で 1 度直したが不十分だった）。
#   ⭐ `comm`（実行ファイル名）が choreonoid のものだけを数える。シェルは bash なので混入しない。
njobs(){ ps -eo comm,args 2>/dev/null | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
launch(){ local RUN=$1; shift
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py "$@" \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)"; sleep 60; }

log "=== キュー開始"

# ① fix 良化の seed=1（対照と固定版の両方）
launch e2e_a1_reach_auto13_s2 \
  cfg=pusher_gearonly xml_name=e2e_a1 num_threads=4 max_epoch_num=200 \
  enable_wandb=false fix_skeleton=true seed=1 +robot_param_scale=1 \
  +reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 \
  +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 \
  +env_specs.check_init_contact=false

launch e2e_a1_fix3_auto13_s2 \
  cfg=pusher_gearonly xml_name=e2e_a1_fix3 num_threads=4 max_epoch_num=200 \
  enable_wandb=false fix_skeleton=true seed=1 +robot_param_scale=1 \
  +reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 \
  +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 \
  +env_specs.check_init_contact=false

log "=== キュー終了"
