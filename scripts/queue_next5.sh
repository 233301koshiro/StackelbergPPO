#!/usr/bin/env bash
# 2026-09-18: GPU を空けないためのキュー（実験系譜 9-117）。
#
#  ① e2e_a1v_obs_tall_reach_s3   柱を高くした障害物 Reach の seed=2
#     9-112 の 2 seed が出揃うので、3 本目で再現性を固める。
#     読み: ① 3 本とも迂回して到達 → ✅ 障害物回避が成立
#           ② 到達できない run が混ざる → ⚠️ 迂回は seed 依存（それも結果）
#
# ⚠️ 9-111 の njobs 不具合を修正済みの数え方を使う。
# 起動: nohup bash scripts/queue_next5.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_next5.log
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
launch e2e_a1v_obs_tall_reach_s3 \
  cfg=pusher_tripo_v3 xml_name=e2e_a1v_obs_tall num_threads=4 max_epoch_num=200 \
  enable_wandb=false fix_skeleton=true seed=2 +robot_param_scale=1 \
  +reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 \
  +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 \
  +env_specs.check_init_contact=false +reward_specs.init_contact_penalty=1900 \
  +env_specs.arm_safe_init=true
log "=== キュー終了"
