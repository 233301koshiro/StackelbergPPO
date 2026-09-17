#!/usr/bin/env bash
# 2026-09-17: GPU を空けないためのキュー（実験系譜 9-109）。
#
#  ① e2e_a1_fix3_auto13   ⭐ **fix が良くなる側を「手で」確かめる**
#       9-106 の自動化は**検出しなかった**（関節3 が 12 % で閾値 10 % を割らず）。
#       ⚠️ だが 9-27/9-61 では同じ形態で固定が良化している（-5.83 → -3.85）。
#       ⭐ **自動化が拾えなくても良化するのか**を、同条件の固定版を回して確かめる。
#       読み: 良化すれば「**閾値 10 % は良い助言を取り逃している**」＝ 9-102 の逆向きの限界。
#             悪化すれば「**検出しなかったのは正しかった**」＝ 閾値の妥当性の傍証。
#       ⚠️ どちらでも結論が出る。
#
#  ② e2e_a1v_obs_reach_fix  障害物 Reach で関節固定が効くか（9-104 の続き）
#       ⚠️ ①の結果を見てから判断するので、**いまは積まない**。
#
# 起動: nohup bash scripts/queue_next2.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_next2.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
njobs(){ ps -eo args 2>/dev/null | grep -cE "[c]horeonoid_train\.py" || true; }
launch(){ local RUN=$1; shift
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py "$@" \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)"; sleep 60; }

log "=== キュー開始"

# ① 現行版 e2e_a1 の関節3 を固定（auto13 と同条件・同 seed）
launch e2e_a1_fix3_auto13 \
  cfg=pusher_gearonly xml_name=e2e_a1_fix3 num_threads=4 max_epoch_num=200 \
  enable_wandb=false fix_skeleton=true seed=0 +robot_param_scale=1 \
  +reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 \
  +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 \
  +env_specs.check_init_contact=false

# ② 障害物 Reach の関節固定（使用率が低い関節があれば固定して確かめる）
launch e2e_a1v_obs_reach_s3 \
  cfg=pusher_tripo_v3 xml_name=e2e_a1v_obs num_threads=4 max_epoch_num=200 \
  enable_wandb=false fix_skeleton=true seed=2 +robot_param_scale=1 \
  +reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 \
  +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 \
  +env_specs.check_init_contact=false +reward_specs.init_contact_penalty=1900 \
  +env_specs.arm_safe_init=true

log "=== キュー終了"
