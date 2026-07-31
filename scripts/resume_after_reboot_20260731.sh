#!/usr/bin/env bash
# 2026-07-31: 7/30リブートで中断した4本の学習を最新チェックポイントから再開する。
# reset_epoch は指定しない(=epochカウンタは継続。max_epoch_numまで残りを消化する)。
set -u
cd /userdir/StackelbergPPO
LOG=single_run/resume_after_reboot_launcher.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

REACH="+reward_specs.use_reach=true +reward_specs.target_x=0.8 \
+reward_specs.target_y=0.0 +reward_specs.target_z=0.15 \
+reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false"

resume() {  # $1=xml_name $2=run名 $3=extra_overrides $4=max_epoch_num $5=epoch $6=seed
  local xml=$1 name=$2 extra=$3 maxep=$4 ep=$5 seed=$6
  say "resume $name from epoch=$ep (max_epoch_num=$maxep)"
  nohup bash scripts/run_cnoid_train.sh \
    cfg=pusher_gearonly xml_name=$xml num_threads=4 max_epoch_num=$maxep \
    enable_wandb=false fix_skeleton=true seed=$seed +robot_param_scale=1 \
    $extra \
    +restore_dir=single_run/$name epoch=$ep \
    hydra.run.dir=single_run/$name \
    > /dev/null 2>&1 &
  say "$name relaunched (PID $!)"
}

resume tripo_arm_v2c_pj_short tripo_pj_short "$REACH" 200 20 0
sleep 60
resume tripo_arm_v2c_pj_long  tripo_pj_long  "$REACH" 200 20 0
sleep 60

M2B="+reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 \
+reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true"

resume rrbot_arm rrbot_arm_pusher_M2b_gearonly_1000    "$M2B" 1000 390 0
sleep 60
resume rrbot_arm rrbot_arm_pusher_M2b_gearonly_1000_s2 "$M2B" 1000 370 1

say "all 4 resume launches issued"
