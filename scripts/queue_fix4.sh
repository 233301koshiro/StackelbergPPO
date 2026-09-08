#!/usr/bin/env bash
# 2026-09-08: 縦型で第3層の「助言が出ない」判断が正しいかを検証する（9-32）。
# 9-27 は平面 A1 で「助言に従うと良化・助言が出ていない側は悪化」を示したが、
# 縦型 A1v は Reach/Pusher とも全関節が閾値超えで**助言が出ない**。
# 最低は Reach の関節4 で 10%（UNUSED_FRAC=0.10 のすぐ上、実値 0.100〜0.105）。
# **境界のすぐ上で助言を出さなかった判断が正しいか**を、固定して悪化するかで確かめる。
# 予測: 悪化する（＝助言を出さなかったのは正しい）。良化したら閾値が緩すぎる。
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_fix4.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
done_(){ grep -q "training done!" "single_run/$1/log/log_train.txt" 2>/dev/null; }
log "キュー開始: 前段（e2e_a2v_reach_s2 / e2e_b2v_reach_s2）の完走を待つ"
while ! { done_ e2e_a2v_reach_s2 && done_ e2e_b2v_reach_s2; }; do sleep 120; done
log "前段が完走。fix4 を投入する"
for t in reach pusher; do
  run="e2e_a1v_fix4_${t}"
  [ -s "single_run/$run/log/log_train.txt" ] && { log "$run は既存。スキップ"; continue; }
  mkdir -p "single_run/$run"
  if [ "$t" = reach ]; then
    EX="+reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false +reward_specs.init_contact_penalty=1900"
  else
    EX="+reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true"
  fi
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_tripo_v3 xml_name=e2e_a1v_fix4 num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=0 +robot_param_scale=1 $EX \
    hydra.run.dir="single_run/$run" > "single_run/$run/stdout.log" 2>&1 &
  log "$run launched (PID $!)"
  sleep 60
done
log "投入完了"
