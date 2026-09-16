#!/usr/bin/env bash
# 2026-09-16: GPU を空けないためのキュー（実験系譜 9-104）。
#
# ⭐ 上から順に、**2 枠空くたびに 1 本ずつ**投入する。
#
#  ① e2e_a1v_obs_reach    障害物のある Reach（指摘17・9-104）。**未着手だった案件**
#                          ⚠️ 障害物は <body> に入れてある（9-98 の是正）。腕とも当たる
#                          読み: 目標へ届くか。届かないなら第1層が「無理」と言えるかを見る
#  ② e2e_a1v_reach_pilot13_s2   指摘13 の seed=1（9-102 は 1 seed）
#                          読み: 関節4 の使用率がまた 10 % 未満になり、固定版が
#                                また悪化するか。⭐ **再現すれば「使用率は固定可否を
#                                含意しない」が 2 seed で言える**
#  ③ e2e_a1v_obs_reach_s2  ①の seed=1（①が回った後に判断）
#
# 起動: nohup bash scripts/queue_next.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_next.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
njobs(){ ps -eo args 2>/dev/null | grep -cE "[c]horeonoid_train\.py" || true; }
wait_slot(){ while [ "$(njobs)" -ge 2 ]; do sleep 180; done; }

launch(){  # $1=run名  $2...=overrides
  local RUN=$1; shift
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  wait_slot
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py "$@" \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)"
  sleep 60          # 起動が重なると落ちるので間隔を空ける
}

log "=== キュー開始"

# ① 障害物のある Reach
launch e2e_a1v_obs_reach \
  cfg=pusher_tripo_v3 xml_name=e2e_a1v_obs num_threads=4 max_epoch_num=200 \
  enable_wandb=false fix_skeleton=true seed=0 +robot_param_scale=1 \
  +reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 \
  +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 \
  +env_specs.check_init_contact=false +reward_specs.init_contact_penalty=1900 \
  +env_specs.arm_safe_init=true

# ② 指摘13 の seed=1
launch e2e_a1v_reach_pilot13_s2 \
  cfg=pusher_tripo_v3 xml_name=e2e_a1v num_threads=4 max_epoch_num=200 \
  enable_wandb=false fix_skeleton=true seed=1 +robot_param_scale=1 \
  +reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 \
  +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 \
  +env_specs.check_init_contact=false +reward_specs.init_contact_penalty=1900

# ③ 障害物 Reach の seed=1
launch e2e_a1v_obs_reach_s2 \
  cfg=pusher_tripo_v3 xml_name=e2e_a1v_obs num_threads=4 max_epoch_num=200 \
  enable_wandb=false fix_skeleton=true seed=1 +robot_param_scale=1 \
  +reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 \
  +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 \
  +env_specs.check_init_contact=false +reward_specs.init_contact_penalty=1900 \
  +env_specs.arm_safe_init=true

# ④ ⭐ **fix で「良くなる」パターンを自動化で拾う**（9-106）
#    9-102 は縦型 A1v で「固定すると悪化」を自動で捕まえた（助言の棄却）。
#    一方 9-27/9-61 の**平面 A1 Reach では固定が良化する**（現行 XML の版で −5.83 → −3.85）。
#    ⚠️ **ただし良化の方は人手で回したもので、自動化が拾った実績ではない。**
#    ⭐ **同じ自動化に平面 A1 を通し、今度は「採用」が出るかを見る。**
#    これが出れば「fix は必ずしも良くならないが、**並列実行のおかげで良くなる場合を
#    捨てずに拾える**」と言える。⚠️ 出なければその主張はしない。
#
#    ⚠️ **`e2e_a1` は 9-105 で版の混入が判明している。**現行版（総リーチ 1.010 m）で統一する。
launch e2e_a1_reach_auto13 \
  cfg=pusher_gearonly xml_name=e2e_a1 num_threads=4 max_epoch_num=200 \
  enable_wandb=false fix_skeleton=true seed=0 +robot_param_scale=1 \
  +reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 \
  +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 \
  +env_specs.check_init_contact=false

# ⭐ 監視デーモンを付ける（これが本体。検出 → 固定版の自動起動 → 自動判定）
sleep 30
nohup python3 scripts/joint_fix_watch.py --run e2e_a1_reach_auto13 \
  > single_run/e2e_a1_reach_auto13/joint_fix_watch_stdout.log 2>&1 &
log "e2e_a1_reach_auto13 に監視デーモンを付けた (PID $!)"

log "=== キュー終了（3 本とも投入済み）"
