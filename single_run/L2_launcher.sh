#!/usr/bin/env bash
# K1/K2 完走を待って L2（Pusher, scratch, ctrl_cost_coeff=0.2）を起動する。
# 第4世代（L 系）: 比較設計是正版。L1（Reach, target 0.8）と対で
# 「ctrl コスト論文水準・同条件・タスク差のみ」の Phase 2 本比較を構成する。
# 背景: docs/研究応用/方針レビュー_2026-07-08.md 懸念1・懸念2（2026-07-09 決定）
set -u
cd /userdir/StackelbergPPO

echo "[$(date)] waiting for K1/K2 to finish..."
while pgrep -f "hydra.run.dir=single_run/rrbot_arm_pusher_K" > /dev/null; do
  sleep 300
done
echo "[$(date)] K1/K2 finished. launching L2..."

mkdir -p single_run/rrbot_arm_pusher_L2
nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
  --no-window --python scripts/choreonoid_train.py \
  cfg=pusher xml_name=rrbot_arm num_threads=4 max_epoch_num=1000 \
  enable_wandb=false fix_skeleton=true +robot_param_scale=1 \
  +reward_specs.ctrl_cost_coeff=0.2 \
  hydra.run.dir=single_run/rrbot_arm_pusher_L2 \
  > single_run/rrbot_arm_pusher_L2/stdout.log 2>&1 &
echo "[$(date)] L2 launched (PID $!)"
