#!/bin/bash
# curriculum_transfer: curriculum_v1 で学んだ形態を引き継ぎ、難しい本番設定で再学習する
#
# 使い方（curriculum_v1 が cube を押せるようになったら実行）:
#   bash scripts/launch_curriculum_transfer.sh
#
# やること:
#   - single_run/rrbot_arm_cnoid_curriculum_v1/models/best.p から
#     形態設計ネットワーク（skel_*/attr_* の重み）だけを読み込む
#   - 制御方策ネットワークは初期化（新たにゼロから学習）
#   - cube_x_offset=0.5（本番設定）・target_x=2.0 で学習再開

cd "$(dirname "$0")/.."

nohup bash scripts/run_cnoid_train.sh \
  cfg=pusher xml_name=rrbot_arm num_threads=4 max_epoch_num=1000 enable_wandb=false fix_skeleton=true \
  +reward_specs.use_target_reward=true +reward_specs.target_x=2.0 \
  +env_specs.cube_x_offset=0.5 +env_specs.cube_x_noise=0.2 +env_specs.arm_safe_init=true \
  +reward_specs.reach_bonus_scale=2.0 +reward_specs.reach_bonus_k=3.0 \
  +restore_dir=single_run/rrbot_arm_cnoid_curriculum_v1 \
  morph_prior=true reset_epoch=true \
  hydra.run.dir=single_run/rrbot_arm_cnoid_curriculum_transfer_v1 \
  > /dev/null 2>&1 &
disown
echo "curriculum_transfer_v1 launched (pid=$!)"
echo "monitor: tail -f single_run/monitor.log"
