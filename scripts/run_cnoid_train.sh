#!/bin/bash
# Choreonoid 学習起動ラッパー
#
# 使い方:
#   scripts/run_cnoid_train.sh [Hydra オーバーライド ...]
#
# hydra.run.dir=single_run/XXX を含めると、stdout/stderr が自動で
# single_run/XXX/stdout.log に保存される。
#
# 例:
#   scripts/run_cnoid_train.sh \
#     cfg=pusher xml_name=rrbot_arm \
#     num_threads=4 max_epoch_num=1000 enable_wandb=false \
#     fix_skeleton=true \
#     +reward_specs.use_target_reward=true \
#     +reward_specs.target_x=2.0 \
#     +env_specs.cube_x_offset=0.5 \
#     +env_specs.cube_x_noise=0.2 \
#     hydra.run.dir=single_run/rrbot_arm_cnoid_v5

set -euo pipefail

# hydra.run.dir=... を引数から抽出
RUN_DIR=""
for arg in "$@"; do
  case "$arg" in
    hydra.run.dir=*)
      RUN_DIR="${arg#hydra.run.dir=}"
      ;;
  esac
done

if [ -z "$RUN_DIR" ]; then
  echo "Error: hydra.run.dir=<path> を引数に含めてください" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/stdout.log"

echo "[run_cnoid_train] run_dir : $RUN_DIR"
echo "[run_cnoid_train] log     : $LOG_FILE"

USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
  /choreonoid_ws/install/bin/choreonoid --no-window \
  --python scripts/choreonoid_train.py \
  "$@" \
  2>&1 | tee "$LOG_FILE"
