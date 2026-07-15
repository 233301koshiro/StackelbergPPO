#!/usr/bin/env bash
# M 系 ablation の追加 seed（seed=1）を起動する。
# 前提: M1/M2b の seed=0 は 2026-07-15 に起動済み（rrbot_arm_pusher_M1_lenonly / M2b_gearonly）。
# 本スクリプトは seed=1 版（M1_lenonly_s2 / M2b_gearonly_s2）を、メモリが空き次第起動する。
#
# 旧版（2026-07-15 初版）は L2_launcher.sh を流用しており実行すると別実験（L2 設定）が
# 起動する欠陥があったため全面書き直し（同日レビュー指摘）。Hydra への seed 指定は
# overrides ファイルの手書きでは効かず、コマンドライン引数 `seed=1` で渡す必要がある。
set -u
cd /userdir/StackelbergPPO
LOG=single_run/m_ablation_s2_launcher.log
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# 共通フラグ（M1/M2b の seed=0 と完全一致させること。差分は seed のみ）
COMMON="xml_name=rrbot_arm num_threads=4 max_epoch_num=200 enable_wandb=false \
fix_skeleton=true seed=1 +robot_param_scale=1 +reward_specs.ctrl_cost_coeff=0.2 \
+reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50"

launch() {  # $1=cfg $2=run名 $3=追加フラグ（空可）
  mkdir -p "single_run/$2"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=$1 $COMMON $3 \
    hydra.run.dir="single_run/$2" \
    > "single_run/$2/stdout.log" 2>&1 &
  say "$2 launched (PID $!)"
}

say "launcher start: waiting for memory"
# メモリ 14GB 以上が空くまで待つ（最大12時間）
for i in $(seq 1 72); do
  avail_gb=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
  [ "$avail_gb" -ge 14 ] && break
  say "waiting for memory (available=${avail_gb}GB < 14GB) [$i/72]"
  sleep 600
done
avail_gb=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
if [ "$avail_gb" -lt 14 ]; then
  say "ABORT: memory still ${avail_gb}GB after 12h"
  exit 1
fi

launch pusher_lenonly  rrbot_arm_pusher_M1_lenonly_s2  ""
sleep 120   # 起動ピークをずらす
launch pusher_gearonly rrbot_arm_pusher_M2b_gearonly_s2 "+env_specs.arm_safe_init=true"

# 起動確認（10分後に log が生まれているか）
sleep 600
for run in rrbot_arm_pusher_M1_lenonly_s2 rrbot_arm_pusher_M2b_gearonly_s2; do
  if [ -s "single_run/$run/log/log_train.txt" ] || grep -q "epoch" "single_run/$run/stdout.log" 2>/dev/null; then
    say "$run startup confirmed"
  else
    say "WARNING: $run may have failed to start. check stdout.log"
  fi
done
say "done"
