#!/usr/bin/env bash
# 前向き判定テスト（PJ実験）の起動スクリプト
#
# 目的:「入力スケッチが異なると co-design スコアも正しく変わるか」の実証
# 指導教員指摘2（2026-07-16）への対応。
#
# 2×2 設計:
#   アーム長:   短腕（0.55m、XMLデフォルト凍結）vs 長腕（~1.0m、tight bounds）
#   タスク:     Pusher vs Reach
#
# Pusher の結果は既存データで代替（短腕=M2b_gearonly:33.2 / 長腕≈L2:102.2）
# → Reach 2本のみ新規起動（PJ_Reach_short / PJ_Reach_long）
#
# 期待スコア（exec_R_eps ＠ 200ep）:
#   PJ_Reach_short: 非常に低い（0.55m腕では0.8m目標に物理的に届かない）
#   PJ_Reach_long:  高い（〜-0.014 程度、L1 と同水準）
set -u
cd /userdir/StackelbergPPO
LOG=single_run/pj_launcher.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Reach タスク共通オーバーライド（L1 と完全一致）
REACH="+reward_specs.use_reach=true +reward_specs.target_x=0.8 \
+reward_specs.target_y=0.0 +reward_specs.target_z=0.15 \
+reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false"

# 共通フラグ
COMMON="xml_name=rrbot_arm num_threads=4 max_epoch_num=200 enable_wandb=false \
fix_skeleton=true seed=0 +robot_param_scale=1"

launch() {  # $1=cfg $2=run名 $3=追加フラグ
  mkdir -p "single_run/$2"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=$1 $COMMON $3 \
    hydra.run.dir="single_run/$2" \
    > "single_run/$2/stdout.log" 2>&1 &
  say "$2 launched (PID $!)"
}

# メモリ確認（M1/M2b_s2 走行中なので 10GB 確保で十分）
avail_gb=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
say "available memory: ${avail_gb}GB"
if [ "$avail_gb" -lt 10 ]; then
  say "WARNING: low memory (${avail_gb}GB < 10GB). Launch with care."
fi

# --- 実験1: PJ_Reach_short ---
# 短腕（XMLデフォルト凍結 ≈0.55m）＋ Reach タスク
# 物理的にゴール(0.8m)に届かない → スコアが低いことを確認
say "--- PJ_Reach_short: short arm (frozen at XML default ~0.55m) + Reach task ---"
launch pusher_gearonly rrbot_arm_reach_PJ_short "$REACH"

sleep 120  # 起動ピークをずらす

# --- 実験2: PJ_Reach_long ---
# 長腕（tight bounds ≈1.0m）＋ Reach タスク
# ゴール(0.8m)に届く → L1 と同水準のスコアを確認
say "--- PJ_Reach_long: long arm (constrained ~1.0m) + Reach task ---"
launch pusher_pj_long rrbot_arm_reach_PJ_long "$REACH"

# 起動確認（10分後）
sleep 600
for run in rrbot_arm_reach_PJ_short rrbot_arm_reach_PJ_long; do
  if [ -s "single_run/$run/log/log_train.txt" ] || grep -q "epoch\|train_R" "single_run/$run/stdout.log" 2>/dev/null; then
    ep=$(tail -1 "single_run/$run/log/log_train.txt" 2>/dev/null | awk '{print $1}')
    say "$run startup confirmed (ep=$ep)"
  else
    say "WARNING: $run may have failed. check single_run/$run/stdout.log"
  fi
done
say "PJ launcher done"
