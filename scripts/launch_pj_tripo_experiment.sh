#!/usr/bin/env bash
# tripo系でのPJ実験（前向き判定テスト、良い/悪いスケッチの判別）
#
# 目的: 4.3.3.1のPJ実験（rrbotの短腕/長腕、Reach）と同じロジックを、
# パイプライン生成形態（tripo_arm_v2c）でも再現し、「同一タスクで良い/悪い
# スケッチ形態を判別できるか」という判定器としての妥当性を確認する。
# 第6章6.5「今後の課題」項目2・想定問答.md Q3で明記した、研究の最終目標に
# 対して最も本質的に残るギャップへの対応。
#
# 設計（2026-07-29、既存PJ実験と同じ「XMLでリンク長を凍結・gear/太さを最適化」方式）:
#   tripo_arm_v2c_pj_short.xml: tripo_arm_v2cの0.5倍スケール（総リーチ 0.403m）
#   tripo_arm_v2c_pj_long.xml:  tripo_arm_v2cの1.8倍スケール（総リーチ 1.451m）
#   タスク: Reach（target_x=0.8、既存のtripo_arm_v2c_reach・rrbot L1/PJと同一設定）
#   → short は物理的に届かない（0.403 << 0.8）、long は余裕で届く（1.451 >> 0.8）
#   cfg=pusher_gearonly（body_params: {} で bone_offset のみ凍結（gear・size・ext_start は可変）。
#   xml_nameを変えるだけで再利用できる。PJ_short/PJ_longと同じ機構）
#
# 期待スコア（exec_R_eps、200ep）:
#   PJ_tripo_short: 非常に低い（0.403mでは0.8mに届かない）
#   PJ_tripo_long:  高い（tripo_arm_v2c_reachのbest -8.27〜rrbot PJ_longの-4.9程度）
#
# 実行前の注意（2026-07-29時点）:
#   現在GPU/CPUスロットは3本（tripo_arm_v2b_reach・M1_lenonly_1000×2）で埋まっている。
#   スロットが空くまでは起動しないこと（CLAUDE.md: 複数学習の同時起動はETA確認必須）。
#   auto_launch_queue2.sh の完走待ちキューに追加するか、3本完走を確認してから
#   このスクリプトを手動起動する。
set -u
cd /userdir/StackelbergPPO
LOG=single_run/pj_tripo_launcher.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Reach タスク共通オーバーライド（tripo_arm_v2c_reach・rrbot L1/PJ と同一設定）
REACH="+reward_specs.use_reach=true +reward_specs.target_x=0.8 \
+reward_specs.target_y=0.0 +reward_specs.target_z=0.15 \
+reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false"

# 共通フラグ（PJ実験と同じ200ep・1seed。判定は較正済みの通りep100前後で足りる想定）
COMMON="num_threads=4 max_epoch_num=200 enable_wandb=false \
fix_skeleton=true seed=0 +robot_param_scale=1"

launch() {  # $1=xml_name $2=run名
  mkdir -p "single_run/$2"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_gearonly xml_name=$1 $COMMON $REACH \
    hydra.run.dir="single_run/$2" \
    > "single_run/$2/stdout.log" 2>&1 &
  say "$2 launched (PID $!)"
}

avail_gb=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
say "available memory: ${avail_gb}GB"
if [ "$avail_gb" -lt 10 ]; then
  say "WARNING: low memory (${avail_gb}GB < 10GB). Launch with care."
fi

say "--- PJ_tripo_short: tripo_arm_v2c 0.5x scale (total reach 0.403m, frozen) + Reach task ---"
launch tripo_arm_v2c_pj_short tripo_pj_short

sleep 120  # 起動ピークをずらす

say "--- PJ_tripo_long: tripo_arm_v2c 1.8x scale (total reach 1.451m, frozen) + Reach task ---"
launch tripo_arm_v2c_pj_long tripo_pj_long

sleep 600
for run in tripo_pj_short tripo_pj_long; do
  if [ -s "single_run/$run/log/log_train.txt" ]; then
    ep=$(tail -1 "single_run/$run/log/log_train.txt" | awk '{print $1}')
    say "$run startup confirmed (ep=$ep)"
  else
    say "WARNING: $run may have failed. check single_run/$run/stdout.log"
  fi
done
say "PJ tripo launcher done"
