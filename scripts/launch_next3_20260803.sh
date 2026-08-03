#!/usr/bin/env bash
# 2026-08-03: PJ実験・M系が全完走してスロットが空いたため、次の3本を起動する。
# 3並列を超えないこと（実験系譜.md の運用ルール）。
#
#  ① tripo_arm_v3_pusher : 縦型4関節（非平面）の Pusher 本走。第6章6.5-1「一般性の拡張」
#  ② tripo_pj_long_s2    : PJ実験(4.4.4)の seed=1。1シードのみという限界を埋める
#  ③ tripo_pj_mid        : 1.1倍スケール(0.887m)。到達可能な形態同士の判別解像度を測る
#                          （第6章6.5-2「優劣が微妙な形態同士は未検証」への回答）
set -u
cd /userdir/StackelbergPPO
LOG=single_run/launch_next3_20260803.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# PJ実験共通（4.4.3/4.4.4 と同一設定。リンク長のみ凍結（gear・size・ext_start は可変））
REACH="+reward_specs.use_reach=true +reward_specs.target_x=0.8 \
+reward_specs.target_y=0.0 +reward_specs.target_z=0.15 \
+reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false"

launch() {  # $1=cfg $2=xml $3=run名 $4=max_epoch $5=seed $6=追加override
  mkdir -p "single_run/$3"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=$1 xml_name=$2 num_threads=4 max_epoch_num=$4 \
    enable_wandb=false fix_skeleton=true seed=$5 +robot_param_scale=1 \
    $6 \
    hydra.run.dir="single_run/$3" \
    > "single_run/$3/stdout.log" 2>&1 &
  say "$3 launched (PID $!)"
}

avail_gb=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
say "available memory: ${avail_gb}GB / load: $(cut -d' ' -f1-3 /proc/loadavg)"

say "--- ① tripo_arm_v3_pusher: 縦型4関節 Pusher 本走 (1000ep) ---"
# 注意: 過去に epoch0 の処理が異常に遅くなる事例あり（床貫通ペナルティで
# 即終了エピソードが大量発生するため）。ハングではないので数十分は待つこと。
launch pusher_tripo_v3 tripo_arm_v3 tripo_arm_v3_pusher 1000 0 ""

sleep 90  # 起動ピークをずらす

say "--- ② tripo_pj_long_s2: PJ実験 seed=1 (200ep) ---"
launch pusher_gearonly tripo_arm_v2c_pj_long tripo_pj_long_s2 200 1 "$REACH"

sleep 90

say "--- ③ tripo_pj_mid: 1.1倍スケール=0.887m (200ep) ---"
launch pusher_gearonly tripo_arm_v2c_pj_mid tripo_pj_mid 200 0 "$REACH"

sleep 420
say "--- 起動確認 ---"
for run in tripo_arm_v3_pusher tripo_pj_long_s2 tripo_pj_mid; do
  ep=$(grep -E "^[0-9]+\s+T_sample" "single_run/$run/stdout.log" 2>/dev/null | tail -1 | awk '{print $1}')
  if [ -n "$ep" ]; then
    say "$run: ep=$ep まで到達"
  else
    say "WARNING: $run はまだ epoch 0 未到達（v3 は正常。他は要確認）"
  fi
done
say "done"
