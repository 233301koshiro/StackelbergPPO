#!/usr/bin/env bash
# 2026-09-02: A1（Bug 27 修正後）の seed=1 を、seed=0 の完走後に自動投入する。
#
# なぜ直列にするか: 4 本同時に走らせたとき **GPU が 92 % で頭打ち**になり、
#   T_update が 130 秒 → 290〜345 秒（2.4 倍）に伸びた。GPU が律速なので
#   同時実行しても総スループットは変わらず、**結果が出るのが遅くなるだけ**。
#   2 本ずつ直列にすれば、判断に必要な seed=0 が約 6 時間で先に出る。
#
# 安全策:
#   - 完走判定は **`training done!` の有無**で行う。`pgrep -f <run名>` は
#     自分自身のコマンドラインにマッチして誤判定する（Bug 19）ので使わない。
#   - 空きメモリが足りなければ投入しない（1 run 約 8.7 GB）。不足のまま起動すると
#     OOM で稼働中の学習まで巻き込んで落ちる。
#   - 既に run ディレクトリがあれば投入しない（再実行しても二重起動しない）。
#
# 起動: nohup bash scripts/queue_e2e_a1_s2.sh > /dev/null 2>&1 & disown
# 進捗: single_run/queue_e2e_a1_s2.log
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_e2e_a1_s2.log
MIN_FREE_GB=20          # 2 run 分 + 余裕
DEADLINE=$(date -d '2026-09-03 12:00' +%s)

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
done_p() { grep -q "training done!" "single_run/$1/log/log_train.txt" 2>/dev/null; }

launch() {  # $1=run名 $2=seed $3=task
  mkdir -p "single_run/$1"
  if [ "$3" = "reach" ]; then
    EXTRA="+reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false"
  else
    EXTRA="+reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true"
  fi
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_gearonly xml_name=e2e_a1 num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=$2 +robot_param_scale=1 $EXTRA \
    hydra.run.dir="single_run/$1" > "single_run/$1/stdout.log" 2>&1 &
  log "$1 launched (PID $!, seed=$2, $3)"
}

log "キュー開始: e2e_a1_reach / e2e_a1_pusher（seed=0）の完走を待つ"
while true; do
  now=$(date +%s)
  if [ "$now" -ge "$DEADLINE" ]; then log "期限超過。投入せず終了。人の判断を待つ"; exit 0; fi
  if done_p e2e_a1_reach && done_p e2e_a1_pusher; then
    log "seed=0 の 2 本が完走"
    break
  fi
  sleep 300
done

for spec in "e2e_a1_reach_s2 1 reach" "e2e_a1_pusher_s2 1 pusher"; do
  set -- $spec
  if [ -d "single_run/$1/log" ]; then log "$1 は既に存在。スキップ"; continue; fi
  while true; do
    f=$(free -g | awk 'NR==2{print $7}')
    [ "$f" -ge "$MIN_FREE_GB" ] && break
    log "空きメモリ ${f} GB < ${MIN_FREE_GB} GB。待機"; sleep 300
  done
  launch "$1" "$2" "$3"
  sleep 60
done
log "キュー完了: seed=1 の 2 本を投入した"
