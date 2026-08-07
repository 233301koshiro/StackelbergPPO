#!/usr/bin/env bash
# 2026-08-07（金）夕方〜08-10（月）朝: 無人運転用のキュー。
#
# 目的: 稼働中の run が完走してメモリが空いたとき、**遊ばせずに次を投入する**。
#   週末に手を入れられないため、空きスロットの発生に自動で追従させる。
#
# 投入する実験（軸3 = 診断フィードバックの有用性を補強するもの）:
#   1. tripo_pjr_minp   Pusher の助言下限 2.11 倍（総リーチ 0.8505 m、200ep、seed=0）
#      → 稼働中の tripo_pjr_min（Reach 下限 1.99 倍）と対になる。
#        「下限ちょうどでは苦しい」が Reach 固有かタスク共通かを分ける。
#        設定は tripo_pjp_mid と xml_name 以外すべて同一（overrides.yaml で照合済み）。
#   2. tripo_pjr_min_s2  Reach の助言下限 1.99 倍の seed=1
#      → 軸3 の結論を 1 シードで述べずに済むようにする。
#
# 安全策:
#   - 完走判定は**完走マーカー**で行う。`pgrep -f <run名>` は自分自身のコマンドラインに
#     マッチして誤判定する（Bug 19）ため使わない。
#   - メモリに余裕（>= MIN_FREE_GB）が無ければ投入しない。1 run 約 8.7 GB 必要で、
#     不足のまま起動すると **OOM で稼働中の学習まで巻き込んで落ちる**。
#   - ディスクが少なければ投入しない。
#   - 既に run ディレクトリがあれば投入しない（再実行しても二重起動しない）。
#
# 起動: nohup bash scripts/weekend_queue.sh > /dev/null 2>&1 & disown
# 進捗: single_run/weekend_queue.log
set -u
cd /userdir/StackelbergPPO
LOG=single_run/weekend_queue.log
MIN_FREE_GB=11          # 1 run 8.7 GB + 余裕
MIN_DISK_GB=20
DEADLINE=$(date -d '2026-08-10 07:00' +%s)   # 月曜朝以降は新規投入しない

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

free_gb() { free -g | awk 'NR==2{print $7}'; }
disk_gb() { df -BG --output=avail /userdir | tail -1 | tr -dc '0-9'; }

launch() {  # $1=run名 $2...=hydra 引数
  local run=$1; shift
  mkdir -p "single_run/$run"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py "$@" \
    hydra.run.dir="single_run/$run" > "single_run/$run/stdout.log" 2>&1 &
  log "$run launched (PID $!)"
}

launch_pjr_minp() {
  launch tripo_pjr_minp \
    cfg=pusher_gearonly xml_name=tripo_arm_v2c_pj_recmin_p num_threads=4 \
    max_epoch_num=200 enable_wandb=false fix_skeleton=true seed=0 \
    +robot_param_scale=1 +reward_specs.ctrl_cost_coeff=0.2 \
    +reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 \
    +env_specs.arm_safe_init=true
}

launch_pjr_min_s2() {
  launch tripo_pjr_min_s2 \
    cfg=pusher_gearonly xml_name=tripo_arm_v2c_pj_recmin num_threads=4 \
    max_epoch_num=200 enable_wandb=false fix_skeleton=true seed=1 \
    +robot_param_scale=1 +reward_specs.use_reach=true +reward_specs.target_x=0.8 \
    +reward_specs.target_y=0.0 +reward_specs.target_z=0.15 \
    +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false
}

QUEUE=(tripo_pjr_minp tripo_pjr_min_s2)

log "週末キュー開始（投入予定: ${QUEUE[*]}）"
for run in "${QUEUE[@]}"; do
  if [ -d "single_run/$run/log" ]; then
    log "$run は既に存在。スキップ"; continue
  fi
  # 空きが出るまで待つ
  while true; do
    now=$(date +%s)
    if [ "$now" -ge "$DEADLINE" ]; then
      log "月曜朝の期限を過ぎたので $run 以降は投入しない。人の判断を待つ"; exit 0
    fi
    f=$(free_gb); d=$(disk_gb)
    if [ "$f" -ge "$MIN_FREE_GB" ] && [ "$d" -ge "$MIN_DISK_GB" ]; then
      log "空きメモリ ${f} GB / ディスク ${d} GB → $run を投入"
      break
    fi
    sleep 600
  done
  case "$run" in
    tripo_pjr_minp)    launch_pjr_minp ;;
    tripo_pjr_min_s2)  launch_pjr_min_s2 ;;
  esac
  sleep 300                       # 起動直後のメモリ確保が落ち着くまで待ってから次を見る
done
log "週末キュー: 予定していた投入をすべて終えた"
