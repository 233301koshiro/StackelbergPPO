#!/usr/bin/env bash
# 2026-09-02: A1 の学習キュー。GPU が律速なので **2 本ずつ直列**に回す。
#
# 4 本同時に投入したとき GPU 使用率が 92 % で頭打ちになり、T_update が
# 130 秒 → 290〜345 秒（2.4 倍）に伸びた。CPU は load average 4.9/20 コアで
# 空いており律速は GPU。同時実行しても総スループットは変わらず結果が遅く出るだけ。
#
# 段取り（優先順）:
#   1. 平面版 seed=0（稼働中） … pj_short/mid/long との比較用。既存実験に接続する
#   2. 縦型版 seed=0            … 発表用の「スケッチが動く」映像 + 6.4.2(1) の限界解消
#   3. 縦型版 seed=1            … 2 シード化
#
# なぜ平面版を捨てないか: 既に走っており、pj_* との比較は縦型では成立しない
#   （形態クラスが違う）。両方あれば 9-18b の比較と非平面の証拠が両立する。
#
# 完走判定は `training done!` の有無で行う。`pgrep -f` / `pkill -f` は
# **自分のコマンドラインにマッチして誤爆する**（Bug 19。2026-09-02 に再度踏んだ）。
#
# 起動: nohup bash scripts/queue_e2e_a1.sh > /dev/null 2>&1 & disown
# 進捗: single_run/queue_e2e_a1.log
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_e2e_a1.log
MIN_FREE_GB=20
DEADLINE=$(date -d '2026-09-04 12:00' +%s)

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
done_p() { grep -q "training done!" "single_run/$1/log/log_train.txt" 2>/dev/null; }

launch() {  # $1=run名 $2=seed $3=task $4=xml名
  mkdir -p "single_run/$1"
  if [ "$3" = "reach" ]; then
    EXTRA="+reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false"
  else
    EXTRA="+reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true"
  fi
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_gearonly xml_name="$4" num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed="$2" +robot_param_scale=1 $EXTRA \
    hydra.run.dir="single_run/$1" > "single_run/$1/stdout.log" 2>&1 &
  log "$1 launched (PID $!, seed=$2, $3, xml=$4)"
}

wait_free() {
  while true; do
    [ "$(date +%s)" -ge "$DEADLINE" ] && { log "期限超過。終了"; exit 0; }
    f=$(free -g | awk 'NR==2{print $7}')
    [ "$f" -ge "$MIN_FREE_GB" ] && return 0
    log "空きメモリ ${f} GB < ${MIN_FREE_GB} GB。待機"
    sleep 300
  done
}

stage() {  # $1 $2 = 前段として完走を待つ run 名、以降 "run seed task xml"
  local a="$1" b="$2"; shift 2
  log "前段の完走を待つ: $a / $b"
  while ! (done_p "$a" && done_p "$b"); do
    [ "$(date +%s)" -ge "$DEADLINE" ] && { log "期限超過。終了"; exit 0; }
    sleep 300
  done
  log "前段が完走: $a / $b"
  for spec in "$@"; do
    set -- $spec
    if [ -d "single_run/$1/log" ]; then log "$1 は既存。スキップ"; continue; fi
    wait_free
    launch "$1" "$2" "$3" "$4"
    sleep 60
  done
}

log "キュー開始"
stage e2e_a1_reach e2e_a1_pusher \
  "e2e_a1v_reach 0 reach e2e_a1v" \
  "e2e_a1v_pusher 0 pusher e2e_a1v"
stage e2e_a1v_reach e2e_a1v_pusher \
  "e2e_a1v_reach_s2 1 reach e2e_a1v" \
  "e2e_a1v_pusher_s2 1 pusher e2e_a1v"
log "キュー完了: 予定していた投入をすべて終えた"
