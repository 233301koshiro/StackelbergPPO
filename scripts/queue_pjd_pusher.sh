#!/usr/bin/env bash
#
# 2026-08-24: 配分判別の Pusher 版キュー。
#
# 問い: 総リーチを 0.887 m に固定して配分だけを反転させた2形態（根元重 / 先端重）の優劣は、
#   **タスクによって逆転するか**。Reach では根元重が勝つことが 2 seed で確定している
#   （9-9、−1.65/−1.81 対 −4.39/−5.75）。第4章 4.3.3.4 の限界として
#   「押しタスクでは先端の質量が運動量伝達に寄与するため逆転しうるが未検証」と書いた箇所を閉じる。
#
#   逆転すれば「配分の優劣もタスク依存」＝タスク識別の主張がスケール以外の軸へ広がる。
#   逆転しなければ「根元重が両タスクで有利」という別の知見になる。どちらでも価値がある。
#
# 投入する4本（Reach 版と対になるよう 2 seed）:
#   tripo_pjdp_prox / _s2   根元重 0.500/0.250/0.137
#   tripo_pjdp_dist / _s2   先端重 0.137/0.250/0.500
#
# 設定は `tripo_pjp_mid`（4.3.3.3 の Pusher 中間スケール）と **xml_name と seed 以外すべて同一**。
# .hydra/overrides.yaml で照合済み。
#
# 安全策:
#   - 完走判定は**完走マーカー**。`pgrep -f <run名>` は自分のコマンドラインに当たる（Bug 19）ので使わない。
#   - メモリの空きを見てから投入する。不足のまま起動すると OOM で稼働中の学習を巻き込む。
#   - **投入前に XML の実効リーチを検算する**（Bug 23）。公称と食い違う XML では学習しない。
#   - run ディレクトリが既にあればスキップ（再実行しても二重起動しない）。
#
# 起動: nohup bash scripts/queue_pjd_pusher.sh > /dev/null 2>&1 & disown
# 進捗: single_run/queue_pjd_pusher.log
set -uo pipefail
cd /userdir/StackelbergPPO
LOG=single_run/queue_pjd_pusher.log
MIN_FREE_GB=8          # 実測 1 run 約 3 GB。2本同時でも余裕を見てこの値
MAX_PARALLEL=2         # 同時に走らせる本数の上限（既存の 2 本と合わせて 4 本まで）

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }
free_gb() { free -g | awk 'NR==2{print $7}'; }
# ⚠️ ワーカープロセスはメインと**同じコマンドライン**（hydra.run.dir 込み）を持つ。
# 単純に grep -c すると 1 run を 4〜5 本と数えてしまい、次の投入が永久に待たされる
# （2026-08-24 に実際に踏んだ: 3 run のところ 7 と数えた）。
# run ディレクトリの**異なり数**を数えること。
n_running() {
  ps -eo cmd --no-headers | grep -F 'choreonoid_train.py' \
    | grep -oP 'hydra\.run\.dir=\S+' | sort -u | wc -l
}

launch() {  # $1=run名 $2=xml $3=seed
  local run=$1 xml=$2 seed=$3
  if [ -d "single_run/$run/log" ]; then log "$run は既に存在。スキップ"; return 0; fi
  mkdir -p "single_run/$run"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_gearonly xml_name="$xml" num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed="$seed" +robot_param_scale=1 \
    +reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 \
    +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true \
    hydra.run.dir="single_run/$run" > "single_run/$run/stdout.log" 2>&1 &
  log "$run launched (PID $!, xml=$xml, seed=$seed)"
  sleep 120        # 起動直後のメモリ確保が落ち着くまで待つ
}

log "=== 配分判別 Pusher 版キュー開始 ==="

# --- 投入前の検算（Bug 23）---
if ! python3 scripts/audit_xml_reach.py \
      assets/mujoco_envs/tripo_arm_v2c_pj_prox.xml \
      assets/mujoco_envs/tripo_arm_v2c_pj_dist.xml >> "$LOG" 2>&1; then
  log "⛔ XML の公称/実効リーチが食い違う。投入を中止する（Bug 23 と同型）"
  exit 1
fi
log "✅ XML 検算を通過（公称=実効）"

# run名 xml seed の3つ組
QUEUE=(
  "tripo_pjdp_prox    tripo_arm_v2c_pj_prox 0"
  "tripo_pjdp_dist    tripo_arm_v2c_pj_dist 0"
  "tripo_pjdp_prox_s2 tripo_arm_v2c_pj_prox 1"
  "tripo_pjdp_dist_s2 tripo_arm_v2c_pj_dist 1"
)

for spec in "${QUEUE[@]}"; do
  read -r run xml seed <<< "$spec"
  [ -d "single_run/$run/log" ] && { log "$run は既に存在。スキップ"; continue; }
  # 空きが出るまで待つ
  while true; do
    f=$(free_gb); n=$(n_running)
    if [ "$f" -ge "$MIN_FREE_GB" ] && [ "$n" -lt $((MAX_PARALLEL + 2)) ]; then
      log "空き ${f} GB / 稼働 ${n} 本 → $run を投入"
      break
    fi
    sleep 300
  done
  launch "$run" "$xml" "$seed"
done

log "全4本の投入を終えた。完走は各 run の stdout.log の完走マーカーで確認する"
