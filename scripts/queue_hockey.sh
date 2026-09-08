#!/usr/bin/env bash
# 2026-09-08: ホッケー系（実験系譜 9-33 の設計 → 9-38 で B+C+D に作り直し）。
#
# ⚠️ **この系列は既存の Pusher・Target-Pusher・Reach と比較できない。**
#   パックの大きさ・質量・初期位置・可動域をすべて変えたため（9-38）。
#   比較は**ホッケー系の内部**（パック正面固定 ↔ 左右に来る）でのみ行う。
#
# 変数は cube_y_noise の 1 つだけ:
# ⚠️ 2026-09-09: ctrl_cost_coeff を 0.2 → 0.001 に下げた（9-41）。
#   ホッケー台はパックの可動範囲を -0.5〜0.8 m に制限しており、Pusher 報酬は
#   速度の積算＝総移動距離なので **報酬の上限が構造的に 0.8** になる。
#   一方 ctrl_cost 0.2 は報酬 100〜350 の通常 Pusher 用に較正された値で、
#   実測の制御コストは 23.35（上限報酬の 29 倍）だった。**動かないのが最適**になり、
#   初回の 2 本は 32 epoch 全部 fwd_cube=0.0000 で終わった。
#   逆算: コストを上限報酬の 1〜2 割に収めるには 0.2 × 0.08/23.35 ≒ 0.0008 → 0.001 を採用。
#   ⚠️ 0.01 では上限報酬の 146 % で不足（最初にこの値を提案して外した）。
#   **報酬スケールが変わったらコストもペナルティも逆算し直す**（9-14 と同型）。
#
#   hockey_fixed  0.0  パックは毎回 (0.70, 0.00)
#   hockey_shot   0.4  パックの y が毎回 ±0.4 m で変わる
#
# 仮説（9-33）: 9-29④ で B2v は根元ヨーを 2 % しか使っていなかった（目標が正面 y=0 固定）。
#   **パックが左右に来れば根元が要るはず。** 見るのはスコアだけでなく
#   **収束形態と関節使用率（とくに関節1＝根元ヨー）**。
#   同一形態内の比較なので、9-34 の「形態どうしの順位が不安定」とは独立に見られる。
#
# ⚠️ 床貫通ペナルティは Pusher 系なので既定 50 のまま。Reach の 1900 は付けない（9-14）。
# ⚠️ Reach は回さない。ホッケーは Pusher 系の話。
#
# 起動: nohup bash scripts/queue_hockey.sh > /dev/null 2>&1 & disown
# 進捗: single_run/queue_hockey.log
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_hockey.log
XML=e2e_hockey_court
DEADLINE=$(date -d '2026-09-11 12:00' +%s)

log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
done_(){ grep -q "training done!" "single_run/$1/log/log_train.txt" 2>/dev/null; }
alive_(){ ps -eo args 2>/dev/null | grep -qE "[c]horeonoid_train\.py.*hydra\.run\.dir=single_run/$1( |\$)"; }
ckpt_(){ local n; n=$(ls "single_run/$1"/models/epoch_*.p 2>/dev/null \
        | sed 's|.*/epoch_||; s|\.p$||' | sort -n | tail -1); [ -n "$n" ] && echo $((10#$n)); }

launch(){  # $1=run名 $2=seed $3=cube_y_noise $4=再開引数
  mkdir -p "single_run/$1"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_tripo_v3 xml_name="$XML" num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed="$2" +robot_param_scale=1 \
    +reward_specs.ctrl_cost_coeff=0.001 +reward_specs.contact_weight=0 \
    +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true \
    +env_specs.cube_y_noise="$3" ${4:-} \
    hydra.run.dir="single_run/$1" >> "single_run/$1/stdout.log" 2>&1 &
  log "$1 launched (PID $!, seed=$2, cube_y_noise=$3${4:+, $4})"
}

stage(){  # 前段2つの完走を待ってから、以降の "run seed noise" を投入
  local a="$1" b="$2"; shift 2
  log "前段の完走を待つ: $a / $b"
  while ! { done_ "$a" && done_ "$b"; }; do
    [ "$(date +%s)" -ge "$DEADLINE" ] && { log "期限超過。終了"; exit 0; }
    sleep 180
  done
  log "前段が完走: $a / $b"
  for spec in "$@"; do
    set -- $spec
    R=""
    if done_ "$1"; then log "$1 は完走済み。スキップ"; continue
    elif alive_ "$1"; then log "$1 は稼働中。スキップ"; continue
    elif [ -s "single_run/$1/log/log_train.txt" ]; then
      C=$(ckpt_ "$1")
      if [ -n "$C" ]; then R="+restore_dir=single_run/$1 epoch=$C"
        log "$1 は epoch $C まで進んで止まっている → そこから再開する"
      else D="single_run/$1.dead_$(date +%m%d%H%M)"; mv "single_run/$1" "$D"
        log "$1 は checkpoint 無し → $D へ退避し最初から"
      fi
    fi
    launch "$1" "$2" "$3" "$R"
    sleep 60
  done
}

log "キュー開始（ホッケー系、XML=$XML）"
stage e2e_a1v_fix4_reach e2e_a1v_fix4_pusher \
  "hockey_fixed 0 0.0" \
  "hockey_shot 0 0.4"
stage hockey_fixed hockey_shot \
  "hockey_fixed_s2 1 0.0" \
  "hockey_shot_s2 1 0.4"
log "キュー完了: 予定していた投入をすべて終えた"
