#!/usr/bin/env bash
# 2026-09-09: 9-27（助言に従うと落ちないか）の seed=1。実験系譜 9-52。
#
# ⚠️ **これは新しい実験ではなく、既存の主結果の seed を埋める作業である。**
#   9-27 は各条件 1 seed で、**中間発表スライドの 3 つ目の主結論**に載っている
#   （「助言は増やす方向・減らす方向の両方で閉ループが成立する」、★付き）。
#   この repo は 9-31・9-34 で「1 seed の結果が seed=1 で崩れる」を 2 回踏んでいる。
#
# 変数は seed だけ。他は seed=0 版の overrides.yaml と完全に同一。
#
# 停止条件（先に決めておく。9-50 の教訓）:
#   ① Reach 良化・Pusher 悪化が再現 → 9-27 は 2 seed で成立。スライドの★を維持
#   ② どちらかが逆転          → 9-27 を「1 seed・向きのみ」に弱める。★を落とす
#   ③ 差が seed 幅に埋もれる   → 同上（**帯の外に出たことだけを根拠にしない**。9-34）
#
# ⚠️ 9-34 で崩れた比較とは性質が違う。あちらは**別形態どうしの順位**、
#   こちらは**同じ腕で関節を固定するかどうか**の 1 変数対照である。
#
# 2 本ずつ直列（GPU が律速。4 本同時は T_update が 2.4 倍に伸びて総時間が変わらない。CLAUDE.md §6）
# タスクごとに組む: 片方のバッチだけ終わっても**1 タスク分の比較が完結する**。
#
# 起動: nohup bash scripts/queue_fix3_s2.sh > /dev/null 2>&1 & disown
# 進捗: single_run/queue_fix3_s2.log
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_fix3_s2.log
DEADLINE=$(date -d '2026-09-11 06:00' +%s)

log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
done_(){ grep -q "training done!" "single_run/$1/log/log_train.txt" 2>/dev/null; }
alive_(){ ps -eo args 2>/dev/null | grep -qE "[c]horeonoid_train\.py.*hydra\.run\.dir=single_run/$1( |\$)"; }

launch(){  # $1=run名 $2=XML $3=タスク(reach|pusher)
  if done_ "$1"; then log "$1 は完走済み。スキップ"; return; fi
  if alive_ "$1"; then log "$1 は稼働中。スキップ"; return; fi
  if [ -s "single_run/$1/log/log_train.txt" ]; then log "$1 は既存。スキップ（手で確認すること）"; return; fi
  mkdir -p "single_run/$1"
  if [ "$3" = reach ]; then
    EXTRA="+reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 +reward_specs.target_z=0.15 +env_specs.check_init_contact=false"
  else
    EXTRA="+reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true"
  fi
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_gearonly xml_name="$2" num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=1 +robot_param_scale=1 \
    +reward_specs.ctrl_cost_coeff=0.2 $EXTRA \
    hydra.run.dir="single_run/$1" >> "single_run/$1/stdout.log" 2>&1 &
  log "$1 launched (PID $!, seed=1, xml=$2, task=$3)"
}

wait_for(){  # 2 本の完走を待つ
  log "完走を待つ: $1 / $2"
  while ! { done_ "$1" && done_ "$2"; }; do
    [ "$(date +%s)" -ge "$DEADLINE" ] && { log "期限超過。終了"; exit 0; }
    sleep 180
  done
  log "完走: $1 / $2"
}

log "キュー開始（9-27 の seed=1、実験系譜 9-52）"
launch e2e_a1_reach_s2       e2e_a1      reach   ; sleep 60
launch e2e_a1_fix3_reach_s2  e2e_a1_fix3 reach
wait_for e2e_a1_reach_s2 e2e_a1_fix3_reach_s2
launch e2e_a1_pusher_s2      e2e_a1      pusher  ; sleep 60
launch e2e_a1_fix3_pusher_s2 e2e_a1_fix3 pusher
wait_for e2e_a1_pusher_s2 e2e_a1_fix3_pusher_s2
log "キュー完了: 4 本すべて完走"
