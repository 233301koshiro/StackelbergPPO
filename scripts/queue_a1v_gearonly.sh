#!/usr/bin/env bash
# 2026-09-10: 縦型 A1v の**リンク長を凍結**して自由度だけを分離する（実験系譜 9-64）。
#
# **なぜやるか**: 第6章 6.4.2(1) が「縦型の差は腕が約 2 倍に伸びたことだけで説明でき、
#   非平面でのタスク識別の強弱は言えない」と自認している。知見総括表 E 節の取り下げ済み主張
#   「自由度は Reach では負債・Pusher では資産」も、復活条件が
#   **「縦型 cfg の body_params を {} にして揃える」**と明記されている。それをやる。
#
# **変数は 1 つ**: cfg を pusher_tripo_v3 → pusher_tripo_v3_gearonly（body_params だけ空）。
#   XML・報酬・seed・epoch はすべて e2e_a1v_* と同一。max_body_depth は 6 のまま
#   （4 に落とすと Bug 32 の IndexError で即死する）。
#
# **比較対象**（どちらも既に 2 seed 完走済み）:
#   e2e_a1v_reach   −13.65 / −14.29     e2e_a1v_pusher   348.56 / 340.39
#   e2e_a1_reach     −6.30 /  −5.83     e2e_a1_pusher    203.40 / 258.98（平面・gearonly）
#
# 停止条件（先に決めておく。9-50 の教訓）:
#   ① 縦型 gearonly でも平面と違う向きが出る → 差は自由度に帰属しうる。E 節の主張を再検討
#   ② 平面 gearonly と同水準になる         → 縦型の差はリンク長で説明できる。
#                                            第6章 6.4.2(1) の限界がそのまま確定する（これでも成果）
#   ③ 学習が成立しない（床ペナルティ −1900 に張り付く等）
#                                          → 「凍結したリンク長では縦型が成立しない」の記録。
#                                            第1層の判定（A1v は余裕 20 %）と突き合わせる材料になる
#   ⚠️ **どれに転んでも書ける。** 性能を上げるための実験ではない（CLAUDE.md §5-2 ⑤-2）。
#
# 2 本ずつ直列（GPU が律速。CLAUDE.md §6）。**タスクごとに組む** ので、
# バッチ2 が終わった時点で Reach の 2 seed 比較が完結する。
#
# 起動: nohup bash scripts/queue_a1v_gearonly.sh > /dev/null 2>&1 & disown
# 進捗: single_run/queue_a1v_gearonly.log
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_a1v_gearonly.log
DEADLINE=$(date -d '2026-09-13 00:00' +%s)

log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
done_(){ grep -q "training done!" "single_run/$1/log/log_train.txt" 2>/dev/null; }
alive_(){ ps -eo args 2>/dev/null | grep -qE "[c]horeonoid_train\.py.*hydra\.run\.dir=single_run/$1( |\$)"; }

launch(){  # $1=run名 $2=タスク(reach|pusher) $3=seed
  if done_ "$1"; then log "$1 は完走済み。スキップ"; return; fi
  if alive_ "$1"; then log "$1 は稼働中。スキップ"; return; fi
  if [ -s "single_run/$1/log/log_train.txt" ]; then log "$1 は既存。スキップ（手で確認すること）"; return; fi
  mkdir -p "single_run/$1"
  if [ "$2" = reach ]; then
    # 床ペナルティ 1900 は 9-14 の較正値（a1v）。正直な最悪の戻り値より大きくする
    EXTRA="+reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 +reward_specs.target_z=0.15 +env_specs.check_init_contact=false +reward_specs.init_contact_penalty=1900"
  else
    EXTRA="+reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true"
  fi
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_tripo_v3_gearonly xml_name=e2e_a1v num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed="$3" +robot_param_scale=1 \
    +reward_specs.ctrl_cost_coeff=0.2 $EXTRA \
    hydra.run.dir="single_run/$1" >> "single_run/$1/stdout.log" 2>&1 &
  log "$1 launched (PID $!, cfg=gearonly, seed=$3, task=$2)"
}

wait_all(){  # 引数の run すべての完走を待つ
  log "完走を待つ: $*"
  while :; do
    ok=1; for r in "$@"; do done_ "$r" || ok=0; done
    [ "$ok" = 1 ] && break
    [ "$(date +%s)" -ge "$DEADLINE" ] && { log "期限超過。終了"; exit 0; }
    sleep 180
  done
  log "完走: $*"
}

log "キュー開始（9-64、縦型 gearonly）"
# バッチ1 は手で起動済み: e2e_a1v_gearonly_reach(seed0) + e2e_a1v_fix4_pusher_s2(9-32 の決着、別実験)
wait_all e2e_a1v_gearonly_reach e2e_a1v_fix4_pusher_s2

log "バッチ2: Reach の seed=1 と Pusher の seed=0"
launch e2e_a1v_gearonly_reach_s2  reach  1 ; sleep 60
launch e2e_a1v_gearonly_pusher    pusher 0
wait_all e2e_a1v_gearonly_reach_s2 e2e_a1v_gearonly_pusher
log "★ ここで Reach の 2 seed 比較が完結した"

log "バッチ3: Pusher の seed=1"
launch e2e_a1v_gearonly_pusher_s2 pusher 1
wait_all e2e_a1v_gearonly_pusher_s2
log "キュー完了: 9-64 の 4 本すべて完走"
