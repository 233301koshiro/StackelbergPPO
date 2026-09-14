#!/usr/bin/env bash
# 2026-09-14: 段1a＋段1b「実機相当の駆動系」で Pusher を回す（実験系譜 9-83 → 9-97）。
#
# **なぜやるか**: 9-82 で「45 g の腕に 400 N·m の関節」が設計空間に入っていると数字で出た。
#   段1a（反射慣性 ∝ gear²、9-81）は gear を下へ押し、上限 400 から降りた。
#   段1b（駆動系の質量 = gear / トルク密度、9-83）は**単独では gear を動かさない**と計算で示されている
#   （トルクも質量も gear 比例で余裕が消える）ので、**両方入れた 1 本**だけを回す。
#
# **変数は 1 つ**: e2e_a1v_actuator_pusher（段1a）との差は cfg の `torque_density: 25` の 1 行。
#   XML・報酬・seed・epoch・ペナルティは同一（launch の引数は queue_actuator_tp.sh と 9-81 の overrides を写した）。
#
#   | 条件 | run | 状態 |
#   |---|---|---|
#   | 対照 | e2e_a1v_pusher | 完走済み |
#   | 1a のみ | e2e_a1v_actuator_pusher | 完走済み（9-81） |
#   | **両方** | **e2e_a1v_actuator_mass_pusher** | **本スクリプト** |
#
# 手順: ① 3 epoch のスモーク（NaN・発散の有無だけを見る。性能は判定しない。CLAUDE.md §5-2 ⑤-2）
#       ② スモークの log に nan/inf が無ければ 200 epoch の本番を投入
#
# 事前に決めた読み（9-83 末尾。**結果を見てから変えない**）:
#   ① gear が中間へ収束 → ✅ 実機相当の代償の下では境界解が消える
#   ② gear が上限へ     → 🟡 1b（上へ押す力）が 1a を上回った。トルク密度の設定次第と書く
#   ③ 学習が成立しない  → ❌ この腕は実機相当の駆動系を積むと動かない。それも判定器の出力
#   ⚠️ 見るのは gear の位置。スコアは物理が違うので比べない。
#   ⚠️ 帰属は 1a のみ（e2e_a1v_actuator_pusher）との差分で行う。
#
# 完走後: COMPARE_EPOCH=200 で同じ epoch どうしを比べる（⚠️ 存在しない epoch を指定するとハング。9-92）。
#   ls single_run/e2e_a1v_actuator_mass_pusher/models/ で実在を確認してから。
#
# GPU は 2 本まで。空くまで待つ。
# 起動: nohup bash scripts/queue_actuator_mass.sh > /dev/null 2>&1 & disown
# 進捗: single_run/queue_actuator_mass.log
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_actuator_mass.log
MAXJOBS=2
CFG=pusher_tripo_v3_actuator_mass
SMOKE=e2e_a1v_actuator_mass_smoke
RUN=e2e_a1v_actuator_mass_pusher

log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
njobs(){ ps -eo args 2>/dev/null | grep -cE "[c]horeonoid_train\.py" || true; }
done_(){ grep -q "training done!" "single_run/$1/log/log_train.txt" 2>/dev/null; }
alive_(){ ps -eo args 2>/dev/null | grep -qE "[c]horeonoid_train\.py.*hydra\.run\.dir=single_run/$1( |\$)"; }
wait_slot(){ while [ "$(njobs)" -ge "$MAXJOBS" ]; do sleep 120; done; }

launch(){  # $1=run名 $2=epoch 数
  if done_ "$1"; then log "$1 は完走済み。スキップ"; return; fi
  if alive_ "$1"; then log "$1 は稼働中。スキップ"; return; fi
  if [ -s "single_run/$1/log/log_train.txt" ]; then log "$1 は既存。スキップ（手で確認）"; return; fi
  wait_slot
  mkdir -p "single_run/$1"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg="$CFG" xml_name=e2e_a1v num_threads=4 max_epoch_num="$2" \
    enable_wandb=false fix_skeleton=true seed=0 +robot_param_scale=1 \
    +reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 \
    +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true \
    hydra.run.dir="single_run/$1" >> "single_run/$1/stdout.log" 2>&1 &
  log "$1 launched (PID $!, cfg=$CFG, epochs=$2)"
  sleep 60
}

log "=== キュー開始（段1a＋段1b の Pusher）"
launch "$SMOKE" 3
while alive_ "$SMOKE"; do sleep 60; done
if ! done_ "$SMOKE"; then log "❌ スモークが完走していない。本番は投入しない（stdout.log を見る）"; exit 1; fi
if grep -qiE "\bnan\b|\binf\b" "single_run/$SMOKE/log/log_train.txt"; then
  log "❌ スモークに NaN/inf。本番は投入しない（トルク密度を上げるか armature_floor を上げて再測定。9-83）"; exit 1
fi
log "✅ スモーク完走・NaN 無し: $(grep -oE 'exec_R_eps [-0-9.]+' "single_run/$SMOKE/log/log_train.txt" | tail -1)"
launch "$RUN" 200
log "=== キュー終了（本番は稼働中。完走したら 9-97 に記録）"
