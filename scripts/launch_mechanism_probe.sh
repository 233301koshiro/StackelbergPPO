#!/usr/bin/env bash
# 2026-08-19: 次の3本。優先度1（非平面のタスク識別）と優先度2（余裕が害になる機序）。
#
# ── 優先度1: v3 Reach のやり直し（第6章 6.4.2(1)・6.5-1 を閉じる）
#   前回（tripo_v3_reach）は 153 epoch 全区間で exec_R_eps = −50.00 のまま学習が成立しなかった。
#   床貫通ペナルティで全エピソードが打ち切られていた（実験系譜 9-13）。
#   仮説: 目標高さ z=0.15 が肩（z=0.222）より下にあり、目標へ近づく勾配が形態を床方向へ引く一方、
#   床拘束がそれを罰するため脱出経路が無かった。
#   対策: **目標を肩より上（z=0.25）に置く**。第1層で検算済み（水平到達限界 0.799 m に対し余裕 10%）。
#   ⚠️ まず 30ep のスモークで **−50 を脱するかだけ**を見る。脱しなければ仮説が外れているので
#      1000ep を投じない。前回はこれを怠って 2.5 日を失った。
#
# ── 優先度2: 制御コストを外した対照（第5章 5.6.2 の「(a) 慣性と (c) 制御コストが分離できていない」）
#   余裕 0% → 5.3% → 10% で成績が単調に悪化する（実験系譜 9-11）。原因の候補は
#   (a) 腕が長いほど慣性が大きく微調整しにくい、(c) ctrl コスト下で長い腕を動かす代償が大きい。
#   どちらも「腕の長さの単調関数」なので既存データでは分離できない。
#   そこで **ctrl_cost_coeff だけを 0.2 → 0.0 に変えた**両端（余裕 0% と 10%）を回し、
#   差分の差分を見る:
#     ctrl=0 でも 0% と 10% の差が残る → (a) 慣性が主因
#     ctrl=0 で差が消える             → (c) 制御コストが主因
#   既存の ctrl=0.2 の run がそのまま対照群になる（`ctrl_cost_coeff` 以外は完全同一）。
#
# usage: launch_mechanism_probe.sh {smoke|c0|all}
set -u
cd /userdir/StackelbergPPO

running() { pgrep -f "hydra\.run\.dir=single_run/$1\$" >/dev/null 2>&1; }
finished() { grep -q "training done!" "single_run/$1/log/log_train.txt" 2>/dev/null; }

launch() {  # launch <run名> <ep数> <cfg> <xml> <フラグ...>
  local run=$1 ep=$2 cfg=$3 xml=$4; shift 4
  if running "$run"; then echo "[$(date '+%F %T')] $run は稼働中。スキップ"; return; fi
  if finished "$run"; then echo "[$(date '+%F %T')] $run は完走済み。スキップ"; return; fi
  mkdir -p "single_run/$run"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
    /choreonoid_ws/install/bin/choreonoid --no-window \
    --python scripts/choreonoid_train.py \
    cfg="$cfg" xml_name="$xml" num_threads=4 max_epoch_num="$ep" \
    enable_wandb=false fix_skeleton=true seed=0 +robot_param_scale=1 \
    "$@" hydra.run.dir="single_run/$run" \
    > "single_run/$run/stdout.log" 2>&1 &
  echo "[$(date '+%F %T')] $run launched (PID $!, ${ep}ep, xml=$xml)"
}

# 到達タスクの共通フラグ。既存の pj 系と target 以外は完全一致
REACH_BASE="+reward_specs.use_reach=true +reward_specs.target_x=0.8 \
+reward_specs.target_y=0.0 +reward_specs.target_z=0.15 \
+env_specs.check_init_contact=false"

case "${1:-all}" in
  smoke|all)
    # v3 Reach スモーク: 目標を肩より上へ。−50 を脱するかだけを見る
    launch tripo_v3_reach_smoke 30 pusher_tripo_v3 tripo_arm_v3 \
      +reward_specs.use_reach=true +reward_specs.target_x=0.72 \
      +reward_specs.target_y=0.0 +reward_specs.target_z=0.25 \
      +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false
    ;;&
  c0|all)
    # 制御コストのみ 0.2 → 0.0。他は tripo_pjr199 / tripo_pj_mid と完全一致
    launch tripo_pjr199_c0 200 pusher_gearonly tripo_arm_v2c_pj_rec199 \
      $REACH_BASE +reward_specs.ctrl_cost_coeff=0.0
    launch tripo_pj_mid_c0 200 pusher_gearonly tripo_arm_v2c_pj_mid \
      $REACH_BASE +reward_specs.ctrl_cost_coeff=0.0
    ;;
esac
