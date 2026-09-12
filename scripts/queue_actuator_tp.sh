#!/usr/bin/env bash
# 2026-09-13: 現実的な物理（反射慣性 ∝ 減速比²）で Target-Pusher を回す（実験系譜 9-81 / 9-82 段1a）。
#
# **なぜやるか**: 9-82 で、設計空間に「45 g の腕に 400 N·m の関節」が入っていることが
#   数字で出た（腕を水平に保つトルクの 1800 倍）。段1a として反射慣性を減速比の 2 乗に
#   連動させた。Pusher は `e2e_a1v_actuator_pusher` で走行中。**Target-Pusher には
#   e2e_a1v の対照が存在しない**ので、対照と実験の 2 本を組む。
#
# **変数は 1 つ**: cfg の actuator_params.gear に 3 行（armature_ref_gear / base / floor）。
#   XML・報酬・seed・epoch は 2 本とも同一。
#
#   | タスク | 対照 | 実験 |
#   |---|---|---|
#   | Pusher | e2e_a1v_pusher（完走済み） | e2e_a1v_actuator_pusher（走行中） |
#   | Target-Pusher | **本スクリプトの 1 本目** | **本スクリプトの 2 本目** |
#
# 停止条件（先に決めておく。9-50 の教訓）:
#   ① gear が上限より内側へ収束 → 境界解は「代償が無かったこと」の産物。修論の弱点に答えが出る
#   ② gear が上限に張り付いたまま → 代償があっても高 gear が最適。境界解は物理の要請
#   ③ 学習が壊れる（発散・NaN）→ timestep 0.01 では低 gear 側を支えられない。floor を上げる
#   ⚠️ **どれに転んでも書ける。性能を上げるための実験ではない**（CLAUDE.md §5-2 ⑤-2）。
#   ⚠️ **見るのは gear の位置であってスコアではない。** 物理が違うので既存 110 run と
#      スコアは比べられない（9-53・9-65 と同じ規律）。
#
# GPU は 2 本まで（4 本同時は T_update が 2.4 倍。CLAUDE.md §6）。**空くまで待つ。**
#
# 起動: nohup bash scripts/queue_actuator_tp.sh > /dev/null 2>&1 & disown
# 進捗: single_run/queue_actuator_tp.log
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_actuator_tp.log
MAXJOBS=2

log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
njobs(){ ps -eo args 2>/dev/null | grep -cE "[c]horeonoid_train\.py" || true; }
done_(){ grep -q "training done!" "single_run/$1/log/log_train.txt" 2>/dev/null; }
alive_(){ ps -eo args 2>/dev/null | grep -qE "[c]horeonoid_train\.py.*hydra\.run\.dir=single_run/$1( |\$)"; }

wait_slot(){
  while [ "$(njobs)" -ge "$MAXJOBS" ]; do sleep 120; done
}

launch(){  # $1=run名 $2=cfg
  if done_ "$1"; then log "$1 は完走済み。スキップ"; return; fi
  if alive_ "$1"; then log "$1 は稼働中。スキップ"; return; fi
  if [ -s "single_run/$1/log/log_train.txt" ]; then log "$1 は既存。スキップ（手で確認）"; return; fi
  wait_slot
  mkdir -p "single_run/$1"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg="$2" xml_name=e2e_a1v num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=0 +robot_param_scale=1 \
    +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true \
    hydra.run.dir="single_run/$1" >> "single_run/$1/stdout.log" 2>&1 &
  log "$1 launched (PID $!, cfg=$2)"
  sleep 60
}

log "=== キュー開始（Target-Pusher の対照と実験）"
launch e2e_a1v_tp                 target_pusher_tripo_v3
launch e2e_a1v_actuator_tp        target_pusher_tripo_v3_actuator
log "=== 全部投入した。完走は各 run の log を見る"
