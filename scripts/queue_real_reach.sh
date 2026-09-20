#!/usr/bin/env bash
# 2026-09-21: ⭐⭐ **実寸条件でタスク識別が残るか**（実験系譜 9-156）。
#
# ⛔⛔ **現実的な物理（密度 189.9・ギア 25〜150）では Pusher しか回しておらず、
#   本研究の中心的主張「タスクごとに異なる形態へ収束する」を一度も確かめていない。**
# ⚠️ **「学習が成立する」と「判定器として機能する」は別である**（§5-2 ⑤-2）。
#   これまで確かめたのは前者だけ。
#
# ⭐ 条件は `e2e_a1v_real_pusher` と完全に同じ物理（cfg=pusher_tripo_v3_real・xml=e2e_a1v_real）で、
#   **タスクだけを Reach に変える。**⭐ **変数はタスクのみ。**
#
# ⭐ 対照の署名（実測済み。探索範囲 size [0.030, 0.100]）:
#
#   条件          L1      L2      L3      L4(先端)
#   旧 Reach     0.0477  0.0404  0.0300  0.0487
#   旧 Pusher    0.1000  0.0350  0.0300  0.0660
#   ⭐ 実寸 Pusher 0.0930  0.0300  0.0300  **0.1000（上限）**
#
#   ⭐⭐ **実寸 Pusher は先端が上限に張り付いた。**旧 Pusher（0.0660・内点）より強い署名である。
#     トルクが限られると、先端に質量を集めて運動量を稼ぐしかないためと読める。
#
# 読み方（**結果を見る前に決める**。§5-2 ⑤-2）:
#   ⭐ ① **先端リンクが実寸 Pusher の 0.100 より明確に細い**（0.05 以下）
#        → ⭐⭐ **識別は物理の現実性に依存しない。研究の主張が実在条件でも立つ**
#   ⛔ ② **先端が同程度に太い**（0.08 以上）
#        → ⛔ **従来の識別は非現実的な物理（軽すぎる腕・強すぎるアクチュエータ）に依存していた**
#   ⚠️ ③ 中間（0.05〜0.08）または L1 側で逆転
#        → ⚠️ **識別はするが機序が違う。**形態を実測して読み直す
#
# ⚠️ **1 seed では判定しない。2 seed で帯を作る。**
# ⚠️ **結論前に `check_before_conclusion.py`（6 項目）と `plot_run.py` を回し、
#   目視記録.md へ 1 行書く**（9-155）。
#
# 起動: nohup bash scripts/queue_real_reach.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_real_reach.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
njobs(){ ps -eo comm,args 2>/dev/null | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
launch(){ local RUN=$1 S=$2
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_tripo_v3_real xml_name=e2e_a1v_real num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 \
    +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 \
    +env_specs.check_init_contact=false +reward_specs.init_contact_penalty=1900 \
    +env_specs.arm_safe_init=true \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)"; sleep 60; }

log "=== キュー開始（実寸条件でタスク識別が残るか・Reach 2 seed）"
launch e2e_a1v_real_reach 0
launch e2e_a1v_real_reach_s2 1
log "=== キュー終了"
