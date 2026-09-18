#!/usr/bin/env bash
# 2026-09-19: **一度「できる」を作るための甘い版**ホッケー（実験系譜 9-121）。
#
# ⚠️⚠️ **変数を 2 つ同時に動かす。** 通常は §5-2 ② で禁じているが、
#   **ユーザーの明示の判断**（「大幅に甘々にすることも大事」2026-09-19）による。
#   ⭐ **狙いは「解が存在する設定でなら学習できるか」の 1 点。**
#   ⚠️ **どちらの変更が効いたかは分離できない。分離が要るなら後で 1 つずつ戻す。**
#
# 変更点（9-103 で「4 分の 3 で解が存在しない」と判明したため）:
#   ① ゴール口の半幅  0.15 → **0.35**（`--easy`）
#   ② パックの初期 y  ±0.40 → **±0.25**（`cube_y_noise`）
#
# ⭐ **幾何で先に確かめた**（学習を回す前に決めた）。初期 y ごとの直線解の本数:
#     y=0.00 → 2 本 / 0.10 → 3 本 / 0.20 → 6 本 / 0.25 → 6 本
#   **全範囲で解が存在する。**（従来は y≤0.10 で 0 本だった）
#
# 読み方（**結果を見る前に決める**。§5-2 ⑤-2）:
#   ① ゴールを通る → ✅ **デモ成立。**⭐ 反射したかを軌跡から別途記録する
#   ② 通らないが壁で止まる → 🟡 **解はあるのに学習が届かない。**探索の問題として記録
#   ③ 壁を貫通する → ❌ 物理が壊れている。⚠️ 9-118 の形状型を疑う
#
# ⚠️ **結論を書く前に `check_before_conclusion.py` を回す**（9-114）。
# ⚠️ **判定は再生して軌跡を図にする。**動く物体すべての極値を先に出す（9-96）。
#
# 起動: nohup bash scripts/queue_hockey_easy.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_hockey_easy.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
njobs(){ ps -eo comm,args 2>/dev/null | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
launch(){ local RUN=$1 S=$2
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_tripo_v3 xml_name=e2e_hockey_easy num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +env_specs.cube_y_noise=0.25 +env_specs.arm_safe_init=true \
    +reward_specs.use_target_reward=true +reward_specs.target_x=1.55 +reward_specs.target_y=0.0 \
    +reward_specs.contact_weight=0 +reward_specs.ctrl_cost_coeff=0.001 \
    +reward_specs.init_contact_penalty=1.0 \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)"; sleep 60; }

log "=== キュー開始（甘い版・2 seed）"
launch hockey_easy 0
launch hockey_easy_s2 1
log "=== キュー終了"
