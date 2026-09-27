#!/usr/bin/env bash
# 2026-09-27: ⭐⭐ **反射 ＋ 腕の壁貫通ブロックでホッケーを学習**（実験系譜 9-185 予定）。
#
# ⭐ 9-181 の結果:
#   ⭐⭐ seed1 は**事前登録の読み②**（パックが側壁で 281 回跳ね返り y=+0.48↔−0.45 を往復）。
#       **腕は壁の中に 0 標本**で、壁を使った経路を実際に探索していた。⛔ だが x=1.18 で未到達。
#   ⛔⛔ seed0 は**腕が壁の中に 1676 標本**。**壁越しに押せるなら反射を使う必要が無い**ので、
#       この seed は仮説を検証していない ＝ ②と③を区別できていなかった。
#
# ⭐ 9-184 で腕の壁貫通を塞いだ（**貫通 730 → 46 標本、94 % 削減**）。
#   ⭐ cube の最大 x が 1.800（可動限界）→ 1.492 ＝ **壁越しに押し込めなくなった**ことの裏づけ。
#
# ⚠️ **変数は 1 つだけ。**`HOCKEY_ARM_BLOCK=1` の有無のみ。
#   ⭐ **対照は `hockey_bank` / `_s2`**（9-181。反射あり・ブロックなし）。他のフラグは同一。
#
# 読み方（**結果を見る前に決める**。§5-2 ⑤-2）:
#   ⭐ ① **ゴールが入る**（x=1.55 を |y|<0.35 で通過する話がある）
#        → ⭐⭐⭐ **反射＋貫通ブロックでホッケーが成立。**判定器が「できる」と言った形態で実際にできた
#   ⭐ ② **入らないが、2 seed とも跳ね返りを使う**（seed0 も seed1 のような挙動になる）
#        → ⭐⭐ **交絡が消えて初めて反射の効果が測れた。**
#          **判定器は正しく、制御が限界**（9-161・5.6.2 と同型）。⚠️ **これも修論に書ける**
#   ⛔ ③ **9-181 と変わらない**（跳ね返りが増えず、到達も伸びない）
#        → ⛔ **貫通は交絡ではなかった。**反射そのものが方策の役に立っていない
#
# ⚠️ **報酬だけで判定しない。**必ず再生して
#   ①壁で跳ね返ったか ②ゴール口を通過したか ③**腕が壁の中に入っていないか** を軌跡で確かめる。
# ⚠️ **固定 body があるタスクは再生ごとに揺れる**（9-138）。1 エピソードで判定しない。
#
# 起動: nohup bash scripts/queue_hockey_bank2.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_hockey_bank2.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
njobs(){ ps -eo comm,args 2>/dev/null | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
launch(){ local RUN=$1 S=$2
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  # ⚠️ **環境変数は env の引数として渡す**（Bug 49。展開結果は代入にならない）
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
    HOCKEY_WALL_RESTITUTION=0.75 HOCKEY_ARM_BLOCK=1 \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_tripo_v3 xml_name=e2e_hockey_easy num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +env_specs.cube_y_noise=0.25 +env_specs.arm_safe_init=true \
    +reward_specs.use_target_reward=true +reward_specs.target_x=1.55 \
    +reward_specs.target_y=0.0 +reward_specs.contact_weight=0 \
    +reward_specs.ctrl_cost_coeff=0.001 +reward_specs.init_contact_penalty=1.0 \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!, seed=$S, e=0.75, arm_block=1)"; sleep 60; }

log "=== 反射＋貫通ブロック（2 seed）開始"
launch hockey_bank2    0
launch hockey_bank2_s2 1
log "=== キュー終了"
