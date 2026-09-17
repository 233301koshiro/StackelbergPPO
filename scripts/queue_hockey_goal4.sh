#!/usr/bin/env bash
# 2026-09-14: 壁が**実在する**ホッケー（実験系譜 9-99）。
#
# ⭐⭐ **hockey_goal3 までとの決定的な差: 壁が Choreonoid のシミュレーションに存在する。**
#   9-98 で判明したとおり、変換器は `worldbody/body` しか読まないため、
#   **worldbody 直下の geom として置かれていた壁は 8 回の試行すべてで存在しなかった。**
#   `--wall-as-body` で壁を 1 枚ずつ <body> に入れ、変換器に拾わせた（e2e_hockey_wall3）。
#     変換結果: body 7 個（腕・パック・壁 5 枚）、壁はすべて joint_type: fixed
#
# ⚠️ **これは 9 回目の環境いじりではない。** 前 8 回は**存在しない壁**を直していた。
#   今回初めて壁が物理として存在する条件で回す。
#
# hockey_goal3 との差は **XML だけ**（報酬・epoch・seed・形態は同一）。
#
# 読み方（**結果を見る前に決める**。§5-2 ⑤-2）:
#   ① パックがゴール口を通る（x=1.55 で |y|<=0.15）
#        → ✅ 成立。**反射したかを軌跡から別途記録する**（通ることと反射は別問題）
#   ② 通らないが**壁で止まる**（パックの |y| が 0.45 前後で頭打ち・x が 1.50 前後で停止）
#        → 🟡 **壁は効いている。残るのは探索。** 9-65 の測定では初期 y=±0.2 以内は反射必須で
#          難度が高い。ここで止まるのは判定器の出力であって壊れてはいない
#   ③ また x=1.80 / |y|=0.55（可動域の端）へ張り付く
#        → ❌ **壁がまだ効いていない。** その場合は A 案が不成立なので B を検討する
#
# ⚠️ **判定は再生して軌跡を図にする。** かつ **9-96 の規律どおり、腕とパックの両方の極値を先に出す。**
# ⚠️ **壁の検証は `probe_wall_choreonoid.py`（Choreonoid 経由）で行う。MuJoCo で測らない。**
#
# 起動: nohup bash scripts/queue_hockey_goal4.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_hockey_goal4.log
RUN=hockey_goal4
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
# ⚠️ **自分のシェルのコマンドラインが同じ文字列を含むと誤って数える**（Bug 19 と同型）。
#   ⭐ **実体のパスとの AND** にすると、シェルのコマンドラインには一致しない。
njobs(){ ps -eo args 2>/dev/null | awk '/choreonoid_train\.py/ && /install\/bin\/choreonoid/' | wc -l; }

log "=== キュー開始。GPU が 2 本空くまで待つ"
while [ "$(njobs)" -ge 2 ]; do sleep 180; done
mkdir -p "single_run/$RUN"
nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
  --no-window --python scripts/choreonoid_train.py \
  cfg=pusher_tripo_v3 xml_name=e2e_hockey_wall3 num_threads=4 max_epoch_num=200 \
  enable_wandb=false fix_skeleton=true seed=0 +robot_param_scale=1 \
  +env_specs.cube_y_noise=0.4 +env_specs.arm_safe_init=true \
  +reward_specs.use_target_reward=true +reward_specs.target_x=1.55 +reward_specs.target_y=0.0 \
  +reward_specs.contact_weight=0 +reward_specs.ctrl_cost_coeff=0.001 \
  +reward_specs.init_contact_penalty=1.0 \
  hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
log "$RUN launched (PID $!)"
