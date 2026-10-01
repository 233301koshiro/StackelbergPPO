#!/usr/bin/env bash
# ⭐⭐⭐ ホッケー第4版 — **初期姿勢をリンクの内側から始める**（系譜 9-193、2026-10-01）。
#
# ⛔⛔ **9-192 で配置そのものが壊れていると分かった。**
#   `arm_safe_init` は qpos[0]=π/2 ＝ 腕を +y へ向けるが、リーチ 1.0425 m では
#   **先端が側壁の外面 0.90 の 14 cm 外**から始まる。
#   ⛔ **パックへ届くには側壁を通り抜けるしかない配置**だった。
#   → 貫通を許せば「ホッケーでない」、塞げば「永久に届かない」（`fwd_cube` 全 epoch 0）。
#
# ⭐⭐ **是正**: `arm_init_clear_y` を与えると**リーチから初期肩角を計算する**。
#   上限 = arcsin(clear_y / R)    … 先端が壁の内側に収まる
#   下限 = arcsin(need / d_obj)   … 対象と初期接触しない（need = 半幅＋腕半径＋margin）
#   ⭐ 実測で窓は **[15.8°, 25.6°] → 初期肩角 20.7°**。
#   ⚠️ **窓が無ければ例外で止まる。**⛔ 黙って 90° へ落とさない（§5-2 ⑤-3-2）。
#   ⭐ **既定（clear_y なし）は従来どおり 90° なので、他の全実験は無影響。**
#
# ⚠️⚠️ **起動したら stdout.log の先頭を必ず見ること。**
#   `[arm_safe_init] リーチ … → 窓 [..] → ⭐ 初期肩角 20.7°` が出る。
#   ⛔ **出ていなければ clear_y が届いていない。**即座に止める。
#
# ⚠️ **リーチが約 1.65 m を超えると窓が消える。**cfg はリンク長を最適化するので、
#   ⭐ **例外で止まったら「この設計空間ではホッケーの初期姿勢が作れない」という結果である。**
#
# ⭐ 対照は **`hockey_bank` / `_s2`**（9-181。反射あり・ブロックなし・初期姿勢は旧）。
# ⚠️ **変数は 2 つ動く**（ブロック＋初期姿勢）。⛔ **9-192 で片方だけでは成立しないと分かったので、
#   分離できない。**⭐ **これは「成立する条件を 1 つ見つける」実験であって、要因分解ではない。**
#
# ⭐ 事前登録した読み（**結果を見てから変えない**）
#   ⭐ ① パックが動く（`fwd_cube` 非ゼロが出る）→ ⭐⭐⭐ **配置が直った。**ここから初めて中身を論じられる
#   ⚠️ ② 動くがゴールに入らない            → ⭐ **バンクショットの学習の話に進める**（9-190 の読み ②③ へ）
#   ⛔ ③ 全 epoch 0 のまま                → ⛔⛔ **初期姿勢以外にも原因がある。**軌跡を録って腕が動いているか見る
#   ⛔ ④ 例外で止まる                     → ⭐ **窓が消えた。**設計空間かリンクの寸法を見直す
#
# ⚠️ **完走後に `check_before_conclusion.py`（6 項目）と `plot_run.py`。目視記録へ 1 行。**
# ⚠️ **`_s2` を飛ばさない**（9-188 の癖）。
#
# 起動: nohup bash scripts/queue_hockey_bank4.sh > /dev/null 2>&1 & disown
# 進捗: single_run/queue_hockey_bank4.log
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_hockey_bank4.log
: > "$LOG"
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }

launch(){ local RUN=$1 S=$2
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
    HOCKEY_WALL_RESTITUTION=0.75 HOCKEY_ARM_BLOCK=1 \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_tripo_v3 xml_name=e2e_hockey_easy num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +env_specs.cube_y_noise=0.25 +env_specs.arm_safe_init=true \
    +env_specs.arm_init_clear_y=0.45 \
    +reward_specs.use_target_reward=true +reward_specs.target_x=1.55 \
    +reward_specs.target_y=0.0 +reward_specs.contact_weight=0 \
    +reward_specs.ctrl_cost_coeff=0.001 +reward_specs.init_contact_penalty=1.0 \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!, seed=$S, clear_y=0.45)"; sleep 60; }

log "=== 初期姿勢を内側から（2 seed）開始"
launch hockey_bank4    0
launch hockey_bank4_s2 1
log "=== キュー終了"
