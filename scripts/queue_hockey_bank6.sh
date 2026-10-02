#!/bin/bash
# ⭐⭐ hockey_bank6 / _s2 — **初期姿勢を順運動学で決めるようにした後の再投入**（系譜 9-197）
#
# ⭐⭐ 変えたのは **コードだけ**。⚠️ **設定は hockey_bank5 と 1 文字も変えていない。**
#   ⭐ したがって `bank5 → bank6` の差は **`_safe_init_angle` の是正だけ**に帰属する。
#
# ⛔⛔ bank5 で何が起きていたか（9-196 / Bug 52）
#   旧 `_safe_init_angle` は腕を「原点から伸びる長さ R の直線の棒」と見なし
#   `tip_y = R·sin(θ)` で初期角を決めていた。⛔ **その形状は存在しない。**
#   ⛔ `qpos[0]` は **ヨー（z 軸回り）**なのに、式は**ピッチ**を記述していた。
#   ⛔ 結果、先端が x>0 に入った step は **1201 中 0**。腕はリンクの外に居続けた。
#
# ⭐ 是正（9-197）: XML から運動連鎖を読み、**FK を実際に回して**条件を満たすヨー角を探す。
#   ⭐ 形態ごとに再計算する（旧実装は初回値を固定していた）。
#   ⭐ 実形態（bank5 ep70）で検算済み:
#       旧 22.3°  → 先端 (−1.381,−0.566) 対象まで 2.012 m  ⛔ リンクの外
#       新 −180°  → 先端 (+1.492, 0.000) 対象まで 0.942 m  ⭐ 全リンクが内側
#
# ────────────────────────────────────────────────────────────────────
# ⭐⭐ この実験で問うこと — **「腕がパックを打てるか」だけ**（ユーザー指示 2026-10-02）
#
#   ⛔ **得点は問わない。**⚠️ パック可動 x≤1.25 に対しゴール壁の内面は 1.55 で
#      **0.25 m 足りず、原理的に得点できない**（9-196）。⭐ **それは次の段で直す。**
#   ⭐ **ここで見るのは `fwd_cube` が 0 を脱するかどうか。**
#
# ⭐ 前提: **腕がパックを打てること自体は実証済み**である。
#   `hockey_wall` / `_wall2` が **exec_R_eps 29.47 / 29.80 → 正味 1.18 / 1.19 m**（2 seed）。
#   ⚠️ **ただしあの環境ではパックが壁を突き抜けていた**（Bug 39 / 9-63）。
#   ⭐ **つまり「打つ能力」は確認済みで、壊れていたのは常に環境側である。**
#
# ⭐ 事前登録した読み（**結果を見てから変えない**）
#   ⭐ ① `fwd_cube` が非ゼロになる（正味 0.1 m 以上）
#        → ⭐⭐⭐ **初期姿勢が原因だったと確定。**ここから初めてゴールの話に進める
#   ⚠️ ② 動くが 0.1 m 未満
#        → ⭐ 届いてはいる。**打撃の学習**の話（9-190 の読み ②③ へ）
#   ⛔ ③ 全 epoch 0 のまま
#        → ⛔⛔ **初期姿勢の是正では足りない。**⭐ **軌跡を録って先端が x>0 に入るか見る**
#          （bank5 は 1201 中 0 だった。これが 0 でなくなっていれば是正自体は効いている）
#   ⛔ ④ 例外「FK で条件を満たすヨー角が一つも無い」で止まる
#        → ⭐ **これは結果である。**「この設計空間ではこの環境の初期姿勢が作れない」
#
# ⚠️⚠️ **epoch 0 で bank5 と差が出ないのは正常である。**
#   ⭐ **epoch 0 の形態は設計値に近く、骨がほぼ真上を向いているのでヨーは効かない**
#     （FK 探索は 1441/1441 すべてを可と答える）。
#   ⛔ **bank5 が壊れたのは co-design が骨を −x へ倒した後**（ep70 で先端 x=−0.935）。
#   ⭐ **差が出るとすれば ep30〜70 あたりからである。**早合点しないこと。
#   実測: bank5 ep0 は exec_R_eps −1.00 / bank6 ep0 も −1.00（同じ）
#
# ⚠️⚠️ **投入後 10 分で `log/log_train.txt` に epoch が出ているか必ず確かめること。**
#   ⛔ **0 バイトのままならハング**（Bug 51 の型）。即座に止める。
# ⛔⛔ **`[arm_safe_init]` は stdout.log には出ない。**⚠️ **worker の stdout は捕捉されない**
#   （bank5 でも 0 件だった。trace は単一プロセスなので出ていただけ）。
#   ⭐ **新実装が効いているかは `fwd_cube` と軌跡で見る。**
#   ⭐ 初期姿勢だけを単体で確かめたいなら `scripts/selfcheck_fk_init.py`（Choreonoid 不要）。
#
# ⚠️ **完走後に `check_before_conclusion.py`（6 項目）と `plot_run.py`。目視記録へ 1 行。**
# ⚠️ **`_s2` を飛ばさない**（9-188 の癖）。
# ────────────────────────────────────────────────────────────────────
set -u
Q=single_run/queue_hockey_bank6.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
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
  log "$RUN launched (PID $!, seed=$S) ⭐ 設定は bank5 と同一。コードだけ是正"; sleep 60; }

log "=== FK 初期姿勢で再投入（2 seed）開始"
launch hockey_bank6    0
launch hockey_bank6_s2 1
log "=== キュー終了"
