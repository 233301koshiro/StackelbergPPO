#!/bin/bash
# ⭐⭐ hockey_bank7 / _s2 — **初期姿勢の探索にピッチを入れた後の再投入**（系譜 9-208）
#
# code: 51cb017 (feat/init-pitch-search)
#
# ⭐⭐ 変えたのは **初期姿勢の探索だけ**。⚠️ 他の設定は hockey_bank6 と 1 文字も同じ。
#   `+env_specs.arm_init_pitch_search=true` を足しただけ。
#
# ⛔⛔ bank6 で何が残ったか（9-207）
#   ⭐ `_s2` は 3 話中 1 話でパックが **0.909 m** 動いた（読み ① 達成）
#   ⛔ だが **3 話中 1 話**・ログでは **5/200 epoch**・**ゴールには一度も入らず**
#   ⚠️ **seed0 は腕を下ろせていない**（高さ差 0.269 m 対 seed1 の 0.040 m）
#   ⭐⭐ **seed 間の差が「下ろせたかどうか」に集約している**ので、そこを初期姿勢で揃える
#
# ⭐ 是正の中身（9-208）
#   ⛔ 旧: FK 探索は **ヨーしか変えない**。高さはピッチで決まるので先端が下りてこない（9-203）
#   ⭐ 新: **ピッチ（第2関節）も探索に入れる**。⚠️ ヨーの刻みは 1441 → 361 に粗くして計算量を保つ
#   ⛔⛔ **投入前の検算で床下 z=−0.303 を踏んだ**ので、⭐ **床のクリアランス制約を足した**
#
# ⭐ 投入前の検算（最適化後の実形態で）
#   | 形態 | ⛔ ヨーのみ | ⭐ ヨー＋ピッチ＋床 |
#   |---|---|---|
#   | bank6    | 高さ差 0.274 m | ⭐ **0.112 m** |
#   | bank6_s2 | 高さ差 0.378 m | ⭐ **0.176 m** |
#
# ⭐ 事前登録した読み（**結果を見てから変えない**）
#   ⭐ ① **2 seed とも**パックが動く（正味 0.1 m 以上の話が出る）
#        → ⭐⭐⭐ **seed 間の差は初期姿勢だった**と確定。ゴールの話へ進める
#   ⚠️ ② 片方だけ動く（bank6 と同じ）
#        → ⭐ **初期姿勢では揃わない。**seed を増やして分布を見る（案 a）
#   ⛔ ③ 両方とも動かない
#        → ⛔⛔ **初期姿勢の高さは原因ではなかった。**読みを取り下げ、打撃の学習へ移る
#   ⛔ ④ 例外「FK で条件を満たす角が一つも無い」
#        → ⭐ **床の制約が厳しすぎる。**`arm_init_floor_clear` を下げる
#
# ⛔⛔ **2026-10-03: 1 度目の投入を ep1〜2 で打ち切った**（`_aborted_hockey_bank7_slow`）。
#   ⛔ **FK 探索が逐次で 752.5 ms かかり、ETA が 2 日 5 時間になった**（9-209）。
#   ⭐ **ベクトル化して 19.9 ms。**⭐ **ETA は bank6 並み（約 21 h）に戻る見込み。**
#
# ⚠️⚠️ **投入後 15 分で `log/log_train.txt` に epoch が出ているか確かめる**（Bug 51 の型）
# ⚠️ **起動は遅い。**1 度目は 15 分で 0 バイトだったが、その後動き出した。⭐ **25 分見る**
# ⚠️ **`[arm_safe_init]` は stdout に出ない**（worker の出力は捕捉されない。9-193 の訂正）
# ⚠️ **完走後に `check_before_conclusion.py`（6 項目）と `plot_run.py`。目視記録へ 1 行。**
# ⛔⛔ **30 epoch では判定できない。**bank6_s2 で動いたのは **ep116 以降**（9-207）
set -u
Q=single_run/queue_hockey_bank7.log
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
    +env_specs.arm_init_clear_y=0.45 +env_specs.arm_init_pitch_search=true \
    +reward_specs.use_target_reward=true +reward_specs.target_x=1.55 \
    +reward_specs.target_y=0.0 +reward_specs.contact_weight=0 \
    +reward_specs.ctrl_cost_coeff=0.001 +reward_specs.init_contact_penalty=1.0 \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!, seed=$S) ⭐ bank6 との差は arm_init_pitch_search のみ"; sleep 60; }

log "=== 初期姿勢にピッチ探索を入れて再投入（2 seed）開始"
launch hockey_bank7    0
launch hockey_bank7_s2 1
log "=== キュー終了"
