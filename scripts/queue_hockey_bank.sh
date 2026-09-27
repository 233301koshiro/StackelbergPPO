#!/usr/bin/env bash
# 2026-09-27: ⭐⭐ **反射ありでホッケーを学習する**（実験系譜 9-181 予定）。
#
# ⭐ ここまでに揃ったもの:
#   9-176: 壁の反射を env の Python 経路に実装。**比 0.750（期待 0.75）で検証済み**
#   9-177: **反射で入る撃ち方が 0 → 7 通り**（Choreonoid の実走で計数）
#   9-179: ⭐⭐ **9-135 の「届かない」の真因が判明** — 摩擦でも飛距離不足でもなく、
#          **ゴール口（|y|<0.35）の 8.7 cm 外側 y=0.437 で横の壁 `wall_goal_p` に当たって停止**していた。
#          ⛔ 9-135 の「狙いも合う」「摩擦で 6〜9 cm 手前」は**両方とも誤り**だった。
#   9-178: 判定器が「狙える向きに当てられるか」を判定できる。**この形態は通過する**
#
# ⚠️ **変数は 1 つだけ。**`HOCKEY_WALL_RESTITUTION=0.75` の有無のみ。
#   ⭐ **対照は既存の `hockey_easy` / `hockey_easy_s2`**（9-135）をそのまま使う。
#   他のフラグは `.hydra/overrides.yaml` と文字どおり同一にしてある。
#
# ⚠️ **甘々版（ゴール口 0.35・初期 y ±0.25）から始めるのは対症療法ではない**（CLAUDE.md §5-2 ①-2）。
#   ⭐ **ここを出発点にして、後で 1 つずつ本来の条件へ戻す**のが計画である。
#   ⛔ 9-121 はゴール口と初期 y を**同時に**緩めており未分離なので、戻すときは 1 つずつ。
#
# 読み方（**結果を見る前に決める**。§5-2 ⑤-2）:
#   ⭐ ① **ゴールが入る**（x=1.55 を |y|<0.35 で通過する話がある）
#        → ⭐⭐⭐ **反射の導入でホッケーが成立。**判定器が「できる」と言った形態で実際にできた
#   ⭐ ② **入らないが、壁で跳ね返って別経路を試す挙動が出る**
#        → ⭐ **反射は効いているが制御が足りない。**9-161・5.6.2 と同型
#          （**判定器は正しく、制御が限界**）。⚠️ **これも修論に書ける結果である**
#   ⛔ ③ **9-135 と変わらない**（壁で跳ね返らない／跳ね返っても軌道が変わらない）
#        → ⛔ **反射は学習の役に立っていない。**バンクショットを方策が見つけられない
#
# ⚠️ **報酬だけで判定しない。**必ず再生して
#   ①壁で跳ね返ったか ②ゴール口を通過したか を**軌跡で確かめる**（9-135 で 2 回誤読した）。
# ⚠️ **固定 body があるタスクは再生ごとに揺れる**（9-138。118/64/308 mm）。**1 エピソードで判定しない。**
#
# 起動: nohup bash scripts/queue_hockey_bank.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_hockey_bank.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
njobs(){ ps -eo comm,args 2>/dev/null | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
launch(){ local RUN=$1 S=$2
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 HOCKEY_WALL_RESTITUTION=0.75 \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_tripo_v3 xml_name=e2e_hockey_easy num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +env_specs.cube_y_noise=0.25 +env_specs.arm_safe_init=true \
    +reward_specs.use_target_reward=true +reward_specs.target_x=1.55 \
    +reward_specs.target_y=0.0 +reward_specs.contact_weight=0 \
    +reward_specs.ctrl_cost_coeff=0.001 +reward_specs.init_contact_penalty=1.0 \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!, seed=$S, e=0.75)"; sleep 60; }

log "=== 反射ありホッケー（2 seed）開始"
launch hockey_bank    0
launch hockey_bank_s2 1
log "=== キュー終了"
