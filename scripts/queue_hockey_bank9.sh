#!/bin/bash
# ⭐⭐ hockey_bank9 / _s2 — **制御コストを外す**（系譜 9-226。9-225 の仮説の介入実験）
#
# code: a31bce5 (main)
#
# ⭐ bank8 との差は **ctrl_cost_coeff 0.001 → 0 だけ**（他の引数・環境変数は bank8 と 1 文字も変えない）。
# ⚠️ 自己干渉は bank8 と同じく**無効のまま**（有効にすると変数が 2 つになる。bank8 はパックに触れていないので
#   9-224 の結論には効かないが、この run で打撃が出たら check_self_penetration.py で量を必ず報告する）。
# ⭐ 9-225: 探索ではパックに 20〜80 % の話で当たるが、パックの報酬（初期 +0.02/話）より制御コスト（−0.25/話）が
#   約 12 倍大きく、方策は「力を出さない」方へ進んで平均の行動は動かなくなった。**制御コストが主因かを確かめる。**
# ⏳ ③（queue_selfcol.sh）が全部投入し終わってから、空いた枠で走る（ユーザー判断 2026-10-06: ③ の後ろに並べる）。
#
# ⭐ 事前登録した読み（**結果を見てから変えない**）
#   判定: best を 5 話（頭 2 話を捨てて 3 話）＝評価の行動、＋ TRACE_STOCHASTIC=1 で ep200 を 10 話＝探索の行動
#   ⭐ ① 評価の行動で、**置いた瞬間の隙間 ≥ 10 cm** のパックを 0.1 m 以上動かす話が、どちらかの seed で 1 つ以上ある
#        → ⭐⭐ 制御コストが主因だった。次はゴール（入れば修論に入れる。ミーティング資料「修論での扱い」）
#   ⚠️ ② 評価の行動では動かないが、探索で当たった話の報酬（Δφ）の平均が ep200 で＋
#        → 信号は出たが方策が収束していない。epoch を延ばすかを決める
#   ⛔ ③ 評価の行動で動かず、探索で当たった話の Δφ 平均も ≤ 0
#        → ⛔ 制御コストは主因ではない。当たり方（掃き当てで遠ざける）の側を調べる
#   ⚠️ 共通: 制御コストが無いと全力で振り回しうる。腕の壁貫通（ブロックの発火数）と形の伸び（ボーン長）を併せて見る
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_hockey_bank9.log
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
    +env_specs.arm_init_after_noise=true +env_specs.init_contact_all_links=true \
    +reward_specs.use_target_reward=true +reward_specs.target_x=1.55 \
    +reward_specs.target_y=0.0 +reward_specs.contact_weight=0 \
    +reward_specs.ctrl_cost_coeff=0 +reward_specs.init_contact_penalty=1.0 \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!, seed=$S) ⭐ bank8 との差は ctrl_cost_coeff 0.001 → 0 のみ"; sleep 60; }

log "=== 制御コストを外したホッケー（2 seed）。③ の投入完了を待つ"
# ⭐ ③ が最後の 2 本を投入し終えるまで待つ（その後は空いた枠から順に入る）
until grep -q "=== キュー終了" single_run/queue_selfcol.log 2>/dev/null; do sleep 600; done
log "③ の投入完了を確認"
launch hockey_bank9    0
launch hockey_bank9_s2 1
log "=== キュー終了"
