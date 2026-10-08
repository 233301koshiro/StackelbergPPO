#!/bin/bash
# ⭐⭐ ホッケー（制御コスト 0）を「自己干渉あり＋自分のリンクの門」で取り直す（系譜 9-230）
#
# code: main（9-229 の bank9 と同じ起動行＋自己干渉の 2 つ）
#
# ⭐ ユーザー判断（2026-10-09）: ホッケーを修論に入れる。修論は自己干渉を最初から考慮した設定で書くので（9-227）、
#   9-229（自己干渉なしでゴール）をこの設定で取り直す。順番は A 案＝修論用キュー（9-228）が全部投入し終わってから。
# ⭐ bank9 との差: `CNOID_SELF_COLLISION=1` と `+env_specs.init_self_contact=true` だけ。
#   ⚠️ `init_contact_penalty=1.0` は bank9 のまま。正直な話の報酬は −0.3〜+0.5 程度（PBRS、制御コスト 0）なので、−1 は
#     それより悪い＝門が得にならない（Bug 9 の原則を満たす）。
#
# ⭐ 事前登録した読み（**結果を見てから変えない**）— best を 5 話（頭 2 話を捨てて 3 話）＝評価の行動
#   ⭐ ① どちらかの seed で、置いた瞬間の隙間 ≥ 10 cm のパックが**ゴールに入る**話がある → 自己干渉ありでもゴールは成立。修論にこの値で入れる
#   ⚠️ ② ゴールは無いが、離れたパックを 0.1 m 以上動かす話がある → 打撃は成立。ゴールは主張しない
#   ⛔ ③ どの話でも動かない → 9-229 のゴールはリンクどうしのすり抜けに頼っていた可能性。原因を調べる
#   共通: 再生は学習と同じ環境変数（HOCKEY_* と CNOID_SELF_COLLISION）で行う（9-229 の訂正）。門の発火・関節の固着・壁の内側かを見る
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_hockey_bank9_scg.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
launch(){ local RUN=$1 S=$2
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 CNOID_SELF_COLLISION=1 \
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
    +env_specs.init_self_contact=true \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!, seed=$S) ⭐ bank9 との差は自己干渉あり＋init_self_contact のみ"; sleep 60; }
log "=== ホッケー（制御コスト 0）を自己干渉ありで取り直す（2 seed）。修論用キュー（9-228）の投入完了を待つ"
until grep -q "=== キュー終了" single_run/queue_thesis_selfcol.log 2>/dev/null; do sleep 600; done
log "9-228 の投入完了を確認"
launch hockey_bank9_scg    0
launch hockey_bank9_scg_s2 1
log "=== キュー終了"
