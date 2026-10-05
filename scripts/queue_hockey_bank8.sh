#!/bin/bash
# ⭐⭐ hockey_bank8 / _s2 — **置き方の重なりを塞いで最初から 200 epoch**（系譜 9-219 / Bug 55）
#
# code: df54809 (feat/init-pitch-search)
#
# ⭐ bank7 との差は 2 スイッチだけ:
#   arm_init_after_noise=true   — 初期姿勢の探索を毎話、揺らした後の位置で行う
#   init_contact_all_links=true — 初期接触の門が全リンクを見る（安全網）
# ⛔ bank7 でパックが動いた話はすべて置いた瞬間の重なりからの押し出しだった（9-218）。
# ⭐ 投入前の確認: 門の発火 0/12 話・12/12 完走（9-219）。
#
# ⭐ 事前登録した読み（**結果を見てから変えない**）— best を 5 話再生し頭 2 話を捨てた 3 話＋ep_init の隙間で判定
#   ⭐ ① **離れて置かれた（隙間 ≥ 10 cm の）パックを取りに行って動かす話**が、どちらかの seed で 1 つ以上ある
#        → ⭐⭐⭐ 抜け道を塞いでも方策は本当の打撃を学習する。次はゴール
#   ⚠️ ② パックが動く話はあるが、すべて隙間 < 10 cm
#        → ⚠️ 置き方がまだ近すぎる。打撃と区別できない（門の発火率も確認）
#   ⛔ ③ どの話でもパックが動かない
#        → ⛔⛔ bank7 の「動いた」はすべて抜け道だった。ホッケーは「打つ」の手前に戻る
#   ⛔ ④ 門の発火率が高い（ログで exec_R_eps に −1 付近が多発 / 再生で init_gate が出る）
#        → 探索が漏らしている。9-219 の確認と食い違うので原因を調べる
#
# ⚠️ 投入後に見るもの
#   python3 scripts/check_before_conclusion.py hockey_bank8      # ⭐ exec 到達
#   cut -f1 single_run/hockey_bank8/log/arm_init_stage.log | sort | uniq -c   # 段の内訳（stdout は見ない。9-193）
# ⛔⛔ 30 epoch では判定できない。bank6_s2 で動き出したのは ep116 以降
set -u
Q=single_run/queue_hockey_bank8.log
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
    +reward_specs.ctrl_cost_coeff=0.001 +reward_specs.init_contact_penalty=1.0 \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!, seed=$S) ⭐ bank7 との差は arm_init_after_noise と init_contact_all_links のみ"; sleep 60; }

log "=== 置き方の重なりを塞いで最初から（2 seed）開始"
launch hockey_bank8    0
launch hockey_bank8_s2 1
log "=== キュー終了"
