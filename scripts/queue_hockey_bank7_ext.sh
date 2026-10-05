#!/bin/bash
# ⭐⭐ hockey_bank7_ext / hockey_bank7_s2_ext — **bank7 の続きを 200 epoch**（系譜 9-217）
#
# code: 03c09b0 (feat/init-pitch-search)
#
# ⭐ 目的: 9-214 で「seed1 は打てる形（全力 2.16 m/s @+32°）なのに方策は 0.82 m/s」、
#   「seed0 は打てない形（パックの脇に先端を置けない）」と分かった。
#   ⭐⭐ **学習の量を倍にしたら何が変わるか**だけを見る。⚠️ 変えるのは epoch の上限だけ（設定は bank7 と同一）。
#   ⭐ epoch_0200.p の重み（Leader・Follower・obs_norm 全部）から再開し、200 → 400 まで回す。
#
# ⭐ 事前登録した読み（**結果を見てから変えない**）
#   seed1（hockey_bank7_s2_ext）— best を 5 話再生し頭 2 話を捨てた 3 話で判定
#   ⭐ ① ゴール（x ≥ 1.55 かつ |y| ≤ 0.35）に入る話が 1 つ以上ある
#        → ⭐⭐⭐ **学習の量が足りなかっただけ。**ホッケーは完遂。4 章に入れるかを決める
#   ⚠️ ② ゴールは 0 だが、パックの最大速度が 1.2 m/s 以上に上がる
#        → ⭐ **強さは学習できる。足りないのは向き・精度**
#   ⛔ ③ パックの最大速度が 1.2 m/s 未満のまま
#        → ⛔ **量の問題ではない。**PBRS（終点の位置だけで決まる）が強い打撃に報いていない可能性。
#          ⚠️ 報酬を変えるのは最後の手段（§5-2 ⑤）。次は「何が強い打撃を妨げているか」を測る
#   seed0（hockey_bank7_ext）— 対照
#   ⭐ ① 形が変わり、判定器の第2層（学習後の形）で打てる構えが 1 個以上になる
#        → Leader は打てない形から抜け出せる
#   ⚠️ ② 0 個のまま
#        → 形は局所解に固着している。⭐ 判定器の「打てない」はこの run について最後まで正しい
#
# ⚠️ 投入後に見るもの（Bug 53 の再発検知）
#   python3 scripts/check_before_conclusion.py hockey_bank7_s2_ext   # ⭐ exec 到達
#   cut -f1 single_run/hockey_bank7_s2_ext/log/arm_init_stage.log | sort | uniq -c
#   ⛔ stdout を grep しない（worker の出力は残らない。9-193）
set -u
Q=single_run/queue_hockey_bank7_ext.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }

launch(){ local RUN=$1 S=$2 SRC=$3
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
    HOCKEY_WALL_RESTITUTION=0.75 HOCKEY_ARM_BLOCK=1 \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_tripo_v3 xml_name=e2e_hockey_easy num_threads=4 max_epoch_num=400 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +env_specs.cube_y_noise=0.25 +env_specs.arm_safe_init=true \
    +env_specs.arm_init_clear_y=0.45 +env_specs.arm_init_pitch_search=true \
    +reward_specs.use_target_reward=true +reward_specs.target_x=1.55 \
    +reward_specs.target_y=0.0 +reward_specs.contact_weight=0 \
    +reward_specs.ctrl_cost_coeff=0.001 +reward_specs.init_contact_penalty=1.0 \
    +restore_dir=single_run/$SRC epoch=200 \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!, seed=$S) ⭐ $SRC の epoch 200 から 400 まで"; sleep 60; }

log "=== bank7 の続き（2 seed）開始"
launch hockey_bank7_s2_ext 1 hockey_bank7_s2
launch hockey_bank7_ext    0 hockey_bank7
log "=== キュー終了"
