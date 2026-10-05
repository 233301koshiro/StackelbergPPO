#!/bin/bash
# ⭐⭐ e2e_a1v_obspen_reach_ext / _s2_ext — **障害物 Reach を 200 → 400 epoch へ続ける**（D-5、系譜 9-220）
#
# code: 4d9dbae (main)
#
# ⭐ 目的（デモ用・ユーザー判断 2026-10-05）: **柱を避けて、目標に届く** run を得る。
#   ⛔ 今ある 2 本は片方ずつ: seed0 は柱を完全回避だが 210 mm 手前、seed1 は届くが柱の中を 1187/1201 step（9-153・9-159）。
#   ⭐ 9-161: 罰の係数を変えても解けない（3.8 倍以上で「避けて届かない」が勝つだけ）。両立解は届く姿勢の 24 %（9-144）。
#      **純粋に探索の失敗**。seed1 は ep150 まで seed0 と同水準で、**最後の 50 epoch で急に良くなった**（−220 → −112）。
#   ⭐⭐ **変えるのは学習の量だけ**（epoch_0200.p の重みから 400 まで）。報酬・形態・物理・初期姿勢は元のキュー（queue_obspen.sh）と同一。
#   ⚠️ この構成は arm_init_clear_y を使わない（初期姿勢は旧来の固定角）ので、10-03〜05 の初期姿勢・門の変更は効かない。
#
# ⭐ 事前登録した読み（**結果を見てから変えない**）— best を 5 話再生し頭 2 話を捨てた 3 話で、柱の中の step と目標への最小距離を見る
#   ⭐ ① どちらかの seed で「柱の中 ≤ 12 step（1 %）かつ最小距離 ≤ 10 mm」の話が出る
#        → ⭐⭐ 学習の量で両立解に届いた。**デモに採用**（D-5 完了）
#   ⚠️ ② seed1 が到達を保ったまま柱の中の step が半分以下（≤ 600）に減る
#        → 両立の方向へ向かっている。量をさらに足すかを判断する
#   ⛔ ③ どちらも 200 epoch の解のまま（seed0 は回避・未到達、seed1 は貫通・到達）
#        → 局所解に固着。**量では解けない**。次は seed を増やす（別の変数、1 つずつ）
#   ⛔ ④ 回避も到達も失う
#        → 記録し、デモは今の 2 本（obstacle_avoid / obstacle_reach）を使う
#
# ⚠️ 投入後に見るもの: python3 scripts/check_before_conclusion.py e2e_a1v_obspen_reach_ext
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_obspen_ext.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
launch(){ local RUN=$1 S=$2 SRC=$3
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_tripo_v3 xml_name=e2e_a1v_obs_tall2 num_threads=4 max_epoch_num=400 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 \
    +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 \
    +reward_specs.obstacle_penalty_scale=1.0 \
    +env_specs.check_init_contact=false +reward_specs.init_contact_penalty=1900 \
    +env_specs.arm_safe_init=true \
    +restore_dir=single_run/$SRC epoch=200 \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!, seed=$S) ⭐ $SRC の epoch 200 から 400 まで"; sleep 60; }
log "=== 障害物 Reach を 400 epoch へ続ける（2 seed）開始"
launch e2e_a1v_obspen_reach_ext    0 e2e_a1v_obspen_reach
launch e2e_a1v_obspen_reach_s2_ext 1 e2e_a1v_obspen_reach_s2
log "=== キュー終了"
