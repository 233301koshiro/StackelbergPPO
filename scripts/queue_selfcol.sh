#!/bin/bash
# ⭐⭐ 押しの柱を**自己干渉あり**で取り直す（系譜 9-223、③）
#
# code: c224757 (main)
#
# ⛔ 学習は腕の自己干渉を見ていなかった（9-221: 185 run 中 118 run ですり抜け）。
# ⛔⛔ 9-6 押し側の「長い腕が勝つ」は、長い腕が根元を 196〜199 mm 突き抜けて振り回していた可能性がある。
#   9-222: 同じ方策を自己干渉ありで再生すると箱の移動が 3.7〜4.1 → 1.65〜2.5 m。
# ⭐ 到達（9-6 到達側・9-9）と閉ループ助言（9-11）はすり抜け 0 なので取り直さない。
#
# ⭐⭐ 変えるのは **CNOID_SELF_COLLISION=1 だけ**（元の overrides.yaml と 1 文字も変えない。案 A＝Choreonoid の既定どおり
#   **離れたリンクの組だけ**判定。隣り合う組の折り返しは残るので check_self_penetration.py で量を報告する）。
#
# ⭐ 事前登録した読み（**結果を見てから変えない**）— best_exec_R_eps を 2 seed で比べ、帯（2 seed の範囲）が重なるかで判定
#   9-6 押し側（元: long 96.18〜99.37 ＞ mid 34.32〜44.94、帯は離れていた）
#   ⭐ ① long ＞ mid で帯が離れたまま → ⭐⭐ 自己干渉ありでも順位の反転（9-6）は成立。修論は「再確認した」と書く
#   ⚠️ ② 帯が重なる → 押し側の順位は主張できない。9-6 は「到達側では反転、押し側は区別できない」へ弱める
#   ⛔ ③ mid ＞ long で帯が離れる → ⛔⛔ 9-6 押し側を訂正（すり抜けが生んだ順位だった）
#   9-17 配分の判別・押し（元: prox 41.56〜41.77 対 dist 27.89〜40.51、帯は重なり「区別できない」）
#   ⭐ ④ 帯が重なったまま → 9-17 の読み（押しでは配分の効きが弱い）は変わらない
#   ⚠️ ⑤ 帯が離れる → 押しでも配分で区別できる（すり抜けが差を消していた）
#   共通: check_self_penetration.py で離れた組のすり抜けが ≈0 になっているかを必ず確かめる（なっていなければ自己干渉が効いていない）
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_selfcol.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
launch(){ local RUN=$1 XML=$2 S=$3
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 CNOID_SELF_COLLISION=1 \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_gearonly xml_name=$XML num_threads=4 max_epoch_num=200 enable_wandb=false \
    fix_skeleton=true seed=$S +robot_param_scale=1 +reward_specs.ctrl_cost_coeff=0.2 \
    +reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!, seed=$S) ⭐ 元との差は CNOID_SELF_COLLISION=1 のみ"; sleep 60; }
log "=== 押しの柱を自己干渉ありで取り直す（8 本）開始"
launch tripo_pjp_long_sc     tripo_arm_v2c_pj_long 0
launch tripo_pjp_mid_sc      tripo_arm_v2c_pj_mid  0
launch tripo_pjp_long_sc_s2  tripo_arm_v2c_pj_long 1
launch tripo_pjp_mid_sc_s2   tripo_arm_v2c_pj_mid  1
launch tripo_pjdp_prox_sc    tripo_arm_v2c_pj_prox 0
launch tripo_pjdp_dist_sc    tripo_arm_v2c_pj_dist 0
launch tripo_pjdp_prox_sc_s2 tripo_arm_v2c_pj_prox 1
launch tripo_pjdp_dist_sc_s2 tripo_arm_v2c_pj_dist 1
log "=== キュー終了"
