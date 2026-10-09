#!/bin/bash
# ⭐⭐ R1（形の最適化は前提として要る）を 3 軸で示し直す — 形を凍結した押し 2 本（系譜 9-233）
#
# code: 2632409 (main)
#
# ⭐ ユーザー判断（2026-10-09）: 修論から rrbot を落とす（3 軸で足りる・ダイエット）。R1 だけは rrbot の形の凍結（各 1 seed、
#   しかも箱に 57 mm めり込んだまま＝Bug 57）しか根拠が無かったので、案 A＝3 軸で取り直す。
# ⭐ 比べる相手は `tripo_pjp_mid_scgc` / `_s2`（9-232、形を最適化する側）。**差は `robot_param_scale` 1 → 0.0001 だけ**
#   （形の更新幅を実質 0 にする。rrbot の C と同じ値）。自己干渉・門・`CNOID_CULLING_DEPTH=1.0` もそろえる。
#   形はギア比既定 100・太さは M4 の値のまま（`pusher_gearonly` はリンク長を最適化しない）。
#
# ⭐ 事前登録した読み（**結果を見てから変えない**）— 2 seed の best の帯で判定
#   ⭐ ① 凍結の帯が `tripo_pjp_mid_scgc` の帯より下で重ならない → R1 は 3 軸でも成立。修論 4.1.4 をこの値で書く
#   ⚠️ ② 帯が重なる → この設定では形の最適化が要ることを示せない。R1 は「原論文の前提」として引用に留め、本文で主張しない
#   ⛔ ③ 凍結の帯の方が上で重ならない → 形の最適化が害になっている。原因を調べる（報告する）
#   共通: 再生で対象が動くか・離れた組のめり込み < 50 mm・根元側の関節の振れ幅 > 10°・門が発火しないか
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_r1_frozen.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
launch(){ local RUN=$1; shift
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 CNOID_SELF_COLLISION=1 CNOID_CULLING_DEPTH=1.0 \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    "$@" +env_specs.init_self_contact=true \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!) ⭐ tripo_pjp_mid_scgc との差は robot_param_scale=0.0001 のみ"; sleep 60; }
C="num_threads=4 enable_wandb=false fix_skeleton=true"
PUSH="+reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true"
log "=== R1 を 3 軸で（形を凍結した押し 2 本）。9-232 のキューの投入完了を待つ"
until grep -q "=== キュー終了" single_run/queue_selfcol_cull.log 2>/dev/null; do sleep 600; done
log "9-232 の投入完了を確認"
launch tripo_pjp_mid_frozen_scgc    cfg=pusher_gearonly xml_name=tripo_arm_v2c_pj_mid max_epoch_num=200 seed=0 $C +robot_param_scale=0.0001 $PUSH
launch tripo_pjp_mid_frozen_scgc_s2 cfg=pusher_gearonly xml_name=tripo_arm_v2c_pj_mid max_epoch_num=200 seed=1 $C +robot_param_scale=0.0001 $PUSH
log "=== キュー終了"
