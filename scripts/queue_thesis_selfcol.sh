#!/bin/bash
# ⭐⭐ 修論の主張のうち、離れたリンクどうしのすり抜けが出た実験を「自己干渉あり＋自分のリンクの門」で取り直す（系譜 9-228）
#
# code: aa08258 (main)
#
# ⭐ ユーザー判断（2026-10-07）: 修論は「最初から自己干渉を考慮した設定」として書く（9-227）。B 案＝結果が変わりうる実験だけ取り直す。
# ⭐ 選び方: 修論に値が載る run の軌跡で**離れた組**（案 A が判定する組）のすり抜けを測り、出た実験は**比較の組ごと**取り直す。
#   出なかった実験（到達マトリクス・閉ループ助言・到達の配分 従来版）は取り直さず「軌跡で検査して 0」と書く。
#   ┌ 4.6.3〜4.6.4（軸5、押し 3 条件＋到達 2 条件）: e2e_a1v_pusher_s2 195 mm・rho 160〜197 mm・real 193 mm・reach_s2 61 mm(100%)・real_reach 45 mm(99%)
#   ├ 4.6.5（配分の判別・実在）: tripo_pjd_dist_real / _s2 が 0×11 で 54 / 42 mm（100 %、Bug 56 と同じ「最初から重なる」型）
#   └ 付録D（非平面 4 関節）: tripo_v3_reach2 が 1×111 で 95 mm（100 %）。タスク間の比較なので押しの 2 本も取り直す
# ⭐ 元 run との差: `CNOID_SELF_COLLISION=1` と `+env_specs.init_self_contact=true` だけ（9-227 と同じ設定）。
#   ⚠️ tripo_pjd_*_real（到達）だけ `+reward_specs.init_contact_penalty=1900` を足す。既定 50 では到達の正直な話（数百の負）より
#     罰が軽く、重なる形が得になる（Bug 9 の型）。⭐ この run は `check_init_contact=false` なので**この値は新しい門にしか効かない**。
#     値は同じ目標の到達 run（e2e_a1v_reach）と揃えた。
#
# ⭐ 事前登録した読み（**結果を見てから変えない**）— 2 seed の best の帯で判定
#   4.6.3 ⭐ ① 実在アクチュエータの押しの帯が従来の帯より下で重ならない → 「代価（低下）」は成立。重なれば弱める
#         ⭐ ② 実寸密度の押しの帯が従来の帯より上で重ならない → 「密度で向上」は成立。重なれば弱める
#   4.6.4 ⭐ ③ 従来・実在の両条件で、表 4.23 の関節のギア比が到達と押しで分かれる（帯が重ならない）→ タスク識別は成立
#   4.6.5 ⭐ ④ prox_real の帯が dist_real の帯より良く（0 に近く）重ならない → 実在条件の順序づけは成立。重なれば 4.6.5 を弱める
#   付録D ⭐ ⑤ 先端リンクの太さが到達は 2 seed とも下限付近・押しは 2 seed とも上限付近 → 分岐は成立。崩れたら付録 D を直す
#   共通: 再生で離れた組のすり抜けが ≈0、各関節の振れ幅 > 10°（固着しない）。best の形で門が発火したら報告する
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_thesis_selfcol.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
launch(){ local RUN=$1; shift
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 CNOID_SELF_COLLISION=1 \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    "$@" +env_specs.init_self_contact=true \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!) ⭐ 元 run との差は自己干渉あり＋init_self_contact"; sleep 60; }
C="num_threads=4 enable_wandb=false fix_skeleton=true +robot_param_scale=1"
PUSH="+reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true"
REACH8="+reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false"
REACHV3="+reward_specs.use_reach=true +reward_specs.target_x=0.72 +reward_specs.target_y=0.0 +reward_specs.target_z=0.25 +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false +reward_specs.init_contact_penalty=1600"
log "=== 修論の主張を自己干渉ありで取り直す（18 本）。9-227 のキューの投入完了を待つ"
until grep -q "=== キュー終了" single_run/queue_selfcol_gate.log 2>/dev/null; do sleep 600; done
log "9-227 の投入完了を確認"
for S in 0 1; do X=$([ $S = 1 ] && echo _s2)
  # 4.6.3〜4.6.4: 押し 3 条件（元 run の起動行どおり）
  launch e2e_a1v_pusher_scg$X      cfg=pusher_tripo_v3      xml_name=e2e_a1v      max_epoch_num=200 seed=$S $C $PUSH
  launch e2e_a1v_rho_pusher_scg$X  cfg=pusher_tripo_v3      xml_name=e2e_a1v_rho  max_epoch_num=200 seed=$S $C +env_specs.arm_safe_init=true
  launch e2e_a1v_real_pusher_scg$X cfg=pusher_tripo_v3_real xml_name=e2e_a1v_real max_epoch_num=200 seed=$S $C +env_specs.arm_safe_init=true
  # 4.6.4: 到達 2 条件
  launch e2e_a1v_reach_scg$X       cfg=pusher_tripo_v3      xml_name=e2e_a1v      max_epoch_num=200 seed=$S $C $REACH8 +reward_specs.init_contact_penalty=1900
  launch e2e_a1v_real_reach_scg$X  cfg=pusher_tripo_v3_real xml_name=e2e_a1v_real max_epoch_num=200 seed=$S $C $REACH8 +reward_specs.init_contact_penalty=1900 +env_specs.arm_safe_init=true
  # 4.6.5: 配分の判別（実在）
  launch tripo_pjd_prox_real_scg$X cfg=pusher_gearonly_real xml_name=tripo_arm_v2c_pj_prox_real max_epoch_num=200 seed=$S $C $REACH8 +reward_specs.init_contact_penalty=1900
  launch tripo_pjd_dist_real_scg$X cfg=pusher_gearonly_real xml_name=tripo_arm_v2c_pj_dist_real max_epoch_num=200 seed=$S $C $REACH8 +reward_specs.init_contact_penalty=1900
done
# 付録D: 非平面 4 関節（1000 epoch、最後に回す）
for S in 0 1; do X=$([ $S = 1 ] && echo _s2)
  launch tripo_v3_reach2_scg$X      cfg=pusher_tripo_v3 xml_name=tripo_arm_v3 max_epoch_num=1000 seed=$S $C $REACHV3
  launch tripo_arm_v3_pusher_scg$X  cfg=pusher_tripo_v3 xml_name=tripo_arm_v3 max_epoch_num=1000 seed=$S $C
done
log "=== キュー終了"
