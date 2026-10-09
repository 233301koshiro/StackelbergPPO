#!/bin/bash
# ⭐⭐ 修論の自己干渉ありの取り直し 28 本を、Bug 57 を直した設定でやり直す（系譜 9-232）
#
# code: ab87894 (main)
#
# ⛔ Bug 57（9-227）: AIST は既定で 5 cm より深い接触を捨てる。押しの長い腕 seed0 は**打撃の最中に自分の台座を
#   175 mm すり抜けて**強く打っていた（閾値を直すと箱 4.1〜4.4 → 2.6〜3.6 m）。
# ⭐ ユーザー判断（2026-10-09）: 「隣り合わないリンクが 5 cm 以上重なるのは見逃せない」→ 走っていたキューを止めて全部やり直す。
# ⭐ 元のキューとの差は `CNOID_CULLING_DEPTH=1.0`（深い接触を捨てない）だけ。run 名は `_scg` → `_scgc`。
#   起動行は 3 本のキューのまま: queue_selfcol_gate.sh（9-227）・queue_thesis_selfcol.sh（9-228）・queue_hockey_bank9_scg.sh（9-230）
#   ⭐ 1.0 m はリンク・箱・パックの取りうる最大のめり込み（約 0.25 m）を超える＝捨てない。閾値は 5 cm を越えた接触にしか
#     効かないので、深い接触が起きない場面の物理は変わらない。投入前に long seed0 の best で、既知の重なり
#     79〜175 mm が 25 mm・0.1 % に消え、物理が暴れないことを確かめた（9-227 の訂正）。
#
# ⭐ 事前登録した読み（**結果を見てから変えない**）— 2 seed の best の帯で判定
#   押しの柱（9-227 と同じ）
#     ① long ＞ mid で帯が離れる → 9-6 押し側は成立 ／ ② 重なる → 順位は主張できない ／ ③ mid ＞ long で離れる → 9-6 押し側を訂正
#     ④ prox と dist の帯が重なる → 9-17 の読みは変わらない ／ ⑤ 離れる → 押しでも配分で区別できる
#   軸5・付録D（9-228 と同じ）
#     4.6.3 ① 実在アクチュエータの押しの帯が従来の帯より下で重ならない → 「代価（低下）」は成立。重なれば弱める
#           ② 実寸密度の押しの帯が従来の帯より上で重ならない → 「密度で向上」は成立。重なれば弱める
#     4.6.4 ③ 従来・実在の両条件で、表 4.23 の関節のギア比が到達と押しで分かれる（帯が重ならない）→ タスク識別は成立
#     4.6.5 ④ prox_real の帯が dist_real の帯より良く（0 に近く）重ならない → 実在条件の順序づけは成立。重なれば弱める
#     付録D ⑤ 先端リンクの太さが到達は 2 seed とも下限付近・押しは 2 seed とも上限付近 → 分岐は成立。崩れたら付録D を直す
#   ホッケー（9-230 と同じ）
#     ① どちらかの seed で、置いた瞬間の隙間 ≥ 10 cm のパックがゴールに入る話がある → 修論にこの値で入れる
#     ② ゴールは無いが離れたパックを 0.1 m 以上動かす → 打撃は成立、ゴールは主張しない ／ ③ どの話でも動かない → 原因を調べる
#   共通（再生は学習と同じ環境変数で）:
#     ⭐ 離れた組の最大めり込み < 50 mm（`check_self_penetration.py` の「うち離れた組」）。超えたら Bug 57 が直っていない → 報告する
#     ⭐ 固着しない ＝ **根元側の関節**の振れ幅 > 10°、かつ対象が動く（押し・ホッケー）。⚠️ 末端関節は押しの方策がもとから
#       ほとんど使わないので判定に使わない（9-227 の教訓。投入前に書き直した）
#     ⭐ best の形で門が発火しない（再生 5 話がすべて最後の step まで走る）
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_selfcol_cull.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
EXTRA=""
launch(){ local RUN=$1; shift
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 CNOID_SELF_COLLISION=1 CNOID_CULLING_DEPTH=1.0 $EXTRA \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    "$@" +env_specs.init_self_contact=true \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!) ⭐ 元キューとの差は CNOID_CULLING_DEPTH=1.0 のみ"; sleep 60; }
C="num_threads=4 enable_wandb=false fix_skeleton=true +robot_param_scale=1"
PUSH="+reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true"
REACH8="+reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false"
REACHV3="+reward_specs.use_reach=true +reward_specs.target_x=0.72 +reward_specs.target_y=0.0 +reward_specs.target_z=0.25 +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false +reward_specs.init_contact_penalty=1600"
log "=== Bug 57 を直した設定で自己干渉ありの取り直し 28 本（押しの柱 8 → 軸5 14 → ホッケー 2 → 付録D 4）"
# 押しの柱（9-227 の起動行）
for S in 0 1; do X=$([ $S = 1 ] && echo _s2)
  for A in "pjp_long tripo_arm_v2c_pj_long" "pjp_mid tripo_arm_v2c_pj_mid" "pjdp_prox tripo_arm_v2c_pj_prox" "pjdp_dist tripo_arm_v2c_pj_dist"; do
    set -- $A
    launch tripo_${1}_scgc$X cfg=pusher_gearonly xml_name=$2 max_epoch_num=200 seed=$S $C $PUSH
  done
done
# 軸5（9-228 の起動行）
for S in 0 1; do X=$([ $S = 1 ] && echo _s2)
  launch e2e_a1v_pusher_scgc$X      cfg=pusher_tripo_v3      xml_name=e2e_a1v      max_epoch_num=200 seed=$S $C $PUSH
  launch e2e_a1v_rho_pusher_scgc$X  cfg=pusher_tripo_v3      xml_name=e2e_a1v_rho  max_epoch_num=200 seed=$S $C +env_specs.arm_safe_init=true
  launch e2e_a1v_real_pusher_scgc$X cfg=pusher_tripo_v3_real xml_name=e2e_a1v_real max_epoch_num=200 seed=$S $C +env_specs.arm_safe_init=true
  launch e2e_a1v_reach_scgc$X       cfg=pusher_tripo_v3      xml_name=e2e_a1v      max_epoch_num=200 seed=$S $C $REACH8 +reward_specs.init_contact_penalty=1900
  launch e2e_a1v_real_reach_scgc$X  cfg=pusher_tripo_v3_real xml_name=e2e_a1v_real max_epoch_num=200 seed=$S $C $REACH8 +reward_specs.init_contact_penalty=1900 +env_specs.arm_safe_init=true
  launch tripo_pjd_prox_real_scgc$X cfg=pusher_gearonly_real xml_name=tripo_arm_v2c_pj_prox_real max_epoch_num=200 seed=$S $C $REACH8 +reward_specs.init_contact_penalty=1900
  launch tripo_pjd_dist_real_scgc$X cfg=pusher_gearonly_real xml_name=tripo_arm_v2c_pj_dist_real max_epoch_num=200 seed=$S $C $REACH8 +reward_specs.init_contact_penalty=1900
done
# ホッケー（9-230 の起動行）
EXTRA="HOCKEY_WALL_RESTITUTION=0.75 HOCKEY_ARM_BLOCK=1"
for S in 0 1; do X=$([ $S = 1 ] && echo _s2)
  launch hockey_bank9_scgc$X cfg=pusher_tripo_v3 xml_name=e2e_hockey_easy num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +env_specs.cube_y_noise=0.25 +env_specs.arm_safe_init=true \
    +env_specs.arm_init_clear_y=0.45 +env_specs.arm_init_pitch_search=true \
    +env_specs.arm_init_after_noise=true +env_specs.init_contact_all_links=true \
    +reward_specs.use_target_reward=true +reward_specs.target_x=1.55 \
    +reward_specs.target_y=0.0 +reward_specs.contact_weight=0 \
    +reward_specs.ctrl_cost_coeff=0 +reward_specs.init_contact_penalty=1.0
done
EXTRA=""
# 付録D: 非平面 4 関節（1000 epoch、最後に回す）
for S in 0 1; do X=$([ $S = 1 ] && echo _s2)
  launch tripo_v3_reach2_scgc$X      cfg=pusher_tripo_v3 xml_name=tripo_arm_v3 max_epoch_num=1000 seed=$S $C $REACHV3
  launch tripo_arm_v3_pusher_scgc$X  cfg=pusher_tripo_v3 xml_name=tripo_arm_v3 max_epoch_num=1000 seed=$S $C
done
log "=== キュー終了"
