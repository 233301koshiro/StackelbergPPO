#!/bin/bash
# ⭐⭐ 系譜 9-246（ユーザー判断 10-13「全部実在の値でやる」）: 修論の全実験を、実在の物理の値で最初からやり直す（*_phys）。
#
# ⭐ なぜ: 物理の値に根拠が無い（密度 5.0 で腕 41〜108 g、トルク上限 400、回転子の慣性 1）うえ、3 つの欠陥
#   （Bug 60 制御入力を切らない・Bug 61 減衰が ÷ 総質量・armature が対象のスライドにも 1）が重なり、全部やり直すことになったため。
#   軸5 の「従来 対 実在」は無くなり、全部を実在の値で検証する。
#
# ⭐ 物理（すべて run の launch_env.json と XML に残る）:
#   - リンク: アルミ A6061 の肉厚 2 mm パイプの実効密度（形ごと、189.9〜294.2 kg/m³）＝ XML の *_phys（make_realistic_density_xml.py）
#   - 回転子の慣性（armature）: UR5e の関節 1〜3 の実測 6.40e-5 kg·m²（モーター側）× 減速比 100² ＝ 0.64（腕の関節）・0（対象のスライド。回転子は無い）
#   - トルク: ±1 で切る（CNOID_CTRL_CLIP、Bug 60）× ギア比の範囲 縦型 25〜150・平面 20〜150 N·m（上限は UR5e、cfg の *_real）
#   - 速度: 3.1416 rad/s で飽和（UR5e の関節 1〜3 の定格 180°/s、CNOID_JOINT_MAX_VEL、9-245）
#   - 減衰: トルク −b·q̇、b = 1（CNOID_HINGE_DAMP_TORQUE、Bug 61。⚠️ b の値には根拠が無い＝原論文のまま。修論 5.3 に書く）
#   - 接触: 自己干渉あり（隣り合わない組）・深い接触を捨てない（閾値 1.0）・初期形態の門
#   ⚠️ 根拠が無いまま残る値: 減衰 b = 1・可動域・制御コスト 0.2・リンク長の探索幅（縦型 ±0.5 m）・接触の摩擦（Choreonoid 既定）
#
# ⭐ armature の確かめ（ユーザー判断 10-13「1 個だけ実在の値から出してみて、全く方向性が変わったら 1 のままでやろう」）:
#   押しの柱 long・mid（seed0）を armature 1 でも回す（*_phys_arm1、XML の armature だけが違う）。
#
# ⭐ 事前登録した読み（**結果を見てから変えない**）— 2 seed の best の帯で判定（9-227・9-228・9-230・9-234 の読みを引き継ぐ）
#   押しの柱: ① long ＞ mid で帯が離れる → 9-6 押し側は成立 ／ ② 重なる → 順位は主張できない ／ ③ mid ＞ long → 9-6 押し側を訂正
#             ④ prox と dist の帯が重なる → 9-17 の読みは変わらない ／ ⑤ 離れる → 押しでも配分で区別できる
#   到達マトリクス: ① 中間 ＞ 長腕 ＞ 短腕 ② 配分 prox ＞ dist（4.3.3.4。⚠️ 平面の台座の球に dist が当たると機序が違う＝9-241）
#             ③ 助言どおり 1.99 倍が短腕より桁で良い ④ 関節 3 を固定しても落ちない（固定の帯が可動より下で重ならなければ 4.4.5 の到達側を訂正）
#   R2（縦型 e2e_a1v の到達 対 押し）: ⑥ ギア比（関節ごと）が到達と押しで分かれる（帯が重ならない）→ タスク識別は成立。
#             ⚠️ 速度の上限でギア比の役割が変わる（9-245）ので、**分かれなければ R2 を「ギア比で」から訂正する**
#   付録D: ⑦ 先端リンクの太さが到達は下限付近・押しは上限付近 → 分岐は成立。⚠️ 旧い物理では「太い＝重い＝減衰が弱い」得があった（Bug 61）
#   ホッケー（9-237 で 20 話に書き直した）: ⑧ どちらかの seed で隙間 ≥ 10 cm のパックがゴール ／ ⑨ ゴールは無いが 0.1 m 以上動く ／ ⑩ 動かない
#   R1: ⑪ 形を凍結した帯が tripo_pjp_mid_phys の帯より下で重ならない → R1 は成立 ／ ⑫ 重なる → 主張しない ／ ⑬ 上 → 原因を調べる
#   armature: ⑭ long 対 mid（seed0）の best の上下と「押せるか」が armature 0.64 と 1 で同じ → 0.64 で続ける
#             ⑮ 上下が入れ替わるか、片方だけ押せない → **armature 1 で全部やり直す**（ユーザー判断）
#   共通: 離れた組のめり込み < 50 mm・根元側の関節の振れ > 10°・best の形で門が発火しない・不変条件（check_invariants.py）
# ⚠️ ひっくり返りうると結果の前に書いておく主張（9-246）: 押しの順位・R2 のギア比・付録D の太さ・R4（判定に要る学習量）・ホッケー
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_phys.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
EXTRA=""
PHYS="CNOID_SELF_COLLISION=1 CNOID_CULLING_DEPTH=1.0 CNOID_CTRL_CLIP=1 CNOID_HINGE_DAMP_TORQUE=1 CNOID_JOINT_MAX_VEL=3.1416"
launch(){ local RUN=$1; shift
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 $PHYS $EXTRA \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    "$@" +env_specs.init_self_contact=true \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)"; sleep 60; }
C="num_threads=4 enable_wandb=false fix_skeleton=true +robot_param_scale=1"
FB="+fix_base=true"   # 系譜 9-235: 縦型アームの台座（根元の柱）を設計変数から外す
PUSH="+reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true"
REACH8="+reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false +env_specs.cube_x_offset=100"
REACHV3="+reward_specs.use_reach=true +reward_specs.target_x=0.72 +reward_specs.target_y=0.0 +reward_specs.target_z=0.25 +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false +reward_specs.init_contact_penalty=1600 +env_specs.cube_x_offset=100"
P="cfg=pusher_gearonly_real max_epoch_num=200 $C $PUSH"
log "=== 実在の物理で全部やり直す（9-246）: 押しの柱 → armature 1 の対照 → 押しの柱 seed1 → R2 → ホッケー → 到達 → R1 → 付録D"
# 押しの柱 seed0（long・mid を先に。armature の対照と比べる）
launch tripo_pjp_long_phys   xml_name=tripo_arm_v2c_pj_long_phys seed=0 $P
launch tripo_pjp_mid_phys    xml_name=tripo_arm_v2c_pj_mid_phys  seed=0 $P
launch tripo_pjp_long_phys_arm1 xml_name=tripo_arm_v2c_pj_long_phys_arm1 seed=0 $P
launch tripo_pjp_mid_phys_arm1  xml_name=tripo_arm_v2c_pj_mid_phys_arm1  seed=0 $P
launch tripo_pjdp_prox_phys  xml_name=tripo_arm_v2c_pj_prox_phys seed=0 $P
launch tripo_pjdp_dist_phys  xml_name=tripo_arm_v2c_pj_dist_phys seed=0 $P
for A in "pjp_long tripo_arm_v2c_pj_long" "pjp_mid tripo_arm_v2c_pj_mid" "pjdp_prox tripo_arm_v2c_pj_prox" "pjdp_dist tripo_arm_v2c_pj_dist"; do
  set -- $A
  launch tripo_${1}_phys_s2 xml_name=${2}_phys seed=1 $P
done
# R2: 縦型 e2e_a1v の到達 対 押し
for S in 0 1; do X=$([ $S = 1 ] && echo _s2)
  launch e2e_a1v_pusher_phys$X cfg=pusher_tripo_v3_real xml_name=e2e_a1v_phys max_epoch_num=200 seed=$S $C $PUSH $FB
  launch e2e_a1v_reach_phys$X  cfg=pusher_tripo_v3_real xml_name=e2e_a1v_phys max_epoch_num=200 seed=$S $C $REACH8 +reward_specs.init_contact_penalty=1900 $FB
done
# ホッケー（9-230 の起動行。物理と cfg だけ実在）
EXTRA="HOCKEY_WALL_RESTITUTION=0.75 HOCKEY_ARM_BLOCK=1"
for S in 0 1; do X=$([ $S = 1 ] && echo _s2)
  launch hockey_bank9_phys$X cfg=pusher_tripo_v3_real xml_name=e2e_hockey_easy_phys num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +env_specs.cube_y_noise=0.25 +env_specs.arm_safe_init=true \
    +env_specs.arm_init_clear_y=0.45 +env_specs.arm_init_pitch_search=true \
    +env_specs.arm_init_after_noise=true +env_specs.init_contact_all_links=true \
    +reward_specs.use_target_reward=true +reward_specs.target_x=1.55 \
    +reward_specs.target_y=0.0 +reward_specs.contact_weight=0 \
    +reward_specs.ctrl_cost_coeff=0 +reward_specs.init_contact_penalty=1.0 $FB
done
EXTRA=""
# 到達（箱なし）
R="cfg=pusher_gearonly_real max_epoch_num=200 $C $REACH8 +reward_specs.init_contact_penalty=1900"
for S in 0 1; do X=$([ $S = 1 ] && echo _s2)
  launch tripo_pj_mid_phys$X       xml_name=tripo_arm_v2c_pj_mid_phys    seed=$S $R
  launch tripo_pj_long_phys$X      xml_name=tripo_arm_v2c_pj_long_phys   seed=$S $R
  launch tripo_pjd_prox_phys$X     xml_name=tripo_arm_v2c_pj_prox_phys   seed=$S $R
  launch tripo_pjd_dist_phys$X     xml_name=tripo_arm_v2c_pj_dist_phys   seed=$S $R
  launch tripo_pjr199_phys$X       xml_name=tripo_arm_v2c_pj_rec199_phys seed=$S $R
  launch e2e_a1_reach_phys$X       xml_name=e2e_a1_phys                  seed=$S $R
  launch e2e_a1_fix3_reach_phys$X  xml_name=e2e_a1_fix3_phys             seed=$S $R
done
launch tripo_pj_short_phys         xml_name=tripo_arm_v2c_pj_short_phys  seed=0 $R
# R1: 形を凍結した押し（tripo_pjp_mid_phys との差は robot_param_scale だけ）
for S in 0 1; do X=$([ $S = 1 ] && echo _s2)
  launch tripo_pjp_mid_frozen_phys$X cfg=pusher_gearonly_real xml_name=tripo_arm_v2c_pj_mid_phys max_epoch_num=200 seed=$S num_threads=4 enable_wandb=false fix_skeleton=true +robot_param_scale=0.0001 $PUSH
done
# 付録D: 非平面 4 関節（1000 epoch、最後に回す）
for S in 0 1; do X=$([ $S = 1 ] && echo _s2)
  launch tripo_v3_reach2_phys$X   cfg=pusher_tripo_v3_real xml_name=tripo_arm_v3_phys max_epoch_num=1000 seed=$S $C $REACHV3 $FB
  launch tripo_arm_v3_pusher_phys$X cfg=pusher_tripo_v3_real xml_name=tripo_arm_v3_phys max_epoch_num=1000 seed=$S $C $PUSH $FB
done
log "=== キュー終了"
