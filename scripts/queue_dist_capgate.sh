#!/bin/bash
# ⭐ Bug 72 の取り直し（ユーザー判断 10-20「門を直して取り直す」「到達の残りの後」、系譜 9-264）:
#   配分 dist の 3 本（到達 seed0・seed1、押し seed0）は、`11` のカプセルが関節の後ろへ 0.18〜0.20 m はみ出して（設計変数 ext_start）
#   台座の球に 13 cm 食い込んだ形で学習していた。自分のリンクどうしの初期接触の門（Bug 56）がボーンだけで測り、通していた。
#
# ⭐ 変えるのは 1 つ: `+env_specs.init_self_contact_caps=true`（門が本当のカプセルで測る。食い込んだ形は 1 step でペナルティ終了）。
#   起動行はそれ以外 queue_phys.sh の元の run と同じ（launch() も同じ）。
#   prox・mid・long は 1 本目が長く構造上台座に食い込めないので、この設定の有無で環境は変わらない
#   （投入前の確かめ: 既存の best の形で、dist の 3 本だけ門が 1 step で働き、押し dist seed1・到達 prox・mid は働かない）。
#   押し dist seed1（tripo_pjdp_dist_phys_s2）ははみ出し 0 で門が働かないので取り直さず、帯は seed0 の取り直しと組む。
#
# ⭐ 事前登録した読み（結果を見てから変えない）:
#   ⑲ 門が効く: 3 本とも最終形態の `0`×`11` の食い込みが 20 mm 未満（audit_capsule_ext.py / check_before_conclusion の「離れた組」）。
#      効かなければ以下は読まない（門の実装を疑う）
#   到達の読み ②（queue_phys.sh 30 行、配分 prox ＞ dist）を、prox（tripo_pjd_prox_phys / _s2）と dist（_cg / _cg_s2）の 2 seed の best の帯で判定:
#      ②-a prox が上で帯が重ならない → 4.3.3.4 の到達側は成立 ／ ②-b 重なる → 順位は主張しない ／ ②-c dist が上 → 4.3.3.4 の到達側を訂正
#   押しの読み ⑤（queue_phys.sh 29 行）を、prox（tripo_pjdp_prox_phys / _s2）と dist（tripo_pjdp_dist_phys_cg と既存の _s2）で読み直す:
#      ⑤-a 帯が離れたまま → 9-259 の読み ⑤ は保たれる ／ ⑤-b 重なる → 9-259 の ⑤ を訂正（修論 4.3.3.4「押しタスクでの確認」も）
#
# ⭐ 順番: queue_phys.sh が到達の最後（tripo_pj_short_phys）を起動した後、R1 より先に入る。
#   ⚠️ queue_phys.sh・queue_hockey_phys_g10_blk3d.sh と 2 本の枠を取り合う。こちらは 15 秒ごとに見る（向こうは 180 秒）ので先に取る
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_dist_capgate.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }
PHYS="CNOID_SELF_COLLISION=1 CNOID_CULLING_DEPTH=1.0 CNOID_CTRL_CLIP=1 CNOID_HINGE_DAMP_TORQUE=1 CNOID_JOINT_MAX_VEL=3.1416 CNOID_JOINT_MAX_VEL_STRICT=1 CNOID_TICK_SYNC=1"
EXTRA=""
launch(){ local RUN=$1; shift
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 15; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 $PHYS $EXTRA \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    "$@" +env_specs.init_self_contact=true \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)"; sleep 60; }
C="num_threads=4 enable_wandb=false fix_skeleton=true +robot_param_scale=1"
PUSH="+reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true"
REACH8="+reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false +env_specs.cube_x_offset=100"
P="cfg=pusher_gearonly_real max_epoch_num=200 $C $PUSH"
R="cfg=pusher_gearonly_real max_epoch_num=200 $C $REACH8 +reward_specs.init_contact_penalty=1900"
CG="+env_specs.init_self_contact_caps=true"

log "=== dist の取り直し（門を本当のカプセルで、Bug 72・系譜 9-264）HEAD=$(git rev-parse --short HEAD)。tripo_pj_short_phys の起動を待つ"
until [ -d single_run/tripo_pj_short_phys/log ]; do sleep 60; done
log "tripo_pj_short_phys が起動した。枠が空き次第入る"
launch tripo_pjd_dist_phys_cg     xml_name=tripo_arm_v2c_pj_dist_phys_arm171 seed=0 $R $CG
launch tripo_pjd_dist_phys_cg_s2  xml_name=tripo_arm_v2c_pj_dist_phys_arm171 seed=1 $R $CG
launch tripo_pjdp_dist_phys_cg    xml_name=tripo_arm_v2c_pj_dist_phys_arm171 seed=0 $P $CG
log "=== 3 本とも起動した"
