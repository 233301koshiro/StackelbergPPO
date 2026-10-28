#!/bin/bash
# ⛔ 系譜 9-281・Bug 74（ユーザー判断 10-22「いいよ」＝ (a) config.py を FLAG から読むように直す）: R1 の形を凍結した押しを取り直す。
#
# ⭐ なぜ: queue_phys.sh の tripo_pjp_mid_frozen_phys / _s2 は `+robot_param_scale=0.0001` が config.py に捨てられ（yml の 1 が効いた）、
#   凍結しない tripo_pjp_mid_phys / _s2 と exec_R_eps が全 epoch 一致した（seed0 は 200 epoch、seed1 は ep58 で止めた）。R1 の読み ⑪〜⑬ は判定できない。
#   旧い 2 本は消さずに残す（Bug 74 の証拠）。名前を _fx にするのは、launch() が既にあるフォルダを飛ばすため。
#
# ⭐ 起動行は queue_phys.sh の tripo_pjp_mid_frozen_phys$X と同じ（launch() も同じ）。違うのは run 名と、コードが直ったことだけ。
# ⭐ 読みは queue_phys.sh の ⑪〜⑬ をそのまま（結果を見てから変えない）:
#   ⑪ 形を凍結した帯が tripo_pjp_mid_phys の帯 [9.85, 10.42] より下で重ならない → R1 は成立 ／ ⑫ 重なる → 主張しない ／ ⑬ 上 → 原因を調べる
# ⭐ 起動の確認（Bug 74 の再発を ep2 で止める）: 一次ログの頭の `⭐ robot_param_scale = 0.0001（flag から）` と、
#   ep0〜2 の exec_R_eps が tripo_pjp_mid_phys（同じ seed）と違うことをこのキューのログに出す。一致したら ⛔⛔ を出す（止めるのは人が判断する）。
# ⚠️ queue_postrun_replay.sh は動いている bash なので一覧に足していない。完走したら再生・図・check_before_conclusion.py を手で回す。
#
# ⭐ 順番: 空いた枠に 15 秒ごとに入る（queue_phys.sh は 180 秒ごとなので、こちらが先に取る。queue_rec211p_s2.sh と同じ方式）
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_r1_fx.log
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
PUSH="+reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true"
curve(){ grep -o "exec_R_eps\s*-\?[0-9.]*" single_run/$1/log/log_train.txt 2>/dev/null | head -3 | awk '{print $2}' | tr '\n' ' '; }
verify(){ local RUN=$1 REF=$2   # ep2 まで待って、効いた値と曲線を比べる（同期せず裏で回す）
  until [ "$(grep -c 'exec_R_eps' single_run/$RUN/log/log_train.txt 2>/dev/null)" -ge 3 ]; do sleep 60; done
  log "$RUN: $(grep -o '⭐ robot_param_scale = .*' single_run/$RUN/log/log_train.txt | head -1)"
  local a b; a=$(curve $RUN); b=$(curve $REF)
  if [ "$a" = "$b" ]; then log "⛔⛔ $RUN の ep0〜2 [$a] が $REF と同じ。凍結が効いていない（Bug 74 の再発）"
  else log "✅ $RUN の ep0〜2 [$a] は $REF [$b] と違う（凍結が学習に届いた）"; fi; }

log "=== R1 の取り直し（Bug 74・系譜 9-281）HEAD=$(git rev-parse --short HEAD)"
for S in 0 1; do X=$([ $S = 1 ] && echo _s2)
  launch tripo_pjp_mid_frozen_phys_fx$X cfg=pusher_gearonly_real xml_name=tripo_arm_v2c_pj_mid_phys_arm171 max_epoch_num=200 seed=$S num_threads=4 enable_wandb=false fix_skeleton=true +robot_param_scale=0.0001 $PUSH
  verify tripo_pjp_mid_frozen_phys_fx$X tripo_pjp_mid_phys$X &
done
wait
log "=== 起動と確認が済んだ"
