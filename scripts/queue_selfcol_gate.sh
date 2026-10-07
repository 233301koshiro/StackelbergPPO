#!/bin/bash
# ⭐⭐ 押しの柱 4 形態 × 2 seed を「自己干渉あり＋自分のリンクどうしの初期接触の門」で取り直す（系譜 9-227）
#
# code: aa08258 (main)
#
# ⭐ ユーザー判断（2026-10-07）: Bug 56 は限界として残さず**対処してから回し直す**。
#   修論は「最初から自己干渉を考慮した設定」として、この 8 本の値を使う（9-223 の `*_sc` は使わない）。
# ⭐ 9-223（`*_sc`）との差は `+env_specs.init_self_contact=true` だけ（`CNOID_SELF_COLLISION=1` は同じ）。
#   ⛔ Bug 56: 太さの設計で離れたリンクどうしが最初から重なると、自己干渉ありでは関節が固着した（dist seed0）。
#   ⭐ 門は重なる形の話を 1 step でペナルティ終了にする（Bug 55 と同じ仕組み・`init_contact_penalty=50`）。
#   ⭐ 投入前の確認: 固着した形を Choreonoid 内で 3/3 話捕まえ、正常な形（prox）は 0 回（9-227）。
#
# ⭐ 事前登録した読み（**結果を見てから変えない**）— best の 2 seed の帯で判定（9-223 と同じ基準）
#   ⭐ ① long ＞ mid で帯が離れる → 9-6 押し側は成立（修論はこの値で書く）
#   ⚠️ ② 帯が重なる → 押し側の順位は主張できない
#   ⛔ ③ mid ＞ long で帯が離れる → 9-6 押し側を訂正
#   ⭐ ④ prox と dist の帯が重なる → 9-17 の読み（押しでは配分の効きが弱い）は変わらない
#   ⚠️ ⑤ prox と dist の帯が離れる → 押しでも配分で区別できる
#   門について: ⭐ 8 本すべてで関節が固着しない（再生で各関節の振れ幅 > 10°）。⚠️ 学習初期は門が発火しうるが、
#   best の形は門を通る（再生で init_gate が出ない）。出たら Leader が回避を学べていない → 報告する
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_selfcol_gate.log
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
    +env_specs.init_self_contact=true \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!, seed=$S) ⭐ 9-223 との差は init_self_contact=true のみ"; sleep 60; }
log "=== 自己干渉あり＋自分のリンクの門で押しの柱を取り直す（8 本）。空き枠を待つ"
launch tripo_pjp_long_scg     tripo_arm_v2c_pj_long 0
launch tripo_pjp_mid_scg      tripo_arm_v2c_pj_mid  0
launch tripo_pjdp_prox_scg    tripo_arm_v2c_pj_prox 0
launch tripo_pjdp_dist_scg    tripo_arm_v2c_pj_dist 0
launch tripo_pjp_long_scg_s2  tripo_arm_v2c_pj_long 1
launch tripo_pjp_mid_scg_s2   tripo_arm_v2c_pj_mid  1
launch tripo_pjdp_prox_scg_s2 tripo_arm_v2c_pj_prox 1
launch tripo_pjdp_dist_scg_s2 tripo_arm_v2c_pj_dist 1
log "=== キュー終了"
