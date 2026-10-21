#!/bin/bash
# ⭐ 障害物 Reach（D-5）を今の物理で 2 seed 回し直す（ユーザー決定 10-21、系譜 9-274・進捗 D-5）。
#   「柱を避けて届く」まで学習させない。「避けて届く姿勢は幾何的にあるのに、学習が届かない」（9-144・9-161）を
#   結果として扱う。旧い 2 本（e2e_a1v_obspen_reach / _s2）は旧い物理（armature 1・密度 5・自己干渉なし・
#   5 cm より深い接触を捨てる）で、seed1 の到達はその「捨てる」穴によるもの（9-274）なので、修論に書くには取り直しが要る。
#
# ⭐ 起動行は queue_phys.sh の e2e_a1v_reach_phys / _s2 と同じ（launch() も同じ）。違うのは 2 つだけ（ユーザー指定 10-21）:
#   ① xml_name=e2e_a1v_obs_tall2_phys_arm171（e2e_a1v_phys_arm171 に、旧い _obs_tall2 と同じ柱を 1 本足しただけ。
#      make_obstacle_reach_xml.py --src … --height 0.85。柱は (0.45, 0, 0〜0.85)・半幅 0.07 の box、body に入れてある（9-98））
#   ② +reward_specs.obstacle_penalty_scale=1.0（旧い obspen と同じ係数。リンクの中心線が柱の箱に入った深さ [m] を引く、9-145）
#   ⚠️ 比べる相手は e2e_a1v_reach_phys / _s2（柱なし）。上の 2 つが同時に違うので、柱の効果と罰の効果は分けない。
#   ⚠️ 旧い obspen の起動行にあった arm_safe_init は付けない（e2e_a1v_reach_phys に無いため）。
#      投入前の確かめ（10-21）: e2e_a1v_reach_phys / _s2 の best の再生 10 話で、初期姿勢は柱と反対の −x を向き、
#      柱との隙間は全リンク 280 mm 以上。柱なしで学習した到達の軌跡は柱の場所を約 97 % の step で通る（柱は通り道を塞ぐ）。
#   ⭐ 投入前の確かめ（10-21、probe_obstacle_collision_choreonoid.py・今の物理の環境変数）: 同じ姿勢・ゼロ制御で、
#      柱ありは末端リンクが z 0.52〜0.83 m で止まり、柱なしは z 0.33〜0.37 m まで落ちる＝柱は実走に存在して腕を止める。
#   ⚠️ 罰は中心線で測るので、太いリンクが柱の表面で止まると罰は 0 のまま（物理は止める）。9-274 の軌跡（深い接触を
#      捨てない設定、リンク半径 30 mm）では中心線が 17〜38 mm 入り、罰は出ていた。
#
# ⭐ 事前登録した読み（結果を見てから変えない）。判定は best の再生 5 話（check_before_conclusion.py・check_obstacle_clearance.py）:
#   ㉑ 柱が効く（前提）: 2 本とも、全話・全 step でリンクの中心線の柱への食い込みが柱の半幅 70 mm 未満（中心を越えて
#      通り抜けていない）、かつ不変条件 7 項目が通る。成り立たなければ以下は読まない（物理か柱の実装を疑う）
#   ㉒ 到達（閾値は check_before_conclusion の 10 mm）:
#      ㉒-a 2 seed とも全話で届かない → 予想どおり。「両立する姿勢は幾何的にあるのに、学習が届かない」（9-161）を
#            今の物理の結果として書く。先端と目標の最小距離の帯を記録する
#      ㉒-b どちらかの seed で 5 話中 3 話以上が届く（㉑ を満たしたまま）→ 「避けて届く」が出た。seed で割れたら割れたと書く
#      ㉒-c 届くが ㉑ を満たさない → 通り抜けで届いた。㉑ に戻る
#   予想（確かめていない）: ㉒-a。すり抜けの道が閉じるので、旧い seed1 の「届くが柱を通る」は出ない（9-274）
#
# ⭐ 順番（10-21 ユーザー「ホッケーの後で」）: 本キュー（queue_phys.sh）の最後の付録D（tripo_arm_v3_pusher_phys_s2）が
#   起動した後 → ホッケー seed1 → 本キューの 2 本。ホッケー seed1 は 10-20 に hockey キューの bash を止めたので起動役がいない。
#   そこで queue_hockey_phys_g10_blk3d.sh を**書き換えずにそのまま 1 回呼ぶ**（seed0 は log/ があるので飛ばされ、seed1 だけが
#   空き枠を待って入る。起動行は hockey キューのものがそのまま使われる）。ホッケー seed1 が起動してから本キューの 2 本が枠を待つ。
#   見込み（10-21 7 時 50 分の計算）: 10-27 8 時ごろ 1 枠目 → ホッケー seed1、16 時ごろ 2 枠目 → obspen seed0、seed1 はその次の空き。
#   D-7 は未定（決めたら後ろに足す）。✅ 投入はユーザーの確認済み（10-21 11 時「キューも投入して」）
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_obspen_phys.log
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
FB="+fix_base=true"
REACH8="+reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false +env_specs.cube_x_offset=100"
OBS="+reward_specs.obstacle_penalty_scale=1.0"

log "=== 障害物 Reach を今の物理で 2 seed（D-5、系譜 9-274）HEAD=$(git rev-parse --short HEAD)。tripo_arm_v3_pusher_phys_s2 の起動 → ホッケー seed1 の起動を待つ"
until [ -d single_run/tripo_arm_v3_pusher_phys_s2/log ]; do sleep 60; done
log "本キューの最後が起動した。先にホッケー seed1（queue_hockey_phys_g10_blk3d.sh をそのまま呼ぶ）"
bash scripts/queue_hockey_phys_g10_blk3d.sh
until [ -d single_run/hockey_bank9_phys_g10_blk3d_s2/log ]; do sleep 60; done
log "ホッケー seed1 が起動した。枠が空き次第入る"
for S in 0 1; do X=$([ $S = 1 ] && echo _s2)
  launch e2e_a1v_obspen_reach_phys$X cfg=pusher_tripo_v3_reallen xml_name=e2e_a1v_obs_tall2_phys_arm171 max_epoch_num=200 seed=$S $C $REACH8 +reward_specs.init_contact_penalty=1900 $FB $OBS
done
log "=== 2 本とも起動した"
