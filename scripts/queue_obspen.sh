#!/usr/bin/env bash
# 2026-09-20: ⭐⭐ **障害物への接触ペナルティ**（実験系譜 9-145）。
#
# ⭐ 9-144 が「幾何的には避けて届く姿勢が 21244 個ある」と示したので、
#   **避けないのはタスク定義の欠落**だと言い切れる。→ 報酬に入れる。
#
# ⭐⭐ **係数は 1（無次元）。** 報酬が `-距離 [m]` なので、罰も `侵入深さ [m]` にして揃えた。
#   ⚠️ **根拠のないパラメータを 1 個も増やさない**のが要点（弱点②を増やさない）。
#   実測: 距離 15.6〜16.2 mm に対し 侵入深さ 51.8〜68.0 mm ＝ **3.3〜4.2 倍**。
#   上限は柱の半幅 70 mm で、**幾何が決める**（人が決める値ではない）。
#
# ⚠️ 対照は `e2e_a1v_obs_tall2_reach` / `_s2`（同じ XML・同じ設定で罰だけ無し）。
#
# 読み方（**結果を見る前に決める**。§5-2 ⑤-2）:
#   ⭐ ① 柱に入らず目標にも届く → ✅ **報酬の欠落だったことが確定。障害物 Reach が成立**
#   🟡 ② 柱には入らないが届かない → **幾何では可能だが学習が届かない**（9-144 が 21244 個と言っている）
#   ⚠️ ③ 柱に入り続ける → 罰が弱いか探索シグナルが消えた。⭐ **係数を上げる前に学習曲線を見る**
#
# ⚠️ **結論を書く前に check_before_conclusion.py と check_obstacle_clearance.py を回す。**
#
# 起動: nohup bash scripts/queue_obspen.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_obspen.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
njobs(){ ps -eo comm,args 2>/dev/null | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
launch(){ local RUN=$1 S=$2
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_tripo_v3 xml_name=e2e_a1v_obs_tall2 num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 \
    +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 \
    +reward_specs.obstacle_penalty_scale=1.0 \
    +env_specs.check_init_contact=false +reward_specs.init_contact_penalty=1900 \
    +env_specs.arm_safe_init=true \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)"; sleep 60; }

log "=== キュー開始（障害物ペナルティ・2 seed）"
launch e2e_a1v_obspen_reach 0
launch e2e_a1v_obspen_reach_s2 1
log "=== キュー終了"
