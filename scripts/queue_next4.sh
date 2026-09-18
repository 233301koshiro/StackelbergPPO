#!/usr/bin/env bash
# 2026-09-18: GPU を空けないためのキュー（実験系譜 9-112）。
#
#  ① e2e_a1v_obs_tall2_reach  ⭐ **柱を高くした障害物 Reach**（2 seed）
#     9-110 で「腕が柱を 4 mm の余裕でまたいでいた」と判明。柱を 0.45 → 0.85 m にした。
#     ⚠️ **投入前に第1層で確認済み**: 「幾何的な障害はありません（余裕 20 %、残り 0.001 m）」
#        ＝ **到達可能性は消えていない。** またいだ経路（z 0.454〜0.483）だけが塞がる。
#     読み: ① 迂回して到達 → ✅ 本当の障害物回避。② 到達できない → ⚠️ 第1層は「届く」と
#           言っているので、**幾何では届くが制御では届かない**例になる（それも結果）
#
# 起動: nohup bash scripts/queue_next4.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_next4.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
# ⚠️ **args だけで絞ると自分のシェルを数える**（Bug 19 と同型。9-111 で 1 度直したが不十分だった）。
#   ⭐ `comm`（実行ファイル名）が choreonoid のものだけを数える。シェルは bash なので混入しない。
njobs(){ ps -eo comm,args 2>/dev/null | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
launch(){ local RUN=$1; shift
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py "$@" \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)"; sleep 60; }

log "=== キュー開始（柱を高くした障害物 Reach）"
for S in 0 1; do
  launch e2e_a1v_obs_tall2_reach$([ $S -eq 1 ] && echo _s2) \
    cfg=pusher_tripo_v3 xml_name=e2e_a1v_obs_tall2 num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 \
    +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 \
    +env_specs.check_init_contact=false +reward_specs.init_contact_penalty=1900 \
    +env_specs.arm_safe_init=true
done
log "=== キュー終了"
