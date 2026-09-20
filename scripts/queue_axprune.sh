#!/usr/bin/env bash
# 2026-09-20: ⭐⭐ **枝刈りの妥当性検証**（実験系譜 9-141）。
#
# ⛔ **9-127 で「第1層で 81 通り中 43 通りを学習前に棄却できる」と示したが、
#   棄却された配置を実際に学習させたことは一度も無い。**
#   ⭐ 「棄却側だけが確実」は 9-78 の設計だが、**軸を変えた場合は未検証**である。
#
# ⭐ **最も際どい棄却を選んだ**: `z-x-x-z`（残り 0.209 m）。
#   ⚠️ **枝刈りが誤るなら境界で誤る。**余裕のある棄却を試しても検証にならない。
#
#   対照 `e2e_a1v`（z-y-y-y、第1層 ✅ 通過）: best −13.65、⭐ **到達（最小 4 mm）**
#   本条件 `e2e_a1v_ax_zxxz`（第1層 ⛔ 棄却）:
#       「関節の可動域では目標に届きません（残り 0.209 m）」
#       「腕が短すぎて目標に届きません（届く範囲は 0.636 m、目標は 0.800 m 先）」
#
# ⭐⭐ **定量的な予測**: 先端は目標に **0.164〜0.209 m より近づけない**。
#
# 読み方（**結果を見る前に決める**。§5-2 ⑤-2）:
#   ⛔ ① **到達する（先端が目標 10 mm 以内）**
#        → ⛔⛔ **枝刈りは誤っていた。「棄却側だけが確実」が崩れる。**
#          **9-127 の削減率 53 % は主張できなくなり、修論 3.10 の枝刈りの記述も要訂正**
#   ⭐ ② **到達せず、残距離が 0.164〜0.209 m 付近**
#        → ✅ **枝刈りは正しい。第1層の予測が学習でも成り立つ**
#   ⚠️ ③ **到達しないが残距離が予測と大きく違う**
#        → ⚠️ **棄却は正しいが理由が違う。**第1層の機序を見直す
#
# ⚠️ **1 seed では「たまたま学習が失敗した」と区別できない。**
#   ⭐ **2 seed 走らせ、両方とも②なら棄却を支持する。**
#   ⚠️ **片方でも①なら、それだけで枝刈りは崩れる**（棄却の確実性は全称命題だから）。
#
# ⚠️ **結論を書く前に `check_before_conclusion.py` を回す**（9-114）。
# ⚠️ **固定 body は無いので再生の揺れは出ないはず**（9-138）。念のため 3 回再生して確認する。
#
# 起動: nohup bash scripts/queue_axprune.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_axprune.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
# ⚠️ comm で絞る。args だけだと自分のシェルを数える（Bug 19 と同型）
njobs(){ ps -eo comm,args 2>/dev/null | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
launch(){ local RUN=$1 S=$2
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_tripo_v3 xml_name=e2e_a1v_ax_zxxz num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 \
    +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 \
    +env_specs.check_init_contact=false +reward_specs.init_contact_penalty=1900 \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)"; sleep 60; }

log "=== キュー開始（枝刈りの妥当性検証・棄却された z-x-x-z を 2 seed）"
launch e2e_a1v_axprune_reach 0
launch e2e_a1v_axprune_reach_s2 1
log "=== キュー終了"
