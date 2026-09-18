#!/usr/bin/env bash
# 2026-09-13: 反発係数を是正したホッケー（実験系譜 9-86）。
#
# **9-80 との差は XML だけ。** 報酬（ゴールまでの距離）・epoch・seed は同一。
#   `make_hockey_court_xml.py --wall` を再生成し、壁とパックに solref="0.02 0.25" が入った。
#
# ⚠️ **これは 7 回目の環境いじりではない。** 前 6 回は「学習が進まない理由」の推測だったが、
#   今回は **タスクが物理的に成立していなかったことの測定**に基づく是正である（9-80 の訂正）。
#   投入前に**反射そのものを測った**（従来は一度も測っていなかった）:
#     反発係数 0.13〜0.18 → 0.34〜0.65（1 を超えない）
#     ゴールに入る角度 0/19 → 4/19
#     壁へのめり込み 0.550（埋まる）→ 0.440（埋まらない）
#     閉じ込め 510 通り 脱出 0
#
# 停止条件（先に決める。9-50 の教訓）:
#   ① ゴール口を通る          → ✅ デモ成立。反射の有無を軌跡で確認して記録
#   ② 通らないが壁に埋まらない → 🟡 物理は直った。残るのは探索。**そこで打ち切る**
#   ③ また壁に埋まる          → ❌ 是正が効いていない。**打ち切って物理の限界として記録**
#   ⚠️ **②③のどちらでも打ち切る。** 8 回目は無い（§5-2 ④）。
#
# 判定は**再生して軌跡を見る**。⚠️ ログの値では判定できない（9-63・9-77 で 2 回誤診）。
# 判定の基準は「x が台の端 1.55 を越えた瞬間の y が |y|<=0.15 か」の 1 点。
#
# 起動: nohup bash scripts/queue_hockey_goal2.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_hockey_goal2.log
RUN=hockey_goal2
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
# ⚠️ **args だけで絞ると自分のシェルを数える**（Bug 19 と同型。9-111 で 1 度直したが不十分だった）。
#   ⭐ `comm`（実行ファイル名）が choreonoid のものだけを数える。シェルは bash なので混入しない。
njobs(){ ps -eo comm,args 2>/dev/null | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }

log "=== キュー開始。GPU が 2 本空くまで待つ"
while [ "$(njobs)" -ge 2 ]; do sleep 180; done
mkdir -p "single_run/$RUN"
nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
  --no-window --python scripts/choreonoid_train.py \
  cfg=pusher_tripo_v3 xml_name=e2e_hockey_wall num_threads=4 max_epoch_num=200 \
  enable_wandb=false fix_skeleton=true seed=0 +robot_param_scale=1 \
  +env_specs.cube_y_noise=0.4 +env_specs.arm_safe_init=true \
  +reward_specs.use_target_reward=true +reward_specs.target_x=1.55 +reward_specs.target_y=0.0 \
  +reward_specs.contact_weight=0 +reward_specs.ctrl_cost_coeff=0.001 \
  +reward_specs.init_contact_penalty=1.0 \
  hydra.run.dir="single_run/$RUN" >> "single_run/$RUN/stdout.log" 2>&1 &
log "$RUN launched (PID $!)"
