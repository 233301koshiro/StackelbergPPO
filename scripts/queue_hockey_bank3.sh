#!/usr/bin/env bash
# ⭐⭐ 反射＋**是正後**の腕ブロックでホッケーを回し直す（系譜 9-190、2026-09-30）。
#
# ⛔⛔ **9-185（`hockey_bank2` / `_s2`）は前提が崩れていた。**
#   腕ブロックが**末端リンクのカプセル本体を見ておらず**、`_s2` は 975/1201 標本が壁の中だった。
#   ⭐ **9-184 でカプセル捕捉を是正**し（4/5 リンク）、⭐ **9-189 で掃過判定と押し戻しを追加**した。
#   ⭐ **プローブで壁の中 0 標本・外面を越えた距離 0 mm を確認済み。**
#   ⚠️ **ただし掃過の必要性は実証できていない**（旧実装でもすり抜けが再現しなかった。9-189）。
#   ⭐ **書けるのは「カプセル捕捉の是正で貫通が止まった」まで。**
#
# ⭐ 対照は **`hockey_bank` / `_s2`**（9-181。反射あり・ブロックなし）。⭐ **変数は 1 つ**（`HOCKEY_ARM_BLOCK`）。
# ⚠️ 9-185 と同じフラグにしてある（**比較可能性のため**。`hockey_bank2` は前提が崩れているが条件は同じ）。
#
# ⚠️⚠️ **Bug 51: パックの初期 x には ±0.1 m のノイズが乗る**（`cube_y_noise=0.25` は y だけを上書きする）。
#   ⭐ **これは 9-185 以前の全ホッケー run にも同じく乗っていた**ので、対照との比較は成立する。
#
# ⭐ 事前登録した読み（**結果を見てから変えない**）
#   ① ゴールが入る（1 話でも）    → ⭐⭐⭐ **物理を整えれば学習できた。**9-93 以来はじめて
#   ② 入らないが壁で反射して狙う   → ⭐⭐ **バンクショットは学習可能だが予算/報酬が足りない。**
#                                  ⭐ **これが出たら「判定器がこの形態を棄却できるか」へ問いを移す**（§5-2 ⑤）
#   ③ 反射せず直接打ちに終始      → ⭐ **報酬（ゴールまでの距離の PBRS）が往路の遠ざかりを罰している**仮説を検証する
#   ④ 壁の中の標本が出る         → ⛔⛔ **ブロックがまだ不完全。**学習結果は読まない
#
# ⚠️ **完走後に `check_before_conclusion.py`（6 項目）と `plot_run.py`。目視記録へ 1 行。**
# ⚠️ **`_s2` を飛ばさない**（9-188: seed0 だけ見る癖が出ていた。9-186 は seed1 で方策が別物だった）。
# ⚠️ **環境変数は env の引数として渡す**（Bug 49）。
#
# 起動: nohup bash scripts/queue_hockey_bank3.sh > /dev/null 2>&1 & disown
# 進捗: single_run/queue_hockey_bank3.log
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_hockey_bank3.log
: > "$LOG"
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }

launch(){ local RUN=$1 S=$2
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
    HOCKEY_WALL_RESTITUTION=0.75 HOCKEY_ARM_BLOCK=1 \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_tripo_v3 xml_name=e2e_hockey_easy num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +env_specs.cube_y_noise=0.25 +env_specs.arm_safe_init=true \
    +reward_specs.use_target_reward=true +reward_specs.target_x=1.55 \
    +reward_specs.target_y=0.0 +reward_specs.contact_weight=0 \
    +reward_specs.ctrl_cost_coeff=0.001 +reward_specs.init_contact_penalty=1.0 \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!, seed=$S, e=0.75, arm_block=1 / 是正版)"; sleep 60; }

log "=== 反射＋是正後の腕ブロック（2 seed）開始"
launch hockey_bank3    0
launch hockey_bank3_s2 1
log "=== キュー終了"
