#!/usr/bin/env bash
# 2026-09-14: 腕も壁に当たるホッケー（実験系譜 9-94）。
#
# **hockey_goal2 との差は XML の conaffinity だけ。** 報酬・epoch・seed・形態は同一。
#   壁 conaffinity 2 → 3。腕は contype=1 なので (1&3)=1 で**当たるようになる**。
#   `diff` で conaffinity 以外が完全に同一であることを確認済み。
#
# ⚠️ **これは 8 回目の「環境いじり」ではない。** 9-93 が特定した**真因の除去**である。
#   前 7 回は症状への対処だったが、今回は
#   「腕が壁をすり抜ける ⇒ 壁越しにパックを押せる ⇒ 反射が不要になる」という
#   **機序を辿って、その入口を塞いだ**。
#
# ⚠️ **元の割り切りの根拠（「腕が引っかかって学習が壊れる」）は測ったら外れた**（9-94）:
#     腕が壁に当たる姿勢          4.6 %
#     台の上の到達範囲            274 → 272 升（**99.3 % 残る**）
#     パックの居る帯              108 → 107 升（99 %）
#     先端が壁の外へ出られるか    74/74 件すり抜け → **0/74**（完全に塞がった）
#   ⭐ 9-65 の「検算」は**すり抜け設定が意図どおりか**を見ただけで、
#     **衝突させたらどうなるか**は一度も測っていなかった。
#
# 読み方（**結果を見る前に決める**。§5-2 ⑤-2）:
#   ① パックがゴール口を通る（x=1.55 で |y|<=0.15）
#        → ✅ 成立。**反射したかどうかも軌跡から記録する**（したか否かは別問題）
#   ② 通らないが壁に埋まらない
#        → 🟡 **物理は直った。残るのは探索。** 9-65 の測定では初期 y=±0.2 以内は反射必須で、
#          探索の難度が高い。ここで止まるのは「判定器の出力」であって壊れてはいない
#   ③ また壁に埋まる
#        → ❌ 真因の特定が誤り。**その場合は 9-93 の結論を訂正する**
#
# ⚠️ **判定は再生して軌跡を図にする。** ログの値では判定できない（9-63・9-77・9-93 で 3 回誤診）。
#
# 起動: nohup bash scripts/queue_hockey_goal3.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_hockey_goal3.log
RUN=hockey_goal3
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
njobs(){ ps -eo args 2>/dev/null | grep -cE "[c]horeonoid_train\.py" || true; }

log "=== キュー開始。GPU が 2 本空くまで待つ"
while [ "$(njobs)" -ge 2 ]; do sleep 180; done
mkdir -p "single_run/$RUN"
nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
  --no-window --python scripts/choreonoid_train.py \
  cfg=pusher_tripo_v3 xml_name=e2e_hockey_wall2 num_threads=4 max_epoch_num=200 \
  enable_wandb=false fix_skeleton=true seed=0 +robot_param_scale=1 \
  +env_specs.cube_y_noise=0.4 +env_specs.arm_safe_init=true \
  +reward_specs.use_target_reward=true +reward_specs.target_x=1.55 +reward_specs.target_y=0.0 \
  +reward_specs.contact_weight=0 +reward_specs.ctrl_cost_coeff=0.001 \
  +reward_specs.init_contact_penalty=1.0 \
  hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
log "$RUN launched (PID $!)"
