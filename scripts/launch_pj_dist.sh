#!/usr/bin/env bash
# 2026-08-07: 「配分」判別実験。総リーチを揃えてリンク長の配分だけを変えた2形態を比較する。
#
# 目的（第4章 4.4.5 限界4 を閉じ、co-design を使う必然性を強める）:
#   これまでの判別実験（4.4.3 / 4.4.4 / 4.4.5）は4形態すべて**同じ配分×スケール違い**で、
#   そのうち「幾何（診断の第1層）では答えが出ず学習が要った」のは Pusher の mid vs long の
#   **1事例だけ**だった。Reach に至っては第1層が順位を的中させている（第3章 3.12.3）。
#   このままだと「スケールの違いなら計算で分かる。なぜ学習が要るのか」に対する答えが弱い。
#
#   そこで**総リーチを 0.887 m に固定したまま配分だけを変えた2形態**を用意した:
#     prox（根元重）: 0.500 / 0.250 / 0.137
#     dist（先端重）: 0.137 / 0.250 / 0.500
#   最大リーチが同一なので**第1層は両者に文字通り同じ判定しか返せない**
#   （実測で確認済み: どちらも「総リーチ 0.887 m」「✅ 目標は可動範囲の内側（余裕 10%）」）。
#   スコアが分かれれば「幾何では分からないが co-design なら分かる」事例になる。
#   分かれなければ「配分は効かない」という negative result で限界節が正確になる。どちらでも価値がある。
#
# ⚠️ Reach を先に回す: 第1層が Reach では順位を的中させており、そこを崩せれば証明力が最も高い。
#    Pusher 版は 1000ep 系が空いてから（メモリが 1 run 約 8.7 GB 必要なため同時は2本まで）。
#
# 設定は `tripo_pj_mid`（4.4.5 の中間スケール）と **xml_name 以外すべて同一**にしてある。
#
# usage: launch_pj_dist.sh prox|dist
set -u
cd /userdir/StackelbergPPO
FORM=${1:-}
case "$FORM" in
  prox|dist) XML=tripo_arm_v2c_pj_${FORM} ;;
  *) echo "usage: $0 prox|dist"; exit 1 ;;
esac
RUN=tripo_pjd_${FORM}
if pgrep -f "hydra.run.dir=single_run/$RUN" >/dev/null 2>&1; then
  echo "[$(date '+%F %T')] $RUN は既に稼働中。スキップ。"; exit 0
fi
mkdir -p "single_run/$RUN"
nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
  --no-window --python scripts/choreonoid_train.py \
  cfg=pusher_gearonly xml_name=$XML num_threads=4 max_epoch_num=200 \
  enable_wandb=false fix_skeleton=true seed=0 +robot_param_scale=1 \
  +reward_specs.use_reach=true +reward_specs.target_x=0.8 \
  +reward_specs.target_y=0.0 +reward_specs.target_z=0.15 \
  +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false \
  hydra.run.dir="single_run/$RUN" \
  > "single_run/$RUN/stdout.log" 2>&1 &
echo "[$(date '+%F %T')] $RUN launched (PID $!, 200ep, xml=$XML)"
