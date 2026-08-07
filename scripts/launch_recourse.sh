#!/usr/bin/env bash
# 2026-08-07: 診断フィードバックの閉ループ検証（第4章「軸1〜4」再編の軸3）。
#
# 問い: 診断が返す反実仮想的な助言（「腕全体を N 倍に伸ばしてください」）に**そのまま従うと**、
#   形態は本当に達成可能な状態へ遷移するか。これまで検証されていたのは「不適合を検出できる」
#   ところまでで、「助言が有効か」は誰も確かめていなかった。
#
# 準備段階で既にバグが1件出ている（Bug 22）:
#   助言の倍率が四捨五入で**切り捨て**られており、表示「1.98 倍以上」に従って直した形態を
#   同じ診断が再び却下した（必要 1.9846 倍・実際は 2 mm 足りない）。切り上げに修正し、
#   さらに余裕 10% を含めた推奨値を併記するようにした。
#
# 本実験が測るのは、修正後の助言のうち**下限ちょうど**（1.99 倍 = 総リーチ 0.802 m、余裕 0%）で
#   タスクが達成できるかである。下限では腕を伸ばしきった特異姿勢でしか目標に触れられないため、
#   Reach のように目標で静止することを要求するタスクでは苦しいと予想される。
#   これが確かめられれば「余裕 10% を推奨値として併記する」という設計判断の実験的根拠になる。
#   達成できてしまえば推奨値は保守的だったというだけで、助言の有効性はより強く言える。どちらでも価値がある。
#
# 比較の3点（リンク長以外すべて同一。capsule 半径・ギア比・目標位置は据え置き）:
#   tripo_pj_short  1.00 倍 / 0.403 m / 診断=❌不適合 / 実測 -396.92
#   tripo_pjr_min   1.99 倍 / 0.802 m / 診断=✅余裕 0%  / ← 本実験
#   tripo_pj_mid    2.20 倍 / 0.887 m / 診断=✅余裕 10% / 実測 -3.42（推奨値 2.19 倍にほぼ一致）
#
# 設定は tripo_pj_mid と **xml_name 以外すべて同一**（.hydra/overrides.yaml で照合済み）。
#
# usage: bash scripts/launch_recourse.sh
set -u
cd /userdir/StackelbergPPO
RUN=tripo_pjr_min
XML=tripo_arm_v2c_pj_recmin

if [ -d "single_run/$RUN/log" ]; then
  echo "[$(date '+%F %T')] $RUN は既に存在する。上書きしないので中止。"; exit 0
fi
# メモリは 1 run あたり約 8.7 GB 必要。空きが足りないまま起動すると OOM で
# **稼働中の他の学習まで巻き込んで落ちる**ので、必ず事前に確認する。
FREE=$(free -g | awk 'NR==2{print $7}')
if [ "$FREE" -lt 10 ]; then
  echo "[$(date '+%F %T')] 空きメモリ ${FREE} GB。10 GB 未満なので起動しない。"; exit 1
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
echo "[$(date '+%F %T')] $RUN launched (PID $!, 200ep, xml=$XML, 空き ${FREE} GB)"
