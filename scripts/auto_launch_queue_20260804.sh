#!/usr/bin/env bash
# 2026-08-04: PJ実験の Pusher 版3本が完走してスロットが空いたら、次の3本を起動する。
# tripo_arm_v3_pusher（8/7頃まで稼働）と合わせて4並列に収める。
#
#  ① tripo_arm_v2c_pusher_ns2 : skel_transform_nsteps=2 の妥当性検証（方針レビュー_2026-08-03 ⑤）
#  ② tripo_arm_v2c_reach_s2   : Reach の seed=1。「各タスク1シードのみ」の限界を1つ解消（6.4.2）
#  ③ tripo_pj_mid_s2          : 判別解像度テストの seed=1。マトリクスの seed 被覆を揃える
#
# 使い方: nohup bash scripts/auto_launch_queue_20260804.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_20260804.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

WAIT_RUNS=("tripo_pjp_short" "tripo_pjp_mid" "tripo_pjp_long")
say "先行 run の完走を待機: ${WAIT_RUNS[*]}"
while true; do
  remaining=0
  for r in "${WAIT_RUNS[@]}"; do
    grep -q "All workers terminated" "single_run/$r/stdout.log" 2>/dev/null && continue
    pgrep -f "hydra.run.dir=single_run/$r\$" > /dev/null 2>&1 && remaining=$((remaining+1))
  done
  [ "$remaining" -eq 0 ] && break
  sleep 300
done
say "先行 run が終了。次の3本を起動する"

launch() {  # $1=run名 $2...=override群
  local name="$1"; shift
  mkdir -p "single_run/$name"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    "$@" hydra.run.dir="single_run/$name" \
    > "single_run/$name/stdout.log" 2>&1 &
  say "$name launched (PID $!)"
}

REACH_OPT="+reward_specs.use_reach=true +reward_specs.target_x=0.8 \
+reward_specs.target_y=0.0 +reward_specs.target_z=0.15 \
+reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false"

# ① ステップ数の検証。元の tripo_arm_v2c_pusher と skel_transform_nsteps 以外は同一設定。
#    元は1000epだが、比較は ep199 同士で行うので200epで足りる（元のep199は 130.41）。
#    ⚠️ skel_transform_nsteps は config.py が FLAG からのみ読む（cfg yml の値は無視される）
#       ため、CLI で明示的に渡す必要がある。
say "--- ① tripo_arm_v2c_pusher_ns2: skel_transform_nsteps=2 (200ep) ---"
launch tripo_arm_v2c_pusher_ns2 \
  cfg=pusher_tripo_v2c xml_name=tripo_arm_v2c fix_skeleton=true \
  max_epoch_num=200 num_threads=4 enable_wandb=false seed=0 \
  skel_transform_nsteps=2

sleep 90

# ② Reach の seed=1。元の tripo_arm_v2c_reach と seed 以外は同一。
say "--- ② tripo_arm_v2c_reach_s2: Reach seed=1 (1000ep) ---"
launch tripo_arm_v2c_reach_s2 \
  cfg=pusher_tripo_v2c xml_name=tripo_arm_v2c fix_skeleton=true \
  max_epoch_num=1000 num_threads=4 enable_wandb=false seed=1 \
  $REACH_OPT

sleep 90

# ③ 判別解像度テストの seed=1。元の tripo_pj_mid と seed 以外は同一。
say "--- ③ tripo_pj_mid_s2: 判別解像度 seed=1 (200ep) ---"
launch tripo_pj_mid_s2 \
  cfg=pusher_gearonly xml_name=tripo_arm_v2c_pj_mid num_threads=4 \
  max_epoch_num=200 enable_wandb=false fix_skeleton=true seed=1 \
  +robot_param_scale=1 $REACH_OPT

sleep 600
say "--- 起動確認 ---"
for run in tripo_arm_v2c_pusher_ns2 tripo_arm_v2c_reach_s2 tripo_pj_mid_s2; do
  ep=$(grep -E "^[0-9]+\s+T_sample" "single_run/$run/stdout.log" 2>/dev/null | tail -1 | awk '{print $1}')
  say "$run: ep=${ep:-未到達}"
done
# ① は起動直後に nsteps が効いているか確認できる（保存された hydra config を見る）
ns=$(grep -oP '^skel_transform_nsteps: \K\d+' single_run/tripo_arm_v2c_pusher_ns2/.hydra/config.yaml 2>/dev/null)
say "① の skel_transform_nsteps 実効値 = ${ns:-取得失敗}（2 なら上書き成功）"
say "done"
