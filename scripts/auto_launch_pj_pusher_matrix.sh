#!/usr/bin/env bash
# 2026-08-03: PJ実験の Pusher 版を起動し、「3形態 × 2タスク」の判定マトリクスを完成させる。
#
# 狙い（第6章6.5-3「E2E・マトリクス判定」への回答）:
#   同一形態を3スケール（0.403 / 0.887 / 1.451 m）に固定し、Reach と Pusher の
#   両タスクで判定させる。予測される順位は次の通りで、**タスクによって順位が
#   入れ替わる**ことが示せれば、「判定器は形態の絶対的な良し悪しではなく
#   タスクへの適合を見ている」という本研究の中心主張を単一の表で示せる。
#
#     形態        Reach(目標0.8m)          Pusher(cube左面 x=0.85)
#     0.403m      届かない → 最下位         届かない → 最下位
#     0.887m      必要十分 → 1位と予測      ぎりぎり届く → 2位と予測
#     1.451m      過剰な慣性 → 2位と予測    先端速度∝長さ → 1位と予測
#
# 設定は M2b_gearonly（rrbot の gearonly Pusher）と同一レシピ。
#   ⚠️ arm_safe_init=true は必須。長腕1.451mは静止姿勢で cube(x=0.85〜1.15)に
#      食い込むため、これがないと初期接触ペナルティで学習が成立しない
#      （M2 が全エピソード -50 に張り付いた事例と同じ罠、実験系譜.md 第7段）。
#   ⚠️ max_epoch_num=200 は Reach 側の PJ 実験と予算を揃えたもの。判定（順位付け）
#      には 4.4.2 の較正（約100ep で順位確定）から十分だが、**絶対値は未収束**
#      である点に注意（M2b は 200ep 時点で未収束だった、実験系譜.md 9-5）。
#
# 使い方: nohup bash scripts/auto_launch_pj_pusher_matrix.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/pj_pusher_matrix.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# 先行する PJ Reach 2本の完走を待つ（3並列を超えないため）
WAIT_RUNS=("tripo_pj_long_s2" "tripo_pj_mid")
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
say "先行 run が終了。Pusher 版を起動する"

PUSHER="+reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 \
+reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true"

launch() {  # $1=xml $2=run名
  mkdir -p "single_run/$2"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_gearonly xml_name=$1 num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=0 +robot_param_scale=1 \
    $PUSHER \
    hydra.run.dir="single_run/$2" \
    > "single_run/$2/stdout.log" 2>&1 &
  say "$2 launched (PID $!)"
}

launch tripo_arm_v2c_pj_short tripo_pjp_short
sleep 90
launch tripo_arm_v2c_pj_mid   tripo_pjp_mid
sleep 90
launch tripo_arm_v2c_pj_long  tripo_pjp_long

sleep 600
say "--- 起動確認 ---"
for run in tripo_pjp_short tripo_pjp_mid tripo_pjp_long; do
  ep=$(grep -E "^[0-9]+\s+T_sample" "single_run/$run/stdout.log" 2>/dev/null | tail -1 | awk '{print $1}')
  say "$run: ep=${ep:-未到達}"
done
say "done"
