#!/usr/bin/env bash
# 2026-09-16: fix で「良くなる」パターンを自動化で拾う（実験系譜 9-106）。
#
# ⭐ **狙い**: 9-102 は縦型 A1v で「固定すると悪化」を自動で捕まえた（助言の棄却）。
#    一方 9-27/9-61 の**平面 A1 Reach では固定が良化する**（現行版で −5.83 → −3.85）。
#    ⚠️ **ただし良化の方は人手で回したもので、自動化が拾った実績ではない。**
#    ⭐ **同じ自動化に平面 A1 を通し、今度は「採用」が出るかを見る。**
#
#    これが出れば「**fix は必ずしも良くならないが、並列実行のおかげで
#    良くなる場合を捨てずに拾える**」と言える。⚠️ **出なければその主張はしない。**
#
# ⚠️ `e2e_a1` は 9-105 で版の混入が判明している。**現行版（総リーチ 1.010 m）で統一する。**
#
# 読み方（結果を見る前に決める）:
#   ① 監視が関節3 を検出し、固定版が良化（帯が重ならず）→ ✅ **主張が成立**
#   ② 検出したが悪化 → 🟡 9-102 と同じ。**「必ずしも良くならない」の 2 例目**
#   ③ そもそも検出しない → ⚠️ 使用率が 10 % を超えている。**自動化の適用範囲の話**
#
# 起動: nohup bash scripts/queue_auto13.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_auto13.log
RUN=e2e_a1_reach_auto13
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
# ⚠️ **自分のシェルのコマンドラインが同じ文字列を含むと誤って数える**（Bug 19 と同型）。
#   ⭐ **実体のパスとの AND** にすると、シェルのコマンドラインには一致しない。
njobs(){ ps -eo args 2>/dev/null | awk '/choreonoid_train\.py/ && /install\/bin\/choreonoid/' | wc -l; }

log "=== キュー開始（fix が良くなる側を自動化で拾う）"
while [ "$(njobs)" -ge 2 ]; do sleep 180; done
mkdir -p "single_run/$RUN"
nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
  --no-window --python scripts/choreonoid_train.py \
  cfg=pusher_gearonly xml_name=e2e_a1 num_threads=4 max_epoch_num=200 \
  enable_wandb=false fix_skeleton=true seed=0 +robot_param_scale=1 \
  +reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 \
  +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 \
  +env_specs.check_init_contact=false \
  hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
log "$RUN launched (PID $!)"

# ⭐ 監視デーモン（これが本体。検出 → 固定版の自動起動 → 完走後の自動判定）
sleep 90
nohup python3 scripts/joint_fix_watch.py --run $RUN \
  > single_run/$RUN/joint_fix_watch_stdout.log 2>&1 &
log "監視デーモンを付けた (PID $!)"
