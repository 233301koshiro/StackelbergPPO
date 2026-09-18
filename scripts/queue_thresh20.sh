#!/usr/bin/env bash
# 2026-09-18: 閾値 20 % で自動化を回し直す（実験系譜 9-120）。
#
# ⭐ **狙い**: 9-119 で「使用率 12 % の関節を固定したら良化した」＝
#    **閾値 10 % が良い助言を取り逃している**と判明した。
#    ⭐ **閾値を 20 % に上げれば 12 % が候補に入り、自動化が良化を拾えるはず。**
#    これが出れば「**fix は必ずしも良くならないが、並列実行のおかげで
#    良くなる場合を捨てずに拾える**」が実績として言える。
#
# ⚠️ **診断本体（diagnose_morphology.py）の既定 10 % は変えない。**既存の結論が動くため。
#    `joint_fix_watch.py --unused-frac 0.20` で監視側だけ上書きする。
#
# ⚠️ **9-105 の版の混入を避ける**: `e2e_a1` は現行版（総リーチ 1.010 m）で統一。
#
# 読み方（**結果を見る前に決める**）:
#   ① 関節3（12 %）を検出し、固定版が良化 → ✅ **主張が完成する**
#   ② 検出したが悪化 → 🟡 閾値 20 % は緩すぎる。**較正の 3 点目になる**
#   ③ 検出しない → ⚠️ 使用率が 20 % も超えている。seed 依存を疑う
#
# ⚠️ **結論を書く前に `check_before_conclusion.py` を回す**（9-114）。
#    ⭐ **とくに「到達しているか」**。9-113 で未到達どうしを比較して取り下げている。
#
# 起動: nohup bash scripts/queue_thresh20.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_thresh20.log
RUN=e2e_a1_reach_th20
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
# ⚠️ comm で絞る（9-117。args だけだと自分のシェルを数える）
njobs(){ ps -eo comm,args 2>/dev/null | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }

log "=== キュー開始（閾値 20 % の較正）"
while [ "$(njobs)" -ge 2 ]; do sleep 180; done
mkdir -p "single_run/$RUN"
nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
  --no-window --python scripts/choreonoid_train.py \
  cfg=pusher_gearonly xml_name=e2e_a1 num_threads=4 max_epoch_num=200 \
  enable_wandb=false fix_skeleton=true seed=1 +robot_param_scale=1 \
  +reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 \
  +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 \
  +env_specs.check_init_contact=false \
  hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
log "$RUN launched (PID $!)"

# ⭐ 監視デーモン（閾値 20 %）
sleep 90
nohup python3 scripts/joint_fix_watch.py --run $RUN --unused-frac 0.20 \
  > single_run/$RUN/joint_fix_watch_stdout.log 2>&1 &
log "監視デーモンを付けた（閾値 20 %、PID $!）"
