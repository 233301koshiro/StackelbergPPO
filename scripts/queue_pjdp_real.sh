#!/usr/bin/env bash
# 2026-09-23: ⭐⭐ **実寸条件の配分判別を Pusher へ広げる**（実験系譜 9-165 予定）。
#
# ⭐ **9-163 で Reach 側は決着した**: 実寸（ギア 20〜150）でも
#   根元重 [-1.76, -1.45] 対 先端重 [-4.41, -2.88] で**帯が分離**した。
# ⛔ **残るのは Pusher 側である。**旧条件（ギア 20〜400）では両タスクとも分離しており
#   （修論 4.3.3.4。Pusher は 根元重 [41.56, 41.77] 対 先端重 [27.89, 40.51]）、
#   ⛔ **実寸で確かめたのは 2 タスクのうち Reach だけ**という非対称が残っている。
#
# ⚠️ **変数は 1 つだけ。**旧 Pusher（`tripo_pjdp_*`）との差は
#   `cfg=pusher_gearonly → pusher_gearonly_real`（ギア上限 400 → 150）と
#   `xml_name` の `_real` 接尾（実効密度 294.2）のみ。他のフラグは
#   `queue_pjd_pusher.sh` と文字どおり同一にしてある。
#   ⭐ **Reach 側（9-160）が旧 Reach に対して加えた差分と完全に同型である。**
#
# ⭐ 読み方（**結果を見る前に決める**。§5-2 ⑤-2）:
#   ① **帯が分離し、根元重が上** → ⭐⭐ **R3 が実寸条件で両タスクとも成立。**
#        ⭐ 事前の予測はこれ: 実寸はアクチュエータが弱いので、
#          先端に質量を寄せた形態ほど根元の必要トルクが増えて不利になり、**差は旧条件より開く**はず。
#   ② **帯が重なる** → ⛔ **Pusher の順序づけは強すぎるアクチュエータに依存していた。**
#        ⚠️ その場合でも Reach（9-163）は無傷なので、「タスクによって順序づけの成否が違う」と読む。
#   ③ **分離するが向きが逆（先端重が上）** → ⚠️ **タスクによる順位反転。**
#        ⭐ 旧条件では反転しなかったので、**反転は物理の現実性が生む**ことになる。形態を実測して読み直す。
#
# ⚠️ **結論前に `check_before_conclusion.py`（6 項目）と `plot_run.py` を回し、目視記録へ 1 行書く。**
#
# 起動: nohup bash scripts/queue_pjdp_real.sh > /dev/null 2>&1 & disown
# 進捗: single_run/queue_pjdp_real.log
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_pjdp_real.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
njobs(){ ps -eo comm,args 2>/dev/null | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
launch(){ local RUN=$1 XML=$2 S=$3
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_gearonly_real xml_name=$XML num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 \
    +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)"; sleep 60; }

log "=== キュー開始（実寸条件で配分だけを変えた判別・Pusher 各 2 seed）"

# --- 投入前の検算（Bug 23）---
if ! python3 scripts/audit_xml_reach.py \
      assets/mujoco_envs/tripo_arm_v2c_pj_prox_real.xml \
      assets/mujoco_envs/tripo_arm_v2c_pj_dist_real.xml >> "$LOG" 2>&1; then
  log "⛔ XML の公称/実効リーチが食い違う。投入を中止する（Bug 23 と同型）"
  exit 1
fi
log "✅ XML 検算を通過（公称=実効）"

launch tripo_pjdp_prox_real tripo_arm_v2c_pj_prox_real 0
launch tripo_pjdp_dist_real tripo_arm_v2c_pj_dist_real 0
launch tripo_pjdp_prox_real_s2 tripo_arm_v2c_pj_prox_real 1
launch tripo_pjdp_dist_real_s2 tripo_arm_v2c_pj_dist_real 1
log "=== キュー終了"
