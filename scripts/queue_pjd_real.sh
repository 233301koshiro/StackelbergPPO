#!/usr/bin/env bash
# 2026-09-22: ⭐⭐ **実寸条件で「配分だけを変えた判別」が成立するか**（実験系譜 9-160）。
#
# ⭐ **9-157 でタスク識別（R2）は実在条件でも成立すると分かった。**
#   ⛔ **残るのは形態の順序づけ（R3・修論 4.3.3。第4章の 43.7 %）である。**
#
# ⚠️ **形態の順位は旧条件ですら帯が重なる**（A1v/A2v/B2v。9-34）。
#   ⭐ **唯一、旧条件で帯が分離しているのが「総リーチ固定・配分だけを変えた対」**（修論 4.3.3.4）:
#       Reach  根元重 [-1.81, -1.65]  対  先端重 [-5.75, -4.39]  ⭐ 分離
#       Pusher 根元重 [41.56, 41.77]  対  先端重 [27.89, 40.51]  ⭐ 分離
#   ⭐⭐ **しかもこれは「幾何が原理的に区別できない形態対を順序づける」という本研究固有の主張**
#     （第1層は両者に文字どおり同一の出力を返す）。
#
# ⚠️ **変数はギア上限のみ**（400 → 150）。密度は XML 側（`*_real`）。
#   ⭐ ギア範囲は **この形態で導出し直した**（9-129 の 25〜150 は e2e_a1v 用なので流用しない）:
#       実効密度 294.2（アルミ肉厚 2 mm）→ 腕 1.311 kg
#       自重 4.84 ＋ 押し 11.74 ＝ 必要 16.58 N·m → 下限 20（旧と同じ）／上限 150（UR5e）
#
# 読み方（**結果を見る前に決める**。§5-2 ⑤-2）:
#   ⭐ ① **根元重と先端重の帯が分離し、根元重が上**
#        → ⭐⭐ **形態の順序づけも実在条件で成立する。R2（9-157）と合わせて判定器の両輪が揃う**
#   ⛔ ② **帯が重なる**
#        → ⛔ **従来の順序づけは非現実的な物理（強すぎるアクチュエータ）に依存していた**
#   ⚠️ ③ **分離するが向きが逆**
#        → ⚠️ **順序づけはするが機序が違う。**形態を実測して読み直す
#
# ⚠️ **結論前に `check_before_conclusion.py`（6 項目）と `plot_run.py` を回し、目視記録へ 1 行書く。**
#
# 起動: nohup bash scripts/queue_pjd_real.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_pjd_real.log
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
    +reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 \
    +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 \
    +env_specs.check_init_contact=false \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)"; sleep 60; }

log "=== キュー開始（実寸条件で配分だけを変えた判別・Reach 各 2 seed）"
launch tripo_pjd_prox_real tripo_arm_v2c_pj_prox_real 0
launch tripo_pjd_dist_real tripo_arm_v2c_pj_dist_real 0
launch tripo_pjd_prox_real_s2 tripo_arm_v2c_pj_prox_real 1
launch tripo_pjd_dist_real_s2 tripo_arm_v2c_pj_dist_real 1
log "=== キュー終了"
