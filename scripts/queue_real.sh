#!/usr/bin/env bash
# 2026-09-19: 現実的な物理をすべて入れた条件（実験系譜 9-129）。
#
# ⭐ **9-124 の弱点②「物理パラメータに根拠がない」への総合的な答え。**
#   ⚠️ 9-124 の時点で 7 個中 4 個が根拠なしだった。本条件で 3 個が解消する。
#
# | パラメータ | 旧 | ⭐ 新 | 根拠 |
# |---|---|---|---|
# | リンク密度 | 5.0（原論文＝移動ロボット） | **189.9** | アルミ A6061・肉厚 2 mm の薄肉パイプ（9-126） |
# | 接触 solref | 既定（e=0.135） | **0.02 0.4**（e=0.382） | 実物の金属↔箱 e≈0.2〜0.5（9-128） |
# | ギア比の範囲 | 20〜400（出所不明） | **25〜150** | 下限＝必要トルク 19.9 の 1.25 倍／上限＝UR5e の関節（9-129） |
#
# ⚠️ **変数が 3 つ同時に変わる。** ⭐ **ただし分離できる:**
#   対照 `e2e_a1v_pusher`（既存）→ `e2e_a1v_rho_pusher`（密度のみ・走行中）
#   → 本条件（密度＋接触＋ギア比）
#   **rho との差が「接触＋ギア比」に帰属する。**⚠️ 接触とギア比の分離はしない（本条件では）。
#
# 読み方（**結果を見る前に決める**。§5-2 ⑤-2）:
#   ① 学習が成立し gear が内点へ
#        → ✅ ⭐ **実在アクチュエータの範囲で、境界解のない設計が得られた。**
#          **フェーズ0 の主張にとって最も強い結果**
#   ② gear が上限 150 に張り付く
#        → 🟡 **範囲がまだ狭い。**⭐ **ただし「UR5e 級では足りない」という実機に翻訳できる主張になる**
#   ③ 学習が成立しない（cube が動かない）
#        → ⚠️ **腕が 38 倍重く、上限が 2.7 分の 1 になった。**⭐ **それも結果**
#          （「実在部材・実在アクチュエータではこのタスクは解けない」）
#
# ⚠️ **結論を書く前に `check_before_conclusion.py` を回す。⭐ とくに「対象が動いたか」。**
#
# 起動: nohup bash scripts/queue_real.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_real.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
njobs(){ ps -eo comm,args 2>/dev/null | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
launch(){ local RUN=$1 S=$2
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_tripo_v3_real xml_name=e2e_a1v_real num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +env_specs.arm_safe_init=true \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)"; sleep 60; }

log "=== キュー開始（現実的な物理・2 seed）"
launch e2e_a1v_real_pusher 0
launch e2e_a1v_real_pusher_s2 1
log "=== キュー終了"
