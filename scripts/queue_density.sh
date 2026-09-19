#!/usr/bin/env bash
# 2026-09-19: リンク密度に根拠を与えた条件（実験系譜 9-126）。
#
# ⭐ **9-124 の弱点② への最初の一手。**
#   リンク密度 5.0 kg/m³ は原論文（移動ロボット）からの引き継ぎで根拠が無かった。
#   ⭐ **薄肉パイプの実効密度（アルミ A6061・肉厚 2 mm）＝ 189.9 kg/m³** へ。
#
# ⚠️ **既存 run とは物理が違うので直接比較しない。条件を足す形。**（9-105 の版の混入を避け、
#   元の e2e_a1v.xml は変更していない。新 XML は e2e_a1v_rho.xml）
#
# 実測した効果:
#   腕の質量        0.108 kg → **4.096 kg**
#   自重保持トルク   0.25 N·m → **9.31 N·m**
#   cube/腕 の比    25.0 倍  → **0.66 倍**（cube 2.7 kg と同程度になった）
#   ギア上限との比   1632 倍  → ⭐ **43 倍**（桁が 2 つ下がった）
#
# 読み方（**結果を見る前に決める**。§5-2 ⑤-2）:
#   ① 学習が成立し、gear が内点へ
#        → ✅ **密度の是正だけで境界解が消える。**⭐ 9-81（反射慣性）と同じ効果が
#          **より単純な変更で得られたことになる**
#   ② 学習は成立するが gear は境界のまま
#        → 🟡 **密度は必要だが十分ではない。**9-81 の反射慣性と併用する
#   ③ 学習が成立しない（押せない）
#        → ⚠️ **腕が 38 倍重くなったので、ギア比 400 でも動かない可能性がある。**
#          ⭐ **それも結果**（「この設計空間では実在材料の腕を動かせない」）
#
# ⚠️ **結論を書く前に `check_before_conclusion.py` を回す**（9-114）。
#    ⭐ **とくに「対象が動いたか」。** 動いていなければ ③ である。
#
# 起動: nohup bash scripts/queue_density.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_density.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
njobs(){ ps -eo comm,args 2>/dev/null | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
launch(){ local RUN=$1 S=$2
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_tripo_v3 xml_name=e2e_a1v_rho num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +env_specs.arm_safe_init=true \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)"; sleep 60; }

log "=== キュー開始（リンク密度の根拠づけ・2 seed）"
launch e2e_a1v_rho_pusher 0
launch e2e_a1v_rho_pusher_s2 1
log "=== キュー終了"
