#!/bin/bash
# ⭐ 修論のための学習（⭐ 9-246 から実在の物理の *_phys 37 本。それ以前は 9-232〜9-235 の *_scgc）を、1 本終わるたびに
#   「学習時の環境で再生 → 図 → 結論前の検査」まで済ませておく（系譜 9-236）。学習はしない。
#
# ⭐ なぜ: ユーザーが 10-11 から不在。結果が出るたびに手で再生すると、環境変数の渡し忘れ（9-229）や
#   測り忘れ（9-223 の「離れた組 0」）が起きる。⭐ 判断は人がする。ここは材料をそろえるだけ。
# ⭐ 再生の環境は run のフォルダの `launch_env.json` から自動で読む（choreonoid_train.py が起動時に書く）。
#   ⚠️ 9-236 より前に起動した 2 本（押しの柱の long・mid seed0）にはファイルが無いので、キューの起動行と同じ値を書く。
# ⚠️ 目視（⑤-3）は人がする。検査の「目視」が ⛔ のまま残るのは正しい状態。図は single_run/<run>/trace/view.png
#
# 出力: single_run/postrun_summary.log（run ごとに best・検査 12 項目（不変条件を含む）・反実仮想の再生・仲間うちの外れ値）
set -u
cd /userdir/StackelbergPPO
OUT=single_run/postrun_summary.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $OUT; }
# ⭐ 10-21（系譜 9-270）: 環境変数 RUNS で一覧を差し替えられる（動いている本体を止めずに、後から足した run だけ別に回すため）
RUNS="${RUNS:-tripo_pjp_long_phys tripo_pjp_mid_phys tripo_pjp_long_phys_arm1 tripo_pjp_mid_phys_arm1 tripo_pjdp_prox_phys tripo_pjdp_dist_phys tripo_pjp_long_phys_s2 tripo_pjp_mid_phys_s2 tripo_pjdp_prox_phys_s2 tripo_pjdp_dist_phys_s2 e2e_a1v_pusher_phys e2e_a1v_reach_phys e2e_a1v_pusher_phys_s2 e2e_a1v_reach_phys_s2 hockey_bank9_phys hockey_bank9_phys_s2 tripo_pj_mid_phys tripo_pj_long_phys tripo_pjd_prox_phys tripo_pjd_dist_phys tripo_pjr199_phys e2e_a1_reach_phys e2e_a1_fix3_reach_phys tripo_pj_mid_phys_s2 tripo_pj_long_phys_s2 tripo_pjd_prox_phys_s2 tripo_pjd_dist_phys_s2 tripo_pjr199_phys_s2 e2e_a1_reach_phys_s2 e2e_a1_fix3_reach_phys_s2 tripo_pj_short_phys tripo_pjp_mid_frozen_phys tripo_pjp_mid_frozen_phys_s2 tripo_v3_reach2_phys tripo_arm_v3_pusher_phys tripo_v3_reach2_phys_s2 tripo_arm_v3_pusher_phys_s2 tripo_pjp_short_phys tripo_pjr211p_phys e2e_a1_pusher_phys e2e_a1_pusher_phys_s2 e2e_a1_fix3_pusher_phys e2e_a1_fix3_pusher_phys_s2 tripo_pjd_dist_phys_cg tripo_pjd_dist_phys_cg_s2 tripo_pjdp_dist_phys_cg tripo_pjr211p_phys_s2 tripo_pj_short_phys_cg tripo_pjp_short_phys_cg}"
log "=== 再生と検査の自動化を開始（$(echo $RUNS | wc -w) 本）"
while true; do
  left=0
  for R in $RUNS; do
    D=single_run/$R
    [ -f $D/trace/.postrun_done ] && continue
    left=$((left+1))
    grep -q "training done" $D/log/log_train.txt 2>/dev/null || continue
    best=$(grep -o 'save best checkpoint with rewards [-0-9.]*' $D/log/log_train.txt | tail -1 | awk '{print $NF}')
    log "--- $R 完走（best $best）。再生する"
    mkdir -p $D/trace
    EVAL_RESTORE_DIR=$D EVAL_CHECKPOINT=best USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
      timeout -k 30 2400 /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/record_arm_trace.py \
      > $D/trace_record.log 2>&1
    log "    再生 exit=$? $(grep -h '話ごとの' $D/trace_record.log | tail -1)"
    python3 scripts/plot_run.py $R >> $D/trace_record.log 2>&1
    python3 scripts/check_before_conclusion.py $R 2>/dev/null | grep -v '^合計' | tee -a $OUT > /dev/null
    # ⭐ 2026-10-12（異常を能動的に見つける方法.md の方法 3）: 物理のつまみを 1 つずつ変えて再生し、結果の動きを並べる
    python3 scripts/counterfactual_replay.py $R 2>/dev/null | tee -a $OUT > /dev/null
    # ⭐ 方法 6: 比べる仲間の中で外れた run（新しく終わった run があるたびに全体を見直す）
    { echo "    [仲間うちの外れ値]"; python3 scripts/peer_outliers.py 2>/dev/null | sed 's/^/    /'; } | tee -a $OUT > /dev/null
    touch $D/trace/.postrun_done
    left=$((left-1))
  done
  [ $left -eq 0 ] && break
  sleep 600
done
log "=== 全部済んだ"
