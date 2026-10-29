#!/bin/bash
# ⭐ 学習の途中で「あからさまにおかしい戦略」に気づくための見張り（系譜 9-256、2026-10-14 ユーザー提案）。学習はしない・止めない。
#
# なぜ: 不変条件の検査（check_invariants.py: 置いた瞬間の重なり・触れる前に動く・床・壁・制御入力）は
#   完走後の自動再生（queue_postrun_replay.sh）でしか回っていなかった。9-212 のホッケーは置いた瞬間の重なりで
#   パックを押し出していただけだったが、気づいたのは完走後だった。⭐ 当時の best をこの検査に掛けると ⛔ が出る（9-256 で確認）。
#
# やること: 走っている学習（ps の hydra.run.dir）ごとに、epoch 10 と 50 ごとの checkpoint ができたら
#   その epoch を再生（trace_mid/ep<N>.npz）→ check_invariants.py と離れた組のめり込み（10-21 追加）→ single_run/midrun_check.log に 1 行。
#   ⛔ が出ても止めない（学習を止めるのはユーザーの確認が要る。CLAUDE.md §6）。ログを見て人が決める。
# 終わり方: 学習が 1 時間（6 回の見回り）続けて無ければ終わる。
#
#   nohup setsid bash scripts/watch_midrun.sh > /dev/null 2>&1 < /dev/null &
#   tail single_run/midrun_check.log
set -u
# ⭐ 9-329（軸6、ブランチだけの変更）: 作業場所を PROJ_DIR で変えられる。軸6 の run は worktree の cfg と XML で再生する（worktree の single_run は main の single_run へのリンク）。既定は main
cd "${PROJ_DIR:-/userdir/StackelbergPPO}"
OUT=single_run/midrun_check.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $OUT; }
# ⛔ 2026-10-22（系譜 9-283 ③）: cut -c はバイトで切るので日本語の途中で切れて UTF-8 が壊れ、ugrep がログを二進とみなして
#   「⛔⛔」を 0 件と返していた（実際は 17 件。10-22 以前の行は壊れたまま残っている、読むときは grep -a）。文字で切る。
trunc(){ python3 -c 'import sys; n = int(sys.argv[1]); [print(l.rstrip("\n")[:n]) for l in sys.stdin]' "$1"; }
log "=== 途中の見張りを開始（PID $$）"
idle=0
while [ $idle -lt 6 ]; do
  runs=$(ps -eo args | grep "[c]horeonoid_train.py" | grep -o "hydra.run.dir=single_run/[^ ]*" | sed 's|.*/||' | grep -E "${RUN_FILTER:-.}")   # 9-329: RUN_FILTER（軸6 は ^ra15_）
  [ -z "$runs" ] && idle=$((idle+1)) || idle=0
  for R in $runs; do
    D=single_run/$R
    # 一番新しい「10 か 50 の倍数」の checkpoint だけ見る（遅れて始めても溜めない）
    E=$(ls $D/models/epoch_*.p 2>/dev/null | sed 's/.*epoch_0*//; s/\.p$//' | awk '$1==10 || ($1>0 && $1%50==0)' | sort -n | tail -1)
    [ -n "$E" ] || continue
    T=$D/trace_mid/ep$(printf %04d $E)
    [ -f $T.txt ] && continue
    mkdir -p $D/trace_mid
    EVAL_RESTORE_DIR=$D EVAL_CHECKPOINT=$E TRACE_OUT=$T.npz USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
      timeout -k 30 2400 /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/record_arm_trace.py \
      > $T.log 2>&1
    rc=$?
    # ⛔ 2026-10-22（系譜 9-283 ③）: 再生に失敗すると不変条件が ⚠️ になるだけで「✅」と書いていた。失敗は失敗として書く
    #   （.txt を残すので同じ epoch を何度も回さない）
    if [ ! -f $T.npz ]; then
      echo "⚠️ 再生に失敗（exit $rc）" > $T.txt
      log "⚠️ $R ep$E: 再生に失敗（exit $rc）。$(tail -1 $T.log | trunc 120)"
      continue
    fi
    python3 scripts/check_invariants.py --trace $T.npz $R > $T.txt 2>&1
    # ⭐ 2026-10-21（系譜 9-272、ユーザー「はい」）: 離れたリンクどうしのめり込みも見る（完走後の検査 ck_far_pen と同じ量・同じ 50 mm）。
    #   短腕（tripo_pj_short_phys）は ep10 で 118 mm・全 step だったのに、ここでは不変条件しか見ず完走後に気づいた（Bug 72、約 8 時間）
    python3 -c "
import sys; sys.path.insert(0, 'scripts')
from check_self_penetration import measure
w, fr, pr, T, far = measure('$T.npz')
mm, pct = far[0] * 1000, far[1] * 100
pair = 'x'.join(far[3]) if far[3] else '-'
mark = '⛔' if mm >= 50 else ('⚠️' if mm >= 20 else '✅')
print(f'  {mark} 離れた組のめり込み {mm:.0f} mm（step の {pct:.1f} %、{pair}）' + ('' if far[2] else '  ⚠️ 軌跡にカプセルの始点が無い'))
" >> $T.txt 2>&1
    bad=$(grep -c '⛔' $T.txt)
    if [ "$bad" -gt 0 ]; then
      log "⛔⛔ $R ep$E: $(grep '⛔' $T.txt | sed 's/^ *//' | trunc 160 | paste -sd'／')"
    else
      warn=$(grep -c '⚠️' $T.txt)
      log "✅ $R ep$E: $(grep -h '話ごとの' $T.log | tail -1 | trunc 80)$([ "$warn" -gt 0 ] && echo "  （⚠️ $warn 件は $T.txt）")"
    fi
  done
  sleep 600
done
log "=== 学習が 1 時間無いので終わる"
