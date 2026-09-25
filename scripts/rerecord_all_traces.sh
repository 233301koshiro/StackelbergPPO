#!/usr/bin/env bash
# 2026-09-23: **全 run の軌跡を取り直す**（§5-2 ⑤-3 の積み残し 110 本 ＋ 半径を持たない既存 45 本）。
#
# ⭐ 目的は 2 つ。
#   ① **目視記録の空欄を埋める**（155 本中 110 本が未記入。うち 97 本は台帳に載っている＝主張の根拠）
#   ② ⭐⭐ **`record_arm_trace.py` が `geom_size`（リンク半径）を保存するようになった**ので、
#      `check_cube_penetration.py` が**表面どうしの食い込みを確定判定できる**（Bug 45 の未確認部分）。
#      ⛔ 既存の軌跡は半径を持たないため、軸だけの下限判定しかできなかった。
#
# ⚠️ **学習と同時に回してよい。**1 本 10 秒・1 スレッドで、GPU は使わない。
# ⚠️ **パイプへ繋がない。ファイルへリダイレクトする**（Bug 43）。
#
# ⚠️⚠️ **`timeout` は必ず `-k` を付ける**（2026-09-25 に踏んだ）。
# **Choreonoid は SIGTERM を無視する**ので、`timeout 1800` だけだと
# 時間切れの後もプロセスが生き残り、**バッチが 1 本で 1 日 16 時間止まった**。
# ⭐ `-k 30` で 30 秒後に SIGKILL を送る。⭐ **1 本 10 秒なので上限は 600 秒で足りる。**
#
# 起動: nohup bash scripts/rerecord_all_traces.sh > /dev/null 2>&1 & disown
# 進捗: single_run/rerecord_traces.log
set -u
cd /userdir/StackelbergPPO
LOG=single_run/rerecord_traces.log
: > "$LOG"
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

n=0; ok=0; ng=0
for d in single_run/*/models; do
  r=$(basename "$(dirname "$d")")
  # checkpoint が無い run は飛ばす
  ls "$d"/epoch_*.p >/dev/null 2>&1 || { log "$r: checkpoint 無し。飛ばす"; continue; }
  # ⭐ 既に geom_size 入りで録り直してあるものは飛ばす（中断しても続きから回せる）
  if python3 - "$r" <<'PYEOF' 2>/dev/null
import sys, numpy as np, pathlib
f = pathlib.Path('single_run')/sys.argv[1]/'trace'/'arm_trace.npz'
sys.exit(0 if f.exists() and 'geom_size' in np.load(f, allow_pickle=True) else 1)
PYEOF
  then log "⏭  $r: 録り直し済み"; continue; fi
  n=$((n+1))
  if EVAL_RESTORE_DIR="single_run/$r" EVAL_CHECKPOINT=best USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
       timeout -k 30 600 /choreonoid_ws/install/bin/choreonoid --no-window \
       --python scripts/record_arm_trace.py > "single_run/$r/trace_record.log" 2>&1
  then
    if [ -f "single_run/$r/trace/arm_trace.npz" ]; then
      ok=$((ok+1)); log "✅ $r"
    else
      ng=$((ng+1)); log "⛔ $r: 完走したが npz が無い"
    fi
  else
    # ⚠️ Choreonoid は sys.exit(0) を受け付けず exit 137 になることがある（Bug 43）。
    #   終了コードだけで判断せず、npz が出来ているかを見る。
    if [ -f "single_run/$r/trace/arm_trace.npz" ]; then
      ok=$((ok+1)); log "✅ $r（終了コードは非 0 だが npz はある。Bug 43）"
    else
      ng=$((ng+1)); log "⛔ $r: 失敗。single_run/$r/trace_record.log を見ること"
    fi
  fi
done
log "=== 終了: 対象 $n / 成功 $ok / 失敗 $ng"
