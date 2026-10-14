#!/bin/bash
# 撤回した旧い物理の run（名前に _phys を含まないもの。系譜 9-246）の途中のチェックポイントを間引いてディスクを空ける。
# ⭐ 残すもの: best*.p・最後の epoch_*.p・100 epoch ごと・100 epoch 以下・ログ・軌跡（trace/）。
#   docs が参照する途中の epoch（10・20・70・200・360）はすべて残る範囲に入る（系譜 9-254）。
# ⭐ 名前に _phys を含む run（今の物理・走行中・退避）は触らない。
# 旧版（2026-10-13）は *_noclip・*_clip_olddamp の途中を消した（348 ファイル、実行済み）。
# ⚠️ single_run/ の削除は Claude の権限設定で止めてあるので、ユーザーが手で実行する。
#   まず `bash scripts/prune_archived_ckpts.sh` で一覧と合計だけ出し、確かめてから `bash scripts/prune_archived_ckpts.sh --delete`。
cd "$(dirname "$0")/../single_run" || exit 1
n=0; bytes=0
for d in */; do
  d=${d%/}
  case "$d" in *_phys*) continue ;; esac
  last=$(ls "$d"/models/epoch_*.p 2>/dev/null | sort | tail -1)
  for f in "$d"/models/epoch_*.p; do
    [ -e "$f" ] && [ "$f" != "$last" ] || continue
    e=$((10#$(basename "$f" .p | sed 's/epoch_//')))
    [ $((e % 100)) -eq 0 ] || [ "$e" -le 100 ] && continue
    n=$((n + 1)); bytes=$((bytes + $(stat -c%s "$f")))
    if [ "${1:-}" = "--delete" ]; then rm -f -- "$f"; else echo "$f"; fi
  done
done
echo "対象 $n ファイル・$((bytes / 1073741824)) GB"
[ "${1:-}" = "--delete" ] && df -h . | tail -1
