#!/bin/bash
# 退避した run（*_noclip・*_clip_olddamp。系譜 9-242・9-244）の途中のチェックポイントを消してディスクを空ける。
# ⭐ 残すもの: best.p・最後の epoch_*.p・ログ・軌跡（trace/）。消すのは途中の epoch_*.p だけ（約 3.8 GB）。
# ⚠️ single_run/ の削除は Claude の権限設定で止めてあるので、ユーザーが手で実行する（2026-10-13、ディスク不足 9.9 GB）。
#   まず `bash scripts/prune_archived_ckpts.sh` で一覧だけ出し、確かめてから `bash scripts/prune_archived_ckpts.sh --delete`。
cd "$(dirname "$0")/../single_run" || exit 1
for d in *_noclip *_clip_olddamp; do
  last=$(ls "$d"/models/epoch_*.p 2>/dev/null | sort | tail -1)
  for f in "$d"/models/epoch_*.p; do
    [ -e "$f" ] && [ "$f" != "$last" ] || continue
    if [ "${1:-}" = "--delete" ]; then rm -f -- "$f"; else echo "$f"; fi
  done
done
[ "${1:-}" = "--delete" ] && df -h . | tail -1
