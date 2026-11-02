#!/bin/bash
# 撤回した旧い物理の run（名前に _phys を含まないもの。系譜 9-246）の途中のチェックポイントを間引いてディスクを空ける。
# ⭐ 残すもの: best*.p・最後の epoch_*.p・100 epoch ごと・100 epoch 以下・ログ・軌跡（trace/）。
#   docs が参照する途中の epoch（10・20・70・200・360）はすべて残る範囲に入る（系譜 9-254）。
# ⭐ 今の物理の run（名前に _phys を含む、または 10-14 以後に学習した。走行中・退避を含む）は触らない。
# ⭐ --phys を付けると逆に今の物理の run だけを同じ規則で間引く（2026-11-02、ディスクの空き 7.7 GB）。
#   学習中の run（choreonoid_train.py の hydra.run.dir に名前が出るもの）は飛ばす。途中の再生（ep150 など）は trace/ の npz に残っている。
# 旧版（2026-10-13）は *_noclip・*_clip_olddamp の途中を消した（348 ファイル、実行済み）。
# ⚠️ single_run/ の削除は Claude の権限設定で止めてあるので、ユーザーが手で実行する。
#   まず `bash scripts/prune_archived_ckpts.sh` で一覧と合計だけ出し、確かめてから `bash scripts/prune_archived_ckpts.sh --delete`。
cd "$(dirname "$0")/../single_run" || exit 1
phys=0; del=0
for a in "$@"; do case "$a" in --phys) phys=1 ;; --delete) del=1 ;; esac; done
n=0; bytes=0
for d in */; do
  d=${d%/}
  # 今の物理 = 名前に _phys がある、または 10-14 以後にログが書かれた（軸6 の ra15_* は名前に _phys が無い。2026-11-02 に気づいた）
  cur=0
  case "$d" in *_phys*) cur=1 ;; esac
  [ -n "$(find "$d/log/log_train.txt" -newermt 2026-10-14 2>/dev/null)" ] && cur=1
  [ $cur -eq $phys ] || continue
  if [ $phys -eq 1 ] && ps -eo args | grep "[c]horeonoid_train.py" | grep -qE "hydra.run.dir=single_run/$d( |$)"; then
    echo "学習中なので飛ばす: $d"; continue
  fi
  last=$(ls "$d"/models/epoch_*.p 2>/dev/null | sort | tail -1)
  for f in "$d"/models/epoch_*.p; do
    [ -e "$f" ] && [ "$f" != "$last" ] || continue
    e=$((10#$(basename "$f" .p | sed 's/epoch_//')))
    [ $((e % 100)) -eq 0 ] || [ "$e" -le 100 ] && continue
    n=$((n + 1)); bytes=$((bytes + $(stat -c%s "$f")))
    if [ $del -eq 1 ]; then rm -f -- "$f"; else echo "$f"; fi
  done
done
echo "対象 $n ファイル・$((bytes / 1073741824)) GB"
[ $del -eq 1 ] && df -h . | tail -1
