#!/bin/bash
# main の md の変更を、軸6 のブランチ feat/layer4-robotassembler（worktree /userdir/StackelbergPPO_ra）へ取り込む（系譜 9-329、2026-10-29 ユーザー「ブランチの md も変える」）。
# ⭐ 記録（md）の正は main。ブランチの md は main に追随させるだけで、ブランチでは md を直さない（研究方針「ブランチの使い分け」）。
# ⭐ 取り込むのは、main にあってブランチに無い変更が md・docs/ の下のファイル・「新しく足すだけのファイル」のときに限る。ブランチに既にあるそれ以外（コード・スクリプト）を変える変更が混ざれば取り込まずに知らせる:
#   docs/ の下の md 以外（drawio・png・tsv）を読むのは docs の検査の道具だけで、学習・キュー・見張りは読まない（2026-10-30 に grep で確かめた。主張の木.drawio の変更で取り込みが止まったので広げた）。
#   このスクリプトと scripts/hooks/ も通す: どちらも main の側（cd /userdir/StackelbergPPO）からしか実行せず、worktree の bash は読まない。
#   worktree では L-4 のキュー・見張り・自動再生が bash で動いており、bash は実行中のファイルを読み進めるので、書き換わると壊れる（引き継ぎ）。
# ⭐ worktree に未コミットの変更があるとき・衝突したときも取り込まない（衝突したら取り消す）。
# ⭐ 自動: main でコミットすると git の post-commit フック（scripts/hooks/post-commit を .git/hooks/post-commit へリンク）がこれを回す（2026-10-30、ユーザー「自動にする」）。
#   フックが入っていない環境では、コミットの後・/reflect の区切りで手で回す。
#
#   bash scripts/sync_branch_md.sh
set -u
cd /userdir/StackelbergPPO
WT=/userdir/StackelbergPPO_ra
BR=feat/layer4-robotassembler
[ -d "$WT" ] || { echo "worktree $WT が無い"; exit 0; }
[ "$(git rev-parse --abbrev-ref HEAD)" = "main" ] || { echo "main で回す"; exit 1; }
files=$(git -c core.quotepath=false diff --name-only "$BR"...main)
[ -n "$files" ] || { echo "✅ ブランチは main の md に追随している（取り込むものは無い）"; exit 0; }
# md・docs/ 以外で、ブランチに既にあるファイルを変える・消す変更（A＝新しく足すだけのファイルは、走っている bash にも学習中のコードにも効かないので通す）
other=$(git -c core.quotepath=false diff --name-status "$BR"...main | awk '$1 != "A" {print $NF}' | grep -v -e '\.md$' -e '^docs/' -e '^scripts/sync_branch_md\.sh$' -e '^scripts/hooks/')
if [ -n "$other" ]; then
  echo "⚠️ 取り込まない: main にブランチの md・docs/ 以外のファイルを変える変更がある（$(echo "$other" | wc -l) 件）。L-4 の bash が止まってから worktree で git merge main する:"
  echo "$other" | sed 's/^/    /'
  exit 0
fi
if [ -n "$(git -C "$WT" status --porcelain --untracked-files=no)" ]; then
  echo "⚠️ 取り込まない: worktree に未コミットの変更がある"
  git -C "$WT" status --short --untracked-files=no | sed 's/^/    /'
  exit 0
fi
if git -C "$WT" merge --no-edit -q -m "main の md を取り込む（sync_branch_md.sh、系譜 9-329）" main; then
  echo "✅ main の md $(echo "$files" | wc -l) 件をブランチ $BR へ取り込んだ（$(git -C "$WT" rev-parse --short HEAD)）"
else
  git -C "$WT" merge --abort
  echo "⛔ 衝突したので取り消した。worktree で手で取り込む"
  exit 1
fi
