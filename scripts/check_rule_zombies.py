#!/usr/bin/env python3
"""指示の層（CLAUDE.md・メモリ・スラッシュコマンド・新しいチャットの始め方）で、
**訂正の注記が付いたのに、訂正される前の命令文が残っている箇所**を並べる（2026-10-20、ユーザー「禁止事項の溜め込み」の精査）。

⛔ なぜ要るか: CLAUDE.md §5-2 ⑤-3 に「プローブの末尾は main() だけにする」が、次の行の訂正（9-257）の後も
命令として残り、10-19 のプローブがそれに従って 30 分残った。「起動 105 秒なので timeout は 1200 秒以上」も
根拠が訂正（実測 2 秒）されたのに命令が残っていた。注記を足すだけでは、先に読まれる命令が勝つ。

見るもの（人が 1 件ずつ判断する。機械は候補を出すだけ）:
  ① 訂正の注記の行の、すぐ前の行（同じ段落）が命令文
  ② 1 行の中に命令文と、それを打ち消す括弧書きの訂正（「（⛔ … 実測 …）」など）が同居
  ③ 訂正の行のすぐ次の行（同じ段落）が命令文（文が行をまたいで続く場合。「（⛔ 実測 2 秒）かかるので、」→「timeout は 1200 秒以上」）
直し方: 命令を今の規則に書き換え、経緯は台帳・デバッグ戦記へ（CLAUDE.md の更新履歴 2026-10-20）。

    python3 scripts/check_rule_zombies.py              # 今の指示の層を見る
    python3 scripts/check_rule_zombies.py --reproduce  # 既知の 2 件（ed644bf で外す前の CLAUDE.md）を拾えるか
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEMORY = Path.home() / '.claude' / 'projects' / str(ROOT).replace('/', '-') / 'memory'

CORR = re.compile(r'⛔\s*\**\s*訂正|訂正（20|\(⛔|（⛔[^）]*(実測|誤り|取り下げ|撤回)')
# ⭐ 見て、命令が今も正しいと判断したもの（理由を必ず書く。CLAUDE.md §5-2 ①「除外した理由を書く」）
KEEP = {
    '1 エピソードで判定しない': '直後の訂正（9-237）が取り下げたのは「頭の 1〜2 話は当てにならない」で、この規則ではない（2026-10-20 確認）',
}
IMPER = re.compile(r'にする|しない|を見込む|必ず|すること|使わない|書かない|を使う|だけにする|禁止|してはいけない|を回す')


def targets():
    fs = [ROOT / 'CLAUDE.md', ROOT / 'docs/リポジトリ説明/新しいチャットの始め方.md']
    fs += sorted((ROOT / '.claude/commands').glob('*.md'))
    fs += sorted(p for p in MEMORY.glob('*.md') if p.name != 'MEMORY.md')
    return [f for f in fs if f.exists()]


def scan(name, text):
    lines = text.split('\n')
    hits = []
    for i, ln in enumerate(lines):
        if not CORR.search(ln):
            continue
        before = ln[:CORR.search(ln).start()]
        if IMPER.search(before):                                  # ② 同じ行に命令と訂正
            hits.append((i + 1, '同じ行', ln.strip()))
        j = i - 1
        while j >= 0 and not lines[j].strip():
            j -= 1
        if j >= 0 and i - j == 1 and IMPER.search(lines[j]) and not CORR.search(lines[j]):   # ① 直前の行
            hits.append((j + 1, '訂正の直前', lines[j].strip()))
        if i + 1 < len(lines) and IMPER.search(lines[i + 1]) and not CORR.search(lines[i + 1]):         # ③ 直後の行
            hits.append((i + 2, '訂正の直後', lines[i + 1].strip()))
    hits = [h for h in hits if not any(k in h[2] for k in KEEP)]
    return [(name, *h) for h in hits]


def reproduce():
    old = subprocess.run(['git', 'show', 'ed644bf^:CLAUDE.md'], cwd=ROOT, capture_output=True, text=True).stdout
    hits = scan('CLAUDE.md@ed644bf^', old)
    want = {'`main()` だけにする': False, '1200 秒以上': False}
    for *_, ln in hits:
        for k in want:
            want[k] |= k in ln
    for k, ok in want.items():
        print(f"  {'✅' if ok else '⛔'} 既知の残骸「{k}」を拾う")
    if not all(want.values()):
        print('⛔ 既知の失敗を拾えない。検査が壊れている'); return 1
    print(f'✅ 既知の 2 件を拾えた（候補は全部で {len(hits)} 件）'); return 0


def main():
    if '--reproduce' in sys.argv:
        return reproduce()
    hits = []
    for f in targets():
        hits += scan(str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else f'memory/{f.name}', f.read_text())
    print(f'見た: {len(targets())} ファイル（CLAUDE.md・新しいチャットの始め方・コマンド・メモリ）')
    if not hits:
        print('✅ 訂正の注記の前に命令が残っている箇所は無い'); return 0
    print(f'⚠️ {len(hits)} 件。1 件ずつ見て、命令がまだ正しいか（訂正と食い違わないか）を判断する:')
    for name, n, kind, ln in hits:
        print(f'  {name}:{n}  [{kind}]  {ln[:150]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
