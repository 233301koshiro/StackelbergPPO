#!/usr/bin/env python3
"""修論の見出しを章順に書き出す（`docs/研究応用/修論ドラフト/構成.md` の一覧を再生成する）。

**なぜスクリプトにするか**: 構成の一覧を手で書くと、節を足したときに必ず腐る。
2026-09-13 だけで 3 節（3.6.1・3.7.1・3.12.5・6.2.2）を新設し、参照の繰り下げが 3 回起きた。
**一覧は生成物にして、説明文だけを人が持つ。**

    python3 scripts/dump_thesis_outline.py            # 標準出力へ
    python3 scripts/dump_thesis_outline.py --check    # 構成.md と食い違わないか検査
"""
import argparse
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFT = ROOT / 'docs/研究応用/修論ドラフト'
OUTLINE = DRAFT / '構成.md'
# ⚠️ 2026-09-30 の再編で独立した考察章（旧 第5章）を廃止し、旧 第6章（結論）が
# 第5章へ繰り上がった。付録は build_thesis_pdf.py の CHAPTER_ORDER と揃える。
FILES = ['第1章_序論', '第2章_関連研究', '第3章前段_前提', '第3章_提案手法',
         '第4章_実験および評価', '第5章_結論',
         '付録A_プロンプト全文', '付録D_再現情報と補足実験']
BEGIN, END = '<!-- OUTLINE:BEGIN -->', '<!-- OUTLINE:END -->'


def outline() -> str:
    out = []
    for f in FILES:
        p = DRAFT / f'{f}.md'
        if not p.exists():
            out.append(f'- ⚠️ {f}.md が無い'); continue
        s = io.open(p, encoding='utf-8').read().split('## 執筆メモ')[0]
        out.append(f'\n**{f.replace("_", " ")}**\n')
        for lv, t in re.findall(r'^(#{2,4}) (.+)$', s, re.M):
            out.append('  ' * (len(lv) - 2) + f'- {t.strip()}')
    return '\n'.join(out).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='構成.md の一覧と突き合わせる')
    a = ap.parse_args()
    cur = outline()
    if not a.check:
        print(cur); return 0
    if not OUTLINE.exists():
        print('❌ 構成.md が無い'); return 1
    s = io.open(OUTLINE, encoding='utf-8').read()
    if BEGIN not in s or END not in s:
        print(f'❌ 構成.md に {BEGIN} / {END} が無い'); return 1
    old = s.split(BEGIN)[1].split(END)[0].strip()
    if old == cur:
        print('✅ 構成.md の一覧は本文と一致している'); return 0
    print('❌ **構成.md の一覧が古い。** 次で更新する:')
    print('   python3 scripts/dump_thesis_outline.py --write')
    o, c = set(old.split('\n')), set(cur.split('\n'))
    for x in sorted(c - o)[:8]: print(f'   + {x.strip()}')
    for x in sorted(o - c)[:8]: print(f'   - {x.strip()}')
    return 1


if __name__ == '__main__':
    if '--write' in sys.argv:
        s = io.open(OUTLINE, encoding='utf-8').read()
        s = s.split(BEGIN)[0] + BEGIN + '\n\n' + outline() + '\n\n' + END + s.split(END)[1]
        io.open(OUTLINE, 'w', encoding='utf-8').write(s)
        print('✅ 構成.md の一覧を更新した'); sys.exit(0)
    sys.exit(main())
