#!/usr/bin/env python3
"""修論本文の節参照が実在する節を指しているかを検査する。

⛔ 2026-10-02 に新設。**既存の 5 本はどれもこれを検出できなかった。**
考察章を廃止して旧 第6章 を 第5章 へ繰り上げたとき、本文に残った
「6.4.2 節」への参照 8 箇所を 1 週間誰も検出しなかった。
`check_stale_claims.py` は数値と状態語を見るので、
**存在しない節番号は「古い主張」に見えない。**

CLAUDE.md §5-1-2「検査を書けるならスクリプトにする」に従う。
"""
import re
import sys
from pathlib import Path

DRAFT = Path(__file__).resolve().parent.parent / 'docs/研究応用/修論ドラフト'

# build_thesis_pdf.py の CHAPTER_ORDER と同じ集合（PDF に入るものだけを見る）
BODY = [
    '要旨.md', '第1章_序論.md', '第2章_関連研究.md', '第3章前段_前提.md',
    '第3章_提案手法.md', '第4章_実験および評価.md', '第5章_結論.md',
    '付録A_プロンプト全文.md', '付録C_本研究の限界.md', '付録D_再現情報と補足実験.md',
]


def strip_memo(text: str) -> str:
    """執筆メモとドラフト管理 blockquote を落とす。

    build_thesis_pdf.py が PDF から除外する範囲と合わせる。
    ⭐ これらは日付つきの作業記録なので、**古い節番号が出てよい**。
    """
    text = re.sub(r'## 執筆メモ（本文には含めない）.*', '', text, flags=re.DOTALL)
    text = re.sub(r'^> \*\*ドラフト管理\*\*.*?(?=\n(?!>)|\Z)', '',
                  text, flags=re.DOTALL | re.MULTILINE)
    return text


def collect_sections(texts: dict) -> set:
    """見出しから実在する節番号を集める。'4.3' / '4.3.3.4' / 'C.2' の形。"""
    found = set()
    for text in texts.values():
        for m in re.finditer(r'^#{2,5}\s+(?:付録)?([0-9A-D]+(?:\.[0-9]+)+)\s', text,
                             flags=re.MULTILINE):
            num = m.group(1)
            found.add(num)
            # 上位も実在するものとして扱う（4.3.3.4 があれば 4.3.3 と 4.3 も指せる）
            parts = num.split('.')
            for i in range(2, len(parts)):
                found.add('.'.join(parts[:i]))
    return found


def main() -> int:
    texts = {}
    for name in BODY:
        p = DRAFT / name
        if not p.exists():
            print(f'⛔ {name} が無い')
            return 1
        texts[name] = strip_memo(p.read_text(encoding='utf-8'))

    sections = collect_sections(texts)
    # 章そのものへの参照（「第4章」）は節集合では持たないので別に許す
    chapters = {n.split('.')[0] for n in sections}

    bad = []
    for name, text in texts.items():
        for i, line in enumerate(text.splitlines(), 1):
            # 「4.3 節」「付録C.2 (3)」「6.4.2 節 (8)」などを拾う
            for m in re.finditer(r'(?:付録)?([0-9A-D]+(?:\.[0-9]+)+)\s*節', line):
                num = m.group(1)
                if num not in sections:
                    bad.append((name, i, num, line.strip()[:90]))

    if bad:
        print(f'⛔ 実在しない節への参照 {len(bad)} 件\n')
        for name, i, num, line in bad:
            print(f'  {name}:{i}  → 「{num} 節」が無い')
            print(f'      {line}')
        print(f'\n⭐ 実在する節: {len(sections)} 個 / 章: {sorted(chapters)}')
        return 1

    print(f'✅ 節参照はすべて実在する節を指している（参照先候補 {len(sections)} 節）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
