#!/usr/bin/env python3
"""本文の引用と参考文献リストを双方向で突き合わせる。

**なぜ要るか**: 修論本文は 22 種を引用していたのに参考文献リストが存在しなかった
（2026-08-28 の点検で発覚）。リストを作った後も、本文に引用を足してリストへの追加を
忘れる事故は起きうる。

**照合の方向を2つに分けている理由**: 本文の引用形式が一定でないため、
本文からの一括抽出だけでは誤検出が支配的になる（実測: システム名を著者と誤認する、
"Lee et al., SIGGRAPH ETech 2024" のように年が会議名の後ろに来る等）。そこで:

  A. **リストを起点に本文を探す**（確実）— リストの各項目について、
     著者姓と発行年が本文中で近接して現れるかを見る。
  B. **本文から抽出してリストに無いものを探す**（取りこぼし検出）—
     ここは誤検出を避けるため、年が姓の直後に来る**厳しい形だけ**を拾う。

使い方:
    python3 scripts/check_citations.py
    終了コード 1 = 引用漏れあり
"""
import glob
import os
import re
import sys

DRAFT = 'docs/研究応用/修論ドラフト'
BIB = os.path.join(DRAFT, '参考文献.md')
NEAR = 200          # 姓と年がこの文字数以内にあれば「引用されている」とみなす

# 著者名として扱わない語（会議名・製品名・見出し語・手法名）
NOT_AUTHORS = {
    'Bug', 'Table', 'Figure', 'ICLR', 'IROS', 'ICRA', 'NeurIPS', 'SIGGRAPH',
    'Python', 'Ubuntu', 'PyTorch', 'Choreonoid', 'Reach', 'Pusher', 'Target',
    'Congress', 'Technologies', 'Emerging', 'Conference', 'International',
    'Journal', 'Computer', 'Computers', 'Artificial', 'Harvard', 'Proximal',
    'BodyGen', 'Free2CAD', 'Teddy', 'RobotSketch', 'Transform2Act', 'Transform',
    'Stackelberg', 'Neural', 'Evolution', 'CEM', 'NGE', 'PPO', 'CAD', 'GDPR',
}


def body_of(path):
    """執筆メモとドラフト管理 blockquote を除いた本文"""
    t = re.split(r'^## 執筆メモ', open(path, encoding='utf-8').read(), flags=re.M)[0]
    return '\n'.join(l for l in t.split('\n') if not l.strip().startswith('>'))


def bib_entries(text):
    """参考文献リストから (姓, 年) を拾う。番号付き行の先頭の姓を第一著者とみなす。"""
    out = []
    for line in text.split('\n'):
        m = re.match(r'^\d+\.\s+([A-Z][A-Za-z\-]+)', line)
        if not m:
            continue
        y = re.search(r'\((\d{4})\)', line)
        if y and m.group(1) not in NOT_AUTHORS:
            out.append((m.group(1), y.group(1), line.strip()))
    return out


def main():
    if not os.path.exists(BIB):
        print(f'❌ 参考文献リストが無い: {BIB}')
        return 1

    bib = bib_entries(open(BIB, encoding='utf-8').read())
    bodies = {os.path.basename(f): body_of(f)
              for f in sorted(glob.glob(f'{DRAFT}/*.md'))
              if os.path.basename(f) != '参考文献.md'}
    all_body = '\n'.join(bodies.values())

    # --- A: リストの各項目が本文で引用されているか ---
    uncited = []
    for surname, year, line in bib:
        hit = False
        for m in re.finditer(re.escape(surname), all_body):
            window = all_body[m.start(): m.start() + NEAR]
            if year in window:
                hit = True
                break
        if not hit:
            uncited.append((surname, year))

    # --- B: 本文にあってリストに無いもの（厳しい形のみ） ---
    tight = re.compile(
        r'([A-Z][A-Za-z\-]+)'
        r'(?:[,、]?\s*(?:et\s+al\.|ら|と\s*[A-Z][A-Za-z\-]+))?'
        r'\s*[（(,、]\s*((?:19|20)\d{2})')
    known = {(s, y) for s, y, _ in bib}
    missing = {}
    for name, t in bodies.items():
        for m in tight.finditer(t):
            key = (m.group(1), m.group(2))
            if m.group(1) in NOT_AUTHORS or key in known:
                continue
            # 第二著者として載っている可能性（本文「Mishra と Chakrabarty（2025）」に対し
            # リストは「Mishra, & Chakrabarty (2025)」）。同じ年の項目の本文に姓があれば既知とみなす。
            if any(y == key[1] and re.search(rf'\b{re.escape(key[0])}\b', line)
                   for _, y, line in bib):
                continue
            missing.setdefault(key, set()).add(name)

    print(f'参考文献リスト: {len(bib)} 件 / 本文: {len(bodies)} ファイル\n')
    if missing:
        print('❌ 本文にあるがリストに無い（引用漏れ）')
        for (a, y), fs in sorted(missing.items()):
            print(f'    {a} {y}   出現: {", ".join(sorted(fs))}')
        print()
    if uncited:
        print('－ リストにあるが本文で引用が見つからない（表記ゆれか、余分な項目）')
        for a, y in uncited:
            print(f'    {a} {y}')
        print()

    # 実際の未確認マークは `[要確認: 何が不明か]` のようにコロン付きで書く。
    # 説明文中の言及（バッククォートで囲んだ `[要確認]`）や完了済みチェック項目を拾わないよう、
    # **コロンを必須**にする（2026-08-28: 全件確認後も 2 と表示され続けたため厳密化）。
    todo = len(re.findall(r'\[要確認\s*[:：]', open(BIB, encoding='utf-8').read()))
    if todo:
        print(f'⚠️ 書誌情報が未確認の箇所: {todo} 件（提出前に一次情報で確認すること）')
    if not missing and not uncited:
        print('✅ 本文の引用とリストは双方向で一致している。')
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
