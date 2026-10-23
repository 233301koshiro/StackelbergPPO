#!/usr/bin/env python3
"""修論本文の節参照が実在する節を指しているかを検査する。

⛔ 2026-10-02 に新設。**既存の 5 本はどれもこれを検出できなかった。**
考察章を廃止して旧 第6章 を 第5章 へ繰り上げたとき、本文に残った
「6.4.2 節」への参照 8 箇所を 1 週間誰も検出しなかった。
`check_stale_claims.py` は数値と状態語を見るので、
**存在しない節番号は「古い主張」に見えない。**

CLAUDE.md §5-1-2「検査を書けるならスクリプトにする」に従う。

⭐ 2026-10-21 に**欠番**の検査を足した。第4章に 4.3.1・4.3.3.2 が無く 4.3.2 から
始まる節が残っていたのを、人が読んで見つけた（判定 md 型 26・型 27）。参照の検査は
「指した先が在るか」しか見ないので、**在るべき節が無い**ことは拾えない。
項が 1 つだけの節は判断が要るので ⚠️ を出すだけにする。
`--reproduce` で既知の欠番を拾えるかを確かめる。
"""
import re
import sys
from pathlib import Path

DRAFT = Path(__file__).resolve().parent.parent / 'docs/研究応用/修論ドラフト'

# build_thesis_pdf.py の CHAPTER_ORDER と同じ集合（PDF に入るものだけを見る）
BODY = [
    '要旨.md', '第1章_序論.md', '第2章_関連研究.md', '第3章前段_前提.md',
    '第3章_提案手法.md', '第4章_実験および評価.md', '第5章_結論.md',
    '付録A_プロンプト全文.md', '付録D_再現情報と補足実験.md',
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


# 意図して空けている番号と、その理由。⛔ 件数を 0 にするために足さない（CLAUDE.md §5-2 ①）
INTENTIONAL_GAPS = {
    '付録B': '付録は A と D だけ。D を B に改めるかは 50 ページ化の判断と一緒に決める（2026-10-21、引き継ぎ）',
    '付録C': '同上',
}


def collect_headings(texts: dict) -> list:
    """文書順の番号の列。章は '4'・'D'、節は '4.3.2'・'D.1' の形。"""
    out = []
    for text in texts.values():
        for line in text.splitlines():
            m = re.match(r'^#\s+(?:第(\d+)章|付録([A-Z]))', line)
            if m:
                out.append(m.group(1) or m.group(2))
                continue
            m = re.match(r'^#{2,5}\s+([0-9A-Z]+(?:\.[0-9]+)+)\s', line)
            if m:
                out.append(m.group(1))
    return out


def find_gaps(headings: list):
    """親ごとに子の番号が 1, 2, … と抜けなく並ぶかを見る。

    戻り値: (欠番, 項が 1 つだけの親)。欠番は '4.3.1' や '付録B' の形。
    """
    gaps, singles = [], []
    chapters = [h for h in dict.fromkeys(h for h in headings if '.' not in h)]
    nums = [int(c) for c in chapters if c.isdigit()]
    letters = [c for c in chapters if c.isalpha()]
    if nums:
        gaps += [f'第{n}章' for n in range(1, max(nums) + 1) if n not in nums]
    if letters:
        gaps += [f'付録{chr(c)}' for c in range(ord('A'), ord(max(letters)) + 1)
                 if chr(c) not in letters]
    children = {}
    for h in headings:
        if '.' in h:
            parent, last = h.rsplit('.', 1)
            children.setdefault(parent, []).append(int(last))
    for parent, ks in children.items():
        for n in range(1, max(ks) + 1):
            if n not in ks:
                gaps.append(f'{parent}.{n}')
        if len(set(ks)) == 1 and parent.count('.') >= 1:
            singles.append(parent)
    return gaps, singles


# ⭐ 派生の md（2026-10-23 追加）。本文の外から修論の節を指している md。
# 10-21 夜の棚卸しで想定問答に「今は無い節」への参照が 15 か所見つかったが、本検査は PDF に入る本文しか
# 見ていなかったので拾えなかった（10-23 に手で直した、目次の記録）。D-6 で節を刈るとまた起きるので検査にする。
# 見出し（想定問答なら Q）ごとに出す — 親の節を失った Q が分かる（md の木の段 2、引き継ぎ 10-23）。
DERIVED = [
    Path(__file__).resolve().parent.parent / 'docs/研究応用/執筆/想定問答.md',
]

# 修論の節を指していると読める形だけを拾う。数値（2.19 倍）や表番号（表 4.12）と区別するため、
# 2 桁の番号は「修論」「第N章」「付録」の接頭辞か「節」「項」の接尾辞があるものだけ、3 桁以上は裸でも拾う
REF_PATTERNS = [
    r'(?:修論|第\d+章)\s*(?:付録\s*)?(?:[A-D]\s+)?([0-9A-D]+(?:\.[0-9]+)+)',
    r'付録\s*[A-D]?\s*([A-D](?:\.[0-9]+)+)',
    r'(?<![0-9A-Za-z.])([0-9A-D]+(?:\.[0-9]+)+)\s*(?:節|項)',
    r'(?<![0-9A-Za-z.\-])([1-5](?:\.[0-9]+){2,3})(?![0-9.])',
]
# 直前にあれば節参照ではない（表・図の番号、系譜の段、Bug、CLAUDE.md の節、版）
NOT_SECTION = re.compile(r'(表|図|式|系譜|Bug|§|v|版)\s*$')


def derived_refs(text: str, sections: set) -> list:
    """派生の md から、実在しない修論の節への参照を (見出し, 行, 番号, 行の頭) で返す。

    「旧 4.3.5」のように**外したことを明記した経緯の文**と、取り消し線（~~…~~）の中は数えない。
    """
    bad, head = [], ''
    for i, line in enumerate(text.splitlines(), 1):
        if re.match(r'^#{1,2}\s', line) or re.match(r'^#{3,4}\s+Q', line):
            head = line.lstrip('#').strip()[:40]
        body = re.sub(r'~~.*?~~', lambda m: ' ' * len(m.group(0)), line)
        seen = set()
        for pat in REF_PATTERNS:
            for m in re.finditer(pat, body):
                num, pos = m.group(1), m.start(1)
                if num in sections or (num, pos) in seen:
                    continue
                seen.add((num, pos))
                before = body[max(0, m.start() - 14):m.start()]
                if '旧' in before or '旧' in body[m.start():pos] or NOT_SECTION.search(body[:m.start()]):
                    continue
                if num.split('.')[0] in '6789':   # 章は 5 と付録まで。6 以上は数値（6.3 N·m）
                    continue
                bad.append((head, i, num, line.strip()[:90]))
    return bad


def reproduce_derived(sections: set) -> int:
    """10-23 に手で直す前の想定問答（0ee8504 の親）から、既知の古い参照を拾えるかを確かめる。"""
    import subprocess
    old = subprocess.run(['git', 'show', '0ee8504^:docs/研究応用/執筆/想定問答.md'],
                         capture_output=True, text=True, cwd=DRAFT.parent.parent.parent).stdout
    got = {num for _, _, num, _ in derived_refs(old, sections)}
    want = {'4.3.3.2', '5.4.2', '5.7.2'}
    ok = bool(old) and want <= got
    print(('✅' if ok else '⛔') + f' 派生の md: 直す前の想定問答から {sorted(want)} → 検出 {sorted(got)}')
    return 0 if ok else 1


def reproduce() -> int:
    """既知の欠番（2026-10-21 の第4章）を拾えるかを確かめる。"""
    known = ['1', '2', '3', '4', '4.1', '4.2', '4.3', '4.3.2', '4.3.3', '4.3.3.1', '4.3.3.3',
             '4.3.3.4', '4.4', '4.4.1', 'A', 'A.1', 'D', 'D.1']
    gaps, _ = find_gaps(known)
    want = {'4.3.1', '4.3.3.2', '付録B', '付録C'}
    ok = want <= set(gaps)
    print(('✅' if ok else '⛔') + f' 既知の欠番 {sorted(want)} → 検出 {sorted(gaps)}')
    return 0 if ok else 1


def main() -> int:
    reproduce_mode = '--reproduce' in sys.argv
    if reproduce_mode and reproduce():
        return 1
    texts = {}
    for name in BODY:
        p = DRAFT / name
        if not p.exists():
            print(f'⛔ {name} が無い')
            return 1
        texts[name] = strip_memo(p.read_text(encoding='utf-8'))

    sections = collect_sections(texts)
    if reproduce_mode:
        return reproduce_derived(sections)
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

    # ⛔ 見出しの深さと番号の桁数が合っているか（2026-10-02 追加）。
    # md の `####` が 3 桁番号（4.5.5）に付いていると、番号を捨てて振り直す実装では
    # PDF が 4.5.4.1 と印字する。build_thesis_pdf.py は md の番号を正とするよう直したが、
    # **ずれていること自体が md の誤り**なので検出する。
    depth_bad = []
    for name in BODY:
        raw = (DRAFT / name).read_text(encoding='utf-8')
        for i, line in enumerate(raw.splitlines(), 1):
            m = re.match(r'^(#{2,5})\s+((?:\d+\.)+\d*)\s', line)
            if m and len(m.group(1)) != len(m.group(2).rstrip('.').split('.')):
                depth_bad.append((name, i, m.group(2), len(m.group(1))))

    if depth_bad:
        print(f'⛔ 見出しの # の数と節番号の桁数が合わない {len(depth_bad)} 件\n')
        for name, i, num, lv in depth_bad:
            print(f'  {name}:{i}  「{num}」は {len(num.rstrip(".").split("."))} 桁なのに # が {lv} 個')
        print()

    gaps, singles = find_gaps(collect_headings(texts))
    known_gaps = [g for g in gaps if g in INTENTIONAL_GAPS]
    gaps = [g for g in gaps if g not in INTENTIONAL_GAPS]
    if gaps:
        print(f'⛔ 欠番 {len(gaps)} 件（在るべき番号の節が無い）\n')
        for g in gaps:
            print(f'  {g}')
        print()
    for g in known_gaps:
        print(f'ℹ️ 意図した欠番 {g}: {INTENTIONAL_GAPS[g]}')
    for s in singles:
        print(f'⚠️ {s} は項が 1 つだけ（分ける必要があるか、判定 md 型 26）')

    dbad = []
    for p in DERIVED:
        for head, i, num, line in derived_refs(p.read_text(encoding='utf-8'), sections):
            dbad.append((p.name, head, i, num, line))
    if dbad:
        print(f'⛔ 派生の md から、実在しない節への参照 {len(dbad)} 件（見出しごと。外した節なら「旧 N.M」と書くか、今の節へ）\n')
        last = None
        for name, head, i, num, line in dbad:
            if (name, head) != last:
                print(f'  {name} — {head}')
                last = (name, head)
            print(f'      L{i}  「{num}」  {line}')
        print()

    if bad:
        print(f'⛔ 実在しない節への参照 {len(bad)} 件\n')
        for name, i, num, line in bad:
            print(f'  {name}:{i}  → 「{num} 節」が無い')
            print(f'      {line}')
        print(f'\n⭐ 実在する節: {len(sections)} 個 / 章: {sorted(chapters)}')
        return 1

    if depth_bad or gaps or dbad:
        return 1
    print(f'✅ 節参照はすべて実在する節を指している（参照先候補 {len(sections)} 節）')
    print('✅ 見出しの深さと節番号の桁数も一致している')
    print('✅ 欠番は無い（意図したものを除く）')
    print(f'✅ 派生の md（{"・".join(p.stem for p in DERIVED)}）の節参照も実在する')
    return 0


if __name__ == '__main__':
    sys.exit(main())
