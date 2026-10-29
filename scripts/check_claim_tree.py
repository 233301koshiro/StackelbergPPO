#!/usr/bin/env python3
"""修論の主張の木（`docs/研究応用/執筆/主張の木.drawio`）が、今の修論本文とずれていないかを並べる（2026-10-29、ユーザー「棚卸しに修論の MD が入っているなら、主張の木の drawio も見るように」）。

⛔ なぜ要るか: md の検査はどれも `.drawio` を読まない。木は 10-23 14 時台の版 3 で止まり、
同じ日の 16 時台に本文から RQ1〜RQ3 を消し、5.2 を組み直した後も、木には RQ と旧い節番号が残った。

見るもの（人が 1 件ずつ判断する。機械は候補を出すだけ）:
  ① 木が指す節（`4.4.3.1` など）・表・図・付録が、今の本文に在るか
  ② 木に書いた語（`RQ1`・`R4`・`軸3`）が、今の本文に在るか（本文から消した語が木に残っていないか）
  ③ 木を最後に直した版から、木が指す節の字数が大きく変わっていないか（木の字数は刈る判断に使う）
  ④ 木を最後に直した版の後に、本文に増えた節
字数は空白を除く本文（執筆メモ・`>` の注記を除く）で、木を最後に直した版と今を**同じ数え方で**比べる
（木に書いた字数とは数え方が数 % ずれるので、木の数字とは直接比べない）。

    python3 scripts/check_claim_tree.py
"""
import html
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_section_refs import BODY, strip_memo  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TREE = 'docs/研究応用/執筆/主張の木.drawio'
DRAFT = 'docs/研究応用/修論ドラフト/'
DRIFT_RATIO = 0.15   # 字数がこの割合以上、かつ
DRIFT_CHARS = 100    # この字数以上変わった節を出す
NUM = r'[0-9A-D]\.[0-9]+(?:\.[0-9]+)*'


def git(*args) -> str:
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout


def read(path: str, rev: str = None) -> str:
    if rev:
        return git('show', f'{rev}:{path}')
    return (ROOT / path).read_text(encoding='utf-8')


def labels(xml: str) -> list:
    """(id, 平文) の列。id は先頭の語（根・A2a・凡例など）。"""
    out = []
    for v in re.findall(r'value="([^"]*)"', xml):
        t = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', html.unescape(v))).strip()
        if t:
            out.append((t.split(' ', 1)[0], t))
    return out


def sections(rev: str = None) -> dict:
    """節番号 → 空白を除く本文（下位の節を含む）。"""
    out = {}
    for name in BODY:
        lines = strip_memo(read(DRAFT + name, rev)).splitlines()
        heads = []
        for i, line in enumerate(lines):
            m = re.match(r'^(#{1,5})\s+(?:付録)?(' + NUM + r')\s', line)
            if m:
                heads.append((i, len(m.group(1)), m.group(2)))
            elif re.match(r'^#{1,2}\s', line):
                heads.append((i, 1, None))
        for k, (i, depth, num) in enumerate(heads):
            if not num:
                continue
            j = next((i2 for i2, d2, _ in heads[k + 1:] if d2 <= depth), len(lines))
            body = '\n'.join(x for x in lines[i + 1:j] if not x.startswith('>'))
            out[num] = re.sub(r'\s', '', re.sub(r'[*#`|]', '', body))
    return out


def body_text() -> str:
    return '\n'.join(strip_memo(read(DRAFT + n)) for n in BODY)


def refs(text: str) -> list:
    """木の 1 枚が指す節番号。`3.7〜3.9` は両端、`付録D.3` は D.3。倍率・百分率は除く。"""
    text = re.sub(r'[表図]\s*' + NUM, '', text)
    out = []
    for m in re.finditer(r'(?<![0-9.,])(?:付録)?(' + NUM + r')(?![0-9.,]|\s*(?:倍|%|％))', text):
        out.append(m.group(1))
    return out


def main() -> int:
    xml = read(TREE)
    tree_rev = git('log', '-1', '--format=%h', '--', TREE).strip()
    tree_date = git('log', '-1', '--format=%ad', '--date=format:%m-%d %H:%M', '--', TREE).strip()
    paths = [DRAFT + n for n in BODY]
    since = git('log', '--format=%h', f'{tree_rev}..HEAD', '--', *paths).split()
    print(f'主張の木: 最後に直したのは {tree_date}（{tree_rev}）。その後に修論本文を変えたコミット {len(since)} 件')

    now, then = sections(), sections(tree_rev)
    text = body_text()
    found = 0

    # ① 節・表・図が在るか
    for nid, t in labels(xml):
        missing = [r for r in refs(t) if r not in now]
        missing += [m.group(0) for m in re.finditer(r'[表図]\s*[0-9]+\.[0-9]+', t)
                    if not re.search(re.escape(m.group(0)[0]) + r'\s*' + re.escape(m.group(0)[1:].strip()) + r'(?![0-9])', text)]
        for r in dict.fromkeys(missing):
            print(f'  ⚠️ ① {nid}: 「{r}」が今の本文に無い')
            found += 1

    # ② 木の語が本文に在るか
    for nid, t in labels(xml):
        for w in dict.fromkeys(re.findall(r'(?<![A-Za-z])(?:RQ[0-9]|R[0-9]|軸[0-9])', t)):
            if not re.search(re.escape(w) + r'(?![0-9])', text):
                print(f'  ⚠️ ② {nid}: 「{w}」が今の本文に無い（本文から消した語が木に残っていないか）')
                found += 1

    # ③ 木が指す節の字数の変化
    for r in dict.fromkeys(r for _, t in labels(xml) for r in refs(t)):
        if r in now and r in then:
            a, b = len(then[r]), len(now[r])
            if abs(b - a) >= DRIFT_CHARS and abs(b - a) >= DRIFT_RATIO * max(a, 1):
                print(f'  ⚠️ ③ {r}: {a:,} → {b:,} 字（木を直した版から）')
                found += 1

    # ④ 増えた節
    for r in sorted(set(now) - set(then)):
        print(f'  ⚠️ ④ {r}: 木を直した後に本文に増えた節（木に載せるかを見る）')
        found += 1

    if found:
        print(f'\n候補 {found} 件。木を直したら draw.io で開いて保存し、コミットする。直さないと決めたものは理由を目次の棚卸しの記録に書く。')
        return 1
    print('✅ 木と本文のずれの候補は無い。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
