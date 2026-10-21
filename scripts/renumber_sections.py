#!/usr/bin/env python3
"""修論の節番号を、旧 → 新の対応表で一度だけ置き換える。

⭐ 2026-10-21 新設。同じ日に節の組み替えを 2 回した（3.12 と 4.1）。
1 回目は手で grep して直し、2 回目は使い捨ての置き換えを書いた。そのとき分かったこと:

- ⛔ **対応表は 2 回当てると壊れる。**4.3→4.4 と 4.4→4.5 が同じ表にあると、
  2 回目で 4.4 がさらに 4.5 へずれる。権限エラーで途中から再実行したとき、
  どのファイルが済んだかを git status で確かめ直す羽目になった。
  → 当てたファイルを記録し、同じ表を同じファイルへ 2 回当てようとしたら止める
- ⛔ **`check_section_refs.py` はずれを検出できない。**繰り下げた後の番号も実在するので、
  「4.3 節」が旧 4.3 のつもりか新 4.3 のつもりかは見分けられない。だから人手の grep ではなく、
  対応表を 1 回だけ機械で当てる
- 表・図・式の番号（「表 4.3」）、「旧」の付いた番号、執筆メモ以降、日付で始まる引用行
  （`> ⛔⛔ **2026-10-13 …` の形）は置き換えない。日付つきの記録は当時の番号のままが正しい。
  ⚠️ 文中に日付があるだけの引用行（「古い物理（2026-10-13 より前）」）は今の状態の注記なので置き換える。
  記録かどうかは機械では決まらないので、dry-run を読み、残す行は `--skip=` で指す

対応表は「旧 新」を 1 行ずつ（`#` 以降は注釈）。**最も長く一致する旧番号**で置き換えるので、
`4.4 4.5` と書けば 4.4.1〜4.4.9 も 4.5.1〜4.5.9 になる。子だけ別の番号へ動かすなら、
その子を別の行に書く（`4.3.3.3 4.4.3.2`）。表に無い番号は変えない。

使い方:
    python3 scripts/renumber_sections.py --map map.txt FILE...            # 既定は dry-run
    python3 scripts/renumber_sections.py --map map.txt --apply FILE...
    python3 scripts/renumber_sections.py --map map.txt --skip=25,42 FILE  # その行は変えない
    python3 scripts/renumber_sections.py --reproduce                      # 既知の 2 回当てを止められるか

置き換えの後は `check_section_refs.py` と `dump_thesis_outline.py --write` を回す（CLAUDE.md §4-2）。
"""
import datetime
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / 'docs/研究応用/修論ドラフト/節番号の付け替え記録.tsv'


def load_map(text: str) -> dict:
    m = {}
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        old, new = line.split()
        assert old not in m, f'旧番号 {old} が 2 回ある'
        m[old] = new
    return m


def map_number(n: str, m: dict):
    """最も長く一致する旧番号で置き換える。表に無ければ None。"""
    parts = n.split('.')
    for k in range(len(parts), 0, -1):
        head = '.'.join(parts[:k])
        if head in m:
            return '.'.join([m[head]] + parts[k:])
    return None


def renumber(text: str, m: dict, skip=frozenset()):
    """置き換えた本文と、置き換えた箇所の一覧を返す。"""
    chapters = sorted({k.split('.')[0] for k in m}, key=len, reverse=True)
    pat = re.compile(r'(?<![0-9.\w])((?:' + '|'.join(map(re.escape, chapters))
                     + r')\.[0-9]+(?:\.[0-9]+)*)(?![0-9]|\.[0-9])')
    out, hits, inmemo = [], [], False
    for i, line in enumerate(text.split('\n'), 1):
        if line.startswith('## 執筆メモ'):
            inmemo = True
        frozen = (inmemo or i in skip or line.startswith('> 📝')
                  or line.startswith('> **ドラフト管理')
                  or re.match(r'>\s*\W{0,6}\**\s*20\d\d-\d\d-\d\d', line))

        def rep(mo):
            n = mo.group(1)
            pre = line[max(0, mo.start() - 3):mo.start()]
            if frozen or re.search(r'[表図式旧]\s*$', pre):
                return n
            new = map_number(n, m)
            if new is None or new == n:
                return n
            hits.append((i, n, new, line[max(0, mo.start() - 25):mo.end() + 12]))
            return new
        out.append(pat.sub(rep, line))
    return '\n'.join(out), hits


def map_key(m: dict) -> str:
    return hashlib.sha1(repr(sorted(m.items())).encode()).hexdigest()[:12]


def applied_before(key: str, path: Path) -> bool:
    if not LOG.exists():
        return False
    rel = str(path.resolve().relative_to(ROOT))
    return any(l.split('\t')[1:3] == [key, rel]
               for l in LOG.read_text(encoding='utf-8').splitlines()[1:])


def record(key: str, path: Path, n, m: dict):
    if not LOG.exists():
        LOG.write_text('日時\t対応表の鍵\tファイル\t置き換えた数\t対応表\n', encoding='utf-8')
    rel = str(path.resolve().relative_to(ROOT))
    with LOG.open('a', encoding='utf-8') as f:
        table = ', '.join(f'{o}→{n2}' for o, n2 in m.items())
        f.write(f'{datetime.datetime.now():%Y-%m-%d %H:%M}\t{key}\t{rel}\t{n}\t{table}\n')


def chains(m: dict) -> list:
    """新番号が別の行の旧番号でもある組（2 回当てるとさらにずれる）。"""
    return [(o, n) for o, n in m.items() if map_number(n, m) not in (None, n)]


def reproduce() -> int:
    """2026-10-21 の第4章の組み替えで、2 回当てたときのずれを再現し、検出できるかを見る。"""
    m = load_map('4.2 4.3\n4.3 4.4\n4.3.3.3 4.4.3.2\n4.4 4.5\n')
    once, _ = renumber('軸1 は 4.2 節、配分は 4.3.3.3 項、軸3 は 4.4.1 節。表 4.3 は動かさない。', m)
    twice, _ = renumber(once, m)
    ok = (once == '軸1 は 4.3 節、配分は 4.4.3.2 項、軸3 は 4.5.1 節。表 4.3 は動かさない。'
          and twice != once and chains(m))
    print(('✅' if ok else '⛔') + f' 1 回: {once}\n   2 回: {twice}（ずれる）\n   連鎖: {chains(m)}')
    return 0 if ok else 1


def main() -> int:
    args = sys.argv[1:]
    if '--reproduce' in args:
        return reproduce()
    if '--map' not in args:
        print(__doc__)
        return 1
    mapfile = args[args.index('--map') + 1]
    m = load_map(Path(mapfile).read_text(encoding='utf-8'))
    apply = '--apply' in args
    skip = set()
    for a in args:
        if a.startswith('--skip='):
            skip |= {int(x) for x in a.split('=', 1)[1].split(',')}
    files = [Path(a) for a in args if not a.startswith('--') and a != mapfile]
    key = map_key(m)
    if chains(m):
        print(f'⚠️ 連鎖がある（{len(chains(m))} 組）: この表は 2 回当てると二重にずれる。'
              f'当てたファイルは {LOG.name} に記録し、2 回目は止める')
    total = 0
    for p in files:
        if applied_before(key, p):
            print(f'⛔ {p}: この対応表（鍵 {key}）は当て済み。止める（{LOG.name}）')
            continue
        new, hits = renumber(p.read_text(encoding='utf-8'), m, skip)
        print(f'=== {p}  {len(hits)} 件')
        for i, o, n, ctx in hits:
            print(f'  L{i:<5} {o:>9} → {n:<9} …{ctx}…')
        if apply and hits:
            p.write_text(new, encoding='utf-8')
            record(key, p, len(hits), m)
        total += len(hits)
    print(f'\n合計 {total} 件' + ('（書き換えた）' if apply else '（dry-run。--apply で書き換える）'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
