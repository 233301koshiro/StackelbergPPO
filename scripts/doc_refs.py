#!/usr/bin/env python3
"""識別子の逆引き索引 — 「ここを変えたら、どこも変わるか」を機械で答える。

**なぜ要るか**: 方針や結論を 1 つ変えると、それを引用している md が芋づるで古くなる。
2026-09-13 の実測では **3.12.5 を 1 節挿しただけで 7 ファイル 13 箇所**の参照が動いた。
9-49 で機序を否定したときは **配布資料 2 件に取り残しが出た**（§4-2 が警告している事故）。

⚠️ **このリポジトリは「リンクのグラフ」ではなく「識別子のグラフ」でできている。**
md 51 本の内部リンクは 256 本あるが、**修論の全章はリンクを一つも張っていない**。
章どうしは `系譜 9-49` / `修論 4.3.2` / `Bug 27` という**識別子**で参照し合う。
本スクリプトはその識別子で逆引きする。

    python3 scripts/doc_refs.py 9-49          # 9-49 を引用している箇所を全部出す
    python3 scripts/doc_refs.py 3.12.5        # 修論の節番号でも引ける
    python3 scripts/doc_refs.py --undefined   # 定義が無いのに引用されている識別子
    python3 scripts/doc_refs.py --hubs        # 引用が集中している識別子（変更の影響が大きい）
"""
import argparse
import collections
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / 'docs/研究応用/台帳/実験系譜.md'
DEBUG = ROOT / 'docs/デバッグ戦記.md'
DRAFT = ROOT / 'docs/研究応用/修論ドラフト'

# 引用の形。⚠️ 比率や日付と紛れないよう、前後の文脈で絞る
PAT = {
    # ⚠️ 段番号は 1 桁始まり（第0段〜第9段）。2 桁始まりは日付（07-10 等）なので除く
    'ledger': re.compile(r'(?<![\d.\-])(\d-\d{1,2}[a-z]?)(?![\d.\-])'),
    'thesis': re.compile(r'(?<![\d.])(\d\.\d{1,2}(?:\.\d{1,2})?)(?![\d.])'),
    'bug':    re.compile(r'Bug\s+(\d+)'),
}


def mds():
    return [p for p in ROOT.glob('docs/**/*.md') if '/archive/' not in str(p)]


def defined():
    """定義されている識別子の集合。"""
    d = {'ledger': set(), 'thesis': set(), 'bug': set()}
    if LEDGER.exists():
        s = io.open(LEDGER, encoding='utf-8').read()
        d['ledger'] = set(re.findall(r'^#{2,4} .*?(\d+-\d+[a-z]?)\.', s, re.M))
    if DEBUG.exists():
        s = io.open(DEBUG, encoding='utf-8').read()
        d['bug'] = set(re.findall(r'^#{2,4} Bug (\d+)', s, re.M))
    for p in DRAFT.glob('*.md'):
        s = io.open(p, encoding='utf-8').read()
        d['thesis'] |= set(re.findall(r'^#{2,4} (\d\.\d{1,2}(?:\.\d{1,2})?)', s, re.M))
    return d


def cites():
    """識別子 → [(ファイル, 行番号)]。"""
    out = collections.defaultdict(list)
    for p in mds():
        for i, line in enumerate(io.open(p, encoding='utf-8').read().split('\n'), 1):
            if line.lstrip().startswith('#'):
                continue                      # 見出しは定義であって引用ではない
            # ⚠️ CLAUDE.md の節番号（§5-2）と比率（2-3 倍）を除く
            clean = re.sub(r'§\s?\d-\d+', '', line)
            for kind, pat in PAT.items():
                for m in pat.findall(clean):
                    out[(kind, m)].append((str(p.relative_to(ROOT)), i))
            # 接頭辞つきの引用は確実な参照として別に数える（9-11 の規約）
            for m in re.findall(r'系譜\s?(\d-\d{1,2}[a-z]?)', clean):
                out[('ledger_sure', m)].append((str(p.relative_to(ROOT)), i))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('ident', nargs='?', help='例: 9-49 / 3.12.5 / Bug27')
    ap.add_argument('--undefined', action='store_true', help='定義が無いのに引用されている識別子')
    ap.add_argument('--hubs', action='store_true', help='引用が集中している識別子')
    a = ap.parse_args()
    C, D = cites(), defined()

    if a.hubs:
        print("引用が集中している識別子 — **ここを変えると影響が広い**\n")
        for (kind, ident), v in sorted(C.items(), key=lambda x: -len(x[1]))[:30]:
            if kind == 'ledger_sure' or ident not in D.get(kind, set()):
                continue
            files = len({f for f, _ in v})
            print(f"  {kind:7s} {ident:9s} {len(v):3d} 箇所 / {files} ファイル")
        return 0

    if a.undefined:
        print("定義が無いのに引用されている識別子（**壊れ参照の候補**）\n")
        n = 0
        for (kind, ident), v in sorted(C.items()):
            if kind == 'thesis' and ident not in D['thesis'] and len(v) >= 2 \
               and any('修論 ' + ident in io.open(ROOT / f, encoding='utf-8').read()
                       for f, _ in v[:3]):
                print(f"  修論 {ident}: {len(v)} 箇所  例 {v[0][0]}:{v[0][1]}"); n += 1
            if kind == 'ledger_sure' and ident not in D['ledger']:
                print(f"  系譜 {ident}: {len(v)} 箇所  例 {v[0][0]}:{v[0][1]}"); n += 1
        print(f"\n{n} 件。⚠️ **接頭辞（`系譜 9-6` / `修論 4.3.2`）が付いた引用だけを見ている**（9-11 の規約）。\n   接頭辞の無い引用は数値と区別できないので拾えない。**接頭辞を付けるほど検査が効く。**")
        return 0

    if not a.ident:
        ap.print_help(); return 1
    key = a.ident.replace('Bug', '').strip()
    hits = []
    for (kind, ident), v in C.items():
        if ident == key:
            hits.append((kind, v))
    if not hits:
        print(f"'{a.ident}' の引用は見つからない"); return 0
    for kind, v in hits:
        ok = '✅ 定義あり' if key in D[kind] else '⚠️ 定義が見つからない'
        print(f"\n=== {kind} {key}  {ok}  — {len(v)} 箇所 / {len({f for f,_ in v})} ファイル")
        for f, i in sorted(v):
            print(f"   {f}:{i}")
    print("\n⚠️ **この一覧が「変えたら見る場所」である。** 上から 1 件ずつ当たる。")
    return 0


if __name__ == '__main__':
    sys.exit(main())
