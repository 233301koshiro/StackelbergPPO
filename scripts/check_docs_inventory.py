#!/usr/bin/env python3
"""docs の棚卸しのうち、機械的に判定できる部分を自動化する。

**なぜ要るか**: CLAUDE.md §4-2 の定期棚卸しは**週に1回**行う（修論まで残り数か月なので、
記事が推奨する四半期ごとでは1〜2回しか回らない）。しかし毎回 51 ファイルを人手で読むのは
続かない。機械で判定できるところを自動化し、**人は「消すかどうか」の判断だけに集中する**。

判定するもの:
  1. 孤立ファイル   … どこからも参照されず、目次にも載っていない
  2. 壊れた内部リンク
  3. archive の注記漏れ … 「アーカイブ」と冒頭に書いていない
  4. archive にあるのに現役参照されているファイル
  5. 長く更新されていないファイル（内容の正しさは判定しない。目を通す候補）

⚠️ **このスクリプトは「消せ」とは言わない。** 2026-09-02 の棚卸しで、
孤立していた `軸1_パイプライン頑健性.md` は読んでみると**第4章 4.2.2 の表と図の原典**だった。
**孤立＝不要ではない。索引から漏れているだけのこともある。** 判断は必ず人がする。

使い方: python3 scripts/check_docs_inventory.py [--stale-days 60]
"""
import argparse
import datetime
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / 'docs'
TOC = DOCS / '目次.md'


def last_commit_date(p: pathlib.Path):
    out = subprocess.run(['git', 'log', '-1', '--format=%ad', '--date=short', '--', str(p)],
                         cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return out or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stale-days', type=int, default=60,
                    help='これより長く更新されていないファイルを候補として挙げる')
    args = ap.parse_args()

    mds = sorted(DOCS.rglob('*.md'))
    text = {p: p.read_text(encoding='utf-8', errors='replace') for p in mds}
    toc = text.get(TOC, '')
    issues = 0

    # 1. 孤立ファイル
    # ⚠️ **ビルドスクリプトが読むファイルは MD からリンクされていなくても孤立ではない。**
    # 修論 PDF は build_thesis_pdf.py がファイル名で直接読むので、参考文献・付録は
    # どこからもリンクされていないが必要である（2026-09-02 に誤検出したので除外を入れた）。
    built = set()
    for sc in (ROOT / 'scripts').glob('build_*.py'):
        t = sc.read_text(encoding='utf-8', errors='replace')
        for p in mds:
            if p.name in t or p.stem in t:
                built.add(p)

    # archive は「もう参照しない」ことが目的なので孤立していて当然。
    # archive 専用の判定（3・4）で別途見る。
    orphans = []
    for p in mds:
        if p == TOC or p in built or 'archive' in p.parts:
            continue
        referenced = any(p.name in t for q, t in text.items() if q != p)
        if not referenced and p.name not in toc:
            orphans.append(p)
    if orphans:
        issues += len(orphans)
        print(f"⚠️  孤立（どこからも参照されず目次にも無く、ビルド対象でもない）: {len(orphans)} 件")
        print("    → **消す前に必ず全文を読むこと。** 台帳（一次データ）が索引から"
              "漏れているだけの場合がある")
        for p in orphans:
            print(f"      {p.relative_to(DOCS)}  ({p.stat().st_size} bytes)")

    # 2. 壊れた内部リンク
    broken = []
    for p in mds:
        for m in re.finditer(r'\]\((?!http|#)([^)]+\.md)[^)]*\)', text[p]):
            if not (p.parent / m.group(1)).resolve().exists():
                broken.append((p, m.group(1)))
    if broken:
        issues += len(broken)
        print(f"\n❌ 壊れた内部リンク: {len(broken)} 件")
        for p, l in broken:
            print(f"      {p.relative_to(DOCS)} → {l}")

    # 3. archive の注記漏れ
    arch = sorted((DOCS / 'archive').glob('*.md')) if (DOCS / 'archive').is_dir() else []
    missing = [p for p in arch if 'アーカイブ' not in text[p][:400]]
    if missing:
        issues += len(missing)
        print(f"\n⚠️  archive なのに冒頭の注記が無い: {len(missing)} 件")
        print("    → 「アーカイブ。作業の参照先にしない」と後継へのリンクを冒頭に足す")
        for p in missing:
            print(f"      {p.relative_to(DOCS)}")

    # 4. archive にあるのに現役から参照されている
    live_refs = {}
    for p in arch:
        refs = [q.relative_to(DOCS) for q, t in text.items()
                if 'archive' not in q.parts and q != TOC and p.name in t]
        if refs:
            live_refs[p] = refs
    if live_refs:
        print(f"\n💡 archive にあるが現役から参照されている: {len(live_refs)} 件")
        print("    → 内容が生きている。冒頭にその旨を明記するか、archive から戻すか判断する")
        for p, refs in live_refs.items():
            print(f"      {p.relative_to(DOCS)} ← {', '.join(str(r) for r in refs)}")

    # 5. 長期未更新（内容の正しさは判定しない）
    today = datetime.date.today()
    stale = []
    for p in mds:
        if 'archive' in p.parts:
            continue
        d = last_commit_date(p)
        if not d:
            continue
        age = (today - datetime.date.fromisoformat(d)).days
        if age >= args.stale_days:
            stale.append((age, d, p))
    if stale:
        print(f"\n💡 {args.stale_days} 日以上更新なし: {len(stale)} 件"
              "（古い＝不要ではない。目を通す候補）")
        for age, d, p in sorted(stale, reverse=True)[:10]:
            print(f"      {d}  {age:>3}日  {p.relative_to(DOCS)}")

    print()
    if issues == 0:
        print("✅ 要対応（孤立・壊れリンク・注記漏れ）は無し。")
        print("   💡 の項目は判断待ちで、放置してよいこともある。")
        return 0
    print(f"❌ 要対応 {issues} 件。**消す前に必ず全文を読む**（CLAUDE.md §4-2 定期棚卸し）。")
    return 1


if __name__ == '__main__':
    sys.exit(main())
