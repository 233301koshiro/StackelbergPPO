#!/usr/bin/env python3
"""「まだ埋まっていない穴」の印が、**埋めた後も残っていないか**を出す。

⛔⛔ なぜ要るか（2026-10-02）: `工程別_検証状況.md` が 2026-09-12 で止まり、
**M1 の再現性と M2 の seed 固定が「❌ 🔧 未測」のまま 3 週間残っていた。**
⭐ 実際は 9-115・9-191・9-194・9-195 で塞がっていた。
⛔⛔ **放置すると、自分の弱点を過大に申告することになる**（審査前に致命的）。

⚠️⚠️ **`check_stale_claims.py` は検出できない。**数値の矛盾ではなく
「埋まった穴が未測のまま書いてある」型だから。

⭐ **この検査は判断しない。**⭐ **「いま未着手と書いてある項目」を全部並べるだけ。**
⭐ **人が「それは本当にまだ穴か」を見る。**

    python3 scripts/check_open_gaps.py
    python3 scripts/check_open_gaps.py --days 14   # 14 日以上更新の無いものだけ
"""
import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / 'docs'

# 「まだ埋まっていない」を表す印。⚠️ この repo で実際に使われている語に限る
MARKS = [
    ('❌ 🔧', '未着手・未測定（着手すれば埋まる）'),
    ('❌ 🔧', None),
    ('⏳ 未確認', '未確認'),
    ('⏳ **未確認**', '未確認'),
    ('❌ 未測', '未測'),
    ('🔧 未測', '未測'),
]


# ⚠️ 印を含むが「穴」ではない行。⭐ 凡例・この検査自身の説明・過去の失敗の記述
SKIP = [
    '**やる。** 未着手・未測定で、着手すれば埋まる',   # 凡例
    'のまま残っていた',                                 # 過去の失敗の記述
    '未測」のまま',
]


def self_check() -> int:
    """⭐ **本物の穴を見落とさないか**を既知の文字列で確かめる（§5-2 ⑤-3）。"""
    pat = re.compile('|'.join(re.escape(m) for m, _ in MARKS))
    cases = [
        ('| seed 固定での再現性 | ❌ 🔧 未測 | — |', True,  '本物の穴'),
        ('| 関節数 3 以外 | ❌ 🚧 やらないと決めた | — |', False, 'やらないと決めた限界'),
        ('| 禁止物 | ⏳ 未確認 |', True,  '未確認'),
        ('| ❌ 🔧 | **やる。** 未着手・未測定で、着手すれば埋まる |', False, '凡例'),
        ('| 長さ | ✅ 済 |', False, '埋まっている'),
    ]
    bad = 0
    for line, want, why in cases:
        got = bool(pat.search(line)) and not any(x in line for x in SKIP)
        ok = (got == want)
        print(f'  {"✅" if ok else "⛔"} {why:22s} 期待 {want} / 実際 {got}')
        bad += (not ok)
    print('✅ 自己検査 5 項目一致' if not bad else f'⛔ {bad} 件 失敗')
    return bad


def last_commit_days(p: Path):
    try:
        out = subprocess.run(
            ['git', 'log', '-1', '--format=%ct', '--', str(p.relative_to(ROOT))],
            cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        if not out:
            return None
        return (datetime.now(timezone.utc).timestamp() - int(out)) / 86400
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=float, default=0.0,
                    help='これより長く更新されていないファイルだけ出す')
    ap.add_argument('--self-check', action='store_true',
                    help='⭐ 本物の穴を見落とさないかを既知の文字列で確かめる')
    args = ap.parse_args()
    if args.self_check:
        return self_check()

    pat = re.compile('|'.join(re.escape(m) for m, _ in MARKS))
    hits = {}
    for p in sorted(DOCS.rglob('*.md')):
        if 'archive' in p.parts or p.name.startswith('archive_'):
            continue
        text = p.read_text(encoding='utf-8', errors='replace')
        rows = []
        for i, l in enumerate(text.splitlines(), 1):
            if not pat.search(l):
                continue
            if any(x in l for x in SKIP):
                continue
            rows.append((i, l.strip()))
        if rows:
            hits[p] = rows

    if not hits:
        print('✅ 「未着手・未測」の印は 1 件も無い')
        return 0

    total = sum(len(v) for v in hits.values())
    print(f'⭐ 「まだ埋まっていない」と書いてある箇所: {total} 件 / {len(hits)} ファイル\n')
    print('⚠️ **これは違反の一覧ではない。**⭐ **埋まったのに印が残っていないかを人が見る表である。**\n')

    shown = 0
    for p, rows in sorted(hits.items(), key=lambda kv: -len(kv[1])):
        d = last_commit_days(p)
        if args.days and (d is None or d < args.days):
            continue
        age = f'{d:.0f} 日前' if d is not None else '不明'
        print(f'  {p.relative_to(ROOT)}  （最終更新 {age}・{len(rows)} 件）')
        for i, line in rows[:4]:
            print(f'      L{i}: {line[:92]}')
        if len(rows) > 4:
            print(f'      … 他 {len(rows) - 4} 件')
        print()
        shown += 1

    if args.days and shown == 0:
        print(f'  ✅ {args.days:.0f} 日以上更新の無いものは無い')
    print('⭐ 埋まった穴は、**埋めた日に印を消す**（CLAUDE.md §4-2 のトリガー表）。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
