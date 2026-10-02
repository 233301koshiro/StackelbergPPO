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
    args = ap.parse_args()

    pat = re.compile('|'.join(re.escape(m) for m, _ in MARKS))
    hits = {}
    for p in sorted(DOCS.rglob('*.md')):
        if 'archive' in p.parts or p.name.startswith('archive_'):
            continue
        text = p.read_text(encoding='utf-8', errors='replace')
        rows = [(i, l.strip()) for i, l in enumerate(text.splitlines(), 1) if pat.search(l)]
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
