#!/usr/bin/env python3
"""外部の方針レビュー（/review-out）の素材を集める。前回のレビューから何をしたかを機械で並べる。

**なぜ要るか**: 方針の説明は Claude が書くので、Claude が覚えている行動しか載らない。
「方針と行動が噛み合っているか」を外に見せるには、行動の側を記憶ではなく記録から拾う必要がある
（2026-11-28 の初回では、9-366 の予測は形を固定した計算、run はリンク長も動かす設定で、
Claude もユーザーも気づかずに投入直前まで進んでいた。系譜 9-369）。

出すもの（前回のレビューの日 ～ 今日）:
  1. 前回の外部の方針レビューから何日たったか（記録: docs/研究応用/方針/外部の方針レビュー記録.md）
  2. コミットの一覧（件名にユーザーの判断が引用されている）
  3. 実験系譜に足された段（### の見出し）と、事前登録の読み・判定（#### の見出し）
  4. 起動した run（launch_env.json の時刻）と、いま動いているキュー

⚠️ 素材であって、方針の説明ではない。方針ごとにまとめ、リポジトリの用語を開くのは Claude（/review-out）。

使い方:
  python3 scripts/policy_review_material.py              # 前回の記録の日から
  python3 scripts/policy_review_material.py --since 2026-11-20
  python3 scripts/policy_review_material.py --days-only  # 経過日数の 1 行だけ（check_docs_inventory が使う）
"""
import argparse
import datetime
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RECORD = ROOT / 'docs' / '研究応用' / '方針' / '外部の方針レビュー記録.md'
LEDGER = 'docs/研究応用/台帳/実験系譜.md'
CADENCE_DAYS = 7   # 週に 1 回（記録の md の「いつ回すか」と合わせる）


def last_review_date():
    """記録の表の行（| 2026-11-28 | …）から、最新の日付を返す。無ければ None。"""
    if not RECORD.exists():
        return None
    dates = re.findall(r'^\|\s*(\d{4}-\d{2}-\d{2})\s*\|', RECORD.read_text(encoding='utf-8'), re.M)
    return max(datetime.date.fromisoformat(d) for d in dates) if dates else None


def last_review_line():
    last = last_review_date()
    if last is None:
        return '⛔ 外部の方針レビュー: 記録が無い（docs/研究応用/方針/外部の方針レビュー記録.md）  ← /review-out'
    days = (datetime.date.today() - last).days
    mark = '⭐' if days <= CADENCE_DAYS else ('⚠️' if days <= 2 * CADENCE_DAYS else '⛔')
    tail = '' if days <= CADENCE_DAYS else '  ← **1 週間を超えている。/review-out で依頼の一式を作る**'
    return f'{mark} 前回の外部の方針レビュー: {last}（**{days} 日前**）{tail}'


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout


def ledger_headings_added(since):
    """since より後のコミットで実験系譜に足された見出し（### と ####）。"""
    base = git('rev-list', '-1', f'--before={since.isoformat()} 00:00', 'HEAD').strip()
    if not base:
        return [], []
    diff = git('diff', base, 'HEAD', '--', LEDGER)
    added = [l[1:] for l in diff.splitlines() if l.startswith('+') and not l.startswith('+++')]
    sections = [l for l in added if l.startswith('### ')]
    readings = [l for l in added if l.startswith('#### ') and re.search(r'読み|判定', l)]
    return sections, readings


def runs_launched(since):
    out = []
    for p in sorted((ROOT / 'single_run').glob('*/launch_env.json')):
        t = datetime.datetime.fromtimestamp(p.stat().st_mtime)
        if t.date() >= since:
            out.append((t, p.parent.name))
    return sorted(out)


def short(s, n=220):
    s = re.sub(r'\*\*', '', s)
    return s if len(s) <= n else s[:n] + '…'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', help='YYYY-MM-DD（既定: 記録の最新の日）')
    ap.add_argument('--days-only', action='store_true')
    a = ap.parse_args()

    if a.days_only:
        print(last_review_line())
        return
    since = datetime.date.fromisoformat(a.since) if a.since else last_review_date()
    if since is None:
        sys.exit('記録が無いので --since を指定する')

    print(last_review_line())
    print(f'\n# 前回（{since}）からの素材\n')

    log = git('log', f'--since={since.isoformat()} 00:00', '--reverse',
              '--format=%h %ad %s', '--date=format:%m-%d %H:%M')
    print(f'## 1. コミット（{len(log.splitlines())} 件）')
    for l in log.splitlines():
        print('- ' + short(l))

    sections, readings = ledger_headings_added(since)
    print(f'\n## 2. 実験系譜に足された段（{len(sections)} 件）')
    for s in sections:
        print('- ' + short(s[4:]))
    print(f'\n## 3. 事前登録の読み・判定の見出し（{len(readings)} 件）')
    for s in readings:
        print('- ' + short(s[5:]))

    runs = [(t, n) for t, n in runs_launched(since) if 'smoke' not in n]
    print(f'\n## 4. 起動した run（{len(runs)} 本、launch_env.json の時刻。スモークは除く）')
    print('   起動時の指定（.hydra/overrides.yaml）も出す。対応表の「run の実際の設定」は記憶ではなくここから写す')
    for t, name in runs:
        print(f'- {t:%m-%d %H:%M} {name}')
        ov = ROOT / 'single_run' / name / '.hydra' / 'overrides.yaml'
        if ov.exists():
            items = [l[2:].strip() for l in ov.read_text().splitlines() if l.startswith('- ')]
            items = [i for i in items if not i.startswith('hydra.run.dir')]
            print('    ' + ' '.join(items))
    q = subprocess.run(['pgrep', '-af', '^bash scripts/queue_'], capture_output=True, text=True).stdout
    print('\n## 5. いま動いているキュー')
    print(q.strip() or '（無し）')

    print('\n---\n⚠️ 次: /review-out の手順 2 で、方針ごとに「決めたこと → したこと → 予測の計算の前提と run の設定」の表にする。')


if __name__ == '__main__':
    main()
