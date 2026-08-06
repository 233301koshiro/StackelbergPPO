#!/usr/bin/env python3
"""docs/ の記述を学習ログ（一次データ）と突き合わせて、古い記述・矛盾を検出する。

CLAUDE.md §4-2 の4層モデルにおける「① 原本（log_train.txt）」を真として、
②台帳・③派生・④状態の各ドキュメントを検査する。

検査項目:
  A. 状態語の矛盾 — 完走済みの run を「進行中/稼働中/完走待ち」と書いている
  B. 数値の食い違い — run 名の近くにある数値が、その run のどのエポックの値とも一致しない
  C. 存在しない run  — docs にある run 名が single_run/ に無い（改名漏れ・誤記）
  D. 記録漏れ       — checkpoint がある run が台帳（実験系譜/実験一覧）に載っていない

B の考え方: ある run の「running best の全エポック値」の集合を作り、docs の数値が
そのどれかに一致すれば正当とみなす。`ep189=28.68 → ep199=34.32` のような**意図的な
履歴記述を誤検出しない**ため。どのエポックの値でもないのに近い値だけを疑わしいと報告する。

使い方:
    python3 scripts/check_docs_consistency.py            # 全検査
    python3 scripts/check_docs_consistency.py --only A   # 特定の検査だけ
    python3 scripts/check_docs_consistency.py --quiet    # 問題があるものだけ

学習もChoreonoidも不要。ログとMDを読むだけ。
"""
import argparse
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(ROOT, 'single_run')
DOC_DIR = os.path.join(ROOT, 'docs')

# ④状態 層だけが「進行中」を書いてよい。②台帳・③派生に状態語があれば古い可能性が高い
STATUS_WORDS = ['進行中', '稼働中', '学習中', '完走待ち', '起動待ち', '実行中']
# 検査から除く（履歴・アーカイブ・レビュー記録は当時の状態を残してよい）
SKIP_PARTS = ['/archive/', 'セカンドオピニオン', '発表原稿']


def iter_docs():
    for dirpath, _, files in os.walk(DOC_DIR):
        for fn in files:
            if not fn.endswith('.md'):
                continue
            p = os.path.join(dirpath, fn)
            if any(s in p for s in SKIP_PARTS):
                continue
            yield p


def rel(p):
    return os.path.relpath(p, ROOT)


def load_runs():
    """run名 -> {'values': そのrunが取りうる値の集合, 'done': 完走したか, 'last_ep': 最終epoch}"""
    runs = {}
    if not os.path.isdir(RUN_DIR):
        return runs
    for name in os.listdir(RUN_DIR):
        d = os.path.join(RUN_DIR, name)
        if not os.path.isdir(d):
            continue
        log = os.path.join(d, 'log', 'log_train.txt')
        if not os.path.exists(log):
            log = os.path.join(d, 'stdout.log')
        if not os.path.exists(log):
            continue

        per_epoch = {}
        for line in open(log, errors='ignore'):
            m = re.search(r'(?:^|\]\s*)(\d+)\s+T_sample', line)
            if not m:
                continue
            v = re.search(r'exec_R_eps\s+([-\d.]+)', line)
            if v:
                per_epoch[int(m.group(1))] = float(v.group(1))
        if not per_epoch:
            continue

        # 生値と running best の両方を「正当な値」として許容する
        values, best = set(per_epoch.values()), float('-inf')
        for e in sorted(per_epoch):
            best = max(best, per_epoch[e])
            values.add(round(best, 2))
        values = {round(v, 2) for v in values}

        so = os.path.join(d, 'stdout.log')
        done = False
        if os.path.exists(so):
            try:
                done = 'All workers terminated' in open(so, errors='ignore').read()
            except OSError:
                pass
        runs[name] = {'values': values, 'done': done, 'last_ep': max(per_epoch)}
    return runs


def runs_in_line(line, runs):
    """行に現れる run 名を**最長一致**で返す。

    `tripo_arm_v2c_pusher` は `tripo_arm_v2c_pusher_ns2` の部分文字列なので、
    素朴に判定すると別 run の行を誤って拾う（実測で A の誤検出の大半がこれだった）。
    長い名前から順に見て、既に採用した名前に含まれる短い名前は捨てる。
    """
    found = []
    for n in sorted(runs, key=len, reverse=True):
        if n in line and not any(n in f for f in found):
            found.append(n)
    return found


# 当時の状態を書き残す性質のドキュメント/行は、状態語があっても古いとは限らない
HISTORICAL_FILES = ['デバッグ戦記.md', '実験系譜.md', '方針レビュー', '学習中欠陥チェックリスト.md']
DATED_LINE = re.compile(r'20\d\d-\d\d-\d\d')


def close_enough(x, values):
    """docs の数値が、その run の取りうる値のどれかに一致するか（丸め・符号表記を許容）"""
    for v in values:
        if abs(x - v) < 0.02 or abs(abs(x) - abs(v)) < 0.02:
            return True
        # 小数第1位まで丸めた記述（41.4 と 41.41）
        if abs(round(x, 1) - round(v, 1)) < 0.001:
            return True
    return False


def check_A(runs):
    """完走済みの run を状態語で書いている箇所"""
    hits = []
    for p in iter_docs():
        for i, line in enumerate(open(p, encoding='utf-8', errors='ignore'), 1):
            if not any(w in line for w in STATUS_WORDS):
                continue
            # 日付が入った行は「その時点の記録」なので状態語があってよい
            if DATED_LINE.search(line):
                continue
            hist = any(h in p for h in HISTORICAL_FILES)
            for name in runs_in_line(line, runs):
                if not runs[name]['done']:
                    continue
                # 完走を明示している行は誤検出（「✅完走」と「進行中」が同居する表など）
                if '完走' in line and ('✅' in line or '済' in line):
                    continue
                if hist:
                    continue
                w = next(w for w in STATUS_WORDS if w in line)
                hits.append((rel(p), i, name, w, line.strip()[:90]))
                break
    return hits


def check_B(runs):
    """run 名の近くの数値が、その run のどのエポック値とも一致しない箇所"""
    hits = []
    num_re = re.compile(r'(?<![\w.])(-|−)?\d{1,3}\.\d{1,2}(?![\d])')
    for p in iter_docs():
        for i, line in enumerate(open(p, encoding='utf-8', errors='ignore'), 1):
            present = runs_in_line(line, runs)
            if len(present) != 1:
                continue                      # 複数 run が同居する行は帰属が定まらないので見送る
            name = present[0]
            info = runs[name]
            # **「best」の直後にある数値だけ**を対象にする。同じ行にはピーク速度・倍率・
            # リンク長・理論上限が混在しており、行単位で拾うと誤検出が支配的になる（実測13件中0件が真）
            for bm in re.finditer(r'best[^0-9\-−]{0,12}((-|−)?\d{1,4}\.\d{1,2})', line):
                x = float(bm.group(1).replace('−', '-'))
                if close_enough(x, info['values']):
                    continue
                hits.append((rel(p), i, name, x, line.strip()[:90]))
    return hits


def check_C(runs):
    """docs にある run 名が single_run/ に存在しない"""
    known = set(runs)
    if os.path.isdir(RUN_DIR):
        known |= set(os.listdir(RUN_DIR))
    # 「削除済み」「無効」と本文が明示している run は既知の欠落なので報告しない
    deleted = set()
    for p in iter_docs():
        t = open(p, encoding='utf-8', errors='ignore').read()
        for m in re.finditer(r'single_run/([A-Za-z0-9_]+)[^\n]{0,60}?(削除|無効|廃止)', t):
            deleted.add(m.group(1))
    known |= deleted
    cand = re.compile(r'single_run/([A-Za-z0-9_]+)')
    # 出現箇所ごとに並べると数十件になり読めないので、**run 名で束ねて**報告する
    seen = {}
    for p in iter_docs():
        for i, line in enumerate(open(p, encoding='utf-8', errors='ignore'), 1):
            for m in cand.finditer(line):
                n = m.group(1)
                if n in known or n.startswith(('comparison', 'queue', 'diag', 'boundary')):
                    continue
                seen.setdefault(n, []).append(f'{rel(p)}:{i}')
    return [(n, locs) for n, locs in sorted(seen.items())]


def check_D(runs):
    """checkpoint がある run が台帳に載っていない"""
    ledgers = [os.path.join(DOC_DIR, '研究応用', '実験系譜.md'),
               os.path.join(DOC_DIR, '実験一覧_詳細.md')]
    text = ''
    for f in ledgers:
        if os.path.exists(f):
            text += open(f, encoding='utf-8', errors='ignore').read()
    hits = []
    for name, info in runs.items():
        mdir = os.path.join(RUN_DIR, name, 'models')
        if not os.path.isdir(mdir) or not os.listdir(mdir):
            continue                          # checkpoint が無い＝実質未実施
        if name not in text:
            hits.append((name, info['last_ep'], info['done']))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=list('ABCD'), help='特定の検査だけ実行')
    ap.add_argument('--quiet', action='store_true', help='問題があるものだけ表示')
    a = ap.parse_args()

    runs = load_runs()
    print(f'一次データ: {len(runs)} run を読み込み（single_run/*/log/log_train.txt 優先）\n')
    total = 0

    if a.only in (None, 'A'):
        h = check_A(runs)
        total += len(h)
        print(f'【A】完走済みなのに状態語で書かれている: {len(h)} 件')
        for p, i, n, w, s in h:
            print(f'    {p}:{i}  [{n}] 「{w}」\n        {s}')
        if not h and not a.quiet:
            print('    ✅ なし')
        print()

    if a.only in (None, 'B'):
        h = check_B(runs)
        total += len(h)
        print(f'【B】run のどのエポック値とも一致しない数値: {len(h)} 件')
        for p, i, n, x, s in h:
            print(f'    {p}:{i}  [{n}] {x}\n        {s}')
        if not h and not a.quiet:
            print('    ✅ なし')
        print()

    if a.only in (None, 'C'):
        h = check_C(runs)
        total += len(h)
        print(f'【C】single_run/ に存在しない run 名: {len(h)} 種')
        print('    （過去に削除した run・改名前の名前が多い。実害があるのは「今も参照されている」ものだけ）')
        for n, locs in h:
            print(f'    {n}  ({len(locs)}箇所)  {", ".join(locs[:3])}{" ..." if len(locs) > 3 else ""}')
        if not h and not a.quiet:
            print('    ✅ なし')
        print()

    if a.only in (None, 'D'):
        h = check_D(runs)
        total += len(h)
        print(f'【D】checkpoint があるのに台帳に無い run: {len(h)} 件')
        for n, ep, done in h:
            print(f'    {n}  (ep{ep}, {"完走" if done else "未完走"})')
        if not h and not a.quiet:
            print('    ✅ なし')
        print()

    print(f'合計 {total} 件。**すべてが誤りとは限らない** — 意図的な履歴記述や、')
    print('別 run を指す数値が混ざる。1件ずつ判断すること。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
