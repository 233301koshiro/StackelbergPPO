#!/usr/bin/env python3
"""複数 seed の最終スコアの「帯」が重なるかを判定する（9-61/9-62 で毎回手計算していたもの）。

**なぜ要るか**: 「助言に従っても性能が落ちないか」の判定はこれまで人が
`best` の値を並べて目算していた（9-11・9-27・9-61・9-62）。指摘13
（関節固定の並行学習を自動でやる）を組むには、**完走後の判定も機械化**が要る。

帯 = [min(scores), max(scores)]。2 群の帯が重ならず、
「固定した方が同等以上」なら ✅、「固定した方が明確に悪化」なら ❌、
帯が重なれば「向きだけ」（9-61/9-62 の言い方）として区別する。

    python3 scripts/band_overlap.py --a -6.30 -5.98 --b -5.32 -5.71
    python3 scripts/band_overlap.py --run-a single_run/e2e_a1_reach --run-b single_run/e2e_a1_fix3_reach
"""
import argparse
import glob
import os
import re
import sys


def best_score(run_dir):
    """log_train.txt から `save best checkpoint with rewards X` の最終値を読む。"""
    log = f'{run_dir}/log/log_train.txt'
    vals = re.findall(r'save best checkpoint with rewards (-?[\d.]+)', open(log).read())
    if not vals:
        raise SystemExit(f'{log}: best の記録が見つからない')
    return float(vals[-1])


def scores_from_pattern(pattern):
    """glob パターンにマッチする全 run から best を集める（複数 seed 用）。

    ⚠️ ディレクトリだけ残って log/log_train.txt が無い run（削除済み・棚卸し対象）は
    黙ってスキップする。9-27 の `e2e_a1_reach_bug27` のように、run 名の接頭辞が
    別の完走 run と被ることがあるため。
    """
    dirs = sorted(d for d in glob.glob(pattern)
                  if os.path.exists(f'{d}/log/log_train.txt'))
    if not dirs:
        raise SystemExit(f'{pattern}: ログのある run が無い')
    return [best_score(d) for d in dirs], dirs


def judge(a, b, higher_is_better=True):
    """a=可動（対照）, b=固定（提案）。9-27/9-61/9-62 と同じ語彙で返す。"""
    lo_a, hi_a = min(a), max(a)
    lo_b, hi_b = min(b), max(b)
    overlap = not (hi_a < lo_b or hi_b < lo_a)
    b_better = (min(b) >= min(a)) if higher_is_better else (max(b) <= max(a))
    if overlap:
        verdict = '🟡 帯が重なる → 「向きだけ」。断定しない（9-61/9-62 の但し書き）'
    elif b_better:
        verdict = '✅ 帯が重ならず、固定した方が同等以上 → 助言を採用してよい'
    else:
        verdict = '❌ 帯が重ならず、固定した方が明確に悪化 → 助言は棄却'
    return overlap, b_better, verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', nargs='+', type=float, help='対照（可動）の best 値。複数 seed 可')
    ap.add_argument('--b', nargs='+', type=float, help='提案（固定）の best 値')
    ap.add_argument('--run-a', help='対照 run のディレクトリ（単一 run）')
    ap.add_argument('--run-b', help='提案 run のディレクトリ（単一 run）')
    ap.add_argument('--glob-a', help='対照 run の glob（複数 seed 用。例: single_run/e2e_a1_reach*）')
    ap.add_argument('--glob-b', help='提案 run の glob')
    ap.add_argument('--lower-is-better', action='store_true', help='Pusher 等スコアが大きいほど良い場合は指定しない（既定 higher=good）')
    a_ = ap.parse_args()

    if a_.glob_a:
        a, da = scores_from_pattern(a_.glob_a)
    elif a_.run_a:
        a, da = [best_score(a_.run_a)], [a_.run_a]
    else:
        a, da = a_.a, ['(手入力)'] * len(a_.a or [])
    if a_.glob_b:
        b, db = scores_from_pattern(a_.glob_b)
    elif a_.run_b:
        b, db = [best_score(a_.run_b)], [a_.run_b]
    else:
        b, db = a_.b, ['(手入力)'] * len(a_.b or [])
    if not a or not b:
        ap.error('--a/--b か --run-a/--run-b か --glob-a/--glob-b のいずれかを指定してください')

    print(f'対照: {a}  ({da})')
    print(f'提案: {b}  ({db})')

    # ⚠️⚠️ 1 seed は「帯」ではない（9-136 で実際に 1 点を帯として判定しかけた）。
    #   ⭐ 9-34 で「順位は seed の産物だった」を踏んでいるので、機械で止める。
    thin = [(n, v) for n, v in (('対照', a), ('提案', b)) if len(v) < 2]
    if thin:
        for n, v in thin:
            print(f'\n⛔ **{n}側が {len(v)} seed しかない。帯を作れない。**')
        print('⚠️ **この比較で順位を主張してはいけない**（9-34: 順位が seed の産物だった実例）。')
        print('⭐ 2 seed 目の完走を待つこと。参考値として範囲だけ出す:')
        overlap, b_better, verdict = judge(a, b, higher_is_better=not a_.lower_is_better)
        print(f'   （参考）{verdict}')
        return 2

    overlap, b_better, verdict = judge(a, b, higher_is_better=not a_.lower_is_better)
    print(f'\n{verdict}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
