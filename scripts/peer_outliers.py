#!/usr/bin/env python3
"""**比べる仲間の中で 1 本だけ外れた run** を拾い、先に再生して見る候補として並べる（2026-10-12）。

[異常を能動的に見つける方法.md](../docs/リポジトリ説明/異常を能動的に見つける方法.md) の方法 6。
組は `check_before_conclusion.py` の `COMPARISONS`（修論が比べる組）。順位や効果は判定しない（それは事前登録の読みの仕事）。
拾うのは「物理や設定の欠陥を疑う形の外れ方」だけ:
  ① 同じ条件の seed 対で best が大きく違う（差が大きい方の半分を超え、かつ 5 以上）
  ② 同じ組・同じ seed で、1 本だけ課題を達成していない（押しは対象が 0.1 m 未満、到達は誤差 10 mm 超）のに、
     仲間は達成している（押し 0.5 m 以上、到達 5 mm 未満）
実例: 9-240 の配分・実在 dist（prox 0〜1 mm に対し 57〜131 mm → 台座の球に当たっていた、9-241）。

    python3 scripts/peer_outliers.py                    # 取り直しの run
    python3 scripts/peer_outliers.py --suffix _noclip   # 退避した run で確かめる
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from check_before_conclusion import COMPARISONS  # noqa: E402
from counterfactual_replay import metric  # noqa: E402


def best(r):
    p = ROOT / 'single_run' / r / 'log' / 'log_train.txt'
    if not p.exists() or 'training done' not in p.read_text(errors='ignore'):
        return None
    m = re.findall(r'save best checkpoint with rewards ([-0-9.]+)', p.read_text(errors='ignore'))
    return float(m[-1]) if m else None


def task(r):
    p = ROOT / 'single_run' / r / 'trace' / 'arm_trace.npz'
    if not p.exists():
        return None
    try:
        name, v = metric(np.load(p, allow_pickle=True))
    except Exception:
        return None
    return name, float(v.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', default='', help='run 名の末尾に付ける（例: _noclip）')
    a = ap.parse_args()
    out = []
    names = sorted({r for _, runs, _ in COMPARISONS for r in runs})
    # ① seed 対
    for r in names:
        b0, b1 = best(r + a.suffix), best(r + '_s2' + a.suffix)
        if b0 is None or b1 is None:
            continue
        if abs(b0 - b1) > max(5.0, 0.5 * max(abs(b0), abs(b1))):
            out.append(f'⚠️ seed で best が大きく違う: {r}{a.suffix} {b0:.2f} 対 _s2 {b1:.2f} → 両方を再生して形と動きを比べる')
    # ② 組の中で 1 本だけ達成していない
    for gname, runs, _ in COMPARISONS:
        for s2 in ('', '_s2'):
            got = {r: task(r + s2 + a.suffix) for r in runs}
            got = {r: v for r, v in got.items() if v}
            for r, (kind, v) in got.items():
                peers = [pv for pr, (pk, pv) in got.items() if pr != r and pk == kind]
                if not peers:
                    continue
                if kind.startswith('到達') and v > 10 and min(peers) < 5:
                    out.append(f'⚠️ {gname}: {r}{s2}{a.suffix} だけ届かない（誤差 {v:.0f} mm、仲間 {min(peers):.0f} mm）→ 先に再生して見る')
                if kind.startswith('対象') and v < 0.1 and max(peers) > 0.5:
                    out.append(f'⚠️ {gname}: {r}{s2}{a.suffix} だけ対象が動かない（{v:.2f} m、仲間 {max(peers):.2f} m）→ 先に再生して見る')
    print('\n'.join(out) if out else '✅ 外れた run は無い（完走して軌跡がある分だけ）')


if __name__ == '__main__':
    main()
