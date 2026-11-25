#!/usr/bin/env python3
"""到達の配分（根元重 prox 対 先端重 dist）の差を「寄せ」と「保持」に分け、それぞれの出どころを見る（系譜 9-355、2026-11-25）。

学習・再生はしない。各 run の trace/arm_trace.npz（5 話）だけを読む。

量（話ごと）:
- n10: 先端と目標の距離が初めて 10 mm 未満になった step（9-278 と同じ定義）
- |Δq|max: 開始から n10 の姿勢までの関節の回転の最大（関節角は隣り合うリンクの相対の z 回り角）
- n_kin = |Δq|max / (π rad/s × 0.04 s)。速度の上限だけで決まる最短の step 数
- 保持の |e|: n10 より後の先端の誤差の中央（mm）
- 保持の |J⁺e|: その誤差を生む最小の関節の誤差（平面 3 リンクのヤコビアンの擬似逆、mrad）
- A・H: n10 の前・後の距離の和（9-302 と同じ）、ctrl: 記録済みの ep_reward_ctrl

読み（計算の前に書いた。系譜 9-355）:
- A-a: dist/prox の n10/n_kin の比が 25 % 以内 → 寄せの差は向かった姿勢の回転量（運動学）で説明できる
- B-a: |J⁺e| の比が 0.67〜1.5 なのに |e| が dist で大きい → てこ / B-b: |J⁺e| も開く → 関節の空間での保持の精度

使い方:
    python3 scripts/analyze_reach_alloc_phases.py              # 話ごと・全体・seed ごと
    python3 scripts/analyze_reach_alloc_phases.py --self-check # 9-278 の prox seed 0 の n10（19・21・19・19・20）を再現するか
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = ['seed0', 'seed1（元）', 'seed1（新）', 'seed2']
RUNS = {
    'prox': ['tripo_pjd_prox_phys', 'tripo_pjd_prox_phys_s2', 'tripo_pjd_prox_phys_s2_ws', 'tripo_pjd_prox_phys_s3'],
    'dist': ['tripo_pjd_dist_phys_cg', 'tripo_pjd_dist_phys_cg_s2', 'tripo_pjd_dist_phys_cg_s2_ws',
             'tripo_pjd_dist_phys_cg_s3'],
}
STEP_RAD = np.pi * 0.04  # 関節の速度の上限 π rad/s × 1 step 0.04 s


def yaw(R):
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def episodes(run):
    z = np.load(os.path.join(ROOT, 'single_run', run, 'trace', 'arm_trace.npz'), allow_pickle=True)
    tgt = z['target'][:2]
    rows = []
    for e in range(int(z['ep_count'])):
        X = np.array(z['ep_xmat'][e], float)
        P = np.array(z['ep_xpos'][e], float)
        B = np.array(z['ep_bone'][e], float)
        L = B[1:, 0]
        q = np.unwrap(np.stack([yaw(np.einsum('tji,tjk->tik', X[:, i - 1], X[:, i])) for i in (1, 2, 3)], 1), axis=0)
        tip = P[:, 3, :2] + np.einsum('tij,j->ti', X[:, 3], B[3])[:, :2]
        d = np.linalg.norm(tip - tgt, axis=1)
        n10 = int(np.argmax(d < 0.01))
        dq = np.abs(q[n10] - q[0])
        th = np.cumsum(q, 1)
        errs, jerr = [], []
        for t in range(n10, len(d)):
            J = np.zeros((2, 3))
            for j in range(3):
                J[0, j] = -(L[j:] * np.sin(th[t, j:])).sum()
                J[1, j] = (L[j:] * np.cos(th[t, j:])).sum()
            ev = tip[t] - tgt
            errs.append(np.linalg.norm(ev))
            jerr.append(np.linalg.norm(np.linalg.pinv(J) @ ev))
        rows.append(dict(ep=e, n10=n10, dqmax=np.degrees(dq.max()), jmax=int(dq.argmax()) + 1,
                         ratio=n10 / (dq.max() / STEP_RAD), qfin=np.degrees(q[-1]).round(0),
                         e_mm=np.median(errs) * 1e3, je_mrad=np.median(jerr) * 1e3,
                         A=d[:n10].sum(), H=d[n10:].sum(), ctrl=float(z['ep_reward_ctrl'][e])))
    return rows


def med(rows, k):
    return float(np.median([r[k] for r in rows]))


KEYS = ['n10', 'dqmax', 'ratio', 'e_mm', 'je_mrad', 'A', 'H', 'ctrl']


def main():
    if '--self-check' in sys.argv:
        got = [r['n10'] for r in episodes('tripo_pjd_prox_phys')]
        ok = got == [19, 21, 19, 19, 20]
        print(f"{'✅' if ok else '⛔'} prox seed 0 の n10 = {got}（9-278: [19, 21, 19, 19, 20]）")
        sys.exit(0 if ok else 1)
    data = {f: [episodes(r) for r in rs] for f, rs in RUNS.items()}
    for f, per_run in data.items():
        print(f'== {f}')
        for run, rows in zip(RUNS[f], per_run):
            for x in rows:
                print(f"{run:32s} 話{x['ep']} n10={x['n10']:3d} |Δq|max={x['dqmax']:5.1f}°(関節{x['jmax']}) "
                      f"n10/n_kin={x['ratio']:.2f} 最後の角={x['qfin']} 保持|e|={x['e_mm']:.2f}mm "
                      f"|J⁺e|={x['je_mrad']:.2f}mrad A={x['A']:.2f} H={x['H']:.2f} ctrl={x['ctrl']:.3f}")
    allp = sum(data['prox'], [])
    alld = sum(data['dist'], [])
    print('\n全体（4 seed × 5 話の中央）  prox / dist / 比 dist÷prox')
    for k in KEYS:
        print(f'  {k:8s} {med(allp, k):8.3f} {med(alld, k):8.3f} {med(alld, k) / med(allp, k):6.2f}')
    print('\nseed ごと（5 話の中央の比 dist÷prox）')
    for i, s in enumerate(SEEDS):
        p, d = data['prox'][i], data['dist'][i]
        print(f'  {s:10s} ' + ' '.join(f'{k}={med(d, k) / med(p, k):.2f}' for k in KEYS if k != 'ctrl'))


if __name__ == '__main__':
    main()
