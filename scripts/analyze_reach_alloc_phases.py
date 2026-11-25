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
    python3 scripts/analyze_reach_alloc_phases.py --trace-dir <dir>   # 50 話の再生（TRACE_EPISODES=50・TRACE_OUT=<dir>/<run>.npz）を読む

保持は前半（n10 から 50 step ＝ 2 秒、He）と後半（Hl）に分け、前半の関節の速さの RMS（qd_early, rad/s）と
|制御入力| の最大の中央（u_early・u_late、ep_u_abs）も出す。f ＝ 先端重の保持の超過のうち前半が占める割合。
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
TRACE_DIR = None  # --trace-dir <dir>: <dir>/<run>.npz を読む（50 話の再生を TRACE_OUT で別の場所へ書いたとき）
STEP_RAD = np.pi * 0.04  # 関節の速度の上限 π rad/s × 1 step 0.04 s


def yaw(R):
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


EARLY = 50  # 保持の前半 ＝ n10 から 50 step（2 秒）


def trace_path(run):
    if TRACE_DIR:
        return os.path.join(TRACE_DIR, f'{run}.npz')
    return os.path.join(ROOT, 'single_run', run, 'trace', 'arm_trace.npz')


def episodes(run):
    z = np.load(trace_path(run), allow_pickle=True)
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
        qd = np.diff(q, axis=0) / 0.04
        u = np.array(z['ep_u_abs'][e], float)
        early = slice(n10, min(n10 + EARLY, len(d)))
        rows.append(dict(ep=e, n10=n10, R=float(z['ep_reward'][e]), fin_mm=d[-1] * 1e3,
                         He=d[early].sum(), Hl=d[n10 + EARLY:].sum(),
                         qd_early=float(np.sqrt((qd[n10:n10 + EARLY] ** 2).sum(1).mean())),
                         u_early=float(np.median(u[early])), u_late=float(np.median(u[n10 + EARLY:])), dqmax=np.degrees(dq.max()), jmax=int(dq.argmax()) + 1,
                         ratio=n10 / (dq.max() / STEP_RAD), qfin=np.degrees(q[-1]).round(0),
                         e_mm=np.median(errs) * 1e3, je_mrad=np.median(jerr) * 1e3,
                         A=d[:n10].sum(), H=d[n10:].sum(), ctrl=float(z['ep_reward_ctrl'][e])))
    return rows


def med(rows, k):
    return float(np.median([r[k] for r in rows]))


KEYS = ['R', 'n10', 'dqmax', 'ratio', 'e_mm', 'je_mrad', 'A', 'H', 'He', 'Hl', 'fin_mm', 'qd_early', 'u_early', 'u_late', 'ctrl']


def main():
    global TRACE_DIR
    if '--trace-dir' in sys.argv:
        TRACE_DIR = sys.argv[sys.argv.index('--trace-dir') + 1]
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
    print('\n全体（4 seed の全話の中央）  prox / dist / 比 dist÷prox')
    for k in KEYS:
        print(f'  {k:8s} {med(allp, k):8.3f} {med(alld, k):8.3f} {med(alld, k) / med(allp, k):6.2f}')
    print('\nseed ごと（各話の中央の比 dist÷prox）')
    for i, s in enumerate(SEEDS):
        p, d = data['prox'][i], data['dist'][i]
        print(f'  {s:10s} ' + ' '.join(f'{k}={med(d, k) / med(p, k):.2f}' for k in KEYS if k != 'ctrl'))
    print('\nseed ごとの中央（prox / dist）と、保持の超過のうち前半が占める割合 f')
    for i, s in enumerate(SEEDS):
        p, d = data['prox'][i], data['dist'][i]
        exc = med(d, 'H') - med(p, 'H')
        f = (med(d, 'He') - med(p, 'He')) / exc if abs(exc) > 1e-9 else float('nan')
        print(f'  {s:10s} ' + ' '.join(f'{k}={med(p, k):.3g}/{med(d, k):.3g}' for k in KEYS) + f'  H の超過={exc:.3f} f={f:.2f}')


if __name__ == '__main__':
    main()
