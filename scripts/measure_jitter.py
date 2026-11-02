#!/usr/bin/env python3
"""**腕の揺れ（関節の行き来）を数える**: 再生の軌跡から、関節の角度の変化の向きが入れ替わった回数を出す（2026-11-02）。

軸6 の β seed 1 は箱に当たった直後の約 3 秒、3 つの関節が約 4 Hz で行き来した（系譜 9-330 の 11-02 夕方の追記）。
揺れを消す試み（系譜 9-332）の判定に、毎回同じ数え方を使うために作った。

数え方: `trace/arm_trace.npz` の各話について、隣どうしの body の姿勢（`ep_xmat`）の差を回転ベクトルにし、
その話で最も大きく回った軸へ射影して関節の角度にする（`np.unwrap` で ±π の折り返しを消す）。
1 step の変化が `--eps` rad を超える step が続けて 2 つあり、向きが逆なら 1 回と数える。
最初の `--steps` step（既定 100 ＝ 4 秒）だけを数える。

    python3 scripts/measure_jitter.py ra15_pusher_beta ra15_pusher_beta_s2
    python3 scripts/measure_jitter.py ra15_pusher_beta_s2 --trace trace/cf_XXX.npz   # 反実仮想の再生の軌跡

出すもの: 話ごとの入れ替わり回数（関節の合計と関節ごと）・箱の進み（`ep_cube_dx`）、5 話の中央値。
⚠️ 関節の角度は姿勢から逆算したもので、学習の内部の qpos ではない。1 自由度のヒンジの鎖を前提にする。
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def rotvec(R):
    """回転行列 (T,3,3) → 回転ベクトル (T,3)。"""
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    th = np.arccos(c)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    s = np.sin(th)
    k = np.where(s > 1e-9, th / (2 * np.where(s > 1e-9, s, 1)), 0.5)
    return v * k[:, None]


def joint_angles(X):
    """X: (T, B, 3, 3) の各 body の姿勢 → (T, B-1) の関節の角度。"""
    out = []
    for j in range(1, X.shape[1]):
        R = np.einsum('tji,tjk->tik', X[:, j - 1], X[:, j])
        rv = rotvec(R)
        ax = np.argmax(np.abs(rv).sum(0))
        out.append(np.unwrap(rv[:, ax]))
    return np.stack(out, 1)


def reversals(A, steps, eps):
    dq = np.diff(A[:steps + 1], axis=0)
    big = np.abs(dq) > eps
    return ((np.sign(dq[1:]) * np.sign(dq[:-1]) < 0) & big[1:] & big[:-1]).sum(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--trace', default='trace/arm_trace.npz', help='run のフォルダからの相対パス')
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--eps', type=float, default=0.001, help='数える 1 step の変化の下限 [rad]')
    a = ap.parse_args()
    print(f'数え方: 最初の {a.steps} step、1 step の変化 > {a.eps} rad の行き来')
    for r in a.runs:
        p = ROOT / 'single_run' / r / a.trace
        if not p.exists():
            print(f'⚠️ {r}: {p} が無い')
            continue
        z = np.load(p, allow_pickle=True)
        X = np.asarray(z['ep_xmat'], dtype=float)
        names = [str(n) for n in z['body_names']]
        dx = np.asarray(z['ep_cube_dx'], dtype=float)
        tot, per = [], []
        for e in range(X.shape[0]):
            n = reversals(joint_angles(X[e]), a.steps, a.eps)
            per.append(n)
            tot.append(int(n.sum()))
        joints = [f'{names[j-1]}→{names[j]}' for j in range(1, len(names))]
        print(f'{r}  ({p.name}, {X.shape[0]} 話)')
        print(f'  入れ替わり（関節の合計） {tot}  中央値 {np.median(tot):.0f}')
        print(f'  関節ごと {joints}: ' + ' '.join(str(list(map(int, n))) for n in per))
        print(f'  箱の進み [m] {np.round(dx, 3).tolist()}  中央値 {np.median(dx):.3f}')


if __name__ == '__main__':
    main()
