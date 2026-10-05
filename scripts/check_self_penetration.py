#!/usr/bin/env python3
"""⭐ 腕のリンクどうしの**すり抜け（自己干渉）**が、既存の run でどれだけ起きているかを測る（2026-10-05）。

⛔ 学習は自己干渉を見ていない（Choreonoid の既定）。Target-Pusher のデモで上腕が伸びた台座の中に
  折りたたまれていた（ユーザー発見）。⭐ **学習し直す前に、どの run でどれだけ起きているかを測る。**

⭐ 軌跡（`single_run/<run>/trace/arm_trace.npz`）だけを使う。学習も Choreonoid も要らない。
- 各リンク = 関節位置 `xpos` から `xmat @ bone_offset` へ伸びる線分 + 半径 `geom_size`（カプセル）
- ⭐ 離れたリンクの組はそのまま、**隣り合う組は関節のまわり（2 本の半径の和）を両方から削ってから**見る
  （関節での自然な重なりは数えず、**子が親の上へ折り返す**重なりだけを拾う）
- 根元の取り付け用の球（先頭の body）は、骨の向きが運動学に効かない（9-154）ので**半径 r の球**として扱う
- ⚠️ カプセルの延長分（`ext_start`）は軌跡に無いので含めない → **すり抜けを少なめに数える側**にずれる
- ⚠️ 軌跡は採用した話のうち中央値の 1 話（`xpos`）。全話ではない

    python3 scripts/check_self_penetration.py               # 全 run
    python3 scripts/check_self_penetration.py e2e_a1v_reach  # 指定 run
    python3 scripts/check_self_penetration.py --self-check  # 線分間距離の自己検査
"""
import glob
import os
import sys

import numpy as np

TOL = 0.005          # めり込みがこれ [m] を超えた step を「すり抜け」と数える


def seg_seg_dist(p1, q1, p2, q2):
    """線分 p1-q1 と p2-q2 の最短距離（先頭の軸で並べて一度に解く）。Ericson の方法。"""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.einsum('...i,...i', d1, d1); e = np.einsum('...i,...i', d2, d2)
    f = np.einsum('...i,...i', d2, r);  c = np.einsum('...i,...i', d1, r)
    b = np.einsum('...i,...i', d1, d2)
    eps = 1e-12
    den = a * e - b * b
    s = np.where(den > eps, np.clip((b * f - c * e) / np.where(den > eps, den, 1), 0, 1), 0.0)
    s = np.where(a <= eps, 0.0, s)
    t = np.where(e > eps, (b * s + f) / np.where(e > eps, e, 1), 0.0)
    # t を [0,1] に収め、s を取り直す
    t0 = t < 0; t1 = t > 1
    s = np.where(t0 & (a > eps), np.clip(-c / np.where(a > eps, a, 1), 0, 1), s)
    s = np.where(t1 & (a > eps), np.clip((b - c) / np.where(a > eps, a, 1), 0, 1), s)
    t = np.clip(t, 0, 1)
    c1 = p1 + d1 * s[..., None]; c2 = p2 + d2 * t[..., None]
    return np.linalg.norm(c1 - c2, axis=-1)


def self_check():
    P = lambda *v: np.array(v, float)
    cases = [((P(0, 0, 0), P(1, 0, 0), P(0, 1, 0), P(1, 1, 0)), 1.0),      # 平行
             ((P(0, 0, 0), P(1, 0, 0), P(0.5, -1, 1), P(0.5, 1, 1)), 1.0),  # ねじれ・交差の上
             ((P(0, 0, 0), P(1, 0, 0), P(2, 0, 0), P(3, 0, 0)), 1.0),       # 同一直線・離れ
             ((P(0, 0, 0), P(0, 0, 0), P(0, 3, 4), P(0, 3, 4)), 5.0),       # 点と点
             ((P(0, 0, 0), P(2, 0, 0), P(1, -1, 0), P(1, 1, 0)), 0.0)]      # 交差
    for (a, b, c, d), want in cases:
        got = float(seg_seg_dist(a, b, c, d))
        assert abs(got - want) < 1e-9, (got, want)
    print('✅ 線分間距離の自己検査 5 件一致')


def measure(run):
    p = f'single_run/{run}/trace/arm_trace.npz'
    d = np.load(p, allow_pickle=True)
    x, R, b = d['xpos'], d['xmat'], d['bone_offset']
    rad = np.asarray(d['geom_size'], float) if 'geom_size' in d.files else np.full(len(b), 0.03)
    rad = np.where(np.isfinite(rad), rad, 0.03)
    T, L = x.shape[:2]
    p0 = x.copy()
    p1 = x + np.einsum('tlij,lj->tli', R, b)
    p1[:, 0] = p0[:, 0]                       # 根元は球として扱う（9-154）
    worst, frac, pair_w = 0.0, 0.0, None
    pairs = [(i, j, False) for i in range(L) for j in range(i + 2, L)]
    # ⭐ 隣り合う組（根元の球は除く）も見る。⛔ ただし関節では必ず重なるので、
    #   **関節のまわり（2 本の半径の和）を両方から削ってから**重なりを見る → 子が親の上へ**折り返す**ときだけ拾う。
    #   ⛔ 初稿は隣り合う組を丸ごと外しており、Target-Pusher の「上腕が台座の中に折りたたまれる」を数えなかった
    pairs += [(i, i + 1, True) for i in range(1, L - 1)]
    for i, j, adj in pairs:
        a0, a1, c0, c1 = p0[:, i], p1[:, i], p0[:, j], p1[:, j]
        if adj:
            m = rad[i] + rad[j]
            da, dc = a1 - a0, c1 - c0
            la = np.linalg.norm(da, axis=-1, keepdims=True); lc = np.linalg.norm(dc, axis=-1, keepdims=True)
            if float(la.min()) <= m or float(lc.min()) <= m:
                continue                       # 短すぎて削ると残らない
            a1 = a1 - da / la * m              # 親は関節側（先端）を削る
            c0 = c0 + dc / lc * m              # 子は関節側（根元）を削る
        dist = seg_seg_dist(a0, a1, c0, c1)
        depth = rad[i] + rad[j] - dist
        f = float((depth > TOL).mean())
        if depth.max() > worst:
            worst, pair_w = float(depth.max()), (str(d['body_names'][i]), str(d['body_names'][j]))
        frac = max(frac, f)
    return worst, frac, pair_w, T


def main():
    if '--self-check' in sys.argv:
        return self_check()
    runs = [a for a in sys.argv[1:] if not a.startswith('-')]
    if not runs:
        runs = sorted(os.path.basename(os.path.dirname(os.path.dirname(f)))
                      for f in glob.glob('single_run/*/trace/arm_trace.npz') if '_aborted' not in f)
    thesis = ''.join(open(f, encoding='utf-8').read()
                     for f in glob.glob('docs/研究応用/修論ドラフト/第[1-5]章*.md')
                     + glob.glob('docs/研究応用/修論ドラフト/付録*.md'))
    rows = []
    for r in runs:
        try:
            w, fr, pr, T = measure(r)
        except Exception as e:
            print(f'  ⚠️ {r}: 読めない（{e!r}）'); continue
        rows.append((r, w, fr, pr, T, (f'`{r}`' in thesis) or (r in thesis)))
    rows.sort(key=lambda z: -z[1])
    hit = [z for z in rows if z[2] > 0]
    print(f'\n⭐ 対象 {len(rows)} run  /  すり抜けあり（めり込み > {TOL*1000:.0f} mm の step が 1 つ以上）: {len(hit)} run'
          f'  /  そのうち修論本文に名前が出る run: {sum(z[5] for z in hit)}')
    print(f'{"run":44s} {"最大めり込み":>10s} {"step の割合":>10s}  {"組":14s} 修論')
    for r, w, fr, pr, T, th in rows:
        if fr == 0 and len(runs) > 3:
            continue
        print(f'{r:44s} {w*1000:8.0f} mm {fr*100:9.1f} %  {str(pr):14s} {"★" if th else ""}')


if __name__ == '__main__':
    main()
