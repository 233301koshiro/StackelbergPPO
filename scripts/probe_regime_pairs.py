#!/usr/bin/env python3
"""減速比を共設計する物理の上で、運動学の指標と動力学の正解が逆を向く形の対を探す（系譜 9-365）。学習なし・GPU なし。

背景（系譜 9-364）: 今の物理（回転子の慣性 1.71・速度の上限 3.14 rad/s）では腕は速度の上限で決まり、
押しの順位は運動学で出る。同じモーターで減速比を下げると（トルク↓・速度の上限↑・回転子の慣性 ∝ 減速比²）、
トルクと慣性で決まる側に入る。その物理で、運動学の指標（9-295 の「先端の最大の速さ」）の順位と、
動力学の正解の代わり（トルクの上限つきの軌道最適化）の順位が逆になる対があるかを見る。

物理（gear = 関節の最大トルク N·m。3 関節とも同じ値）:
  今      : 速度の上限 3.1416、回転子 1.71、関節の質量なし                       （今の `_vp` の物理）
  共設計  : 速度の上限 3.1416×150/gear、回転子 max(0.05, 1.71×(gear/150)²)、関節 2・3 に質量 gear/25 kg
課題（押しの最初の一打の近似）: 静止・初めの姿勢（+y へ伸ばす、arm_safe_init のヨー 90°）から、先端が箱の手前の面
  （x = 0.85、|y| <= 0.15）を +x 向きに横切る。そのときの先端の +x の速さが大きいほど良い（箱に渡す運動量の目安）。
  途中でリンクが箱（[0.85, 1.15] × [-0.15, 0.15]）に入ったら不成立。制限時間 1.0 s。
正解の代わり: 関節トルクを 8 点の折れ線にして CEM で最適化（自前の剛体の式。コリオリは質量行列の数値微分）。
  ⚠️ Choreonoid ではない。接触・箱の動き・カプセルの太さは見ない。順位の向きを見るための近似。
運動学の指標: 箱の手前の面（x = 0.85、|y| <= 0.15）の逆運動学の全解の中の、+x の先端の最大の速さ（その物理の速度の上限で）。
  今の物理では最適化がこの値に近づくはず（速度の上限で決まる）。これが --check の確かめ。

  python3 scripts/probe_regime_pairs.py            # 全部（数分）
  python3 scripts/probe_regime_pairs.py --check    # 今の物理で、最適化が運動学の上限に届くか（最適化の確かめ）
  python3 scripts/probe_regime_pairs.py --random   # リンク長を振った 24 形で、順位が逆を向く対を数える（約 15 分）。⚠️ 種 1 つなので最適化のばらつき（最大約 4 割）を逆転と数える
  python3 scripts/probe_regime_pairs.py --verify   # 逆を向いた上位の形を種 3 つで測り直す（約 10 分）。9-365 では形 7 対 13 の逆転は消えた
"""
import sys
import numpy as np

RHO = 294.2
RADII = [0.045, 0.0326, 0.032]          # XML の半径（根元から順）。形によらず同じにする
FORMS = {'mid': [0.334, 0.329, 0.224], 'long': [0.547, 0.538, 0.366],
         'prox': [0.5, 0.25, 0.1368], 'dist': [0.1368, 0.25, 0.5]}   # probe_kinematic_indices.py と同じ（9-295）
BOX = (0.85, 1.15, -0.15, 0.15)
DT, T_END, NK = 0.005, 1.0, 8


def physics(kind, g):
    if kind == 'now':
        return dict(g=g, W=3.1416, arm=1.71, mj=0.0)
    return dict(g=g, W=3.1416 * 150 / g, arm=max(0.05, 1.71 * (g / 150) ** 2), mj=g / 25)


def pts(l, q):
    """q: (N,3) → 関節と先端の位置 (N,4,2)"""
    a = np.cumsum(q, axis=1); P = [np.zeros((q.shape[0], 2))]
    for k in range(3):
        P.append(P[-1] + l[k] * np.stack([np.cos(a[:, k]), np.sin(a[:, k])], 1))
    return np.stack(P, 1)


def mass_matrix(l, q, ph):
    N = q.shape[0]; P = pts(l, q); M = np.zeros((N, 3, 3)) + np.eye(3) * ph['arm']
    def Jv(pt, upto):
        J = np.zeros((N, 2, 3))
        for i in range(upto):
            J[:, 0, i] = -(pt[:, 1] - P[:, i, 1]); J[:, 1, i] = pt[:, 0] - P[:, i, 0]
        return J
    for k in range(3):
        mk = RHO * (np.pi * RADII[k] ** 2 * l[k] + 4 / 3 * np.pi * RADII[k] ** 3)
        J = Jv((P[:, k] + P[:, k + 1]) / 2, k + 1); M += mk * np.einsum('nai,naj->nij', J, J)
        w = np.zeros(3); w[:k + 1] = 1; M += mk * l[k] ** 2 / 12 * np.outer(w, w)
    if ph['mj']:
        for j in (1, 2):
            J = Jv(P[:, j], j); M += ph['mj'] * np.einsum('nai,naj->nij', J, J)
    return M


def accel(l, q, dq, tau, ph, eps=1e-5):
    M = mass_matrix(l, q, ph)
    dM = [(mass_matrix(l, q + eps * np.eye(3)[k], ph) - mass_matrix(l, q - eps * np.eye(3)[k], ph)) / (2 * eps) for k in range(3)]
    Mdot = sum(dM[k] * dq[:, k, None, None] for k in range(3))
    h = np.einsum('nij,nj->ni', Mdot, dq) - 0.5 * np.stack([np.einsum('ni,nij,nj->n', dq, dM[k], dq) for k in range(3)], 1)
    return np.linalg.solve(M, (tau - 1.0 * dq - h)[..., None])[..., 0]   # 減衰 1 N·m·s/rad


def rollout(l, ph, U):
    """U: (N, NK, 3) ∈ [-1,1]。戻り値: (+x の速さ, 横切った時刻) 不成立は (-inf, nan)"""
    N = U.shape[0]; q = np.tile([np.pi / 2, 0, 0], (N, 1)).astype(float); dq = np.zeros((N, 3))
    tk = np.linspace(0, T_END, NK); score = np.full(N, -np.inf); thit = np.full(N, np.nan)
    alive = np.ones(N, bool)
    for s in range(int(T_END / DT)):
        t = s * DT
        u = np.stack([np.stack([np.interp(t, tk, U[n, :, j]) for j in range(3)]) for n in range(N)])
        ddq = accel(l, q, dq, ph['g'] * u, ph)
        dq = np.clip(dq + ddq * DT, -ph['W'], ph['W']); q = q + dq * DT
        P = pts(l, q)
        # リンク上の点が箱に入ったか（先端は除く）
        seg = np.concatenate([P[:, k:k + 1] + f * (P[:, k + 1:k + 2] - P[:, k:k + 1]) for k in range(3) for f in (0.2, 0.4, 0.6, 0.8)], 1)
        inside = ((seg[..., 0] > BOX[0]) & (seg[..., 0] < BOX[1]) & (seg[..., 1] > BOX[2]) & (seg[..., 1] < BOX[3])).any(1)
        alive &= ~inside
        tip = P[:, 3]
        J0 = np.stack([-(tip[:, 1] - P[:, i, 1]) for i in range(3)], 1)
        vx = (J0 * dq).sum(1)
        hit = alive & np.isnan(thit) & (tip[:, 0] >= BOX[0]) & (np.abs(tip[:, 1]) <= BOX[3]) & (vx > 0)
        score[hit] = vx[hit]; thit[hit] = t
        # 面より先を上下から回り込んで入った先端も不成立
        alive &= ~(np.isnan(thit) & (tip[:, 0] > BOX[0]) & (tip[:, 0] < BOX[1]) & (np.abs(tip[:, 1]) < BOX[3]))
        if (~np.isnan(thit) | ~alive).all():
            break
    return score, thit


def cem1(l, ph, iters=40, pop=64, elite=8, seed=0):
    rng = np.random.default_rng(seed); mu = np.zeros((NK, 3)); sd = np.ones((NK, 3)) * 0.8
    best = (-np.inf, np.nan)
    for _ in range(iters):
        U = np.clip(mu + sd * rng.standard_normal((pop, NK, 3)), -1, 1)
        sc, th = rollout(l, ph, U)
        o = np.argsort(-sc)
        if sc[o[0]] > best[0]:
            best = (sc[o[0]], th[o[0]])
        ok = o[:elite][np.isfinite(sc[o[:elite]])]
        if len(ok) >= 2:
            mu = U[ok].mean(0); sd = U[ok].std(0) + 0.05
    return best


def cem(l, ph, seeds=(0, 1, 2)):
    """CEM を 3 つの種で回して最良を取る（局所解に落ちた分を減らす）。"""
    return max((cem1(l, ph, seed=s) for s in seeds), key=lambda x: x[0])


def kin_index(l, W, ys=np.linspace(-0.15, 0.15, 7)):
    """箱の手前の面（x = 0.85、|y| <= 0.15）のどこで当ててもよいときの、+x の先端の最大の速さ（速度の上限だけ）。
    最適化と同じ条件にそろえる（面の端では軸の上より速い。9-363 の軸外の計算と同じ理由）。"""
    best = 0.0
    for y in ys:
      for phi in np.linspace(-np.pi, np.pi, 720, endpoint=False):
        w = np.array([0.85, y]) - l[2] * np.array([np.cos(phi), np.sin(phi)]); d = np.linalg.norm(w)
        c = (d * d - l[0] ** 2 - l[1] ** 2) / (2 * l[0] * l[1])
        if abs(c) > 1:
            continue
        for s in (1, -1):
            q2 = s * np.arccos(c); q1 = np.arctan2(w[1], w[0]) - np.arctan2(l[1] * np.sin(q2), l[0] + l[1] * np.cos(q2))
            P = pts(l, np.array([[q1, q2, phi - q1 - q2]]))[0]
            best = max(best, W * sum(abs(P[3, 1] - P[i, 1]) for i in range(3)))
    return best


if __name__ == '__main__' and '--random' not in sys.argv and '--verify' not in sys.argv:
    if '--check' in sys.argv:
        for f in ('prox', 'dist'):
            v, t = cem(FORMS[f], physics('now', 150))
            print(f'今の物理 gear 150 {f}: 最適化 +x の速さ {v:.3f} m/s・横切る時刻 {t:.3f} s'
                  f'（運動学の指標 {kin_index(FORMS[f], 3.1416):.3f}。学習した押しは根元重 0.44 s・先端重 0.72 s で当たる）', flush=True)
        sys.exit(0)
    print('form  物理      gear  速度上限  運動学の指標  最適化 +x の速さ  横切る時刻', flush=True)
    res = {}
    for kind, gears in [('now', [150]), ('co', [20, 40, 75, 150])]:
        for g in gears:
            ph = physics(kind, g)
            for f, l in FORMS.items():
                k = kin_index(l, ph['W']); v, t = cem(l, ph)
                res[(kind, g, f)] = (k, v, t)
                print(f'{f:5s} {kind:4s} {g:5d} {ph["W"]:8.1f} {k:12.3f} {v:14.3f} {t:12.3f}', flush=True)
    print('\n対の向き（運動学の指標 と 最適化 が逆なら ⭐）')
    for kind, gears in [('now', [150]), ('co', [20, 40, 75, 150])]:
        for g in gears:
            for a, b in [('long', 'mid'), ('prox', 'dist')]:
                ka, va, _ = res[(kind, g, a)]; kb, vb, _ = res[(kind, g, b)]
                flip = (ka - kb) * (va - vb) < 0 and abs(va - vb) / max(va, vb) > 0.05
                print(f'  {kind:4s} gear {g:3d} {a}/{b}: 運動学 {ka/kb:.2f} 倍  最適化 {va/vb if vb > 0 else float("nan"):.2f} 倍 {"⭐ 逆" if flip else ""}')
    # 共設計の物理で、各形が gear を選べるとき（形ごとに最良の gear）
    print('\n共設計の物理で gear を形ごとに選べるとき（最良）')
    for f in FORMS:
        g, (k, v, t) = max(((g, res[('co', g, f)]) for g in [20, 40, 75, 150]), key=lambda x: x[1][1])
        print(f'  {f:5s}: 最良の gear {g}  最適化 +x の速さ {v:.3f} m/s（運動学の指標、gear 150 の速度上限で {res[("co", 150, f)][0]:.3f}）')


def random_search(n=24, gears=(20, 40, 75), seed=0, margin=0.10):
    """--random: リンク長を振った n 形で、運動学の指標（今の速度の上限 3.1416、9-295 の指標）の順位と、
    共設計の物理で形ごとに最良の gear を選んだ最適化の順位が、両方とも margin 以上の差で逆を向く対を数える。"""
    rng = np.random.default_rng(seed); forms = []
    while len(forms) < n:
        l = rng.uniform(0.1, 0.6, 3)
        if 0.9 <= l.sum() <= 1.5:
            forms.append(l)
    rows = []
    for i, l in enumerate(forms):
        k = kin_index(l, 3.1416)
        best = max(((g,) + cem1(l, physics('co', g)) for g in gears), key=lambda x: x[1])
        rows.append((l, k, best[1], best[0]))
        print(f'  形 {i:2d} 長さ {np.round(l, 3)} 総長 {l.sum():.3f}: 運動学 {k:.3f}  最適化 {best[1]:.3f}（gear {best[0]}）', flush=True)
    flips = []
    for i in range(n):
        for j in range(i + 1, n):
            ki, vi = rows[i][1], rows[i][2]; kj, vj = rows[j][1], rows[j][2]
            if not (np.isfinite(vi) and np.isfinite(vj)):
                continue
            rk = ki / kj; rv = vi / vj
            if (rk > 1 + margin and rv < 1 - margin) or (rk < 1 - margin and rv > 1 + margin):
                flips.append((i, j, rk, rv))
    npairs = n * (n - 1) // 2
    print(f'\n逆を向く対（両方 {int(margin*100)} % 以上の差）: {len(flips)} / {npairs}')
    for i, j, rk, rv in flips:
        print(f'  形 {i} 対 形 {j}: 運動学 {rk:.2f} 倍  最適化 {rv:.2f} 倍  長さ {np.round(rows[i][0], 3)} / {np.round(rows[j][0], 3)}')
    ks = np.array([r[1] for r in rows]); vs = np.array([r[2] for r in rows])
    rk_ = np.argsort(np.argsort(ks)); rv_ = np.argsort(np.argsort(vs))
    print(f'順位相関（スピアマン）: {1 - 6 * np.sum((rk_ - rv_) ** 2) / (n * (n * n - 1)):.3f}')


if __name__ == '__main__' and '--random' in sys.argv:
    random_search()


def verify(forms=None):
    """--verify: --random で逆を向いた上位の形を、CEM の種 3 つ（cem）で gear ごとに測り直す（9-365）。
    種 1 つの最適化は最大で約 4 割ばらつき（形 7 の gear 20: 9.9 → 14.0）、1 割の逆転はそれより小さい。"""
    forms = forms or {'7': [0.59, 0.443, 0.425], '13': [0.545, 0.214, 0.412],
                      '23': [0.566, 0.157, 0.465], '10': [0.343, 0.545, 0.567]}
    for name, l in forms.items():
        l = np.array(l); out = []
        for kind, g in [('now', 150), ('co', 20), ('co', 40), ('co', 75), ('co', 150)]:
            v, _ = cem(l, physics(kind, g)); out.append(f'{kind}{g} {v:.2f}')
        print(f'形 {name} {l} 運動学(3.14) {kin_index(l, 3.1416):.3f} | ' + '  '.join(out), flush=True)


if __name__ == '__main__' and '--verify' in sys.argv:
    verify()
