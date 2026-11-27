#!/usr/bin/env python3
"""縦型の到達で先端に荷物（質量）を持たせると、どの質量から運動学・静力学の物差しが順位を外すかを計算する（系譜 9-366）。学習なし・GPU なし。

難しさの段（11-27 夜、ユーザー決定）を、荷物の質量 1 つのつまみで振る:
  荷物なし → 簡単（運動学で順位が出るはず） / 中くらい → 頑張ればできる（動力学） / 重い → 無理（目標で止まれない。静力学で分かる）

課題（`e2e_a1v_reach_phys` と同じ形）: 縦の面のお辞儀型。肩は高さ 0.2395 m、ピッチ関節 3 つ（可動域 ±90°）、初めはまっすぐ上（全関節 0）。
  目標 (0.8, 0.15)（x, 高さ）。報酬 = Σ_1000 step [ −先端と目標の距離 − 0.2 × mean(u²) ]（1 step 0.04 s、u = τ / gear ∈ [−1, 1]）。
  物理は今の `_vp`: 回転子 1.71・減衰 1・速度の上限 3.1416 rad/s・gear 150（gear は高いほど止まる費用が下がるので上限を取る）。
  リンクは XML の半径（0.0501・0.0466・0.0535）、実効密度 189.9。荷物は先端の点質量。

物差し（どれも形ごとに 1 つの値。大きいほど良い）:
  K（運動学）: 9-299 の参照の制御。逆運動学の解へ関節空間でまっすぐ、速度の上限だけで動かし −Σ 距離。重力・費用は見ない
  S（静力学）: K の動きに、着いてから終わりまで止まり続ける指令の費用（重力のトルク / gear）を足す。止まれない（|u| > 1）なら −inf
  D（正解の代わり）: 重力の補償つきの PD 制御で逆運動学の解へ関節空間でまっすぐ追わせ、解・かける時間・ゲインを総当たり
     （トルクの上限・速度の上限・重力・指令の費用を入れる）。4 s より後は最後の姿勢で止まり続けるとする（止まれなければ −inf）。
     ⚠️ 最初はトルクの折れ線を CEM で最適化したが、目標でぴたりと止められず K の 3〜7 倍悪かったので替えた（`ruler_D_cem`、9-366）
  ⚠️ 自前の剛体の式（Choreonoid ではない）。可動域は角を切るだけ。床・自己接触は見ない。

段の読み方: K の順位 ＝ D の順位なら「簡単」、K ≠ D で S ＝ D なら「静力学（安い）」、S ≠ D なら「動力学（頑張ればできる）」の段。

  python3 scripts/probe_vertical_payload.py            # 5 形 × 荷物 5 通り（数分〜十数分）
  python3 scripts/probe_vertical_payload.py --check    # 荷物なしで D が K を大きく外れないか（最適化の確かめ）
  python3 scripts/probe_vertical_payload.py --random   # リンク長を振った 24 形で、荷物ごとに K・S が D と逆を向く対を数える（数分）
  python3 scripts/probe_vertical_payload.py --random --nstep=50   # 話を 2 s に縮める（動く途中の比重を上げる）
"""
import sys
import numpy as np

RHO, G0, W, GEAR, ARM, DAMP, CC = 189.9, 9.81, 3.1416, 150.0, 1.71, 1.0, 0.2
RADII = [0.0501, 0.0466, 0.0535]
SH = np.array([0.0, 0.2395])
TGT = np.array([0.8, 0.15])
DT_SIM, DT_STEP, NSTEP = 0.005, 0.04, 1000
# --nstep N: 話の長さ（既定 1000 = 40 s。起動行の done_condition.max_nsteps に当たる）。短いほど動く途中の比重が上がる
for _a in sys.argv:
    if _a.startswith('--nstep='):
        NSTEP = int(_a.split('=')[1])
T_OPT, NK = 2.0, 10
LIM = np.pi / 2
BASE = [0.4031, 0.3308, 0.2763]
FORMS = {'base': BASE, 'prox': [0.5, 0.3, 0.2102], 'dist': [0.2102, 0.3, 0.5],
         'long': [x * 1.3 for x in BASE], 'short': [x * 0.85 for x in BASE]}


def pts(l, q):
    a = np.cumsum(q, axis=1); P = [np.tile(SH, (q.shape[0], 1))]
    for k in range(3):
        P.append(P[-1] + l[k] * np.stack([np.sin(a[:, k]), np.cos(a[:, k])], 1))
    return np.stack(P, 1)


def link_mass(l):
    return [RHO * (np.pi * RADII[k] ** 2 * l[k] + 4 / 3 * np.pi * RADII[k] ** 3) for k in range(3)]


def jac(P, pt, upto):
    """xz 平面の点の速度 = J q̇。角は +z から +x へ測る（d/dq of (sin a, cos a) = (cos a, −sin a)）。"""
    N = P.shape[0]; J = np.zeros((N, 2, 3))
    for i in range(upto):
        r = pt - P[:, i]
        J[:, 0, i] = r[:, 1]; J[:, 1, i] = -r[:, 0]
    return J


def dyn_terms(l, q, mp):
    P = pts(l, q); N = q.shape[0]; ms = link_mass(l)
    M = np.zeros((N, 3, 3)) + np.eye(3) * ARM; Gv = np.zeros((N, 3))
    for k in range(3):
        com = (P[:, k] + P[:, k + 1]) / 2; J = jac(P, com, k + 1)
        M += ms[k] * np.einsum('nai,naj->nij', J, J)
        w = np.zeros(3); w[:k + 1] = 1; M += ms[k] * l[k] ** 2 / 12 * np.outer(w, w)
        Gv += ms[k] * G0 * J[:, 1, :]           # 重力のトルク（∂V/∂q）
    if mp:
        J = jac(P, P[:, 3], 3); M += mp * np.einsum('nai,naj->nij', J, J); Gv += mp * G0 * J[:, 1, :]
    return M, Gv, P


def accel(l, q, dq, tau, mp, eps=1e-5):
    M, Gv, _ = dyn_terms(l, q, mp)
    dM = [(dyn_terms(l, q + eps * np.eye(3)[k], mp)[0] - dyn_terms(l, q - eps * np.eye(3)[k], mp)[0]) / (2 * eps) for k in range(3)]
    Mdot = sum(dM[k] * dq[:, k, None, None] for k in range(3))
    h = np.einsum('nij,nj->ni', Mdot, dq) - 0.5 * np.stack([np.einsum('ni,nij,nj->n', dq, dM[k], dq) for k in range(3)], 1)
    return np.linalg.solve(M, (tau - DAMP * dq - h - Gv)[..., None])[..., 0]


def hold_cost(l, q, mp):
    """姿勢 q で止まり続ける 1 step の指令の費用と、止まれるか。"""
    _, Gv, P = dyn_terms(l, q, mp); u = Gv / GEAR
    return CC * np.mean(u ** 2, axis=1), np.all(np.abs(u) <= 1, axis=1), np.linalg.norm(P[:, 3] - TGT, axis=1)


def ik_all(l, n=720):
    sols = []
    for phi in np.linspace(-np.pi, np.pi, n, endpoint=False):      # 最後のリンクの向き（+z から）
        w = TGT - SH - l[2] * np.array([np.sin(phi), np.cos(phi)]); d = np.linalg.norm(w)
        c = (d * d - l[0] ** 2 - l[1] ** 2) / (2 * l[0] * l[1])
        if abs(c) > 1:
            continue
        for s in (1, -1):
            q2 = s * np.arccos(c)
            a1 = np.arctan2(w[0], w[1]) - np.arctan2(l[1] * np.sin(q2), l[0] + l[1] * np.cos(q2))
            q = np.array([a1, q2, phi - a1 - q2]); q = (q + np.pi) % (2 * np.pi) - np.pi
            if np.all(np.abs(q) <= LIM):
                sols.append(q)
    return np.array(sols)


def ruler_K_S(l, mp):
    S = ik_all(l)
    if len(S) == 0:
        return -np.inf, -np.inf
    bestK = bestS = -np.inf
    hc, ok, _ = hold_cost(l, S, mp)
    for q, c, o in zip(S, hc, ok):
        T = np.abs(q).max() / W; nmove = min(int(np.ceil(T / DT_STEP)), NSTEP)
        f = np.clip(np.arange(nmove) * DT_STEP / T, 0, 1)
        P = pts(l, np.outer(f, q)); d = np.linalg.norm(P[:, 3] - TGT, axis=1).sum()
        bestK = max(bestK, -d)
        if o:
            bestS = max(bestS, -d - (NSTEP - nmove) * c)
    return bestK, bestS


def rollout(l, mp, U):
    N = U.shape[0]; q = np.zeros((N, 3)); dq = np.zeros((N, 3)); tk = np.linspace(0, T_OPT, NK)
    R = np.zeros(N); nsub = int(round(DT_STEP / DT_SIM))
    for s in range(int(T_OPT / DT_STEP)):
        t = s * DT_STEP
        u = np.stack([np.stack([np.interp(t, tk, U[n, :, j]) for j in range(3)]) for n in range(N)])
        for _ in range(nsub):
            ddq = accel(l, q, dq, GEAR * u, mp)
            dq = np.clip(dq + ddq * DT_SIM, -W, W); q = q + dq * DT_SIM
            hit = np.abs(q) > LIM; q = np.clip(q, -LIM, LIM); dq[hit] = 0.0
        R += -np.linalg.norm(pts(l, q)[:, 3] - TGT, axis=1) - CC * np.mean(u ** 2, axis=1)
    c, ok, d = hold_cost(l, q, mp)
    R += (NSTEP - int(T_OPT / DT_STEP)) * (-d - c)
    R[~ok] = -np.inf
    return R


def cem1(l, mp, iters=30, pop=48, elite=8, seed=0):
    rng = np.random.default_rng(seed); mu = np.zeros((NK, 3)); sd = np.ones((NK, 3)) * 0.7; best = -np.inf
    for _ in range(iters):
        U = np.clip(mu + sd * rng.standard_normal((pop, NK, 3)), -1, 1)
        R = rollout(l, mp, U); o = np.argsort(-R); best = max(best, R[o[0]])
        ok = o[:elite][np.isfinite(R[o[:elite]])]
        if len(ok) >= 2:
            mu = U[ok].mean(0); sd = U[ok].std(0) + 0.05
    return best


def ruler_D_cem(l, mp, seeds=(0, 1, 2)):
    """旧 D（トルクの折れ線を CEM）。⚠️ 目標でぴたりと止められず、残りの距離が 38 s 分たまって K の 3〜7 倍悪くなった（9-366）。使わない。"""
    return max(cem1(l, mp, seed=s) for s in seeds)


def ruler_D(l, mp, n_ik=40, t_scales=(1.0, 1.25, 1.5, 2.0, 3.0), gains=((200, 40), (800, 80), (2000, 150)), T_SIM=4.0):
    T_SIM = min(T_SIM, NSTEP * DT_STEP)
    """D（正解の代わり）: 重力の補償つきの PD 制御で、初めの姿勢から逆運動学の解へ関節空間でまっすぐ（時間 T）追わせる。
    解（最大 n_ik）・T（速度の上限だけの時間の倍率）・ゲインの組を総当たりし、トルクの上限（|u| <= 1）・速度の上限・重力・
    指令の費用を入れて 1000 step の報酬を出す。T_SIM 秒より後は最後の姿勢で止まり続けるとする。家族の中の最良なので真の最適の下界。"""
    S = ik_all(l)
    if len(S) == 0:
        return -np.inf
    S = S[np.linspace(0, len(S) - 1, min(n_ik, len(S))).astype(int)]
    cand = [(q, ts, kp, kd) for q in S for ts in t_scales for kp, kd in gains]
    Q = np.array([c[0] for c in cand]); TS = np.array([c[1] * np.abs(c[0]).max() / W for c in cand])
    KP = np.array([c[2] for c in cand], float)[:, None]; KD = np.array([c[3] for c in cand], float)[:, None]
    N = len(cand); q = np.zeros((N, 3)); dq = np.zeros((N, 3)); R = np.zeros(N)
    nsub = int(round(DT_STEP / DT_SIM)); nstep_sim = int(T_SIM / DT_STEP)
    for s_ in range(nstep_sim):
        usum = np.zeros((N, 3))
        for k in range(nsub):
            t = (s_ * nsub + k) * DT_SIM
            f = np.clip(t / TS, 0, 1)[:, None]; qr = f * Q; dqr = np.where(f < 1, Q / TS[:, None], 0.0)
            _, Gv, _ = dyn_terms(l, q, mp)
            tau = np.clip(Gv + KP * (qr - q) + KD * (dqr - dq), -GEAR, GEAR)
            ddq = accel(l, q, dq, tau, mp)
            dq = np.clip(dq + ddq * DT_SIM, -W, W); q = q + dq * DT_SIM
            hit = np.abs(q) > LIM; q = np.clip(q, -LIM, LIM); dq[hit] = 0.0
            usum += tau / GEAR
        u = usum / nsub
        R += -np.linalg.norm(pts(l, q)[:, 3] - TGT, axis=1) - CC * np.mean(u ** 2, axis=1)
    c, ok, d = hold_cost(l, q, mp)
    R += (NSTEP - nstep_sim) * (-d - c)
    R[~ok] = -np.inf
    return R.max()


def order(vals):
    return ' > '.join(sorted(vals, key=lambda k: -vals[k]))


if __name__ == '__main__' and '--random' not in sys.argv:
    payloads = [0.0] if '--check' in sys.argv else [0.0, 5.0, 10.0, 15.0, 20.0]
    for mp in payloads:
        K, S, D = {}, {}, {}
        for f, l in FORMS.items():
            K[f], S[f] = ruler_K_S(np.array(l), mp); D[f] = ruler_D(np.array(l), mp)
            print(f'荷物 {mp:4.1f} kg  {f:5s} 長さ {np.round(l, 3)}: K {K[f]:8.2f}  S {S[f]:8.2f}  D {D[f]:8.2f}', flush=True)
        print(f'  順位 K: {order(K)}\n  順位 S: {order(S)}\n  順位 D: {order(D)}\n', flush=True)


def random_search(n=24, payloads=(0.0, 5.0, 7.5, 10.0, 12.5, 15.0), seed=0, margin=0.10):
    """--random: リンク長を振った n 形（総長 0.85〜1.4 m）で、荷物ごとに K・S が D と逆を向く対（両方 margin 以上の差）を数える。
    D は PD の家族の総当たりなので種によるばらつきは無い（9-365 の CEM と違う）。"""
    rng = np.random.default_rng(seed); forms = []
    while len(forms) < n:
        l = rng.uniform(0.15, 0.6, 3)
        if 0.85 <= l.sum() <= 1.4 and len(ik_all(l, n=180)):
            forms.append(l)
    for mp in payloads:
        K = []; S = []; D = []
        for l in forms:
            k, s = ruler_K_S(l, mp); K.append(k); S.append(s); D.append(ruler_D(l, mp))
        K, S, D = map(np.array, (K, S, D))
        def flips(A):
            c = t = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if not np.all(np.isfinite([A[i], A[j], D[i], D[j]])):
                        continue
                    t += 1
                    ra = (A[i] - A[j]) / max(abs(A[i]), abs(A[j])); rd = (D[i] - D[j]) / max(abs(D[i]), abs(D[j]))
                    if (ra > margin and rd < -margin) or (ra < -margin and rd > margin):
                        c += 1
            return c, t
        def spear(A):
            m = np.isfinite(A) & np.isfinite(D)
            if m.sum() < 3:
                return float('nan')
            a = np.argsort(np.argsort(A[m])); d = np.argsort(np.argsort(D[m])); k = m.sum()
            return 1 - 6 * np.sum((a - d) ** 2) / (k * (k * k - 1))
        (fk, tk), (fs, ts) = flips(K), flips(S)
        print(f'荷物 {mp:4.1f} kg: 順位相関 K-D {spear(K):.3f}  S-D {spear(S):.3f}  |  逆を向く対（両方 {int(margin*100)} % 以上）K {fk}/{tk}  S {fs}/{ts}'
              f'  |  止まれない形 {int((~np.isfinite(S)).sum())}/{n}', flush=True)


if __name__ == '__main__' and '--random' in sys.argv:
    random_search()
