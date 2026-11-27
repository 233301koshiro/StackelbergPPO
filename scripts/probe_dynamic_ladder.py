#!/usr/bin/env python3
"""E2・E3 の前提を、学習の前に計算で確かめる（系譜 9-364）。学習なし・GPU なし・数秒。

難しさの段（11-27 夜、ユーザー決定）: 簡単（運動学で順位が出る）／頑張ればできる（接触の無い動力学）／
頑張ってもできない（接触・タイミング・反応）／無理難題。押しの根元重 0.5/0.25/0.137 と先端重 0.137/0.25/0.5 の対で、
課題だけを段ごとに難しくする案の、投入前の空振りを防ぐ。

  A（E2 の前提）: トルクの上限が、速度の上限より先に効くか。
     指標 t_sat = 運動学で最も速い関節速度のベクトル ω*（各関節 ±3.1416 rad/s、+x の先端の速さが最大になる符号）へ、
     静止から ω* の向きにそろえて加速したときに着くまでの時間。τ = M q̈ を |τ_i| <= gear_i で最大にとる（コリオリは無視）。
     学習した押しが箱に当たるまでの時間（根元重 11 step = 0.44 s、先端重 18 step = 0.72 s、`trace/arm_trace.npz`）と比べる。
     t_sat がそれより十分短ければ、腕は速度の上限で決まり、動力学の差は順位に出ない（E2 は「簡単」の段に戻る）。
  B（E3 の前提）: 軸の上で箱を受け止める姿勢で、安い指標が 2 形態で同点になるか。
  C: 同じモーターで減速比だけ変えると、どの gear からトルクと慣性で決まる側に入るか（9-364 の次の案 1）。
     接触点 (x, 0) ごとに、逆運動学の全解の中の最良値で
       ① +x 方向の先端の最大の速さ（運動学）② x 方向に出せる最大の力（静力学、|τ_i| <= gear_i）
       ③ x 方向の先端の実質の質量 1/(e_x J M^-1 J^T e_x)（動力学。小さいほど当たりが軽い）
     を出す。①② が同点で ③ だけが違えば、運動学・静力学では順位が付かず、動力学の中でしか差が出ない。

モデル: 平面の 3 リンク。リンクは XML の半径のカプセル（実効密度 294.2）を細い棒の慣性で近似、関節の回転子の慣性
armature（既定 1.71、`_ag` では 1.71×(gear/150)²、下限 0.05）、関節 2・3 の位置に質量 m（`torque_density` の仕組みなら gear/25）。
⚠️ Choreonoid ではなく自前の剛体の式。接触・減衰・刻みは見ない。前提を確かめるだけで、正解の代わりにはならない。

    python3 scripts/probe_dynamic_ladder.py
"""
import numpy as np
np.set_printoptions(precision=3, suppress=True)

W = 3.1416
RHO = 294.2
FORMS = {  # リンク長と XML の半径（tripo_arm_v2c_pj_{prox,dist}_phys_arm171.xml）
    'prox': ([0.5, 0.25, 0.1368], [0.045, 0.0326, 0.032]),
    'dist': ([0.1368, 0.25, 0.5], [0.045, 0.0326, 0.032]),
}


def fk(l, q):
    a = np.cumsum(q); p = [np.zeros(2)]
    for li, ai in zip(l, a):
        p.append(p[-1] + li * np.array([np.cos(ai), np.sin(ai)]))
    return np.array(p)


def ik_all(l, goal, n=1440):
    sols = []
    for phi in np.linspace(-np.pi, np.pi, n, endpoint=False):
        w = goal - l[2] * np.array([np.cos(phi), np.sin(phi)]); d = np.linalg.norm(w)
        c = (d * d - l[0] ** 2 - l[1] ** 2) / (2 * l[0] * l[1])
        if abs(c) > 1:
            continue
        for s in (1, -1):
            q2 = s * np.arccos(c); q1 = np.arctan2(w[1], w[0]) - np.arctan2(l[1] * np.sin(q2), l[0] + l[1] * np.cos(q2))
            q3 = phi - q1 - q2; q = np.array([q1, q2, (q3 + np.pi) % (2 * np.pi) - np.pi])
            if np.all(np.abs(q) <= np.pi):
                sols.append(q)
    return sols


def jac_point(p, pt, upto):
    """点 pt（関節 upto より先の物体上）の平面の速度のヤコビアン 2x3。"""
    J = np.zeros((2, 3))
    for i in range(upto):
        J[0, i] = -(pt[1] - p[i][1]); J[1, i] = pt[0] - p[i][0]
    return J


def mass_matrix(l, r, q, m_joint, arm):
    p = fk(l, q); M = np.diag(np.asarray(arm, float))
    for k in range(3):
        mk = RHO * (np.pi * r[k] ** 2 * l[k] + 4 / 3 * np.pi * r[k] ** 3)
        com = (p[k] + p[k + 1]) / 2
        Jv = jac_point(p, com, k + 1); Jw = np.zeros(3); Jw[:k + 1] = 1
        M += mk * Jv.T @ Jv + mk * l[k] ** 2 / 12 * np.outer(Jw, Jw)
    for j, m in zip((1, 2), m_joint):  # 関節 2・3 の位置の点質量
        if m:
            Jv = jac_point(p, p[j], j); M += m * Jv.T @ Jv
    return M


def t_sat(l, r, q, gear, m_joint, arm):
    p = fk(l, q); J = jac_point(p, p[-1], 3)
    w = W * np.sign(J[0]); w[w == 0] = W           # +x の先端の速さを最大にする関節速度
    tau_per = mass_matrix(l, r, q, m_joint, arm) @ w  # q̈ = s·w を出すのに要る τ / s
    s = np.min(np.asarray(gear) / np.abs(tau_per))
    return 1.0 / s, W * np.abs(J[0]).sum()


def model(kind, g, m_fixed=0.0):
    """kind: 'now'（armature 1.71 固定・関節の質量なし）/'ag'（armature ∝ gear²）/'ag_td'（＋関節の質量 gear/25）/'mfix'（固定の質量 m）"""
    gear = [g] * 3
    arm = [1.71] * 3 if kind in ('now', 'mfix') else [max(0.05, 1.71 * (g / 150) ** 2)] * 3
    if kind == 'ag_td':
        mj = (g / 25, g / 25)
    elif kind == 'mfix':
        mj = (m_fixed, m_fixed)
    else:
        mj = (0, 0)
    return gear, arm, mj


print('=== A（E2 の前提）: 速度の上限へ着くまでの時間 t_sat [s]。学習した押しは 0.44 s（根元重）・0.72 s（先端重）で箱に当たる ===')
pt = np.array([0.85, 0.0])
for label, kind, gs, ms in [
    ('今の物理（armature 1.71 固定）', 'now', [20, 60, 150], [0]),
    ('_ag（armature ∝ gear²）', 'ag', [20, 60, 150], [0]),
    ('_ag ＋ 関節の質量 gear/25（torque_density 25）', 'ag_td', [20, 60, 150], [0]),
    ('armature 1.71 固定 ＋ 関節 2・3 に固定の質量 m（gear 150）', 'mfix', [150], [1, 2, 4, 8, 16]),
]:
    print(f'-- {label}')
    for g in gs:
        for m in ms:
            gear, arm, mj = model(kind, g, m)
            row = {}
            for f, (l, r) in FORMS.items():
                # 接触点 (0.85, 0) の逆運動学の解のうち、+x の先端の速さが最大の解で測る（9-363 の指標と同じ姿勢）
                best = max(ik_all(l, pt), key=lambda q: np.abs(jac_point(fk(l, q), fk(l, q)[-1], 3)[0]).sum())
                ts, vk = t_sat(l, r, best, gear, mj, arm)
                # 初めの姿勢（+y へ伸ばす、arm_safe_init のヨー 90°）でも測る
                ts0, _ = t_sat(l, r, np.array([np.pi / 2, 0, 0]), gear, mj, arm)
                row[f] = (ts, ts0, vk)
            tag = f'gear {g:3d}' + (f' m {m:4.1f} kg' if kind == 'mfix' else (f' m {g/25:4.1f} kg' if kind == 'ag_td' else ''))
            print(f'  {tag}: 根元重 t_sat 接触姿勢 {row["prox"][0]:.3f} / 初めの姿勢 {row["prox"][1]:.3f}   '
                  f'先端重 {row["dist"][0]:.3f} / {row["dist"][1]:.3f}   比（根元重/先端重, 接触姿勢）{row["prox"][0]/row["dist"][0]:.2f}')

print('\n=== B（E3 の前提）: 軸の上 (x, 0) で受け止める。逆運動学の全解の中の最良値。gear 150 ===')
for kind, m in [('now', 0), ('mfix', 2), ('mfix', 4)]:
    gear, arm, mj = model(kind, 150, m)
    print(f'-- armature 1.71、関節 2・3 の質量 {m} kg')
    for x in [0.5, 0.6, 0.7, 0.8, 0.85]:
        res = {}
        for f, (l, r) in FORMS.items():
            v = []; F = []; me = []
            for q in ik_all(l, np.array([x, 0.0])):
                p = fk(l, q); J = jac_point(p, p[-1], 3)
                v.append(W * np.abs(J[0]).sum())
                F.append(np.min(np.asarray(gear) / np.maximum(np.abs(J[0]), 1e-9)))
                Minv = np.linalg.inv(mass_matrix(l, r, q, mj, arm))
                me.append(1.0 / (J[0] @ Minv @ J[0]))
            res[f] = (max(v), max(F), min(me), max(me))
        a, b = res['prox'], res['dist']
        print(f'  x {x:.2f}: ① 速さ 根元重 {a[0]:.3f} 先端重 {b[0]:.3f} m/s | ② 力 {a[1]:7.1f} {b[1]:7.1f} N | '
              f'③ 実質の質量 [最小, 最大] 根元重 [{a[2]:.2f}, {a[3]:.1f}] 先端重 [{b[2]:.2f}, {b[3]:.1f}] kg')

print('\n=== C: 同じモーターで減速比だけ変える（トルク＝gear、回転子 ∝ gear²、速度の上限 3.1416×150/gear。今のコードには無い） ===')
print('    初めの姿勢（+y へ伸ばす）の t_sat を、80°（1.4 rad）を速度の上限だけで回す時間 T と比べる。1 を超えるとトルクと慣性で決まる')
W0 = W
for td in [None, 25]:
    for g in [20, 40, 75, 150]:
        W = W0 * 150 / g
        arm = [max(0.05, 1.71 * (g / 150) ** 2)] * 3
        mj = (g / td, g / td) if td else (0, 0)
        T = 1.4 / W
        cells = []
        for f, (l, r) in FORMS.items():
            ts0, _ = t_sat(l, r, np.array([np.pi / 2, 0, 0]), [g] * 3, mj, arm)
            cells.append(f'{f} t_sat {ts0:.3f} s / T {T:.3f} s = {ts0 / T:.2f}')
        print(f'  関節の質量 {"なし" if not td else f"gear/{td} = {g/td:.1f} kg"}  gear {g:3d}（{W:4.1f} rad/s）: ' + ' | '.join(cells))
W = W0
