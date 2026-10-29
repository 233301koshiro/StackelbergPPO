"""お辞儀型の A1v を研究室の部品で組んだ腕（`build_arm3.py` の出力の JSON）について、各関節を回したときに
部品どうしが当たらない角度の範囲（可動域）を、部品の当たり判定から計算する（軸6、系譜 9-324。9-315 は平面の A1 で、A1v では出し直し）。

⭐ 当たりの判定は 9-315（`irsl_joint_range.py`）と同じ: 部品ごとに当たり判定（`collision` のメッシュと箱）を 1 mm の格子で塗りつぶし、
   回す側の格子点が、止まっている側の塗りつぶしを表面から DEPTH（1 mm）削った内側に入った体積を数える。
   0° の姿勢での体積を差し引き、増えた体積が TH（20 mm³）を超えた最初の角度の 1 つ手前を、その向きの限界とする。
⭐ 0° の姿勢 = A1v のモデルと同じく腕をまっすぐ上に伸ばした姿勢（関節 2〜4 のリンクがすべて鉛直）。関節は 1 つずつ回し、他は 0° のまま。
⭐ 向き: 関節 2〜4 は、関節 2 の軸 a で右ねじに回す向きを正（＋）とする。正に回すと手先は f = a × z（鉛直）の側へ倒れる。
   関節 1（旋回）は鉛直の軸。回す側 = その関節のホーンより先のすべての部品、止まっている側 = それより根元のすべての部品。
⚠️ 床・台・箱との当たりは見ない（タスクの環境の側。シミュレーションが扱う）。

    IN=data/axis6/a1v_1x5_parts.json OUT=data/axis6/a1v_joint_range.csv python3 scripts/joint_range_a1v.py
"""
import csv
import json
import os

import numpy as np
import trimesh
import yaml
from scipy.ndimage import binary_erosion

PROJ = '/choreonoid_ws/install/share/choreonoid-2.3/robot_assembler/irsl'
IN = os.environ.get('IN', '/userdir/StackelbergPPO/data/axis6/a1v_1x5_parts.json')
OUT = os.environ.get('OUT', '/userdir/StackelbergPPO/data/axis6/a1v_joint_range.csv')
LABEL = os.environ.get('LABEL', os.path.basename(IN))
PITCH = float(os.environ.get('PITCH', '1.0'))
DEPTH = float(os.environ.get('DEPTH', '1.0'))
TH = float(os.environ.get('TH', '20.0'))
STEP = float(os.environ.get('STEP', '1.0'))


def rotmat(r):
    if r is None:
        return np.eye(3)
    ax = np.array(r[:3], float)
    if np.linalg.norm(ax) == 0:
        return np.eye(3)
    ax /= np.linalg.norm(ax)
    ang = np.radians(r[3])
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def rot_about(o, a, deg):
    M = np.eye(4)
    M[:3, :3] = rotmat([*a, deg])
    M[:3, 3] = o - M[:3, :3] @ o
    return M


class Shape:
    cache = {}

    def __new__(cls, t, P):
        if t in cls.cache:
            return cls.cache[t]
        self = super().__new__(cls)
        pts = []
        for g in P[t]['collision']:
            T = np.eye(4)
            T[:3, :3] = rotmat(g.get('rotation'))
            T[:3, 3] = g.get('translation', [0, 0, 0])
            if g.get('type') == 'mesh':
                m = trimesh.load(os.path.join(PROJ, g['url']), force='mesh', process=False)
                m.apply_scale(g.get('scale', 1.0) * 1000.0)
            elif 'box' in g:
                m = trimesh.creation.box(extents=g['box'])
            else:
                raise ValueError(f'{t}: 知らない当たり判定 {g}')
            m.apply_transform(T)
            pts.append(m.voxelized(PITCH).fill().points)
        g = np.unique(np.round(np.vstack(pts) / PITCH).astype(int), axis=0)
        self.lo = g.min(0)
        self.occ = np.zeros(g.max(0) - self.lo + 1, bool)
        self.occ[tuple((g - self.lo).T)] = True
        k = int(round(DEPTH / PITCH))
        self.core = binary_erosion(self.occ, iterations=k) if k > 0 else self.occ
        self.pts = g * PITCH
        self.c = self.pts.mean(0)
        self.r = float(np.linalg.norm(self.pts - self.c, axis=1).max()) + PITCH
        cls.cache[t] = self
        return self

    def inside(self, wpts, T):
        loc = (wpts - T[:3, 3]) @ T[:3, :3]
        idx = np.round(loc / PITCH).astype(int) - self.lo
        ok = np.all((idx >= 0) & (idx < self.occ.shape), axis=1)
        return int(self.core[tuple(idx[ok].T)].sum())


def main():
    cfg = yaml.safe_load(open(PROJ + '/irsl_assembler_config.yaml'))
    P = {p['type']: p for p in cfg['PartsSettings']}
    d = json.load(open(IN))
    parts = [(p['name'], p['type'], np.array(p['T'])) for p in d['parts']]
    S = [Shape(t, P) for _, t, _ in parts]
    sv = [i for i, (_, t, _) in enumerate(parts) if t == 'XL-430']
    T = [M.copy() for _, _, M in parts]
    up = np.array([0, 0, 1.0])

    def horn(i):
        return T[sv[i]][:3, 3].copy(), T[sv[i]][:3, 2].copy()

    def move_after(k, M):
        for i in range(sv[k] + 1, len(T)):
            T[i] = M @ T[i]

    def tip():
        o, a = horn(len(sv) - 1)
        w = np.vstack([(S[i].pts @ T[i][:3, :3].T) + T[i][:3, 3] for i in range(sv[-1] + 1, len(T))])
        v = w - o
        perp = v - np.outer(v @ a, a)
        return w[np.argmax(np.linalg.norm(perp, axis=1))]

    def signed(u, w, a):
        u = u - (u @ a) * a
        w = w - (w @ a) * a
        return np.degrees(np.arctan2(np.cross(u, w) @ a, u @ w))

    # 0° の姿勢: 関節 2〜4 の先のリンクをまっすぐ上へ
    for k in (1, 2, 3):
        o, a = horn(k)
        b = horn(k + 1)[0] if k + 1 < len(sv) else tip()
        move_after(k, rot_about(o, a, signed(b - o, up, a)))
    a_ref = horn(1)[1]
    f = np.cross(a_ref, up)
    J = [horn(i)[0] for i in range(len(sv))] + [tip()]
    lens = [float(np.linalg.norm(J[i + 1] - J[i])) for i in range(len(J) - 2)]
    v = J[-1] - J[-2]
    a4 = horn(len(sv) - 1)[1]
    lens.append(float(np.linalg.norm(v - (v @ a4) * a4)))   # 手先は軸に垂直な成分（9-319 と同じ。いちばん遠い点は軸方向に約 33 mm ずれる）
    print(f'0° の姿勢: 関節の間 {[round(x, 1) for x in lens]} mm、関節 2〜4 の軸と a の内積 '
          f'{[round(float(horn(k)[1] @ a_ref), 3) for k in (1, 2, 3)]}、f = {np.round(f, 3)}', flush=True)

    T0 = [M.copy() for M in T]

    def overlap(k, M):
        rot = range(sv[k] + 1, len(T0))
        sta = range(0, sv[k] + 1)
        tot = 0
        for m in rot:
            Tm = M @ T0[m]
            cm = Tm[:3, :3] @ S[m].c + Tm[:3, 3]
            wm = None
            for s in sta:
                cs = T0[s][:3, :3] @ S[s].c + T0[s][:3, 3]
                if np.linalg.norm(cm - cs) > S[m].r + S[s].r:
                    continue
                if wm is None:
                    wm = S[m].pts @ Tm[:3, :3].T + Tm[:3, 3]
                tot += S[s].inside(wm, T0[s])
        return tot * PITCH ** 3

    rows = []
    for k in range(len(sv)):
        o, a = T0[sv[k]][:3, 3], T0[sv[k]][:3, 2]
        a_use = up if k == 0 else a_ref
        if k == 0 and a @ up < 0:
            pass
        base = overlap(k, np.eye(4))
        lim = {}
        for sgn in (+1, -1):
            q = 0.0
            hit = None
            while q < 180.0:
                q2 = q + STEP
                v = overlap(k, rot_about(o, a_use, sgn * q2)) - base
                if v > TH:
                    hit = v
                    break
                q = q2
            lim[sgn] = (q, hit)
        print(f'関節 {k+1}: ＋側 {lim[1][0]:.0f}°（当たり {lim[1][1]}）、−側 {lim[-1][0]:.0f}°（当たり {lim[-1][1]}）、0° の重なり {base:.0f} mm³', flush=True)
        rows.append({'形': LABEL, '関節': k + 1, '＋側_度': f'{lim[1][0]:.0f}', '−側_度': f'{-lim[-1][0]:.0f}',
                     '＋側で当たった体積_mm3': '' if lim[1][1] is None else f'{lim[1][1]:.0f}',
                     '−側で当たった体積_mm3': '' if lim[-1][1] is None else f'{lim[-1][1]:.0f}',
                     '0度の重なり_mm3': f'{base:.0f}', '関節の間_mm': f'{lens[k]:.1f}'})
    new = not os.path.exists(OUT)
    with open(OUT, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print('書き出した:', OUT, flush=True)


if __name__ == '__main__':
    main()
