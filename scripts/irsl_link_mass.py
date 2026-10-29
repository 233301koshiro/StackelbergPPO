"""研究室の部品で作るサーボ間のリンク（9-277 の CSV の代表の組み方）ごとに、質量・重心・慣性と、リンクの中で部品がめり込んでいないかを出す
（軸6、系譜 9-316）。

⭐ なぜ要るか:
   ① β で長さを学習するとき、XML のリンクの質量を「長さ → 質量」の表から入れる（スケール表 P1）。
   ② 関節の軸のずれのうち「横のずれ」（リンクの実体が軸を結ぶ線から横に出る）で、重心がどれだけ動くかを量で出す。
   ③ RobotAssembler は部品のめり込みを調べない（9-277）。作れる長さの表から、実物では組めない組み方を除く。
⭐ リンク = A のサーボのホーン → Hinge →（LEGO）→ Frame → B のサーボ。質量と重心は Hinge・LEGO・Frame（構造）と、B のサーボ込みの 2 通り。
   座標はリンクの座標（原点 = A のホーン、x = A のホーンから B のホーンへ（軸に垂直な成分）、z = A のホーンの軸）。mm と g。
⭐ めり込み: 固くつながる部品（Hinge・LEGO・Frame・B のサーボ）のうち、直接つないでいない組の重なりを、`irsl_joint_range.py` と同じ方法
   （格子 PITCH、相手を表面から DEPTH 削った内側、TH を超えたら当たり）で数える。A のサーボとの当たりは関節角で変わるので見ない（可動域の側、9-315）。

    CSV_IN=data/axis6/irsl_link_lengths.csv OUT=data/axis6/irsl_link_mass.csv \\
      PYTHONPATH=/choreonoid_ws/install/lib/choreonoid-2.3/python python3 scripts/irsl_link_mass.py
"""
import csv
import os
import sys

import numpy as np
import trimesh
import yaml
from scipy.ndimage import binary_erosion

PROJ = '/choreonoid_ws/install/share/choreonoid-2.3/robot_assembler/irsl'
CSV_IN = os.environ.get('CSV_IN', 'data/axis6/irsl_link_lengths.csv')
OUT = os.environ.get('OUT', 'data/axis6/irsl_link_mass.csv')
PITCH = float(os.environ.get('PITCH', '0.5'))      # mm
DEPTH = float(os.environ.get('DEPTH', '1.0'))      # mm
TH = float(os.environ.get('TH', '20.0'))           # mm³（9-315 の訂正と同じ）


def rotmat(r):
    if r is None:
        return np.eye(3)
    ax = np.array(r[:3], float)
    ax /= np.linalg.norm(ax)
    ang = np.radians(r[3])
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


class Shape:
    """部品 1 つの当たり判定を、局所座標（mm）で塗りつぶした格子にしたもの（`irsl_joint_range.py` と同じ）。"""

    cache = {}

    def __new__(cls, ptype, P):
        if ptype in cls.cache:
            return cls.cache[ptype]
        self = super().__new__(cls)
        pts = []
        for g in P[ptype]['collision']:
            T = np.eye(4)
            T[:3, :3] = rotmat(g.get('rotation'))
            T[:3, 3] = g.get('translation', [0, 0, 0])
            if g.get('type') == 'mesh':
                m = trimesh.load(os.path.join(PROJ, g['url']), force='mesh', process=False)
                m.apply_scale(g.get('scale', 1.0) * 1000.0)
            elif 'box' in g:
                m = trimesh.creation.box(extents=g['box'])
            else:
                raise ValueError(f'{ptype}: 知らない当たり判定 {g}')
            m.apply_transform(T)
            pts.append(m.voxelized(PITCH).fill().points)
        self.pts = np.unique(np.round(np.vstack(pts) / PITCH).astype(int), axis=0)
        self.lo = self.pts.min(axis=0)
        occ = np.zeros(self.pts.max(axis=0) - self.lo + 1, bool)
        occ[tuple((self.pts - self.lo).T)] = True
        k = int(round(DEPTH / PITCH))
        self.core = binary_erosion(occ, iterations=k) if k > 0 else occ
        self.centroid = self.pts.mean(axis=0) * PITCH
        self.radius = float(np.linalg.norm(self.pts * PITCH - self.centroid, axis=1).max()) + PITCH
        cls.cache[ptype] = self
        return self

    def overlap(self, other, T_self, T_other):
        """self の点のうち、other の削った内側に入る体積（mm³）。"""
        c1 = T_self[:3, :3] @ self.centroid + T_self[:3, 3]
        c2 = T_other[:3, :3] @ other.centroid + T_other[:3, 3]
        if np.linalg.norm(c1 - c2) > self.radius + other.radius:
            return 0.0
        w = (self.pts * PITCH) @ T_self[:3, :3].T + T_self[:3, 3]
        loc = (w - T_other[:3, 3]) @ T_other[:3, :3]
        idx = np.round(loc / PITCH).astype(int) - other.lo
        ok = np.all((idx >= 0) & (idx < other.core.shape), axis=1)
        return float(other.core[tuple(idx[ok].T)].sum()) * PITCH ** 3


def main():
    import cnoid.RobotAssembler as cra
    cfg = yaml.safe_load(open(PROJ + '/irsl_assembler_config.yaml'))
    P = {p['type']: p for p in cfg['PartsSettings']}
    ru = cra.RoboasmUtil(PROJ + '/irsl_assembler_config.yaml')
    act = P['XL-430']['actuators'][0]
    horn_t = np.array(act['translation'], float)
    horn_z = rotmat(act.get('rotation'))[:, 2]

    def pose(parts):
        wc = parts.worldcoords
        wc = wc() if callable(wc) else wc
        cp = wc.cnoidPosition
        T = np.array(cp() if callable(cp) else cp, float)
        T[:3, 3] *= 1000.0
        return T

    def types(t, n):
        return next(c.get('types', []) for c in P[t]['connecting-points'] if c['name'] == n)

    def face(t):
        return [c['name'] for c in P[t]['connecting-points'] if set(c.get('types', [])) & {'PC1', 'SC1'}]

    def build(r):
        """CSV の 1 行どおりに組み、[(名前, 型, 部品)] と、直接つないだ組の集合を返す（`enum_irsl_link_lengths.py` と同じ組み方）。"""
        h, a, f, b, fc, fs, bs = r['Hinge'], r['Hinge の点'], r['Frame'], r['Frame の点'], r['つなぎ方'], r['Frame→サーボ'], r['サーボの点']
        R = ru.makeRobotFromKey('XL-430')
        A = R.rootParts()
        H = ru.makeParts(h)
        assert R.attach(H, H.name + '/side_connect_point', A.name + '/horn', 'fixed')
        parts = [('A', 'XL-430', A), ('Hinge', h, H)]
        links = {('A', 'Hinge')}
        hpol = 'P' if 'PC1' in types(h, face(h)[0]) else 'S'
        hin, hout = ('SK', 'PL') if hpol == 'P' else ('PL', 'SK')
        last, lname, lp = H, 'Hinge', a
        kind = r['LEGO']
        if kind.startswith('棒'):
            beam = kind.split()[1]
            Bm = ru.makeParts(beam)
            assert R.attach(Bm, f'{Bm.name}/{hin}1_0_0', f'{H.name}/{a}', r['LEGO の詳細'])
            parts.append(('LEGO0', beam, Bm)); links.add(('Hinge', 'LEGO0'))
            last, lname, lp = Bm, 'LEGO0', f'{hout}1_0_{int(r["LEGO の段・ポッチ"])}'
        elif kind == '重ね':
            nb, npl = map(int, r['LEGO の詳細'].replace('block×', '').replace('plate×', '').split('+'))
            for i, blk in enumerate(['block_1_1'] * nb + ['plate_1_1'] * npl):
                Bm = ru.makeParts(blk)
                assert R.attach(Bm, f'{Bm.name}/{hin}1_0_0', f'{last.name}/{lp}', 'fixed')
                n = f'LEGO{i}'
                parts.append((n, blk, Bm)); links.add((lname, n))
                last, lname, lp = Bm, n, f'{hout}1_0_0'
        elif kind != 'なし':
            raise NotImplementedError(kind)
        F = ru.makeParts(f)
        assert R.attach(F, F.name + '/' + b, last.name + '/' + lp, fc)
        parts.append(('Frame', f, F)); links.add((lname, 'Frame'))
        S = ru.makeParts('XL-430')
        assert R.attach(S, S.name + '/' + bs, F.name + '/' + fs, 'fixed')
        parts.append(('B', 'XL-430', S)); links.add(('Frame', 'B'))
        return parts, links

    rows = list(csv.DictReader(open(CSV_IN)))
    out = []
    for n_row, r in enumerate(rows):
        parts, links = build(r)
        T = {n: pose(p) for n, _, p in parts}
        pA, zA = T['A'][:3, :3] @ horn_t + T['A'][:3, 3], T['A'][:3, :3] @ horn_z
        pB = T['B'][:3, :3] @ horn_t + T['B'][:3, 3]
        d = pB - pA
        ax = float(d @ zA)
        ex = d - ax * zA
        L = float(np.linalg.norm(ex))
        ex /= L
        E = np.stack([ex, np.cross(zA, ex), zA])            # 世界 → リンクの座標
        want = float(r['軸間距離_mm'])
        if abs(L - want) > 0.06:
            sys.exit(f'⛔ {want} mm の行が {L:.2f} mm に組めた。CSV の組み方と合わない。止める')

        def mass_props(names):
            m_tot, c_sum, I_list = 0.0, np.zeros(3), []
            for n, t, _ in parts:
                if n not in names:
                    continue
                mp = P[t]['mass-param']
                m = float(mp['mass'])
                Rw = T[n][:3, :3]
                c = E @ (Rw @ np.array(mp['center-of-mass'], float) + T[n][:3, 3] - pA)
                I = np.array(mp['inertia-tensor'], float).reshape(3, 3)
                Rl = E @ Rw
                I_list.append((m, c, Rl @ I @ Rl.T))
                m_tot += m
                c_sum += m * c
            com = c_sum / m_tot
            I_tot = np.zeros((3, 3))
            for m, c, I in I_list:                            # 平行軸の定理で重心まわりへ
                rr = c - com
                I_tot += I + m * (np.dot(rr, rr) * np.eye(3) - np.outer(rr, rr))
            return m_tot, com, I_tot

        struct = [n for n, _, _ in parts if n not in ('A', 'B')]
        ms, cs, Is = mass_props(struct)
        mb, cb, Ib = mass_props(struct + ['B'])

        # めり込み（A を除く、直接つないでいない組）
        shp = {n: Shape(t, P) for n, t, _ in parts}
        names = [n for n, _, _ in parts if n != 'A']
        worst, wpair = 0.0, ''
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                if (n1, n2) in links or (n2, n1) in links:
                    continue
                v = max(shp[n1].overlap(shp[n2], T[n1], T[n2]), shp[n2].overlap(shp[n1], T[n2], T[n1]))
                if v > worst:
                    worst, wpair = v, f'{n1} × {n2}'
        clash = worst > TH
        out.append([want, r['組み方の数'], round(ax, 2), r['LEGO'], r['Hinge'], r['Frame'],
                    round(ms, 2), *np.round(cs, 2), round(mb, 2), *np.round(cb, 2),
                    *np.round([Is[0, 0], Is[1, 1], Is[2, 2], Is[0, 1], Is[0, 2], Is[1, 2]], 1),
                    round(worst, 1), wpair, '✗ めり込む' if clash else '○'])
        if n_row % 50 == 0 or clash or want in (80.5, 66.0):
            print(f'[{n_row + 1}/{len(rows)}] {want} mm: 構造 {ms:.1f} g・重心 x {cs[0]:.1f} y {cs[1]:+.1f} z {cs[2]:+.1f} mm'
                  f'（サーボ B 込み {mb:.1f} g）・最大の重なり {worst:.0f} mm³ {wpair} {"⛔ めり込む" if clash else ""}', flush=True)

    os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
    with open(OUT, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['軸間距離_mm', '組み方の数', '軸方向のずれ_mm', 'LEGO', 'Hinge', 'Frame',
                    '構造の質量_g', '構造の重心x_mm', '構造の重心y_mm', '構造の重心z_mm',
                    'サーボB込みの質量_g', '込みの重心x_mm', '込みの重心y_mm', '込みの重心z_mm',
                    '構造のIxx_gmm2', 'Iyy', 'Izz', 'Ixy', 'Ixz', 'Iyz',
                    '最大の重なり_mm3', '重なった組', 'めり込み'])
        w.writerows(out)
    n_bad = sum(1 for x in out if x[-1] != '○')
    print(f'[done] {len(out)} 本・めり込む {n_bad} 本 → {OUT}（格子 {PITCH} mm・削る深さ {DEPTH} mm・閾値 {TH} mm³）', flush=True)


if __name__ == '__main__':
    main()
