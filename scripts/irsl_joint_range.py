"""研究室の部品（IRSL の RobotAssembler 部品ライブラリ）で組んだ A1 の 1/5 のアームについて、各関節を回したときに
部品どうしが当たらない角度の範囲（可動域）を、部品の当たり判定用のメッシュから計算する（軸6、系譜 9-277 の続き）。

⭐ なぜ要るか: RobotAssembler が書き出す可動域は ±180°（設定ファイルの `limit`）で、部品どうしの干渉を見ていない。
   今のモデル（`e2e_a1_phys.xml`）は ±90° で、隣り合うリンクどうしの当たりは判定しない。実物で取れない角度を方策が使うと判定が変わる。
⭐ 組み方: 9-277 の CSV で、軸方向のずれ 0 の組み方から A1 の 1/5 に最も近い長さ（環境変数 L1・L2、既定 80.5・66.0 mm）を選ぶ。
   サーボ A のホーン → リンク 1 → サーボ B のホーン → リンク 2 → サーボ C。根元（台）と手先は未設計なので含めない。
⭐ 当たりの判定: 部品ごとに当たり判定のメッシュ（`meshes_col`）と箱を局所座標で塗りつぶした格子（既定 1 mm）にし、
   回す側の格子点が、止まっている側の塗りつぶしを表面から DEPTH（既定 1 mm）削った内側に入った数（体積）を数える。
   組んだ姿勢（回す前）の体積を差し引き、増えた体積が TH（既定 5 mm³）を超えた最初の角度を、その向きの限界とする。
   ⛔ 削らずに数えると、触れて滑っているだけの面が格子の丸めで 0〜8 mm³ を行き来し、当たりと誤る（10-29 の 1 回目: 関節 2 が幅 17° と出た）。
   関節 1 は台が未設計なのでサーボ A の本体だけ、関節 3 は手先が未設計なので Hinge だけを回す（4 種類）。
⭐ 角度の向きと 0: 回す軸はサーボ A のホーンの z。q = 0 は「回す側のリンクが止まっている側のリンクの延長にある」（モデルの関節角と同じ）。
   関節 2: 止まっている側 = サーボ A のホーン → サーボ B のホーン、回す側 = サーボ B のホーン → サーボ C のホーン。
   関節 1: 止まっている側 = サーボ A の本体の重心 → ホーン、回す側 = サーボ A のホーン → サーボ B のホーン。
   関節 3: 止まっている側 = リンク 2、回す側 = サーボ C のホーン → Hinge の重心。

⚠️ 出力に組んだ長さ・軸方向のずれ・回す前の重なりを必ず出す（狙った組み方になっているかの確かめ、CLAUDE.md ⑤-3-2）。
⚠️ 回す前の重なり（同じ側の部品どうし）は、リンクの中で部品がめり込んでいる疑い（組めない組み合わせ）として別に出す。

    L1=80.5 L2=66.0 OUT=data/axis6/irsl_joint_range.csv \\
      PYTHONPATH=/choreonoid_ws/install/lib/choreonoid-2.3/python python3 scripts/irsl_joint_range.py
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
OUT = os.environ.get('OUT', 'data/axis6/irsl_joint_range.csv')
L1 = float(os.environ.get('L1', '80.5'))
L2 = float(os.environ.get('L2', '66.0'))
PITCH = float(os.environ.get('PITCH', '1.0'))      # mm
TH = float(os.environ.get('TH', '5.0'))            # mm³
DEPTH = float(os.environ.get('DEPTH', '1.0'))      # mm。止まっている側をこの深さだけ削ってから数える（触れているだけの面を当たりにしない）
STEP = float(os.environ.get('STEP', '1.0'))        # 度
NPZ = os.environ.get('NPZ', '')                    # 指定すると、限界の角度での部品の点を保存する（目で見る確かめ用）
snap = {}


def rotmat(r):
    if r is None:
        return np.eye(3)
    ax = np.array(r[:3], float)
    ax /= np.linalg.norm(ax)
    ang = np.radians(r[3])
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def axis_rot(z, deg):
    return rotmat([*z, deg])


class Shape:
    """部品 1 つの当たり判定を、局所座標（mm）で塗りつぶした格子にしたもの。"""

    cache = {}

    def __init__(self, ptype, P):
        if ptype in Shape.cache:
            self.__dict__ = Shape.cache[ptype].__dict__
            return
        pts = []
        for g in P[ptype]['collision']:
            T = np.eye(4)
            T[:3, :3] = rotmat(g.get('rotation'))
            T[:3, 3] = g.get('translation', [0, 0, 0])
            if g.get('type') == 'mesh':
                m = trimesh.load(os.path.join(PROJ, g['url']), force='mesh', process=False)
                m.apply_scale(g.get('scale', 1.0) * 1000.0)        # 設定の scale は m へ。ここでは mm で扱う
            elif 'box' in g:
                m = trimesh.creation.box(extents=g['box'])
            else:
                raise ValueError(f'{ptype}: 知らない当たり判定 {g}')
            m.apply_transform(T)
            pts.append(m.voxelized(PITCH).fill().points)
        self.pts = np.unique(np.round(np.vstack(pts) / PITCH).astype(int), axis=0)   # 格子の番号
        self.lo = self.pts.min(axis=0)
        self.occ = np.zeros(self.pts.max(axis=0) - self.lo + 1, bool)
        self.occ[tuple((self.pts - self.lo).T)] = True
        k = int(round(DEPTH / PITCH))
        self.core = binary_erosion(self.occ, iterations=k) if k > 0 else self.occ   # 表面から DEPTH より内側
        self.centroid = self.pts.mean(axis=0) * PITCH
        self.radius = float(np.linalg.norm(self.pts * PITCH - self.centroid, axis=1).max()) + PITCH
        Shape.cache[ptype] = self

    def count_inside(self, world_pts, T):
        """world_pts（mm）のうち、姿勢 T（4x4、mm）のこの部品の塗りつぶしの中にある数。"""
        loc = (world_pts - T[:3, 3]) @ T[:3, :3]
        idx = np.round(loc / PITCH).astype(int) - self.lo
        ok = np.all((idx >= 0) & (idx < self.occ.shape), axis=1)
        return int(self.core[tuple(idx[ok].T)].sum())


def main():
    import cnoid.RobotAssembler as cra
    cfg = yaml.safe_load(open(PROJ + '/irsl_assembler_config.yaml'))
    P = {p['type']: p for p in cfg['PartsSettings']}
    ru = cra.RoboasmUtil(PROJ + '/irsl_assembler_config.yaml')
    act = P['XL-430']['actuators'][0]
    horn_t = np.array(act['translation'], float)                 # mm
    horn_z = rotmat(act.get('rotation'))[:, 2]
    rows = {round(float(r['軸間距離_mm']), 1): r for r in csv.DictReader(open(CSV_IN))}

    def pose(parts):
        wc = parts.worldcoords
        wc = wc() if callable(wc) else wc
        cp = wc.cnoidPosition
        T = np.array(cp() if callable(cp) else cp, float)
        T[:3, 3] *= 1000.0                                       # m → mm
        return T

    def horn(T):
        return T[:3, :3] @ horn_t + T[:3, 3], T[:3, :3] @ horn_z

    def types(t, n):
        return next(c.get('types', []) for c in P[t]['connecting-points'] if c['name'] == n)

    def face(t):
        return [c['name'] for c in P[t]['connecting-points'] if set(c.get('types', [])) & {'PC1', 'SC1'}]

    def add_link(R, parts, r, servo, tag):
        """CSV の 1 行（9-277 の組み方）どおりに、servo のホーンから次のサーボまでを組む。"""
        h, a, f, b, fc, fs, bs = r['Hinge'], r['Hinge の点'], r['Frame'], r['Frame の点'], r['つなぎ方'], r['Frame→サーボ'], r['サーボの点']
        H = ru.makeParts(h)
        assert R.attach(H, H.name + '/side_connect_point', servo.name + '/horn', 'fixed'), f'{tag}: Hinge が付かない'
        parts.append((f'{tag}.Hinge', h, H))
        last, lp = H, a
        if r['LEGO'] == '重ね':
            nb, npl = map(int, r['LEGO の詳細'].replace('block×', '').replace('plate×', '').split('+'))
            hpol = 'P' if 'PC1' in types(h, face(h)[0]) else 'S'
            hin, hout = ('SK', 'PL') if hpol == 'P' else ('PL', 'SK')
            for i, blk in enumerate(['block_1_1'] * nb + ['plate_1_1'] * npl):
                Bm = ru.makeParts(blk)
                assert R.attach(Bm, f'{Bm.name}/{hin}1_0_0', f'{last.name}/{lp}', 'fixed'), f'{tag}: LEGO が付かない'
                parts.append((f'{tag}.{blk}#{i}', blk, Bm))
                last, lp = Bm, f'{hout}1_0_0'
        elif r['LEGO'] != 'なし':
            raise NotImplementedError(f'{tag}: LEGO「{r["LEGO"]}」の組み方は未対応')
        F = ru.makeParts(f)
        assert R.attach(F, F.name + '/' + b, last.name + '/' + lp, fc), f'{tag}: Frame が付かない'
        parts.append((f'{tag}.Frame', f, F))
        S = ru.makeParts('XL-430')
        assert R.attach(S, S.name + '/' + bs, F.name + '/' + fs, 'fixed'), f'{tag}: サーボが付かない'
        return S

    def build():
        """サーボ A → リンク 1 → サーボ B → リンク 2 → サーボ C を組む。手先の Hinge を付け替えるために毎回組み直す。"""
        R = ru.makeRobotFromKey('XL-430')
        parts = [('A', 'XL-430', R.rootParts())]
        for L, tag, nxt in [(L1, 'L1', 'B'), (L2, 'L2', 'C')]:
            if L not in rows:
                sys.exit(f'⛔ {L} mm は CSV に無い')
            S = add_link(R, parts, rows[L], parts[-1][2], tag)
            parts.append((nxt, 'XL-430', S))
        return R, parts

    R, parts = build()

    T = {n: pose(p) for n, _, p in parts}
    shp = {n: Shape(t, P) for n, t, _ in parts}
    (pA, zA), (pB, zB), (pC, zC) = horn(T['A']), horn(T['B']), horn(T['C'])

    def link_check(p0, z0, p1, z1, want, tag):
        d = p1 - p0
        ax = float(d @ z0)
        dist = float(np.linalg.norm(d - ax * z0))
        par = abs(float(z0 @ z1))
        print(f'[組んだリンク] {tag}: 軸間 {dist:.2f} mm（CSV {want}）・軸方向のずれ {ax:.2f} mm・平行 {par:.4f}', flush=True)
        if abs(dist - want) > 0.06 or par < 0.9999:
            sys.exit(f'⛔ {tag} が CSV の組み方と合わない。止める')

    link_check(pA, zA, pB, zB, L1, 'リンク 1')
    link_check(pB, zB, pC, zC, L2, 'リンク 2')
    print(f'[部品] {len(parts)} 個: ' + '・'.join(n for n, _, _ in parts), flush=True)
    z = zA

    def world_pts(n, Tn):
        return (shp[n].pts * PITCH) @ Tn[:3, :3].T + Tn[:3, 3]

    def overlap(n1, T1, n2, T2):
        c1 = T1[:3, :3] @ shp[n1].centroid + T1[:3, 3]
        c2 = T2[:3, :3] @ shp[n2].centroid + T2[:3, 3]
        if np.linalg.norm(c1 - c2) > shp[n1].radius + shp[n2].radius:
            return 0.0
        return shp[n2].count_inside(world_pts(n1, T1), T2) * PITCH ** 3

    # ⭐ 回す前の重なり（全部の組）。同じリンクの中で大きいものは「組めない」の疑い
    names = [n for n, _, _ in parts]
    base = {}
    print('[回す前の重なり] 1 mm³ 以上の組（格子の目 {PITCH} mm）:'.replace('{PITCH}', str(PITCH)), flush=True)
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            v = overlap(n1, T[n1], n2, T[n2])
            base[(n1, n2)] = v
            if v >= 1:
                print(f'    {n1} × {n2}: {v:.0f} mm³', flush=True)

    def proj(v):
        v = v - (v @ z) * z
        return v / np.linalg.norm(v)

    def signed(a, b):
        return float(np.degrees(np.arctan2(np.cross(a, b) @ z, a @ b)))

    def sweep(label, fixed, moving, pivot, ref_dir, mov_dir, extra=None):
        """moving の部品を pivot のまわりに z で回し、fixed のどれかとの重なりが回す前より TH 増えたら止める。"""
        fixed = fixed + (extra or [])
        q0 = signed(ref_dir, mov_dir)
        pairs = [(m, f) for m in moving for f in fixed]
        b0 = {pr: overlap(pr[0], T[pr[0]], pr[1], T[pr[1]]) for pr in pairs}
        lim, who, curve = {}, {}, []
        for sgn in (+1, -1):
            th = 0.0
            while th < 360.0:
                th += STEP
                Rz = np.eye(4)
                Rz[:3, :3] = axis_rot(z, sgn * th)
                Rz[:3, 3] = pivot - Rz[:3, :3] @ pivot
                worst = (0.0, None)
                for m, f in pairs:
                    v = overlap(m, Rz @ T[m], f, T[f]) - b0[(m, f)]
                    if v > worst[0]:
                        worst = (v, f'{m} × {f}')
                curve.append([label, sgn * th, q0 + sgn * th, worst[0], worst[1] or ''])
                if worst[0] > TH:
                    lim[sgn], who[sgn] = sgn * (th - STEP), worst[1]
                    if NPZ:
                        k = f'{label}|{sgn:+d}'
                        snap[k + '|fixed'] = np.vstack([world_pts(f, T[f]) for f in fixed])
                        snap[k + '|moving'] = np.vstack([world_pts(m, Rz @ T[m]) for m in moving])
                        snap[k + '|z'] = z
                    break
            else:
                lim[sgn], who[sgn] = sgn * 360.0, '（一周しても当たらない）'
        lo, hi = q0 + lim[-1], q0 + lim[+1]
        print(f'[{label}] 組んだ姿勢の関節角 q0 = {q0:+.1f}°・当たらない範囲 q = {lo:+.1f}° 〜 {hi:+.1f}°（幅 {hi - lo:.0f}°）'
              f'\n    + 側で当たる組: {who[+1]}\n    − 側で当たる組: {who[-1]}', flush=True)
        return [label, q0, lo, hi, who[+1], who[-1]], curve

    out, curves = [], []
    L1n = [n for n in names if n.startswith('L1.')]
    L2n = [n for n in names if n.startswith('L2.')]
    # 関節 2: サーボ B のホーン
    r, c = sweep('関節 2（リンク 1 対 リンク 2）', ['A'] + L1n + ['B'], L2n + ['C'], pB, proj(pB - pA), proj(pC - pB))
    out.append(r); curves += c
    # 関節 1: サーボ A のホーン（台は未設計なのでサーボ A の本体だけ）
    r, c = sweep('関節 1（サーボ A の本体だけ。台は未設計）', ['A'], L1n + ['B'] + L2n + ['C'], pA,
                 proj(pA - (T['A'][:3, :3] @ shp['A'].centroid + T['A'][:3, 3])), proj(pB - pA))
    out.append(r); curves += c
    # 関節 3: サーボ C のホーン（手先は未設計なので Hinge だけ、4 種類）
    for h in ['Hinge_Short_Plug', 'Hinge_Long_Plug', 'Hinge_Short_Socket', 'Hinge_Long_Socket']:
        Rt, pt = build()
        Hp = ru.makeParts(h)
        if not Rt.attach(Hp, Hp.name + '/side_connect_point', pt[-1][2].name + '/horn', 'fixed'):
            print(f'[関節 3] {h}: 付かない', flush=True)
            continue
        n = f'tip.{h}'
        T[n], shp[n] = pose(Hp), Shape(h, P)
        hc = T[n][:3, :3] @ shp[n].centroid + T[n][:3, 3]
        r, c = sweep(f'関節 3（手先の {h} だけ）', ['A'] + L1n + ['B'] + L2n + ['C'], [n], pC, proj(pC - pB), proj(hc - pC))
        out.append(r); curves += c

    os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
    with open(OUT, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['関節', '組んだ姿勢の q0_deg', '下限_deg', '上限_deg', '+側で当たる組', '−側で当たる組'])
        w.writerows([[a, round(b, 1), round(c_, 1), round(d, 1), e, f] for a, b, c_, d, e, f in out])
    with open(OUT.replace('.csv', '_curve.csv'), 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['関節', '回した角度_deg', 'q_deg', '増えた重なり_mm3', '最も重なった組'])
        w.writerows(curves)
    if NPZ:
        np.savez_compressed(NPZ, **snap)
    print(f'[done] → {OUT}（L1={L1}・L2={L2}・格子 {PITCH} mm・削る深さ {DEPTH} mm・閾値 {TH} mm³・刻み {STEP}°）', flush=True)


if __name__ == '__main__':
    main()
