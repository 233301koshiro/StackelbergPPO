"""研究室の部品（IRSL の RobotAssembler 部品ライブラリ）で、最後の XL-430（関節 3）から手先までに作れる長さを数え上げる（軸6、系譜 9-319）。

⭐ 手先 = 最後の XL-430 のホーン → Hinge（Short/Long の Plug/Socket）→（LEGO）。次のサーボが無いので、9-277 の表（サーボ間）の対象外だった。
⭐ 手先の長さ = 関節 3 の軸（ホーンの回転軸）から、手先の部品（Hinge と LEGO）の当たり判定のメッシュの頂点までの、軸に垂直な距離の最大。
   モデルの手先のリンク長（関節から先端まで）と同じ定義で、箱や目標に最初に届く点までの距離。
⭐ LEGO の挟み方は 9-277 と同じ 3 通り: なし／長い棒（block_1_16・plate_1_12）を Hinge の面の角のポッチにつなぐ／ブロック m 段＋プレート n 段の重ね。
⭐ 9-315 の訂正: 関節 3 を両側に広く回せるのは手先が長い Hinge のとき（−143°〜+136°）。短い Hinge は参考として並べる。

⚠️ RobotAssembler は部品どうしのめり込みを調べない。⚠️ 強度は見ていない（1×1 のブロックを高く重ねた柱は弱い）。
⚠️ 質量は部品の設定ファイルの mass-param の和（ねじ・ホーンの金具は入らない）。

    OUT=data/axis6/irsl_tip_lengths.csv \\
      /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/irsl_tip_length.py > /tmp/tip.log 2>&1
"""
import csv
import os
import re

import numpy as np
import trimesh
import yaml

PROJ = '/choreonoid_ws/install/share/choreonoid-2.3/robot_assembler/irsl'
OUT = os.environ.get('OUT', 'data/axis6/irsl_tip_lengths.csv')
CF = ['fixed', 'invert', 'rotate90', 'rotate-90']


def rotmat(r):
    if r is None:
        return np.eye(3)
    ax = np.array(r[:3], float)
    ax /= np.linalg.norm(ax)
    ang = np.radians(r[3])
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def main():
    import cnoid.RobotAssembler as cra
    cfg = yaml.safe_load(open(PROJ + '/irsl_assembler_config.yaml'))
    P = {p['type']: p for p in cfg['PartsSettings']}
    ru = cra.RoboasmUtil(PROJ + '/irsl_assembler_config.yaml')
    act = P['XL-430']['actuators'][0]
    horn_t = np.array(act['translation'], float)                 # mm
    horn_z = rotmat(act.get('rotation'))[:, 2]

    verts = {}

    def local_verts(t):
        """部品 t の当たり判定の頂点（部品の局所座標、mm）。"""
        if t not in verts:
            vs = []
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
                vs.append(np.asarray(m.vertices))
            verts[t] = np.vstack(vs)
        return verts[t]

    def pose(parts):
        wc = parts.worldcoords
        wc = wc() if callable(wc) else wc
        cp = wc.cnoidPosition
        T = np.array(cp() if callable(cp) else cp, float)
        T[:3, 3] *= 1000.0                                       # m → mm
        return T

    def types(t, n):
        return next(c.get('types', []) for c in P[t]['connecting-points'] if c['name'] == n)

    def face(t):
        return [c['name'] for c in P[t]['connecting-points'] if set(c.get('types', [])) & {'PC1', 'SC1'}]

    def corners(names):
        ij = [(n, tuple(map(int, re.findall(r'(\d+)_(\d+)$', n)[0]))) for n in names]
        I = [x[1][0] for x in ij]
        J = [x[1][1] for x in ij]
        return [n for n, (i, j) in ij if i in (min(I), max(I)) and j in (min(J), max(J))]

    def measure(A, tip_parts):
        """関節 3 の軸から手先の部品の頂点までの、軸に垂直な距離の最大と、その点の軸方向の位置（ホーンの面から、mm）。質量の和（g）。"""
        TA = pose(A.rootParts())
        p = TA[:3, :3] @ horn_t + TA[:3, 3]
        z = TA[:3, :3] @ horn_z
        best, ax_at, mass = -1.0, 0.0, 0.0
        for t, parts in tip_parts:
            T = pose(parts)
            w = local_verts(t) @ T[:3, :3].T + T[:3, 3] - p
            a = w @ z
            r = np.linalg.norm(w - np.outer(a, z), axis=1)
            i = int(np.argmax(r))
            if r[i] > best:
                best, ax_at = float(r[i]), float(a[i])
            mass += float(P[t]['mass-param']['mass'])
        return best, ax_at, mass

    def start(h, hc):
        A = ru.makeRobotFromKey('XL-430')
        H = ru.makeParts(h)
        if not A.attach(H, H.name + '/side_connect_point', A.rootParts().name + '/horn', hc):
            return None, None
        return A, H

    rows = []
    for h in ['Hinge_Long_Plug', 'Hinge_Long_Socket', 'Hinge_Short_Plug', 'Hinge_Short_Socket']:
        hpol = 'P' if 'PC1' in types(h, face(h)[0]) else 'S'
        hin, hout = ('SK', 'PL') if hpol == 'P' else ('PL', 'SK')
        for hc in CF:
            A, H = start(h, hc)
            if A is None:
                continue
            L, ax, m = measure(A, [(h, H)])
            rows.append(['なし', h, hc, '', '', '', L, ax, m])
            for a in corners(face(h)):
                for beam in ['block_1_16', 'plate_1_12']:
                    for bc in CF:
                        A, H = start(h, hc)
                        Bm = ru.makeParts(beam)
                        if not A.attach(Bm, f'{Bm.name}/{hin}1_0_0', f'{H.name}/{a}', bc):
                            continue
                        L, ax, m = measure(A, [(h, H), (beam, Bm)])
                        rows.append([f'棒 {beam}', h, hc, a, bc, '', L, ax, m])
                for nb in range(0, 13):
                    for npl in range(0, 3):
                        if nb + npl == 0:
                            continue
                        A, H = start(h, hc)
                        last, lp, ok, tp = H, a, True, [(h, H)]
                        for blk in ['block_1_1'] * nb + ['plate_1_1'] * npl:
                            Bm = ru.makeParts(blk)
                            if not A.attach(Bm, f'{Bm.name}/{hin}1_0_0', f'{last.name}/{lp}', 'fixed'):
                                ok = False
                                break
                            tp.append((blk, Bm))
                            last, lp = Bm, f'{hout}1_0_0'
                        if not ok:
                            continue
                        L, ax, m = measure(A, tp)
                        rows.append(['重ね', h, hc, a, '', f'block×{nb}+plate×{npl}', L, ax, m])
        print(f'[tip] {h} まで {len(rows)} 通り', flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['LEGO', 'Hinge', 'Hinge のつなぎ方', 'Hinge の点', '棒のつなぎ方', 'LEGO の詳細',
                    '手先の長さ_mm', 'その点の軸方向_mm', '質量_g'])
        for r in rows:
            w.writerow(r[:6] + [f'{r[6]:.2f}', f'{r[7]:.2f}', f'{r[8]:.2f}'])
    for h in ['Hinge_Long_Plug', 'Hinge_Long_Socket', 'Hinge_Short_Plug', 'Hinge_Short_Socket']:
        Ls = sorted({round(r[6], 1) for r in rows if r[1] == h})
        none = sorted({round(r[6], 1) for r in rows if r[1] == h and r[0] == 'なし'})
        if Ls:
            print(f'[tip] {h}: {len(Ls)} 種類 {Ls[0]:.1f}〜{Ls[-1]:.1f} mm（LEGO なし {none}）', flush=True)
    print(f'[tip] 書いた: {OUT}（{len(rows)} 行）', flush=True)


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
os._exit(0)
