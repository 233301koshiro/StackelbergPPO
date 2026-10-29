"""A1v（お辞儀型: 根元ヨー＋ピッチ 3）を 1/5 にして、研究室の部品（IRSL）で RobotAssembler の attach で組み、
各部品の世界姿勢（mm）と見た目のメッシュを npz に書き出す。描画は render_bow.py。"""
import csv
import json
import os
import sys

import numpy as np
import yaml

PROJ = '/choreonoid_ws/install/share/choreonoid-2.3/robot_assembler/irsl'
CSV = '/userdir/StackelbergPPO/data/axis6/irsl_link_lengths.csv'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bow_parts.json')


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


def main():
    import cnoid.RobotAssembler as cra
    cfg = yaml.safe_load(open(PROJ + '/irsl_assembler_config.yaml'))
    P = {p['type']: p for p in cfg['PartsSettings']}
    ru = cra.RoboasmUtil(PROJ + '/irsl_assembler_config.yaml')
    rows = list(csv.DictReader(open(CSV)))

    def row(L):
        c = [r for r in rows if abs(float(r['軸方向のずれ_mm'])) < 1e-9 and r['LEGO'] == 'なし']
        return min(c, key=lambda r: abs(float(r['軸間距離_mm']) - L))

    R = ru.makeRobotFromKey('XL-430')
    J1 = R.rootParts()
    parts = [('J1', 'XL-430', J1)]
    servos = [J1]

    # 肩: 研究室の腕（/userdir/TA_assembler/arm_history.roboasm）の最初のつなぎ方と同じ
    FS = ru.makeParts('Frame_Short')
    cfg_sh = sys.argv[1] if len(sys.argv) > 1 else 'invert'
    assert R.attach(FS, FS.name + '/horn', J1.name + '/horn', cfg_sh)
    J2 = ru.makeParts('XL-430')
    assert R.attach(J2, J2.name + '/bottom-side1', FS.name + '/side_connect_point2', 'fixed')
    parts += [('肩フレーム', 'Frame_Short', FS), ('J2', 'XL-430', J2)]
    servos.append(J2)

    used = []
    for k, L in enumerate((403.1 * 0.2, 330.8 * 0.2)):
        r = row(L)
        used.append({'目標_mm': round(L, 1), **r})
        A = servos[-1]
        H = ru.makeParts(r['Hinge'])
        assert R.attach(H, H.name + '/side_connect_point', A.name + '/horn', 'fixed')
        F = ru.makeParts(r['Frame'])
        assert R.attach(F, F.name + '/' + r['Frame の点'], H.name + '/' + r['Hinge の点'], r['つなぎ方'])
        S = ru.makeParts('XL-430')
        assert R.attach(S, S.name + '/' + r['サーボの点'], F.name + '/' + r['Frame→サーボ'], 'fixed')
        parts += [(f'リンク{k+1}の Hinge', r['Hinge'], H), (f'リンク{k+1}の Frame', r['Frame'], F), (f'J{k+3}', 'XL-430', S)]
        servos.append(S)
    # 手先: 長い Hinge だけ（系譜 9-319、54.2 mm）
    T = ru.makeParts('Hinge_Long_Plug')
    assert R.attach(T, T.name + '/side_connect_point', servos[-1].name + '/horn', 'fixed')
    parts.append(('手先の Hinge', 'Hinge_Long_Plug', T))

    def pose(p):
        wc = p.worldcoords
        wc = wc() if callable(wc) else wc
        cp = wc.cnoidPosition
        M = np.array(cp() if callable(cp) else cp, float)
        M[:3, 3] *= 1000.0
        return M

    out = {'shoulder_config': cfg_sh, 'links': used, 'parts': []}
    for name, typ, p in parts:
        vis = []
        for g in P[typ].get('visual', []):
            if g.get('type') != 'mesh':
                continue
            Tl = np.eye(4)
            Tl[:3, :3] = rotmat(g.get('rotation'))
            Tl[:3, 3] = g.get('translation', [0, 0, 0])
            vis.append({'url': os.path.join(PROJ, g['url']), 'scale_mm': g.get('scale', 1.0) * 1000.0,
                        'T': Tl.tolist(), 'color': g.get('color')})
        out['parts'].append({'name': name, 'type': typ, 'T': pose(p).tolist(), 'visual': vis,
                             'servo': typ == 'XL-430'})
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1)
    hz = rotmat(P['XL-430']['actuators'][0].get('rotation'))[:, 2]
    for n, t, p in parts:
        if t == 'XL-430':
            M = pose(p)
            print(n, '軸', np.round(M[:3, :3] @ hz, 3), 'ホーン', np.round(M[:3, 3], 1))
    print('書き出した:', OUT)
    os._exit(0)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        os._exit(1)
