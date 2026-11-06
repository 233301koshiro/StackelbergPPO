"""**RobotAssembler で組んだ四脚を、部品のメッシュ付きの Choreonoid の `.body` に書き出す**（系譜 9-342 の追記、2026-11-06、ユーザー「bodyにして」）。

なぜ: `make_quad_ra_xml.py` の XML は当たり判定がカプセルで、部品の形は残らない。組み上がった姿を Choreonoid で見るには、部品のメッシュの付いた `.body` が要る。

やること: 骨格（リンク・関節・質量・当たり判定のカプセル）は学習と同じ変換 `mujoco_xml_to_body` に XML を通したものをそのまま使い、
各リンクの形を `Collision`（カプセル、学習と同じ）と `Visual`（部品のメッシュ）に分ける。部品の姿勢は `make_quad_ra_xml.py` の
`<OUT>_parts_posed.json`（関節 0° の姿勢、胴の中心から mm）から取り、リンクの原点からの相対に直す。
部品の見た目は `irsl_assembler_config.yaml` の `visual`（メッシュは mm で scale 0.001、平行移動・箱は mm）。

確かめ（⑤-3-2）: 読み戻して、リンク数・関節数・質量を XML と並べる。リンクごとに、部品のメッシュの外接箱とカプセルの外接箱を胴の座標で並べる
（ずれていれば部品の置き方か単位の誤り）。

    NAME=quad_tripo_rab /choreonoid_ws/install/bin/choreonoid --no-window \\
        --python scripts/export_quad_ra_body.py > ログ 2>&1
    # → /userdir/StackelbergPPO/assets/choreonoid/bodies/<NAME>/<NAME>.body
"""
import json
import os
import sys
import traceback

import numpy as np
import yaml

MAIN = '/userdir/StackelbergPPO'
sys.path.insert(0, MAIN)
PROJ = '/choreonoid_ws/install/share/choreonoid-2.3/robot_assembler/irsl'
NAME = os.environ.get('NAME', 'quad_tripo_rab')
XML = f'{MAIN}/assets/mujoco_envs/{NAME}.xml'
POSED = f'{MAIN}/data/axis6/quad/{NAME}_parts_posed.json'
OUT = f'{MAIN}/assets/choreonoid/bodies/{NAME}/{NAME}.body'
GRAY = [0.6, 0.6, 0.62]


def log(*a):
    print(*a, flush=True)


def axis_angle(R):
    """回転行列 → [x, y, z, 角度°]（Choreonoid の rotation の書き方）。"""
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if ang < 1e-6:
        return [0.0, 0.0, 1.0, 0.0]
    if abs(ang - 180) < 1e-3:
        ax = np.sqrt(np.clip((np.diag(R) + 1) / 2, 0, 1))
        i = int(np.argmax(ax))
        for j in range(3):
            if j != i:
                ax[j] = np.copysign(ax[j], R[i, j])
    else:
        ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    ax = ax / np.linalg.norm(ax)
    return [float(v) for v in ax] + [float(ang)]


def part_visual(ptype, P):
    """部品 1 個の見た目（部品の座標、m）。"""
    out = []
    for v in P[ptype].get('visual', []):
        node = {'type': 'Transform', 'translation': [x / 1000.0 for x in v.get('translation', [0, 0, 0])]}
        if 'rotation' in v:
            node['rotation'] = [float(x) for x in v['rotation']]
        app = {'material': {'diffuseColor': v.get('color', GRAY)}}
        if v.get('type') == 'mesh':
            sc = v.get('scale', 1.0)
            node['scale'] = [float(sc)] * 3 if not isinstance(sc, list) else sc   # ⚠️ スカラーは読み捨てられる（StdSceneReader は 3 成分だけ読む）
            shape = {'type': 'Shape', 'geometry': {'type': 'Resource', 'uri': f'{PROJ}/{v["url"]}'}, 'appearance': app}
        elif 'box' in v:
            shape = {'type': 'Shape', 'geometry': {'type': 'Box', 'size': [x / 1000.0 for x in v['box']]}, 'appearance': app}
        else:
            raise ValueError(f'{ptype} の見た目の型を扱えない: {v}')
        node['elements'] = [shape]
        out.append(node)
    return out


def main():
    from cnoid.Body import BodyLoader
    from khrylib.rl.envs.common.mujoco_env_choreonoid import mujoco_xml_to_body
    cfg = yaml.safe_load(open(PROJ + '/irsl_assembler_config.yaml'))
    P = {p['type']: p for p in cfg['PartsSettings']}
    posed = json.load(open(POSED))

    body_name, ystr = mujoco_xml_to_body(open(XML).read())[0][0]
    y = yaml.safe_load(ystr)
    links = {l['name']: l for l in y['links']}
    # リンクの原点（胴の座標、m）。変換の出力は回転を持たない（持てば止まる）
    origin = {}
    for l in y['links']:
        assert 'rotation' not in l, f'リンク {l["name"]} に回転がある（扱っていない）'
        t = np.array(l.get('translation', [0, 0, 0]), float)
        origin[l['name']] = t if 'parent' not in l else origin[l['parent']] + t
    root = y['root_link']
    origin = {k: v - origin[root] for k, v in origin.items()}   # 胴の原点を 0 に（部品の JSON と同じ）

    seg2link = lambda p: {'torso': root, 'mount': str(p['leg']), 'thigh': f'1{p["leg"]}', 'shin': f'11{p["leg"]}'}[p['seg']]
    vis = {k: [] for k in links}
    for p in posed['parts']:
        ln = seg2link(p)
        T = np.array(p['T'], float)
        node = {'type': 'Transform', 'translation': [float(x) for x in T[:3, 3] / 1000.0 - origin[ln]],
                'rotation': axis_angle(T[:3, :3]), 'elements': part_visual(p['type'], P)}
        vis[ln].append(node)
    for k, l in links.items():
        col = l.pop('elements', [])
        l['elements'] = [{'type': 'Visual', 'elements': vis[k]}, {'type': 'Collision', 'elements': col}]
    log(f'[body] 部品 {len(posed["parts"])} 個を {sum(len(v) > 0 for v in vis.values())} / {len(links)} リンクに載せた: '
        f'{ {k: len(v) for k, v in vis.items()} }')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        yaml.safe_dump(y, f, allow_unicode=True, sort_keys=False, default_flow_style=None)

    # 確かめ
    b = BodyLoader().load(OUT)
    assert b is not None, f'{OUT} を読み戻せない'
    b.calcForwardKinematics()
    mass = sum(b.link(i).m for i in range(b.numLinks))
    xm = sum(float(i.get('mass')) for i in __import__('xml.etree.ElementTree', fromlist=['x']).parse(XML).getroot().iter('inertial'))
    log(f'[body] {OUT}  リンク {b.numLinks}・関節 {b.numJoints}・質量 {mass:.4f} kg（XML の inertial の合計 {xm:.4f} kg）'
        f'  {"✅" if abs(mass - xm) < 1e-6 else "⛔"}')
    p0 = b.rootLink.p
    for i in range(b.numLinks):
        lk = b.link(i)
        boxes = []
        for which, sh in (('部品', lk.visualShape), ('カプセル', lk.collisionShape)):
            bb = sh.boundingBox()
            if bb.empty():
                boxes.append(f'{which} なし')
                continue
            lo = (lk.R @ np.array(bb.min()) + lk.p - p0) * 1000
            hi = (lk.R @ np.array(bb.max()) + lk.p - p0) * 1000
            boxes.append(f'{which} x[{lo[0]:.0f},{hi[0]:.0f}] y[{lo[1]:.0f},{hi[1]:.0f}] z[{lo[2]:.0f},{hi[2]:.0f}]')
        log(f'[body]   {lk.name:>4s}: ' + ' ／ '.join(boxes) + ' mm')


try:
    main()
except Exception:
    traceback.print_exc()
sys.stdout.flush()
os._exit(0)
