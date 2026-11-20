#!/usr/bin/env python3
"""MuJoCo の XML（胴から直列の鎖が出る形）から、`build_ra_tree.py` の入力（関節の表）を作る（軸6、2026-11-20）。

なぜ: `build_ra_tree.py` は「関節ごとの位置・軸の向き・親（どの鎖の何番目か）」の表から部品で組む。
表は手で書いてもよい（チャットで決めた軸の向きをそのまま書く）が、四脚のように既に XML がある形はここで作る。
Choreonoid は要らない。

読み方（`quad_mesh_to_xml.py`・`quad_ra_approx.py` の XML の形）:
  - 胴 = free joint を持つ体。胴の原点を胴の中心（部品の胴の板の中心）とする。単位は m → mm
  - 胴の子の体ごとに 1 本の鎖。鎖を子へ辿り、hinge の関節を持つ体を関節とする（関節の無い取り付けの体は飛ばす）
  - 関節の軸 = joint の axis（体は回転しない前提。quat・euler・axisangle があれば止める）
  - 先端 = 最後の体の最初の geom（カプセル）の終点 ＋ 半径（カプセルの外形の先）
  - 胴の板の狙う位置（--target）:
      hip   … 最初の関節の xy（build_quad_ra.py の QUAD=tripo の HIPS と同じ）
      plane … x は最初の関節、ホーンの向きの成分は「カプセルの終点（半径を足さない）＋ 19 mm」（同じく QUAD=walk）

    python3 scripts/ra_spec_from_xml.py --xml assets/mujoco_envs/quad_walk_ra235.xml --rail plate_4_8 --target plane \\
        --link-mm 68.7 --tip-mm 68.7 --link-names thigh shin --out data/axis6/tree/quad_walk_ra235_spec.json
"""
import argparse
import json
import xml.etree.ElementTree as ET

import numpy as np

HINGE_IN = 19.0


def vec(s):
    return np.array([float(x) for x in s.split()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xml', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--name', default=None)
    ap.add_argument('--plate', default='plate_16_16')
    ap.add_argument('--rail', default='')
    ap.add_argument('--target', choices=['hip', 'plane', 'none'], default='plane',
                    help='none なら書かない（build_ra_tree.py が先端の点から決める）')
    ap.add_argument('--link-mm', type=float, nargs='*', default=None,
                    help='関節の間の長さをすべての鎖でこの値にする（左右・前後で同じにするとき。鎖の関節の数 −1 個）')
    ap.add_argument('--tip-mm', type=float, default=None, help='先端までの長さをすべての鎖でこの値にする')
    ap.add_argument('--link-names', nargs='*', default=None, help='部分の名前（例 thigh shin。make_quad_ra_xml.py が読む）')
    ap.add_argument('--range-deg', type=float, default=30.0)
    ap.add_argument('--tip-hinge', default='Hinge_Short_Plug')
    a = ap.parse_args()

    root = ET.parse(a.xml).getroot()
    for b in root.iter('body'):
        for k in ('quat', 'euler', 'axisangle', 'xyaxes', 'zaxis'):
            if b.get(k):
                raise SystemExit(f'体 {b.get("name")} が回っている（{k}）。回っていない体だけを読む')
    torso = next(b for b in root.iter('body') if any(j.get('type') == 'free' for j in b.findall('joint')))
    chains = []
    for ci, child in enumerate(torso.findall('body'), start=1):
        joints, p, b, last = [], vec(child.get('pos', '0 0 0')) * 1000, child, None
        while b is not None:
            for j in b.findall('joint'):
                if j.get('type', 'hinge') == 'hinge':
                    joints.append(dict(pos=p.round(3).tolist(), axis=vec(j.get('axis', '0 0 1')).tolist(), body=b.get('name')))
            last = (b, p.copy())
            kids = b.findall('body')
            if len(kids) > 1:
                raise SystemExit(f'体 {b.get("name")} で枝分かれしている。胴から出る直列の鎖だけを扱う')
            b = kids[0] if kids else None
            if b is not None:
                p = p + vec(b.get('pos', '0 0 0')) * 1000
        lb, lp = last
        g = lb.find('geom')
        ft = vec(g.get('fromto')) * 1000
        end = lp + ft[3:]
        u = (ft[3:] - ft[:3]) / np.linalg.norm(ft[3:] - ft[:3])
        tip = end + u * float(g.get('size').split()[0]) * 1000
        c = dict(name=f'鎖{ci}（体 {child.get("name")}）', joints=joints, tip=tip.round(3).tolist())
        if a.target != 'none':
            j0 = np.array(joints[0]['pos'])
            if a.target == 'hip':
                c['mount_target_xy'] = j0[:2].round(2).tolist()
            else:
                ax = np.array(joints[0]['axis']) / np.linalg.norm(joints[0]['axis'])
                h = ax if j0 @ ax >= 0 else -ax          # ホーンは胴の外を向く側
                # build_quad_ra.py の QUAD=walk と同じ: 足先はカプセルの終点（半径を足さない）
                t = j0 + ((end - j0) @ h + HINGE_IN) * h
                c['mount_target_xy'] = t[:2].tolist()
        if a.link_mm is not None:
            c['lengths_mm'] = list(a.link_mm)
        if a.tip_mm is not None:
            c['tip_mm'] = a.tip_mm
        chains.append(c)
    spec = dict(name=a.name or a.out.rsplit('/', 1)[-1].removesuffix('_spec.json'), source=a.xml,
                torso=dict(plate=a.plate, rail=a.rail), range_deg=a.range_deg, tip_hinge=a.tip_hinge, chains=chains)
    if a.link_names:
        spec['link_names'] = a.link_names
    import os
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(spec, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print(f'→ {a.out}  鎖 {len(chains)} 本（関節 {[len(c["joints"]) for c in chains]}）')
    for c in chains:
        print(f'  {c["name"]}: 関節 {[j["pos"] for j in c["joints"]]}・軸 {[j["axis"] for j in c["joints"]]}・先端 {c["tip"]}'
              + (f'・狙う xy {np.round(c["mount_target_xy"], 2).tolist()}' if 'mount_target_xy' in c else ''))


if __name__ == '__main__':
    main()
