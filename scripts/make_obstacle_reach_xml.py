#!/usr/bin/env python3
"""障害物のある Reach の XML を作る（実験系譜 9-104、助教の指摘17）。

**指摘17**: 「障害物を避けて Reach する場合や、お辞儀型以外のロボットを考える場合、
関節軸の組み合わせを総当りすると 3^N で爆発するので枝刈りが要る」。
仕分けは「記録のみ」だったが、**障害物 Reach そのものは未着手**だった。

⚠️⚠️ **障害物は必ず `<body>` に入れる。** Choreonoid の変換器は
`worldbody/body` しか読まないので、**worldbody 直下の `<geom>` は実走に存在しない**（9-98）。
⭐ **1 つの body に複数 geom を入れてはいけない**（最初の 1 個しか読まれない。9-99）。

障害物は**根元と目標を結ぶ直線上**に置く。まっすぐ伸ばすと当たるので、
**腕は迂回する姿勢を取る必要がある**。これが指摘17 の言う「避けて Reach する」状況。

    python3 scripts/make_obstacle_reach_xml.py            # 既定（柱1本）
    python3 scripts/make_obstacle_reach_xml.py --name e2e_a1v_obs2 --oy 0.15
"""
import argparse
import pathlib
import xml.etree.ElementTree as ET

SRC = pathlib.Path('assets/mujoco_envs/e2e_a1v.xml')
ENVS = pathlib.Path('assets/mujoco_envs')
# 目標は reward_specs の既定 (0.8, 0, 0.15)。根元は原点。
TARGET = (0.80, 0.0, 0.15)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='e2e_a1v_obs')
    ap.add_argument('--ox', type=float, default=0.45, help='障害物の x（根元と目標の間）')
    ap.add_argument('--oy', type=float, default=0.0, help='障害物の y')
    ap.add_argument('--radius', type=float, default=0.07, help='柱の半幅（box の一辺の半分）')
    ap.add_argument('--height', type=float, default=0.45, help='柱の高さ')
    a = ap.parse_args()

    root = ET.parse(SRC).getroot()
    wb = root.find('worldbody')

    # ⭐ body として置く（9-98）。geom は body 原点に置き、位置は body が持つ。
    b = ET.SubElement(wb, 'body', {
        'name': 'obstacle', 'pos': f'{a.ox:.4f} {a.oy:.4f} {a.height / 2:.4f}'})
    # ⚠️⚠️ **`cylinder` は Choreonoid の変換器が扱えない**（capsule / sphere / box のみ。9-118）。
    #   円柱で書くと **body は作られるが elements が空になり、衝突しない**。
    #   ⭐ **box（角柱）にする。** 半径 r の円柱の外接角柱として size を r とする。
    ET.SubElement(b, 'geom', {
        'name': 'obstacle_geom', 'type': 'box',
        'pos': '0 0 0',
        'size': f'{a.radius:.4f} {a.radius:.4f} {a.height / 2:.4f}',
        'rgba': '0.85 0.45 0.10 1',
        'contype': '1', 'conaffinity': '3',      # ⭐ 腕とも当たる（9-94 と同じ理屈）
        'friction': '0.5 0.1 0.1',
        'solref': '0.02 0.25'})

    out = ENVS / f'{a.name}.xml'
    out.write_bytes(ET.tostring(root))
    print(f'{out}')
    print(f'  障害物  角柱 半幅 {a.radius} m・高さ {a.height} m  位置 ({a.ox}, {a.oy})')
    print(f'  目標    {TARGET}')
    print(f'  ⚠️ 根元(0,0) と目標({TARGET[0]},{TARGET[1]}) を結ぶ直線上に置いてある。'
          f'まっすぐ伸ばすと当たる')
    print(f'  ⭐ body として置いたので Choreonoid にも存在する（9-98 の是正）')
    return 0


if __name__ == '__main__':
    main()
