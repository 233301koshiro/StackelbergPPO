"""床を 0.5 m 角の箱に分けた .body を書く（Bug 103、系譜 9-342 の 11-10）。

⛔ Bug 103: 1 枚の Box の床（内蔵の 10 m 四方・広い 200 m 四方）では、四脚の脛と床の接触を AIST の衝突検出が
  約半分の場面で返さず、脚が床の中を通り抜けた。0.5 m 角の箱に分けた床では、沈む話が 16〜17 → 5〜6 / 20 に減った
  （checkpoint 3 つ × 20 話）。仕組みは分かっていない（継ぎ目の無い 0.5 m の網・5 m の箱・2 枚に分けた床では直らない）。
⚠️ 腕の run には使わない: 軸6 の押しでは箱の継ぎ目で軌跡が最大 38.8 mm ずれた。腕は今の床で取りこぼしが起きていない。

箱は上面 z = 0・厚さ 0.2 m・材質 Ground（内蔵の床と同じ）。継ぎ目は x = −0.25 + 0.5k・y = 0.5k（検証に使った床と同じ並び）。
範囲の外は上面 z = −0.05 の 200 m 四方（範囲を出ると 5 cm 下がる）。

    python3 scripts/make_floor_tiles.py                     # 四脚用（x −1.25〜25.25・y −2〜2 m）
    python3 scripts/make_floor_tiles.py --x 0 10 --y -1 1 --out <path>
"""
import argparse, os

ap = argparse.ArgumentParser()
ap.add_argument('--x', nargs=2, type=float, default=[-1.25, 25.25], help='敷く範囲の x（箱の端）')
ap.add_argument('--y', nargs=2, type=float, default=[-2.0, 2.0], help='敷く範囲の y（箱の端）')
ap.add_argument('--size', type=float, default=0.5)
ap.add_argument('--out', default='assets/choreonoid/bodies/floor_tiles/floor_tiles_quad.body')
a = ap.parse_args()

h = a.size
nx = round((a.x[1] - a.x[0]) / h); ny = round((a.y[1] - a.y[0]) / h)
L = ['format: ChoreonoidBody', 'formatVersion: 1.0',
     f'# scripts/make_floor_tiles.py が書いた（Bug 103）。{h} m 角の箱 {nx * ny} 個、x {a.x[0]}〜{a.x[1]}・y {a.y[0]}〜{a.y[1]} m。外は上面 −0.05 m の 200 m 四方',
     'name: Floor', 'links:', '  -', '    name: Floor', '    translation: [ 0, 0, 0 ]', '    jointType: fixed',
     '    material: Ground', '    AMOR: true', '    elements:']


def box(x, y, z, sx, sy, sz):
    return ['      - type: Transform', f'        translation: [ {x:.4f}, {y:.4f}, {z} ]', '        elements:',
            '          - type: Shape', f'            geometry: {{ type: Box, size: [ {sx}, {sy}, {sz} ] }}',
            '            appearance: { material: { diffuseColor: [ 0, 0, 1 ] }}']


for i in range(nx):
    for j in range(ny):
        L += box(a.x[0] + h * (i + 0.5), a.y[0] + h * (j + 0.5), -0.1, h, h, 0.2)
L += box(0, 0, -0.15, 200.0, 200.0, 0.2)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
open(a.out, 'w').write('\n'.join(L) + '\n')
print(f'{a.out}: {nx * ny} 個')
