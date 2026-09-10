#!/usr/bin/env python3
"""スライド8 の ② 用に「リンクに切り分けて寸法を測った」図を作る。

**なぜ要るか**: スライド8 で ①（M1 の設計図スタイル画像）と ②（Tripo3D の 3D レンダ）を
並べたところ、**見た目がほとんど同じ**でユーザーから指摘が入った（2026-09-10）。
M2 が入力に忠実なので**設計が効いている証拠なのだが、スライドでは「何も起きていない」に見える**。

②が実際に生むのは「**バラバラのリンクと測った寸法**」なので、そちらを描く。

出力: figures/slide8_links_split.png
"""
import numpy as np, trimesh, matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

# ⚠️ 長さは XML（assets/mujoco_envs/e2e_a1.xml の fromto）の実測値。台帳と揃える
ITEMS = [('base', '台座', '#8a8a8a', None),
         ('upper_arm', '上腕', '#d94141', 0.4031),
         ('forearm', '前腕', '#2f6fd0', 0.3308),
         ('hand', '先端', '#2fa14a', 0.2763)]
SRC = 'data/test/A1/meshes'


def main():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    x = 0.0
    for f, ja, c, L in ITEMS:
        v = np.asarray(trimesh.load(f'{SRC}/{f}.stl').vertices)[:, [0, 2]].copy()
        v -= v.mean(0)
        _, _, vt = np.linalg.svd(v, full_matrices=False)
        v = v @ vt.T                                  # 最長軸を水平へ
        if np.ptp(v[:, 0]) < np.ptp(v[:, 1]):
            v = v[:, ::-1]
        poly = v[ConvexHull(v).vertices]
        poly[:, 0] += x - poly[:, 0].min()
        ax.add_patch(Polygon(poly, closed=True, fc=c, ec='#333', lw=1.2))
        w = np.ptp(poly[:, 0])
        ax.text(x + w / 2, poly[:, 1].min() - 0.055,
                ja if L is None else f'{ja}\n{L:.3f} m',
                ha='center', va='top', fontsize=15)
        x += w + 0.10
    ax.set_xlim(-0.05, x); ax.set_ylim(-0.20, 0.13)
    ax.set_aspect('equal'); ax.axis('off')
    plt.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    plt.savefig('figures/slide8_links_split.png', dpi=170)
    print('figures/slide8_links_split.png')


if __name__ == '__main__':
    main()
