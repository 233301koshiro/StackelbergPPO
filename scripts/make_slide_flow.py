#!/usr/bin/env python3
"""**発表スライド用の「提案システムの中身」の図**（2026-11-02 の研究進捗発表、3 枚目）。

文字だけのフローにせず、各工程の実物の画像を並べる（ユーザー指摘 11-02）。
システム図.drawio は工程が細かすぎるので使わない。

    python3 scripts/make_slide_flow.py            # → docs/研究応用/執筆/slide_flow.png（3200×1800）

使う材料（すべてリポジトリの中）:
  ① 手描き          data/test/A1/sketch/A1_hand.webp
  ② M1 の絵         data/test/A1/sketch/A1_m1.jpeg
  ③④ 3D と分けた形   data/test/A1/meshes/*.stl と joints.json（frame_origins を足すと元の位置）
  ⑤ シミュレーション  demo/phys_pusher.mp4 の 0.6 s のコマ（e2e_a1v_pusher_phys_s2）
  ⑥ 判定と助言       single_run/tripo_pj_short_phys_cg と tripo_pjr199_phys の arm_trace.npz。
                     先端が目標に最も近づいた step の姿勢を真上から描く（系譜 9-279）
"""
import pathlib
import subprocess
import tempfile
import json

import numpy as np
import trimesh
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / 'docs' / '研究応用' / '執筆' / 'slide_flow.png'

# 既定のフォントは日本語を持たない。Noto CJK を明示登録する（render_arm_video.py と同じ）
for _ttc in sorted(pathlib.Path('/usr/share/fonts').rglob('*CJK*')):
    try:
        matplotlib.font_manager.fontManager.addfont(str(_ttc))
    except Exception:
        pass
for _f in ('Noto Sans CJK JP', 'Droid Sans Fallback'):
    if any(_x.name == _f for _x in matplotlib.font_manager.fontManager.ttflist):
        matplotlib.rcParams['font.family'] = _f
        break

INK = '#1f2937'
MUTED = '#6b7280'
TILE = '#f3f6fa'
TILE_EDGE = '#c9d6e6'
ACCENT = '#3b78c2'
NG = '#c0392b'
OK = '#2f8f46'
LINK_COLORS = ['#d93636', '#2f4fd6', '#3a9d4a']   # 上腕・前腕・先端（A1 の色）


def autocrop(im, pad=12, thresh=245):
    a = np.asarray(im.convert('RGB')).astype(int)
    mask = (a < thresh).any(axis=2)
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, a.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, a.shape[1])
    return im.crop((x0, y0, x1, y1))


def render_mesh(split):
    """A1 の STL を描く。split=False は 1 色の 3D モデル、True は関節で分けて色を付ける。"""
    m_dir = ROOT / 'data' / 'test' / 'A1' / 'meshes'
    fo = json.load(open(m_dir / 'joints.json'))['frame_origins']
    names = ['base', 'upper_arm', 'forearm', 'hand']
    cols = [(0.55, 0.57, 0.65), (0.85, 0.15, 0.15), (0.15, 0.2, 0.85), (0.2, 0.62, 0.2)]
    d = np.array(fo['hand']) - np.array(fo['upper_arm'])
    d /= np.linalg.norm(d)
    gap = 0.12 if split else 0.0
    light = np.array([0.3, -0.6, 0.75])
    light /= np.linalg.norm(light)
    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    allv = []
    for i, n in enumerate(names):
        m = trimesh.load(m_dir / f'{n}.stl')
        v = m.vertices + np.array(fo[n]) + d * gap * i
        tri = v[m.faces]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        k = 0.45 + 0.55 * np.clip(nrm @ light, 0, 1)
        base = np.array(cols[i] if split else (0.62, 0.64, 0.68))
        ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(base[None] * k[:, None] + 0.08, 0, 1),
                                             edgecolors='none'))
        allv.append(v)
    if split:
        for i, n in enumerate(names[1:], start=1):
            p = np.array(fo[n]) + d * gap * (i - 0.5)
            ax.scatter(*p, s=160, c='#d02bd0', edgecolors='k', depthshade=False, zorder=10)
    V = np.vstack(allv)
    c = (V.max(0) + V.min(0)) / 2
    r = (V.max(0) - V.min(0)).max() / 2 * 0.92
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=12, azim=-75)
    ax.set_axis_off()
    plt.subplots_adjust(0, 0, 1, 1)
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        fig.savefig(f.name, facecolor='white')
        plt.close(fig)
        return autocrop(Image.open(f.name).copy())


def sim_frame():
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '0.6', '-i', str(ROOT / 'demo' / 'phys_pusher.mp4'),
                        '-frames:v', '1', f.name], check=True)
        im = Image.open(f.name).copy()
    im = im.crop((0, int(im.height * 0.16), im.width, im.height))   # 動画の題を落とす
    im = autocrop(im)
    # 床の左右の端を落として腕と箱に寄せる
    return im.crop((int(im.width * 0.2), 0, int(im.width * 0.88), im.height))


def pose_at_closest(run):
    """先端（最後のカプセルの端）が目標に最も近い step の、各リンクのカプセル (a, b, r) を返す。"""
    d = np.load(ROOT / 'single_run' / run / 'trace' / 'arm_trace.npz', allow_pickle=True)
    xp, xm, cf, gs, tg = d['xpos'], d['xmat'], d['cap_fromto'], d['geom_size'], d['target']

    def caps(t):
        return [(xp[t, b] + xm[t, b] @ cf[b, :3], xp[t, b] + xm[t, b] @ cf[b, 3:], float(gs[b]))
                for b in range(1, len(xp[t]))]
    dist = np.array([np.linalg.norm(caps(t)[-1][1] - tg) for t in range(len(xp))])
    t = int(dist.argmin())
    return caps(t), float(gs[0]), tg, float(dist[t])


def capsule(ax, a, b, r, color):
    a, b = np.asarray(a[:2]), np.asarray(b[:2])
    u = b - a
    u = u / (np.linalg.norm(u) + 1e-12)
    n = np.array([-u[1], u[0]]) * r
    ax.add_patch(Polygon([a + n, b + n, b - n, a - n], closed=True, fc=color, ec='none'))
    ax.add_patch(Circle(a, r, fc=color, ec='none'))
    ax.add_patch(Circle(b, r, fc=color, ec='none'))


def top_view(ax, run):
    caps, r0, tg, dist = pose_at_closest(run)
    ax.add_patch(FancyBboxPatch((-0.2, -0.24), 1.15, 0.48, boxstyle='round,pad=0,rounding_size=0.03',
                                fc='#e7e9ee', ec='none'))
    ax.add_patch(Circle((0, 0), r0, fc='#8c8f9a', ec='none'))
    for (a, b, r), c in zip(caps, LINK_COLORS):
        capsule(ax, a, b, r, c)
    ax.plot([tg[0]], [tg[1]], marker='*', ms=34, color='#f59f00', mec='#a86a00', mew=1.5, zorder=5)
    ax.set_xlim(-0.22, 0.97)
    ax.set_ylim(-0.26, 0.26)
    ax.set_aspect('equal')
    ax.set_axis_off()
    return caps[-1][1], tg, dist


def main():
    fig = plt.figure(figsize=(16, 9), dpi=200)
    fig.patch.set_facecolor('white')

    def tile(x, y, w, h, fc=TILE, ec=TILE_EDGE):
        fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.012',
                                          transform=fig.transFigure, fc=fc, ec=ec, lw=1.5, zorder=-10))

    def arrow(x0, y0, x1, y1, color='#9ca3af', lw=4):
        fig.patches.append(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure,
                                           arrowstyle='-|>', mutation_scale=28, color=color, lw=lw))

    # ===== 上の段: ①〜⑤ =====
    imgs = [
        ('① 手描きの絵', '入力', Image.open(ROOT / 'data/test/A1/sketch/A1_hand.webp').convert('RGB')),
        ('② AI が整えた絵', '画像生成 AI で清書', Image.open(ROOT / 'data/test/A1/sketch/A1_m1.jpeg').convert('RGB')),
        ('③ 3D モデル', '絵から立体を作る', render_mesh(split=False)),
        ('④ 関節で分ける', '部品と関節のあるロボットに', render_mesh(split=True)),
        ('⑤ 形と動かし方を学習', 'シミュレーションで同時に', sim_frame()),
    ]
    n = len(imgs)
    left, right, gapx = 0.02, 0.98, 0.022
    w = (right - left - gapx * (n - 1)) / n
    y0, h = 0.47, 0.50
    for i, (title, cap, im) in enumerate(imgs):
        x = left + i * (w + gapx)
        hl = i == n - 1
        tile(x, y0, w, h, fc='#eef4fb' if hl else TILE, ec=ACCENT if hl else TILE_EDGE)
        fig.text(x + 0.012, y0 + h - 0.05, title, fontsize=17, fontweight='bold', color=INK, va='baseline')
        ax = fig.add_axes([x + 0.008, y0 + 0.07, w - 0.016, h - 0.14])
        ax.imshow(np.asarray(im))
        ax.set_axis_off()
        fig.text(x + w / 2, y0 + 0.028, cap, fontsize=14.5, color=MUTED, ha='center', va='baseline')
        if i < n - 1:
            arrow(x + w + 0.003, y0 + h / 2, x + w + gapx - 0.003, y0 + h / 2)

    # ⑤ → ⑥
    x5 = left + (n - 1) * (w + gapx) + w / 2
    arrow(x5, y0 - 0.005, x5, 0.405, color=ACCENT, lw=5)

    # ===== 下の段: ⑥ 判定と助言 =====
    py, ph = 0.03, 0.37
    tile(left, py, right - left, ph, fc='#fbfcfd', ec=ACCENT)
    fig.text(left + 0.015, py + ph - 0.052, '⑥ 判定と助言', fontsize=20, fontweight='bold', color=INK, va='baseline')
    fig.text(left + 0.135, py + ph - 0.052, '例: 腕の先を ★ に届かせる', fontsize=15, color=MUTED, va='baseline')

    axL = fig.add_axes([0.06, 0.055, 0.33, 0.215])
    tipL, tg, dL = top_view(axL, 'tripo_pj_short_phys_cg')
    axL.plot([tipL[0], tg[0]], [tipL[1], tg[1]], ls=(0, (4, 3)), color=NG, lw=2.5)
    axL.text((tipL[0] + tg[0]) / 2, 0.06, f'{int(dL * 1000)} mm', color=NG, fontsize=17, fontweight='bold',
             ha='center')
    fig.text(0.225, 0.272, '× 短すぎる腕: 届かない', fontsize=18, fontweight='bold', color=NG, ha='center')

    # 助言
    fig.patches.append(FancyBboxPatch((0.415, 0.15), 0.17, 0.11, boxstyle='round,pad=0,rounding_size=0.015',
                                      transform=fig.transFigure, fc='#fff4e5', ec='#ea8a2e', lw=2))
    fig.text(0.5, 0.217, '助言', fontsize=14, color='#c06a12', ha='center', va='baseline')
    fig.text(0.5, 0.172, '「腕を 1.99 倍に伸ばせ」', fontsize=17, fontweight='bold', color=INK, ha='center',
             va='baseline')
    arrow(0.40, 0.205, 0.413, 0.205, color='#ea8a2e', lw=3)
    arrow(0.587, 0.205, 0.605, 0.205, color='#ea8a2e', lw=3)

    axR = fig.add_axes([0.62, 0.055, 0.33, 0.215])
    _, _, dR = top_view(axR, 'tripo_pjr199_phys')
    label = '1 mm 未満' if dR < 0.001 else f'{dR * 1000:.0f} mm'
    fig.text(0.785, 0.272, f'○ 伸ばした腕: 届いた（{label}）', fontsize=18, fontweight='bold', color=OK, ha='center')

    fig.text(right - 0.012, py + 0.012, '上から見た図（上の段とは別の腕）。先端が ★ に最も近づいた瞬間の姿勢（学習した結果）',
             fontsize=12, color=MUTED, ha='right', va='baseline')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor='white')
    print(f'書いた: {OUT.relative_to(ROOT)}  短い腕 {dL * 1000:.1f} mm / 1.99 倍 {dR * 1000:.2f} mm')


if __name__ == '__main__':
    main()
