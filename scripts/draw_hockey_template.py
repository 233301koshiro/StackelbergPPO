#!/usr/bin/env python3
"""ホッケー用アーム（先端がマレット）の**下書きテンプレート**を描く。

⚠️ **これは研究の入力ではない。** 手描きスケッチを人が引くときの下敷きである。
E2E の入力は必ず人が紙に描いたものを使うこと（想定問答 Q12「手描きと言うが実際には
AI が描いた絵を写しただけでは」への回答が崩れる）。
A1 でも「下書きで関節位置を打ってからフリーハンドで引く」手順を採っており、これはその下書きにあたる。

比は台座の高さを 1 とした値（`make_m1_prompt.py` と同じ基準）。

    python3 scripts/draw_hockey_template.py
"""
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib.patches import Circle, Ellipse

for _ttc in sorted(pathlib.Path('/usr/share/fonts').rglob('*CJK*')):
    try:
        matplotlib.font_manager.fontManager.addfont(str(_ttc))
    except Exception:
        pass
for _f in ('Noto Sans CJK JP', 'Noto Serif CJK JP'):
    if any(_x.name == _f for _x in matplotlib.font_manager.fontManager.ttflist):
        matplotlib.rcParams['font.family'] = _f
        break

OUT = pathlib.Path('data/test/hockey/sketch/hockey_template.png')

# 台座の高さを 1 とした比。先端は「柄 + 円盤」の合計。
BASE_H = 1.0
L1, L2, L3 = 1.6, 1.4, 0.9        # 上腕 / 前腕 / 先端（柄+円盤）
DISC_D = 0.36                     # 円盤の直径。L3 / DISC_D = 2.5 ≥ 1.5（OBB 判定を通す）
ANG = [78, 55, 20]                # 各リンクの角度[deg]（水平から）。関節の高さを散らす
JOINT_R = 0.13


def sketchy(p0, p1, n=40, amp=0.012, seed=0):
    """手描き風に少し揺らした線分"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    x = p0[0] + (p1[0] - p0[0]) * t
    y = p0[1] + (p1[1] - p0[1]) * t
    d = np.array([-(p1[1] - p0[1]), p1[0] - p0[0]])
    d = d / (np.linalg.norm(d) + 1e-9)
    w = np.sin(t * np.pi) * rng.normal(0, amp, n).cumsum()
    return x + d[0] * w, y + d[1] * w


fig, ax = plt.subplots(figsize=(6.5, 6.2), dpi=150)
fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.13)

# 台座
bx, by, bw = 0.0, 0.0, 0.9
for i, (x0, y0, x1, y1) in enumerate([
        (bx - bw / 2, by, bx + bw / 2, by), (bx + bw / 2, by, bx + bw / 2, by + BASE_H),
        (bx + bw / 2, by + BASE_H, bx - bw / 2, by + BASE_H), (bx - bw / 2, by + BASE_H, bx - bw / 2, by)]):
    ax.plot(*sketchy((x0, y0), (x1, y1), seed=i), color='#444444', lw=2.2)
ax.text(bx, by + BASE_H / 2, '台座\n高さ=1', ha='center', va='center', fontsize=9, color='#444444')

# リンク（根元から）
pts = [np.array([bx, by + BASE_H])]
cols = ['#e03131', '#3b5bdb', '#2f9e44']
names = ['上腕', '前腕', '先端（マレット）']
lens = [L1, L2, L3]
for i, (L, a) in enumerate(zip(lens, ANG)):
    p0 = pts[-1]
    # 先端は「柄」だけ線を引き、残りを円盤にする
    draw_L = L - DISC_D if i == 2 else L
    p1 = p0 + draw_L * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
    lw = 9 if i == 0 else (7 if i == 1 else 7)   # 柄は前腕と同じ太さ（細い棒にしない）
    ax.plot(*sketchy(p0, p1, seed=10 + i), color=cols[i], lw=lw, solid_capstyle='round', alpha=.9)
    off = np.array([0.02, 0.42]) if i == 2 else np.array([0.34, 0.0])
    ha = 'center' if i == 2 else 'left'
    ax.text(*(p0 + p1) / 2 + off, f'{names[i]}\n{L:g}',
            fontsize=8.5, color=cols[i], ha=ha, va='center')
    pts.append(p1)

# 円盤ヘッド: 横から見た薄い円盤＝「柄に垂直な板」。楕円だと柄と一体に見えるので線で描く
tip_dir = np.array([np.cos(np.radians(ANG[2])), np.sin(np.radians(ANG[2]))])
perp = np.array([-tip_dir[1], tip_dir[0]])
disc_c = pts[-1] + tip_dir * DISC_D * 0.22          # 板の厚みぶんだけ先へ
d0, d1 = disc_c - perp * DISC_D / 2, disc_c + perp * DISC_D / 2
ax.plot([d0[0], d1[0]], [d0[1], d1[1]], color='#2f9e44', lw=8,
        solid_capstyle='butt', zorder=4)
ax.plot([d0[0], d1[0]], [d0[1], d1[1]], color='#1a6b30', lw=9.5,
        solid_capstyle='butt', zorder=3)
ax.annotate('', xy=d0, xytext=d1, arrowprops=dict(arrowstyle='<->', color='#1a6b30', lw=1))
ax.text(disc_c[0] + 0.34, disc_c[1], f'円盤 直径={DISC_D:g}\n先端長の {DISC_D/L3:.0%}',
        fontsize=8.5, color='#1a6b30', va='center')

# 関節（マゼンタ）
for i, p in enumerate(pts[:-1]):
    ax.add_patch(Circle(p, JOINT_R, facecolor='#FF00FF', edgecolor='#a000a0', lw=1.5, zorder=5))
    ax.text(p[0] - 0.28, p[1], f'J{i+1}', fontsize=9, color='#a000a0', ha='right', va='center')

# 目安の寸法（紙に写すときの当たり）
tot = BASE_H + L1 + L2 + L3
ax.set_title(f'ホッケー用アーム 下書きテンプレート\n'
             f'台座1 / 上腕{L1:g} / 前腕{L2:g} / 先端{L3:g}（うち円盤{DISC_D:g}）',
             fontsize=11)
fig.text(0.5, 0.09,
         '【これは下書き】実際の入力は紙に手で描くこと。関節の位置だけ写し、線はフリーハンドで引く。\n'
         '柄は前腕と同じ太さにする（細い棒にしない） / 円盤と手首の球をくっつけない / 腕を寝かせない',
         ha='center', va='top', fontsize=9, color='#555555')

ax.set_aspect('equal')
ax.set_xlim(-1.6, 4.4)
ys = [p[1] for p in pts] + [disc_c[1]]
ax.set_ylim(by - 0.45, max(ys) + 0.75)
ax.axis('off')
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor='white')
print(f'{OUT}')
print(f'  リンク比（台座=1）: {L1:g} / {L2:g} / {L3:g}')
print(f'  先端長 / 円盤直径 = {L3/DISC_D:.2f} （OBB 判定の下限 1.5 を{"満たす" if L3/DISC_D>=1.5 else "満たさない"}）')
