#!/usr/bin/env python3
"""スライド9（①スケッチ整形）用に「同じ入力・同じプロンプトで 4 回生成した結果」を並べる。

**なぜ要るか**: 9-68 で、同じ手描き写真に同じプロンプトを 1 文字も変えず 4 回入れたところ
**1 本だけ向きの指示を破った**（鉛直から 42.5°）。プロンプトには
「スケッチが斜めでも必ず立て直すこと」を明示してある（9-35）。
**言葉で説明するより 4 枚並べた方が速い。**

出力: figures/slide9_m1_variance.png
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
from PIL import Image

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

SRC = 'data/test/A1_m1var/sketch'
# 傾きは 9-68 の実測（マゼンタ球の重心から。概算）
ITEMS = [('A1_m1var_m1.jpeg', '1 回目', 5.4, True),
         ('A1_m1var_r2.jpeg', '2 回目', 42.5, False),
         ('A1_m1var_r3.jpeg', '3 回目', 23.4, True),
         ('A1_m1var_r4.jpeg', '4 回目', 7.6, True)]


def main():
    fig, axes = plt.subplots(1, 4, figsize=(11, 4.2))
    for ax, (f, lab, deg, ok) in zip(axes, ITEMS):
        im = Image.open(f'{SRC}/{f}')
        w, h = im.size
        s = 1024 / h
        ax.imshow(im.resize((int(w * s), 1024)))
        ax.axis('off')
        col = '#1a7f37' if ok else '#c1121f'
        ax.set_title(f'{lab}\n傾き {deg:.1f}°', fontsize=15, color=col, pad=6)
        for sp in ('top', 'bottom', 'left', 'right'):
            ax.spines[sp].set_visible(False)
        if not ok:
            ax.add_patch(plt.Rectangle((0, 0), int(w * s) - 1, 1023, fill=False,
                                       ec='#c1121f', lw=5))
            ax.text(int(w * s) / 2, 975, '寝ている',
                    ha='center', va='center', fontsize=17, color='white',
                    fontweight='bold',
                    bbox=dict(fc='#c1121f', ec='none', pad=5))
    fig.suptitle('同じ手描き写真・同じプロンプトで 4 回生成した結果',
                 fontsize=17, y=0.99)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig('figures/slide9_m1_variance.png', dpi=150)
    print('figures/slide9_m1_variance.png')


if __name__ == '__main__':
    main()
