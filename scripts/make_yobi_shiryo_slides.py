#!/usr/bin/env python3
"""新方針の予備審査資料（docs/研究応用/執筆/予備審査論文/新方針_予備審査資料.md）に貼るスライドの図を作る（2026-11-20）。

先生の指摘（11-20）: 予備審査論文はスライドの補助資料で、スライドの流れに沿い、スライドの図を使っていると良い。
そこで本編のスライド（Google スライド「予備審査_秋田」の 2〜25 枚目）の画像を、題の帯（上の赤黒の帯・題・ロゴ）と
右下のページ番号を落として、資料の図にする。題は資料の節の見出しが持つ。

スライドの画像（2000×1125 px）は Google スライドのコネクタ（read_slide_page_thumbnail、WIDTH2000_PX）で
Claude が取り出し、`s02.png`〜`s25.png` の名前で 1 つのディレクトリに置く（シェルからは取り出せない）。
スライドを直したら、取り出し直してからこれを回し、資料の PDF を作り直す。

使い方:
    python3 scripts/make_yobi_shiryo_slides.py <s02.png〜s25.png のディレクトリ>
    → docs/研究応用/執筆/使えそうな画像データ/予備審査資料/slide_NN.png
"""
import sys
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'docs' / '研究応用' / '執筆' / '使えそうな画像データ' / '予備審査資料'
TOP = 170 / 1125      # 題の帯の下の線（2000 px 幅で y = 162）より下だけを残す
BOTTOM = 1070 / 1125  # 右下のページ番号（y ≈ 1095）より上だけを残す
PAD = 20              # 余白を詰めた後に残す白の幅（px）


def crop(path: Path) -> Image.Image:
    im = Image.open(path).convert('RGB')
    w, h = im.size
    im = im.crop((0, int(h * TOP), w, int(h * BOTTOM)))
    # 白い余白を詰める（スライドの地は白）
    diff = ImageChops.difference(im, Image.new('RGB', im.size, 'white'))
    box = ImageChops.add(diff, diff, 2.0, -20).getbbox()
    if box:
        l, t, r, b = box
        im = im.crop((max(0, l - PAD), max(0, t - PAD), min(im.width, r + PAD), min(im.height, b + PAD)))
    return im


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob('s[0-9][0-9].png'))
    if not files:
        sys.exit(f'[ERROR] {src} に s02.png〜s25.png が無い')
    for f in files:
        im = crop(f)
        dst = OUT / f'slide_{f.stem[1:]}.png'
        im.save(dst, optimize=True)
        print(f'{f.name} → {dst.relative_to(ROOT)}  {im.width}×{im.height}（縦横比 {im.width / im.height:.2f}）')


if __name__ == '__main__':
    main()
