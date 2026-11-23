#!/usr/bin/env python3
"""新方針の予備審査資料（docs/研究応用/執筆/予備審査論文/新方針_予備審査資料.md）の図のうち、切り出しが要るものを作る（2026-11-20）。

方針（ユーザー 11-20「スライドをそのまま使いすぎ。スライドに使った図形などを対応させてほしいだけで、図とパワポで入れた文字まで
一緒に入れてほしいわけではない。文字だけのスライドは入れないほうがいいし、スライドの図を使ってほしいだけでスライドのスクショを
はってほしいわけではない」）: 資料の図は、スライドに貼った画像ファイルそのもの（使えそうな画像データ/ の元のファイル）を使う。
スライドの題・リード文・注記（スライドに打った文字）は入れない。文字だけのスライドには図を置かない。

元のファイルをそのまま使えないのは次の 2 種類で、これが作る:
  1. スライド 12（③ 適性判定）は全体図 sysflow_slide.png をスライド上で切り抜いて貼っている。同じ範囲（左 59.68 %・右 19.41 % を
     落とす。スライドの画像の cropProperties を 11-20 に読んだ値）で切る → sysflow_stage3.png
  2. スライド 13（④ 三層の診断）とスライド 15（評価項目）の図は、画像ではなく図形で組んである。スライドの画像（2000×1125 px）から
     図の段だけ（リード文の下端から、下の注記またはページ番号の上端まで。位置はスライドの図形の座標〔EMU〕から求めた）を切り出す
     → diagnosis_layers.png・eval_items.png
     スライドの画像は Google スライドのコネクタ（read_slide_page_thumbnail、WIDTH2000_PX）で Claude が取り出し、s13.png・s15.png
     の名前で置く（シェルからは取り出せない）。スライドを直したら、取り出し直してからこれを回し、資料の PDF を作り直す。

使い方:
    python3 scripts/make_yobi_shiryo_figures.py <s13.png と s15.png のディレクトリ>
    → docs/研究応用/執筆/使えそうな画像データ/予備審査資料/{sysflow_stage3,diagnosis_layers,eval_items}.png
"""
import sys
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / 'docs' / '研究応用' / '執筆' / '使えそうな画像データ'
OUT = IMG / '予備審査資料'
SLIDE_H_EMU = 5143500          # 16:9（幅 9144000 EMU）
PAD = 20

# スライド 12 の画像の cropProperties（11-20 に read_presentation で読んだ値）
STAGE3_CROP = dict(left=0.5968383, right=0.19405292)
# 図形で組んだ図の縦の範囲（EMU）
DIAGRAMS = {
    # g409ccd5bbe6_0_78（リード文）の下端 〜 yb_f11_note（下の注記）の上端
    's13.png': ('diagnosis_layers.png', 748025 + 0.1666 * 3000000, 4450000),
    # yb_req_c（リード文）の下端 〜 ページ番号の上
    's15.png': ('eval_items.png', 777240 + 0.1524 * 3000000, 4890000),
}


def trim(im: Image.Image) -> Image.Image:
    diff = ImageChops.difference(im, Image.new('RGB', im.size, 'white'))
    box = ImageChops.add(diff, diff, 2.0, -20).getbbox()
    if not box:
        return im
    l, t, r, b = box
    return im.crop((max(0, l - PAD), max(0, t - PAD), min(im.width, r + PAD), min(im.height, b + PAD)))


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    flow = Image.open(IMG / 'スライド用' / 'sysflow_slide.png').convert('RGB')
    w, h = flow.size
    st3 = flow.crop((round(w * STAGE3_CROP['left']), 0, round(w * (1 - STAGE3_CROP['right'])), h))
    st3.save(OUT / 'sysflow_stage3.png', optimize=True)
    print(f'sysflow_slide.png → sysflow_stage3.png  {st3.width}×{st3.height}')
    for name, (dst, top, bottom) in DIAGRAMS.items():
        im = Image.open(src / name).convert('RGB')
        k = im.height / SLIDE_H_EMU
        im = trim(im.crop((0, round(top * k), im.width, round(bottom * k))))
        im.save(OUT / dst, optimize=True)
        print(f'{name} → {dst}  {im.width}×{im.height}')


if __name__ == '__main__':
    main()
