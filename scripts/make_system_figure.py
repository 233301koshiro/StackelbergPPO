#!/usr/bin/env python3
"""システム図（draw.io）を、修論の A4 縦に入る 2 段組みへ並べ替えて書き出す（修論 図 3.1）。

**なぜ要るか**: 原本 `docs/研究応用/執筆/システム図.drawio` はスライド用の横長（幅 1680）で、
A4 縦の本文幅（160 mm）に入れると図中の文字が約 1 mm になり読めない。
フェーズ 1・2 を上段、フェーズ 3・4 を下段に移して幅を約 960 に縮める。

**原本は 1 つ**: 図を直すときは原本（横長）だけを draw.io で直し、このスクリプトで作り直す。
論文用の `.drawio` は生成物なので手で直さない（直すと次の生成で消える）。

    python3 scripts/make_system_figure.py                  # 論文用 .drawio を作る
    DRAWIO=/path/to/drawio python3 scripts/make_system_figure.py --export
        # → figures/system_overview.pdf（ベクター）。draw.io desktop の CLI が要る
        #   （AppImage を --appimage-extract し、xvfb-run 経由で --no-sandbox で呼ぶ）
"""
import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'docs' / '研究応用' / '執筆' / 'システム図.drawio'
OUT = ROOT / 'docs' / '研究応用' / '執筆' / 'システム図_論文用.drawio'
PDF = ROOT / 'figures' / 'system_overview.pdf'

# 下段へ移すもの（フェーズ 3・4 の枠と、その中の部品）
ROW2 = {'g3', 'g4', 'm5', 'd5', 'gate', 'fail1', 'task', 'm6', 'd6', 'm7', 'fail2', 'pass'}
DX, DY = -740, 420          # 下段への移動量（左に余白 40 を残し、topology.json → M5 の矢印を通す）
ROW1_H = 400                # 上段の枠の高さ（原本の 460 は下に空きが多い。中身は y=437 まで）
GAP_Y = 470                 # 段の間を通る矢印の高さ
LEGEND = 'scnuweBpAybWxMy2U-mT-1'


def geom(cell):
    return cell.find('mxGeometry')


def shift_point(p, dx, dy):
    for k, d in (('x', dx), ('y', dy)):
        if p.get(k) is not None:
            p.set(k, str(float(p.get(k)) + d))


def relayout(tree):
    root = tree.getroot()
    cells = {c.get('id'): c for c in root.iter('mxCell')}
    moved = []
    for cid in ('g1', 'g2'):
        geom(cells[cid]).set('height', str(ROW1_H))
    for cid in ROW2:
        g = geom(cells[cid])
        g.set('x', str(float(g.get('x')) + DX))
        g.set('y', str(float(g.get('y')) + DY))
        moved.append(cid)
    # 両端が下段にある矢印は、曲がり点も同じだけ動かす
    for c in cells.values():
        if c.get('edge') and c.get('source') in ROW2 and c.get('target') in ROW2:
            g = geom(c)
            for p in g.iter('mxPoint'):
                if p.get('as') != 'offset':
                    shift_point(p, DX, DY)
    # 段をまたぐ矢印: topology.json の下から段の間を通り、M5 の左から入る
    m5 = geom(cells['m5'])
    m5_mid_y = float(m5.get('y')) + float(m5.get('height')) / 2
    e = geom(cells['e_d4_m5'])
    arr = e.find('Array')
    for p in list(arr):
        arr.remove(p)
    for x, y in ((703, GAP_Y), (40, GAP_Y), (40, m5_mid_y)):
        ET.SubElement(arr, 'mxPoint', {'x': str(x), 'y': str(y)})
    # 凡例: 下段の下へ移し、矢印 3 本を 2 行目に回して幅を詰める
    g4 = geom(cells['g4'])
    width = float(g4.get('x')) + float(g4.get('width')) - 20 + 20   # 凡例の右端を下段の枠より 20 外へ（楕円がはみ出さないように）
    lg = geom(cells[LEGEND])
    lg.set('y', str(float(lg.get('y')) + DY))
    lg.set('width', str(width))
    lg.set('height', '140')
    leg = geom(cells['leg'])
    leg.set('width', str(width))
    leg.set('height', '140')
    for cid in ('lg_a1', 'lg_a2', 'lg_a3', 'lg_a1t', 'lg_a2t', 'lg_a3t'):
        g = geom(cells[cid])
        for p in g.iter('mxPoint'):
            shift_point(p, -1030, 55)
        if g.get('x') is not None:
            shift_point(g, -1030, 55)
    # ページの大きさ
    model = root.find('.//mxGraphModel')
    model.set('pageWidth', str(int(width) + 60))
    model.set('pageHeight', '1200')
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', action='store_true', help='figures/system_overview.pdf も書き出す')
    a = ap.parse_args()
    tree = ET.parse(SRC)
    moved = relayout(tree)
    tree.write(OUT, encoding='utf-8', xml_declaration=False)
    print(f'下段へ移した部品: {len(moved)} 個 → {OUT.relative_to(ROOT)}')
    if a.export:
        exe = os.environ.get('DRAWIO')
        if not exe:
            sys.exit('DRAWIO に draw.io desktop の実行ファイルを指定すること')
        PDF.parent.mkdir(exist_ok=True)
        cmd = ['xvfb-run', '-a', exe, '--no-sandbox', '--disable-gpu',
               '-x', '-f', 'pdf', '--crop', '-o', str(PDF), str(OUT)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not PDF.exists():
            sys.exit(f'書き出しに失敗: exit {r.returncode}\n{r.stderr[-500:]}')
        print(f'→ {PDF.relative_to(ROOT)}（{PDF.stat().st_size:,} B）')


if __name__ == '__main__':
    main()
