#!/usr/bin/env python3
"""システム図（draw.io）を、修論の A4 縦に入る 4 段（フェーズごとに 1 段）へ並べ替えて書き出す（修論 図 3.1）。

**なぜ要るか**: 原本 `docs/研究応用/執筆/システム図.drawio` はスライド用の横長（幅 1680）で、
A4 縦の本文幅（160 mm）に入れると図中の文字が約 1 mm になり読めない。
フェーズ 1〜4 を 1 段ずつ縦に積み、幅を 690 に縮める（2026-11-13 に 2 段から 4 段へ。ユーザー）。
論文用だけのラベル（ファイル名 → 中身の名前）は THESIS_LABELS に持つ。横長の原本はスライド用にそのまま残す。

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

# 2026-11-13 ユーザー「大枠ごとに段数を増やす、縦長のフローにして。つまり 4 段に論文用に組み直して。横長はスライドで使うから 2 パターン残して」
#   → 論文用はフェーズ 1〜4 を 1 段ずつ縦に積む。横長の原本（スライド用）はそのまま残す。
ROWS = [
    ('g1', ['sk', 'nl', 'm1', 'd1'],
     ['e_sk_m1', 'e_nl_m1', 'e_m1_d1']),
    ('g2', ['m2', 'd2', 'm3', 'd3', 'm4', 'd4'],
     ['e_m2_d2', 'e_d2_m3', 'e_m3_d3', 'e_d3_m4', 'e_m4_d4']),
    ('g3', ['m5', 'd5'],
     ['e_m5_d5']),
    ('g4', ['task', 'gate', 'fail1', 'm6', 'd6', 'm7', 'fail2', 'pass'],
     ['e_gate_m6', 'e_gate_fail1', 'e_task_m6', 'e_m6_d6', 'e_d6_m7', 'e_m7_pass', 'e_m7_fail2']),
]
# 段をまたぐ矢印（データ → 次の段の工程）。次の段の工程へは左から入る
CROSS = [('e_d1_m2', 'd1', 'm2'), ('e_d4_m5', 'd4', 'm5'), ('e_d5_gate', 'd5', 'gate')]
LOOP = ('e_fail1_d5', 'fail1', 'd5', 'lbl_loop_auto')
X0 = 30           # 枠の左端（左の余白は閉ループの矢印を通す）
W = 690           # 枠の幅（最も広いフェーズ 4 の 660 に、左から入る矢印の余白を足す）
TITLE = 38        # 枠の上端から工程の段までの高さ（題の帯）
TIER_GAP = 28     # 工程の段とデータの段の間
GAP = 44          # 段と段の間（段をまたぐ矢印が通る）
SPLIT_Y = 300     # 原本でこれより上を工程の段、下をデータの段とみなす
LEGEND = 'scnuweBpAybWxMy2U-mT-1'

# 論文用だけのラベル（型 24: 本文と図にファイル名を書かない。topology.json は 3.9 で定義した名前なので残す）
THESIS_LABELS = {
    'd1': '整形スケッチ<br>画像',
    'd2': '三次元メッシュ<br>(単一・色付き)',
    'd3': 'リンクごとのメッシュ<br>+<br>関節位置',
    'd4': '形態パラメータ<br>(topology.json)',
    'd5': 'XML<br>(MuJoCo 互換)',
    'd6': '学習済みの方策と形態<br>学習曲線・軌跡',
}


def geom(cell):
    return cell.find('mxGeometry')


def shift_point(p, dx, dy):
    for k, d in (('x', dx), ('y', dy)):
        if p.get(k) is not None:
            p.set(k, str(float(p.get(k)) + d))


def box(cell):
    g = geom(cell)
    return [float(g.get(k)) for k in ('x', 'y', 'width', 'height')]


def set_points(cell, pts):
    g = geom(cell)
    arr = g.find('Array')
    if arr is None:
        arr = ET.SubElement(g, 'Array', {'as': 'points'})
    for p in list(arr):
        arr.remove(p)
    for x, y in pts:
        ET.SubElement(arr, 'mxPoint', {'x': str(x), 'y': str(y)})


def relayout(tree):
    root = tree.getroot()
    cells = {c.get('id'): c for c in root.iter('mxCell')}
    for cid, label in THESIS_LABELS.items():
        cells[cid].set('value', label)
    v = cells['m5'].get('value')
    cells['m5'].set('value', v.replace('既定: 縦型', '平面／縦型'))   # 既定は平面（run_tripo_pipeline.sh の VERTICAL=0、修論 3.10）
    row_top = 20.0
    rows = []
    for gid, members, edges in ROWS:
        gx, gy, gw, gh = box(cells[gid])
        dx = X0 + (W - gw) / 2 - gx                      # 中身を枠の中央へ
        P = [m for m in members if box(cells[m])[1] < SPLIT_Y]
        D = [m for m in members if box(cells[m])[1] >= SPLIT_Y]
        dyP = row_top + TITLE - min(box(cells[m])[1] for m in P)
        p_bottom = max(box(cells[m])[1] + box(cells[m])[3] for m in P) + dyP
        dyD = p_bottom + TIER_GAP - min(box(cells[m])[1] for m in D)
        bottom = 0.0
        for m in members:
            g = geom(cells[m])
            dy = dyP if m in P else dyD
            shift_point(g, dx, dy)
            bottom = max(bottom, float(g.get('y')) + float(g.get('height')))
        for eid in edges:
            for p in geom(cells[eid]).iter('mxPoint'):
                if p.get('as') == 'offset':
                    continue
                dy = dyP if float(p.get('y', 0)) < SPLIT_Y else dyD
                shift_point(p, dx, dy)
                bottom = max(bottom, float(p.get('y', 0)))
        bottom += 14
        g = geom(cells[gid])
        g.set('x', str(X0)); g.set('y', str(row_top)); g.set('width', str(W)); g.set('height', str(bottom - row_top))
        # 題は右上へ（左から入る矢印と重ならないように）
        cells[gid].set('style', cells[gid].get('style') + 'align=right;spacingRight=14;')
        rows.append((row_top, bottom))
        row_top = bottom + GAP
    # 段をまたぐ矢印: データの下から段の間へ出て、次の段の工程の左から入る
    for k, (eid, src, tgt) in enumerate(CROSS):
        sx, sy, sw, sh = box(cells[src])
        tx, ty, tw, th = box(cells[tgt])
        gap_y = (rows[k][1] + rows[k + 1][0]) / 2
        set_points(cells[eid], [(sx + sw / 2, gap_y), (tx - 14, gap_y), (tx - 14, ty + th / 2)])
    # 閉ループ: 棄却の出力の左から枠の外（左の余白）を上り、XML の左から入る
    eid, src, tgt, lbl = LOOP
    sx, sy, sw, sh = box(cells[src])
    tx, ty, tw, th = box(cells[tgt])
    set_points(cells[eid], [(12, sy + sh / 2), (12, ty + th / 2)])
    lg = geom(cells[lbl])
    lg.set('x', '0'); lg.set('y', '0')
    cells[lbl].set('style', cells[lbl].get('style') + 'horizontal=0;')   # 縦の線に沿って縦書きに
    # 凡例: 最後の段の下へ。図形 7 つを 2 行、矢印 3 本を 3 行目に
    legend_y = row_top
    lgc = geom(cells[LEGEND])
    lgc.set('x', str(X0)); lgc.set('y', str(legend_y)); lgc.set('width', str(W)); lgc.set('height', '200')
    leg = geom(cells['leg']); leg.set('width', str(W)); leg.set('height', '200')
    for cid in ('lg_mesh', 'lg_diag', 'lg_end'):
        shift_point(geom(cells[cid]), -530, 55)
    for cid in ('lg_a1', 'lg_a2', 'lg_a3', 'lg_a1t', 'lg_a2t', 'lg_a3t'):
        g = geom(cells[cid])
        for p in g.iter('mxPoint'):
            shift_point(p, -1030, 115)
        if g.get('x') is not None:
            shift_point(g, -1030, 115)
    model = root.find('.//mxGraphModel')
    model.set('pageWidth', str(X0 + W + 30))
    model.set('pageHeight', str(int(legend_y + 230)))
    return [r[0] for r in ROWS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', action='store_true', help='figures/system_overview.pdf も書き出す')
    a = ap.parse_args()
    tree = ET.parse(SRC)
    moved = relayout(tree)
    tree.write(OUT, encoding='utf-8', xml_declaration=False)
    print(f'{len(moved)} 段に組み直した → {OUT.relative_to(ROOT)}')
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
