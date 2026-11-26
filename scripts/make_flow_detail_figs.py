#!/usr/bin/env python3
"""**予備審査デッキの「フロー詳細」の枚に貼る横長の図（draw.io）**（2026-11-26、ユーザーのデッキへの指摘
「文字～～～～ 下に横長の画像 というパワポのデファクト・スタンダードな様式美を守ってみないか？」
「並列な関係の説明なのに前のスライドと今のスライドで図の配置のレイアウトが違うときもい」→「図を横長に描き直して 3 枚をそろえる」）。

フロー詳細の 3 枚（①② / ③ / ④）を「1 行目の主張 → 上に短い文字 → 下に横長の図」にそろえるため、
①② と ③ の図を横長に描き直す（④ は図形のカードが横に並んでいて既にこの形）。
部品と画像は システム図_スライド用.drawio（ユーザーが draw.io で手直しした全体図。予備審査論文 図 1 でもある）から
**読むだけ**で、そちらは書き換えない。箱・画像・説明の並びと色は全体図と同じで、位置だけを横長に組み直す。

    python3 scripts/make_flow_detail_figs.py            # → 使えそうな画像データ/フロー詳細_スライド用.drawio（2 ページ）
    python3 scripts/make_flow_detail_figs.py --export   # さらに PNG を書き出す（draw.io desktop、xvfb-run）
    #   → 使えそうな画像データ/スライド用/flow_wide_stages1-2.png（① ②）・flow_wide_stage3.png（③）
"""
import argparse
import copy
import os
import pathlib
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parent.parent
W = ROOT / 'docs' / '研究応用' / '執筆' / '使えそうな画像データ'
SRC = W / 'システム図_スライド用.drawio'
OUT = W / 'フロー詳細_スライド用.drawio'
PNG = {0: W / 'スライド用' / 'flow_wide_stages1-2.png', 1: W / 'スライド用' / 'flow_wide_stage3.png'}

CW, GAP, PAD = 128, 14, 8          # 列の幅・列の間・段の枠の内側の余白
BOX_W, BOX_H = 112, 62
IMG_W, IMG_H = 112, 100            # 画像はこの枠に縦横比を保って収める
FS = 15                            # 箱と説明の字（全体図は 13・14。スライドに縮めて貼るので上げた）
GRAY = 'rounded=1;fillColor=#f5f5f5;strokeColor=#9e9e9e;fontColor=#1f2937;fontSize=15;fontStyle=1;html=1;whiteSpace=wrap;fontFamily=Noto Sans JP;'


def src_cells():
    r = ET.parse(SRC).getroot()
    return {c.get('id'): c for c in r.iter('mxCell')}


def with_fs(style, fs=FS):
    parts = [p for p in style.split(';') if p and not p.startswith('fontSize=')]
    return ';'.join(parts + [f'fontSize={fs}']) + ';'


class Page:
    def __init__(self, name):
        self.name, self.cells = name, []

    def vertex(self, cid, value, style, x, y, w, h):
        c = ET.Element('mxCell', id=cid, value=value, style=style, vertex='1', parent='1')
        ET.SubElement(c, 'mxGeometry', x=f'{x:.1f}', y=f'{y:.1f}', width=f'{w:.1f}', height=f'{h:.1f}', **{'as': 'geometry'})
        self.cells.append(c)

    def copy_vertex(self, S, sid, x, y, w, h, value=None, fs=FS):
        s = S[sid]
        style = s.get('style') or ''
        if 'image=' not in style:
            style = with_fs(style, fs)
        self.vertex(sid, s.get('value') if value is None else value, style, x, y, w, h)

    def image(self, S, sid, cx, y):
        g = S[sid].find('mxGeometry')
        w, h = float(g.get('width')), float(g.get('height'))
        k = min(IMG_W / w, IMG_H / h)
        self.vertex(sid, '', S[sid].get('style'), cx - w * k / 2, y + (IMG_H - h * k) / 2, w * k, h * k)

    def edge(self, eid, style, src, tgt, value=''):
        c = ET.Element('mxCell', id=eid, value=value, style=style, edge='1', parent='1', source=src, target=tgt)
        ET.SubElement(c, 'mxGeometry', relative='1', **{'as': 'geometry'})
        self.cells.append(c)

    def xml(self, idx):
        d = ET.Element('diagram', id=f'flow{idx}', name=self.name)
        m = ET.SubElement(d, 'mxGraphModel', grid='1', gridSize='10', guides='1', page='1', pageScale='1',
                          math='0', shadow='0')
        root = ET.SubElement(m, 'root')
        ET.SubElement(root, 'mxCell', id='0')
        ET.SubElement(root, 'mxCell', id='1', parent='0')
        for c in self.cells:
            root.append(c)
        return d


def col_x(phase_x, i):
    return phase_x + PAD + i * (CW + GAP)


def page12(S):
    p = Page('① ②')
    flow = S['e_m1_m2'].get('style')
    dot = S['d_m1'].get('style')
    top = 6
    box_y, img_y, cap_y = top + 38, top + 38 + BOX_H + 18, top + 38 + BOX_H + 18 + IMG_H + 6
    ph_h = cap_y + 56 - top
    x0 = 5
    w0 = 2 * CW + GAP + 2 * PAD
    x1 = x0 + w0 + 6
    w1 = 4 * CW + 3 * GAP + 2 * PAD
    p.copy_vertex(S, 'ph0', x0, top, w0, ph_h, fs=18)
    p.copy_vertex(S, 'ph1', x1, top, w1, ph_h, fs=18)
    cols = [('sketch', 'img_sketch', 'cap_sketch', col_x(x0, 0)), ('m1', 'img_m1', 'cap_m1', col_x(x0, 1))]
    cols += [(f'm{i}', f'img_m{i}', f'cap_m{i}', col_x(x1, i - 2)) for i in range(2, 6)]
    for box, img, cap, cx in cols:
        value = '手描き<br>スケッチ' if box == 'sketch' else None
        p.copy_vertex(S, box, cx + (CW - BOX_W) / 2, box_y, BOX_W, BOX_H, value=value)
        p.image(S, img, cx + CW / 2, img_y)
        p.copy_vertex(S, cap, cx, cap_y, CW, 40)
        p.edge(f'd_{box}', dot, box, img)
    for (a, *_), (b, *_) in zip(cols, cols[1:]):
        p.edge(f'e_{a}_{b}', flow, a, b)
    return p, x1 + w1 + 5, top + ph_h + 6


def page3(S):
    p = Page('③')
    flow = S['e_m1_m2'].get('style')
    dot = S['d_m1'].get('style')
    top = 6
    task_y = top + 38
    box_y = task_y + 28 + 14
    img_y = box_y + BOX_H + 18
    cap_y = img_y + IMG_H + 6
    ph_h = cap_y + 56 - top
    gap = 40                        # ③ は列が少ないので、列の間を広げて横長にする
    c_in = 5
    x_ph = c_in + CW + gap - PAD
    w_ph = 2 * CW + gap + 2 * PAD
    c_gate = x_ph + PAD
    c_m6 = c_gate + CW + gap
    c_out = x_ph + w_ph + gap - PAD
    p.copy_vertex(S, 'ph2', x_ph, top, w_ph, ph_h, fs=18)
    # 左: ② の出力（段の外。灰色）
    p.vertex('in_model', '② のモデル', GRAY, c_in + (CW - BOX_W) / 2, box_y, BOX_W, BOX_H)
    p.image(S, 'img_m5', c_in + CW / 2, img_y)
    p.copy_vertex(S, 'cap_m5', c_in, cap_y, CW, 40)
    p.edge('d_in', dot, 'in_model', 'img_m5')
    # ③ の中: 第1層 → 同時最適化（タスクが入る）
    p.copy_vertex(S, 'gate', c_gate + (CW - 112) / 2, box_y - 6, 112, BOX_H + 12, fs=14)
    p.copy_vertex(S, 'm6', c_m6 + (CW - BOX_W) / 2, box_y, BOX_W, BOX_H)
    p.copy_vertex(S, 'task', c_m6 + (CW - 76) / 2, task_y, 76, 28)
    p.image(S, 'img_fail1', c_gate + CW / 2, img_y)
    p.copy_vertex(S, 'cap_fail1', c_gate - 6, cap_y, CW + 12, 40)
    p.image(S, 'img_m6', c_m6 + CW / 2, img_y)
    p.copy_vertex(S, 'cap_m6', c_m6, cap_y, CW, 40)
    # 右: ④ へ（段の外。灰色）
    p.vertex('out_diag', '④ 診断と<br>修正指針へ', GRAY, c_out + (CW - BOX_W) / 2, box_y, BOX_W, BOX_H)
    p.edge('e_in_gate', flow, 'in_model', 'gate')
    p.edge('e_gate_m6', flow, 'gate', 'm6')
    p.edge('e_m6_out', flow, 'm6', 'out_diag')
    p.edge('e_task_m6', S['e_task_m6'].get('style'), 'task', 'm6')
    p.edge('d_fail1', with_fs(S['d_fail1'].get('style')), 'gate', 'img_fail1', value='棄却')
    p.edge('d_m6', dot, 'm6', 'img_m6')
    return p, c_out + CW + 5, top + ph_h + 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', action='store_true')
    a = ap.parse_args()
    S = src_cells()
    mx = ET.Element('mxfile', host='make_flow_detail_figs.py')
    for i, (p, w, h) in enumerate((page12(S), page3(S))):
        d = p.xml(i)
        d.find('mxGraphModel').set('pageWidth', str(int(w)))
        d.find('mxGraphModel').set('pageHeight', str(int(h)))
        mx.append(d)
        print(f'ページ {i + 1}（{p.name}）: {w:.0f} × {h:.0f}、縦横比 {w / h:.2f}')
    ET.ElementTree(mx).write(OUT, encoding='utf-8', xml_declaration=False)
    print(f'→ {OUT.relative_to(ROOT)}')
    if a.export:
        exe = os.environ.get('DRAWIO', '/usr/bin/drawio')
        for i, png in PNG.items():
            # xvfb-run はコンテナにだけあり、ホストには無い（ホストは画面があるのでそのまま動く。11-26 に確かめた）
            pre = ['xvfb-run', '-a'] if shutil.which('xvfb-run') else []
            cmd = pre + [exe, '--no-sandbox', '--disable-gpu', '-x', '-f', 'png', '-s', '3',
                         '--crop', '-p', str(i + 1), '-o', str(png), str(OUT)]   # v27.0.2 から 1 始まり
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if r.returncode != 0 or not png.exists():
                sys.exit(f'書き出しに失敗（ページ {i + 1}）: exit {r.returncode}\n{r.stderr[-500:]}')
            print(f'→ {png.relative_to(ROOT)}（{png.stat().st_size:,} B）')


if __name__ == '__main__':
    main()
