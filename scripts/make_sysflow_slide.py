#!/usr/bin/env python3
"""**予備審査のスライド用のシステムフロー図（draw.io）**（2026-11-09、ユーザー「システムフローの drawio をスライドに入れない？
出来ればその時のそれぞれの状態を表す画像を各所に散りばめて」）。

システム図.drawio（修論 図 3.1 の元）は工程の説明文が多く、スライドの幅では字が読めない。
そこで同じ流れを、工程名だけの箱と「その工程を通った後の状態」の実物の画像で描き直す。
段の区切りはスライドの「各フロー詳細①〜④」にそろえる（① スケッチ整形 ② 形にする ③ タスクに向くか調べる ④ フィードバック）。

    python3 scripts/make_sysflow_slide.py
    # → docs/研究応用/執筆/システム図_スライド用.drawio（画像を埋め込んだ draw.io。PNG への書き出しは draw.io で行う）
    # → docs/研究応用/執筆/yobi/sysflow_preview.png（同じ配置を matplotlib で描いた確認用。字の幅は draw.io と少し違う）

画像（すべてリポジトリの中。手描き A1 の流れと、到達の短腕 / 1.99 倍の腕）:
  手描き・M1        data/test/A1/sketch/A1_hand.webp・A1_m1.jpeg
  3D・関節で分ける   docs/研究応用/執筆/slide_flow.png の ③④（make_slide_flow.py が STL から描いたもの）
  長さ・太さ        figures/pipeline_m4_paper.png の上腕（修論 図 3.4）
  モデル            figures/pipeline_m5_paper.png の左（修論 図 3.5）
  学習              slide_flow.png の ⑤（phys_pusher.mp4 のコマ）
  棄却・クリア       slide_flow.png の ⑥ の左右（系譜 9-279 の短腕と 1.99 倍）
  指摘              docs/研究応用/執筆/yobi/three_layers.png の ②（模式図。make_yobi_figures.py）
"""
import base64
import io
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Ellipse, FancyArrowPatch

ROOT = pathlib.Path(__file__).resolve().parent.parent
WRITE = ROOT / 'docs' / '研究応用' / '執筆'
OUT_DRAWIO = WRITE / 'システム図_スライド用.drawio'
OUT_PREVIEW = WRITE / 'yobi' / 'sysflow_preview.png'

for _ttc in sorted(pathlib.Path('/usr/share/fonts').rglob('*CJK*')):
    try:
        matplotlib.font_manager.fontManager.addfont(str(_ttc))
    except Exception:
        pass
for _f in ('Noto Sans CJK JP', 'Droid Sans Fallback'):
    if any(_x.name == _f for _x in matplotlib.font_manager.fontManager.ttflist):
        matplotlib.rcParams['font.family'] = _f
        break

# 色は システム図.drawio の凡例にそろえる
PROC_FILL, PROC_EDGE = '#dae8fc', '#6c8ebf'
JUDGE_FILL, JUDGE_EDGE = '#e1d5e7', '#9673a6'
IN_FILL, IN_EDGE = '#fff2cc', '#d6b656'
END_FILL, END_EDGE = '#d5e8d4', '#82b366'
NG = '#b85450'
INK = '#1f2937'
MUTED = '#6b7280'


def autocrop(im, pad=8, thresh=240):
    a = np.asarray(im.convert('RGB')).astype(int)
    ys, xs = np.where((a < thresh).any(axis=2))
    return im.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0),
                    min(xs.max() + pad, a.shape[1]), min(ys.max() + pad, a.shape[0])))


def load_images():
    sf = Image.open(WRITE / 'slide_flow.png')                       # 3200×1800
    return {
        'sketch': Image.open(ROOT / 'data/test/A1/sketch/A1_hand.webp'),
        'm1': autocrop(Image.open(ROOT / 'data/test/A1/sketch/A1_m1.jpeg')),
        'm2': autocrop(sf.crop((1370, 185, 1830, 825))),             # ③ 3D モデル（枠の線を除く）
        'm3': autocrop(sf.crop((1985, 185, 2475, 825))),             # ④ 関節で分ける
        'm4': Image.open(ROOT / 'figures/pipeline_m4_paper.png').crop((775, 70, 920, 655)),   # 上腕の断面とカプセル（軸の字を除く）
        'm5': Image.open(ROOT / 'figures/pipeline_m5_paper.png').crop((190, 230, 290, 760)),  # XML の模型のカプセルの列（字を除く）
        'm6': sf.crop((2605, 320, 3112, 688)),                       # ⑤ シミュレーションのコマ
        'fail1': sf.crop((293, 1328, 1147, 1685)),                   # ⑥ 左: 短腕（397 mm 届かない）
        'fail2': (Image.open(WRITE / 'yobi/three_layers.png').crop((1100, 215, 1900, 840))),  # ② ギア比が上限に張り付く
        'pass': sf.crop((2083, 1328, 2939, 1685)),                   # ⑥ 右: 1.99 倍で届いた
    }


# ---------------------------------------------------------------- 配置（px、draw.io の座標と同じ）
W, H = 1100, 392
# 段ごとの外枠と題の色。赤（棄却）・緑（使える形）は図の中で意味を持つので避けた
PHASES = [('① スケッチ整形', 5, 213, '#d9822b'), ('② 形にする', 218, 655, '#3b78c2'),
          ('③ タスクに向くか調べる', 660, 882, '#8e5ea2'), ('④ フィードバック', 887, 1095, '#2a8c8c')]
IMG_EDGE = '#c8c8c8'             # 状態の画像の枠（色の枠＝判定の出力 以外）


def tint(c, a=0.92):
    """段の塗り: 外枠の色を白へ a だけ寄せる。"""
    r, g, b = (int(c[i:i + 2], 16) for i in (1, 3, 5))
    return '#%02x%02x%02x' % tuple(round(v + (255 - v) * a) for v in (r, g, b))
COL = {'sketch': 60, 'm1': 160, 'm2': 274, 'm3': 382, 'm4': 490, 'm5': 598,
       'gate': 718, 'm6': 828, 'm7': 940, 'pass': 1042}
PY, PH, PW = 84, 62, 96          # 工程の箱
IY, IH, IW = 172, 120, 98        # 状態の画像の枠
CY, CH = 296, 34                 # 画像の下の説明
NODES = [  # (id, 種類, 字)
    ('sketch', 'input', '手描きスケッチ'),
    ('m1', 'proc', '絵を整える\n（関節に印）'),
    ('m2', 'proc', '立体にする'),
    ('m3', 'proc', '関節で分ける'),
    ('m4', 'proc', '長さ・太さを\n測る'),
    ('m5', 'proc', 'シミュレータの\nモデルにする'),
    ('gate', 'judge', '第1層\n届くか'),
    ('m6', 'proc', '形と動かし方を\n一緒に学習'),
    ('m7', 'judge', '第2・3層\n形と動き'),
    ('pass', 'end', '使える形'),
]
STATES = [  # (画像の key, 列, 説明, 枠の色)
    ('sketch', 'sketch', '入力', None),
    ('m1', 'm1', '整えた絵', None),
    ('m2', 'm2', '3D メッシュ', None),
    ('m3', 'm3', '関節で分けた部品', None),
    ('m4', 'm4', '長さ・太さ', None),
    ('m5', 'm5', 'モデル（XML）', None),
    ('fail1', 'gate', '棄却＋助言\n「腕を 1.99 倍に」', NG),
    ('m6', 'm6', '学習した方策と形', None),
    ('fail2', 'm7', '指摘\n「ギア比が上限に」', NG),
    ('pass', 'pass', '1.99 倍で届いた', END_EDGE),
]
CHAIN = ['sketch', 'm1', 'm2', 'm3', 'm4', 'm5', 'gate', 'm6', 'm7', 'pass']
LOOP_Y = 352
LOOP_LAB = (400, LOOP_Y + 4, 250, 16)   # 閉ループの説明（② の段の中、右端は段の境 655 の手前）


def node_box(nid, kind):
    x = COL[nid]
    if kind == 'judge':
        return x - 52, PY - 6, 104, PH + 12
    if kind == 'end':
        return x - 46, PY + 4, 92, PH - 8
    return x - PW / 2, PY, PW, PH


def image_box(img, col):
    """枠（IW×IH）に縦横比を保って収め、下に寄せる。"""
    w, h = img.size
    s = min(IW / w, IH / h)
    ww, hh = w * s, h * s
    return COL[col] - ww / 2, IY + IH - hh, ww, hh


# ---------------------------------------------------------------- draw.io
def to_data_uri(img, box):
    im = img.convert('RGB')
    im.thumbnail((int(box[2] * 3), int(box[3] * 3)))
    buf = io.BytesIO()
    im.save(buf, format='PNG', optimize=True)
    return 'data:image/png,' + base64.b64encode(buf.getvalue()).decode()


def build_drawio(imgs):
    mxfile = ET.Element('mxfile', host='make_sysflow_slide.py')
    diagram = ET.SubElement(mxfile, 'diagram', id='sysflow_slide', name='スライド用')
    model = ET.SubElement(diagram, 'mxGraphModel', dx='1100', dy='400', grid='1', gridSize='10', guides='1',
                          tooltips='1', connect='1', arrows='1', fold='1', page='1', pageScale='1',
                          pageWidth=str(W), pageHeight=str(H), math='0', shadow='0')
    root = ET.SubElement(model, 'root')
    ET.SubElement(root, 'mxCell', id='0')
    ET.SubElement(root, 'mxCell', id='1', parent='0')

    def vertex(cid, value, style, box):
        c = ET.SubElement(root, 'mxCell', id=cid, value=value, style=style, vertex='1', parent='1')
        ET.SubElement(c, 'mxGeometry', x=f'{box[0]:.1f}', y=f'{box[1]:.1f}', width=f'{box[2]:.1f}',
                      height=f'{box[3]:.1f}', **{'as': 'geometry'})

    def edge(cid, src, dst, style, value='', points=()):
        c = ET.SubElement(root, 'mxCell', id=cid, value=value, style=style, edge='1', parent='1',
                          source=src, target=dst)
        g = ET.SubElement(c, 'mxGeometry', relative='1', **{'as': 'geometry'})
        if points:
            arr = ET.SubElement(g, 'Array', **{'as': 'points'})
            for x, y in points:
                ET.SubElement(arr, 'mxPoint', x=str(x), y=str(y))

    text = 'html=1;whiteSpace=wrap;fontFamily=Noto Sans JP;'
    for i, (t, x0, x1, c) in enumerate(PHASES):
        vertex(f'ph{i}', t, f'rounded=1;arcSize=4;fillColor={tint(c)};strokeColor={c};strokeWidth=2.5;fontColor={c};verticalAlign=top;'
               f'fontStyle=1;fontSize=15;spacingTop=4;{text}', (x0, 6, x1 - x0, 370))
    vertex('task', 'タスク', f'shape=note;size=10;fillColor={IN_FILL};strokeColor={IN_EDGE};fontSize=12;{text}',
           (COL['m6'] - 34, 44, 68, 28))
    styles = {
        'proc': f'rounded=1;fillColor={PROC_FILL};strokeColor={PROC_EDGE};fontSize=13;fontStyle=1;{text}',
        'judge': f'rhombus;fillColor={JUDGE_FILL};strokeColor={JUDGE_EDGE};fontSize=12;fontStyle=1;{text}',
        'input': f'shape=note;size=12;fillColor={IN_FILL};strokeColor={IN_EDGE};fontSize=13;fontStyle=1;{text}',
        'end': f'ellipse;fillColor={END_FILL};strokeColor={END_EDGE};fontSize=13;fontStyle=1;{text}',
    }
    for nid, kind, label in NODES:
        vertex(nid, label.replace('\n', '<br>'), styles[kind], node_box(nid, kind))
    for key, col, cap, border in STATES:
        box = image_box(imgs[key], col)
        st = f'shape=image;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;image={to_data_uri(imgs[key], box)};'
        st += f'imageBorder={border};strokeWidth=2;' if border else f'imageBorder={IMG_EDGE};strokeWidth=1;'
        vertex(f'img_{key}', '', st, box)
        color = border if border in (NG,) else INK
        vertex(f'cap_{key}', cap.replace('\n', '<br>'),
               f'text;align=center;verticalAlign=top;fontSize=11;fontColor={color};{text}',
               (COL[col] - 54, CY, 108, CH))
    flow = 'endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#333333;html=1;'
    for a, b in zip(CHAIN, CHAIN[1:]):
        edge(f'e_{a}_{b}', a, b, flow + 'exitX=1;exitY=0.5;entryX=0;entryY=0.5;')
    edge('e_task_m6', 'task', 'm6', flow + 'exitX=0.5;exitY=1;entryX=0.5;entryY=0;')
    # 工程 → その状態の画像: 出力の矢印と区別して、矢じりの無い点線にする。判定の出力（色の枠）だけ矢印
    down = 'strokeWidth=1;html=1;exitX=0.5;exitY=1;entryX=0.5;entryY=0;'
    for key, col, cap, border in STATES:
        if key == 'sketch':
            continue
        st = (f'endArrow=block;endFill=1;strokeColor={border};' if border
              else 'endArrow=none;dashed=1;dashPattern=1 3;strokeColor=#888888;') + down
        lab = '棄却' if key == 'fail1' else ''
        edge(f'd_{key}', col, f'img_{key}', st + 'fontSize=11;fontColor=#b85450;labelBackgroundColor=#ffffff;', lab)
    edge('e_loop', 'cap_fail1', 'cap_m5',
         f'edgeStyle=orthogonalEdgeStyle;dashed=1;endArrow=block;strokeColor={NG};fontColor={NG};fontSize=11;'
         'exitX=0.5;exitY=1;entryX=0.5;entryY=1;html=1;',
         '', points=[(COL['gate'], LOOP_Y), (COL['m5'], LOOP_Y)])
    # 説明は線のラベルにせず、② の段の中へ右寄せで置く（段の境をまたぐと塗りの違う 2 段に背景が掛かる）
    vertex('loop_lab', '閉ループ：助言どおりに伸ばして再び判定',
           f'text;align=right;verticalAlign=top;fontSize=11;fontColor={NG};{text}', LOOP_LAB)
    ET.indent(mxfile)
    ET.ElementTree(mxfile).write(OUT_DRAWIO, encoding='utf-8', xml_declaration=False)
    print(f'→ {OUT_DRAWIO}（{OUT_DRAWIO.stat().st_size // 1024} KB）')


# ---------------------------------------------------------------- 確認用のプレビュー
def build_preview(imgs):
    s = 2.0
    fig = plt.figure(figsize=(W * s / 100, H * s / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    pt = lambda px: px * s * 0.72      # draw.io の px → matplotlib の pt（この倍率の図で）
    for t, x0, x1, c in PHASES:
        ax.add_patch(FancyBboxPatch((x0, 6), x1 - x0, 370, boxstyle='round,pad=0,rounding_size=8',
                                    fc=tint(c), ec=c, lw=2.5 * s * 0.72))
        ax.text((x0 + x1) / 2, 18, t, ha='center', va='center', fontsize=pt(15), weight='bold', color=c)
    ax.add_patch(FancyBboxPatch((COL['m6'] - 34, 44), 68, 28, boxstyle='square,pad=0', fc=IN_FILL, ec=IN_EDGE))
    ax.text(COL['m6'], 58, 'タスク', ha='center', va='center', fontsize=pt(12))
    for nid, kind, label in NODES:
        x, y, w, h = node_box(nid, kind)
        if kind == 'judge':
            ax.add_patch(Polygon([(x + w / 2, y), (x + w, y + h / 2), (x + w / 2, y + h), (x, y + h / 2)],
                                 fc=JUDGE_FILL, ec=JUDGE_EDGE))
        elif kind == 'end':
            ax.add_patch(Ellipse((x + w / 2, y + h / 2), w, h, fc=END_FILL, ec=END_EDGE))
        else:
            fc, ec = (IN_FILL, IN_EDGE) if kind == 'input' else (PROC_FILL, PROC_EDGE)
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=6', fc=fc, ec=ec))
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=pt(13 if kind != 'judge' else 12),
                weight='bold', color=INK, linespacing=1.2)
    for key, col, cap, border in STATES:
        x, y, w, h = image_box(imgs[key], col)
        ax.imshow(np.asarray(imgs[key].convert('RGB')), extent=(x, x + w, y + h, y), zorder=2)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='square,pad=0', fc='none', ec=border or IMG_EDGE,
                                    lw=2 if border else 1, zorder=3))
        ax.text(COL[col], CY + 2, cap, ha='center', va='top', fontsize=pt(11),
                color=NG if border == NG else INK, linespacing=1.2)
    arr = lambda p, q, c='#333333', lw=1.5, ls='-': ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle='-|>', mutation_scale=12, color=c, lw=lw, ls=ls, shrinkA=0, shrinkB=0, zorder=4))
    for a, b in zip(CHAIN, CHAIN[1:]):
        ka = dict((n, k) for n, k, _ in NODES)
        xa, ya, wa, ha = node_box(a, ka[a])
        xb, yb, wb, hb = node_box(b, ka[b])
        arr((xa + wa, ya + ha / 2), (xb, yb + hb / 2))
    arr((COL['m6'], 72), (COL['m6'], PY))
    kinds = dict((n, k) for n, k, _ in NODES)
    for key, col, cap, border in STATES:
        if key == 'sketch':
            continue
        x, y, w, h = node_box(col, kinds[col])
        ix, iy, iw, ih = image_box(imgs[key], col)
        if border:
            arr((COL[col], y + h), (COL[col], iy), c=border, lw=1)
        else:
            ax.plot([COL[col]] * 2, [y + h, iy], color='#888888', lw=1.2, ls=(0, (1, 3)), zorder=4)
        if key == 'fail1':
            ax.text(COL[col] + 4, (y + h + iy) / 2, '棄却', fontsize=pt(11), color=NG, va='center')
    xs = [COL['gate'], COL['gate'], COL['m5'], COL['m5']]
    ys = [CY + CH, LOOP_Y, LOOP_Y, CY + CH]
    ax.plot(xs[:3], ys[:3], color=NG, ls='--', lw=1.2)
    arr((xs[2], ys[2]), (xs[3], ys[3]), c=NG, lw=1.2, ls='--')
    lx, ly, lw_, lh = LOOP_LAB
    ax.text(lx + lw_, ly + 2, '閉ループ：助言どおりに伸ばして再び判定', ha='right', va='top', fontsize=pt(11), color=NG)
    fig.savefig(OUT_PREVIEW, facecolor='white')
    plt.close(fig)
    print(f'→ {OUT_PREVIEW}')


if __name__ == '__main__':
    imgs = load_images()
    build_drawio(imgs)
    build_preview(imgs)
