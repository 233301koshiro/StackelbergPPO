#!/usr/bin/env python3
"""**予備審査のスライドの図**（2026-11-09、ユーザー「もっとスライドから字を減らしたい」）。

字の箇条書き・表を図に置き換える。値はすべて修論の本文にあるもの（スライドで新しい値を作らない）。

    python3 scripts/make_yobi_figures.py      # → docs/研究応用/執筆/yobi/*.png

作るもの:
  切り抜き（既存の図から）
    crop_matrix_rank.png   matrix_reversal.png の右（順位の入れ替わり）→ 結果①
    crop_arm_forms.png     arm_forms.png の (b)（比較に用いた形態）→ 評価の設定
    crop_flow_advice.png   slide_flow.png の下の段 ⑥（判定と助言）→ 結果③
  新しい図
    alloc_pair.png         結果② 配分だけが違う形の組と、判定の方法ごとの区別（修論 4.4.3.2・表 4.7）
    eval_questions.png     評価の問い R1〜R4・診断・部品で組めるか（修論 4.7）を、システムのどこが壊れるかで置く（修論 4.2.1）
    three_layers.png       ④ 3 層の診断（幾何・収束した形・動き）を 3 コマで（修論 3.12。模式図）
    closed_loop.png        結果④ 閉ループと、見つかった欠陥 3 つ（修論 4.5.2〜4.5.4・5.2.3）
    schedule.png           今後の予定のガント図（スライド「今後の予定」の表と同じ中身）
"""
import pathlib

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Wedge, Rectangle
import datetime as dt
import matplotlib.dates as mdates

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / 'docs' / '研究応用' / '執筆' / 'yobi'
FIG = ROOT / 'figures'

# 既定のフォントは日本語を持たない。Noto CJK を明示登録する（make_slide_flow.py と同じ）
for _ttc in sorted(pathlib.Path('/usr/share/fonts').rglob('*CJK*')):
    try:
        matplotlib.font_manager.fontManager.addfont(str(_ttc))
    except Exception:
        pass
for _f in ('Noto Sans CJK JP', 'Droid Sans Fallback'):
    if any(_x.name == _f for _x in matplotlib.font_manager.fontManager.ttflist):
        matplotlib.rcParams['font.family'] = _f
        break

# 色は make_slide_flow.py にそろえる
INK = '#1f2937'
MUTED = '#6b7280'
TILE = '#f3f6fa'
TILE_EDGE = '#c9d6e6'
ACCENT = '#3b78c2'
NG = '#c0392b'
OK = '#2f8f46'
WARN = '#e08a1e'
LINK_COLORS = ['#e06666', '#5b8fd6', '#5bb36e']   # 1〜3 本目のリンク（arm_forms.png と同じ並び）
DPI = 200


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, facecolor='white')
    plt.close(fig)
    print(f'→ {OUT / name}')


def tile(ax, x, y, w, h, fc=TILE, ec=TILE_EDGE, lw=1.5, r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                                fc=fc, ec=ec, lw=lw))


def arrow(ax, p, q, color=MUTED, lw=2.5, ms=22, cs='arc3'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=cs, shrinkA=0, shrinkB=0))


def draw_arm(ax, x0, y0, lengths, scale, width=0.035, base_r=0.05):
    """上から見た水平 3 関節アーム（全関節 0°）。arm_forms.png (b) と同じ描き方。"""
    ax.add_patch(Circle((x0, y0), base_r, fc='#bdbdbd', ec='#9e9e9e', lw=1.5, zorder=1))
    x = x0
    for L, c in zip(lengths, LINK_COLORS):
        ax.plot([x, x + L * scale], [y0, y0], color=c, lw=width * 400, solid_capstyle='round', zorder=2)
        ax.plot([x], [y0], 'o', color='black', ms=4, zorder=3)
        x += L * scale
    return x


# ---------------------------------------------------------------- 切り抜き
def crops():
    im = Image.open(FIG / 'matrix_reversal.png')          # 1735×755。右の 1 枚（順位の目盛りを含む）は x ≥ 1320
    im.crop((1320, 0, im.width, im.height)).save(OUT / 'crop_matrix_rank.png')
    print(f'→ {OUT / "crop_matrix_rank.png"}')
    im = Image.open(FIG / 'arm_forms.png')                 # 1457×888。(b) は x ≥ 590
    im.crop((590, 0, im.width, im.height)).save(OUT / 'crop_arm_forms.png')
    print(f'→ {OUT / "crop_arm_forms.png"}')
    im = Image.open(ROOT / 'docs' / '研究応用' / '執筆' / 'slide_flow.png')   # 3200×1800。⑥ は y ≥ 1070
    im.crop((40, 1070, im.width - 40, im.height - 20)).save(OUT / 'crop_flow_advice.png')
    print(f'→ {OUT / "crop_flow_advice.png"}')


# ---------------------------------------------------------------- 結果② 配分の組
def alloc_pair():
    """修論 4.4.3.2・表 4.7 の押しの列。提案の 3 シードの値は本文の best exec_R_eps。"""
    root = [11.39, 11.62, 9.72]
    tip = [5.38, 9.52, 8.86]
    frozen_root, frozen_tip = (6.77, 7.04), (6.04, 6.88)   # 形を凍結（範囲だけ本文にある）

    fig = plt.figure(figsize=(13, 5.6))
    # 左: 2 本の腕
    axL = fig.add_axes([0.0, 0.0, 0.34, 1.0])
    axL.set_xlim(-0.1, 1.15)
    axL.set_ylim(0, 1)
    axL.axis('off')
    for y, name, Ls, col in [(0.70, '根元重', (0.500, 0.250, 0.137), ACCENT),
                             (0.30, '先端重', (0.137, 0.250, 0.500), WARN)]:
        axL.text(0.0, y + 0.13, name, fontsize=22, color=col, weight='bold', va='center')
        draw_arm(axL, 0.05, y, Ls, 1.0)
    axL.text(0.5, 0.06, '総リーチはどちらも 0.887 m', fontsize=17, color=INK, ha='center')

    # 右: 判定の方法ごと
    ax = fig.add_axes([0.47, 0.16, 0.50, 0.78])
    rows = ['総リーチ（第1層）', '先端の最大速度', '形を凍結し\n制御だけ学習', '提案\n（形も最適化）']
    ys = [3, 2, 1, 0]
    ax.set_ylim(-0.75, 3.6)
    ax.set_xlim(4.5, 13.6)
    ax.set_yticks(ys)
    ax.set_yticklabels(rows, fontsize=17)
    for t, y in zip(ax.get_yticklabels(), ys):
        if y == 0:
            t.set_color(INK)
            t.set_weight('bold')
    ax.tick_params(axis='y', length=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.set_xlabel('押しの成績（各 3 シードの最良エピソード報酬）', fontsize=15)
    ax.tick_params(axis='x', labelsize=13)
    for y in ys:
        ax.axhline(y - 0.5, color='#e5e7eb', lw=1)
    # 上 2 行: 同じ値 ⇒ 区別できない
    for y, txt in [(3, '0.887 m  ＝  0.887 m'), (2, '0.583 m/s  ＝  0.583 m/s')]:
        ax.text(8.5, y, txt + '　区別できない', fontsize=17, color=MUTED, ha='center', va='center')
    # 凍結: 範囲が重なる
    for (lo, hi), dy, c in [(frozen_root, 0.13, ACCENT), (frozen_tip, -0.13, WARN)]:
        ax.plot([lo, hi], [1 + dy, 1 + dy], color=c, lw=9, solid_capstyle='butt', alpha=0.85)
    ax.text(7.4, 1, '重なる → 区別できない', fontsize=15, color=MUTED, va='center')
    # 提案: 3 シードの点と範囲
    for vals, dy, c in [(root, 0.13, ACCENT), (tip, -0.13, WARN)]:
        ax.plot([min(vals), max(vals)], [dy, dy], color=c, lw=9, solid_capstyle='butt', alpha=0.35)
        ax.plot(vals, [dy] * 3, 'o', color=c, ms=10, mec='white', mew=1.5)
    for xv in (9.52, 9.72):
        ax.plot([xv, xv], [-0.3, 0.3], color=NG, lw=1.2, ls=(0, (2, 2)))
    ax.text(9.62, -0.48, 'すき間 0.20', fontsize=14, color=NG, ha='center', va='center')
    ax.text(11.9, 0.0, '根元重が上', fontsize=16, color=INK, weight='bold', va='center')
    save(fig, 'alloc_pair.png')


# ---------------------------------------------------------------- 評価の問い
def eval_questions():
    """修論 4.2.1 の R1〜R4 と診断（スライドでは中身の名前で呼ぶ）を、パイプラインの「満たさないと壊れる場所」に置く。"""
    fig = plt.figure(figsize=(14, 6.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.2)
    ax.axis('off')
    # パイプライン（中段）
    steps = [('入れた形', 0.3), ('形と制御を\n一緒に学習', 3.5), ('順位を出す', 6.9), ('助言を返す', 10.3)]
    W, H, Y = 3.0, 1.3, 2.6
    for i, (t, x) in enumerate(steps):
        tile(ax, x, Y, W, H, fc='white', ec=ACCENT, lw=2.2, r=0.15)
        ax.text(x + W / 2, Y + H / 2, t, fontsize=21, ha='center', va='center', color=INK, weight='bold')
        if i < len(steps) - 1:
            arrow(ax, (x + W + 0.03, Y + H / 2), (steps[i + 1][1] - 0.03, Y + H / 2), color=ACCENT)

    def tag(x, y, code, q, where, tx, col=NG, up=True):
        w, h = 3.2, 1.75
        tile(ax, x, y, w, h, fc='#fff7f5', ec=col, lw=1.8, r=0.12)
        ax.text(x + 0.18, y + h - 0.32, code, fontsize=19, color=col, weight='bold', va='center')
        ax.text(x + 0.18, y + h - 0.88, q, fontsize=16, color=INK, va='center')
        ax.text(x + w - 0.15, y + 0.25, where, fontsize=13, color=INK, ha='right', va='center')
        ax.plot([x + w / 2, tx], [y if up else y + h, Y + H if up else Y],
                color=col, lw=1.6, ls=(0, (3, 2)))

    # 上段（学習の性質）
    # R1・R2 は学習、R4 は学習から順位までの量
    # 見出しは修論の記号（R1〜R4）ではなく中身の名前にする。聞き手には「R1」の意味が通じない（2026-11-10、ユーザー）。
    # 対応: 形の効き目＝R1、形に出るか＝R2、順序づけ＝R3、判定の早さ＝R4（修論 表 4.3）
    tag(0.6, 4.25, '形の効き目', '形を変えると\n成績が変わるか', '付録', 4.4, up=True)
    tag(4.1, 4.25, '形に出るか', 'タスクで\n収束する形が変わるか', '付録', 5.6, up=True)
    tag(7.6, 4.25, '判定の早さ', 'どれだけ学習すれば\n順位が分かるか', '結果⑤', 8.4, up=True)
    # 下段（出力の性質）
    tag(6.9, 0.25, '順序づけ', 'タスクに応じた\n順位を付けられるか', '結果①②', 8.4, up=False)
    tag(10.3, 0.25, '助言', '従うと\n届く形になるか', '結果③④', 11.8, up=False)
    # 結果⑥ は R1〜R4・診断のどれにも当たらないので、問いとして足す（2026-11-10、ユーザーが A 案を選んだ。修論 4.7）
    tag(3.4, 0.25, '組めるか', '部品で組める長さに\n限っても通るか', '結果⑥', 5.0, up=False)
    ax.text(0.3, 0.9, '満たさないと\nシステムのここが\n壊れる', fontsize=17, color=NG, va='center')
    save(fig, 'eval_questions.png')


# ---------------------------------------------------------------- ④ 3 層
def three_layers():
    """修論 3.12 の 3 層を模式図で。値は助言の文にある 1.99 倍・上限・10 % だけ。"""
    fig = plt.figure(figsize=(15, 5.4))
    titles = [('第1層 幾何', '学習の前'), ('第2層 収束した形', '学習の後'), ('第3層 動き', '学習の後')]   # ①〜③ はスライドの段の番号とぶつかるので使わない
    advice = ['「腕全体を 1.99 倍にすれば届く」',
              '「ギア比が上限に張り付く\n＝出力が足りない」',
              '「可動域の 10 % 未満しか使わない\n関節は、固定を試す価値がある」']
    axes = []
    for i in range(3):
        a = fig.add_axes([0.01 + i * 0.333, 0.0, 0.315, 1.0])
        a.set_xlim(0, 10)
        a.set_ylim(0, 10)
        a.axis('off')
        tile(a, 0.1, 0.1, 9.8, 9.8, r=0.3)
        a.text(0.6, 9.0, titles[i][0], fontsize=24, weight='bold', color=INK, va='center')
        a.text(9.4, 9.0, titles[i][1], fontsize=15, color=INK, va='center', ha='right')
        a.text(5.0, 1.5, advice[i], fontsize=16.5, color=INK, ha='center', va='center', weight='bold')   # 青はスライドの役割の色（形態）なので使わない
        axes.append(a)

    # ① 腕と目標の間のすき間
    a = axes[0]
    a.add_patch(Circle((1.5, 5.6), 0.65, fc='#bdbdbd', ec='#9e9e9e', lw=1.5))
    x = 1.5
    for L, c in zip((1.0, 0.9, 0.7), LINK_COLORS):
        a.plot([x, x + L], [5.6, 5.6], color=c, lw=14, solid_capstyle='round')
        x += L
    a.plot([x + 0.3, 8.2], [5.6, 5.6], color=NG, lw=2.5, ls=(0, (4, 3)))
    a.text((x + 8.2) / 2 + 0.15, 6.25, '届かない', fontsize=17, color=NG, ha='center', weight='bold')
    a.plot([8.5], [5.6], marker='*', ms=34, color='#f2a516', mec='#b77400')

    # ② ギア比のメーター 3 本が上限に張り付く
    a = axes[1]
    bx = [3.0, 5.0, 7.0]
    lo_y, hi_y = 3.0, 7.6
    for k, (xx, fr) in enumerate(zip(bx, (0.97, 0.9, 0.93))):
        a.add_patch(Rectangle((xx - 0.35, lo_y), 0.7, hi_y - lo_y, fc='white', ec=MUTED, lw=1.5))
        a.add_patch(Rectangle((xx - 0.35, lo_y), 0.7, (hi_y - lo_y) * fr, fc=WARN, ec='none', alpha=0.85))
        a.text(xx, lo_y - 0.45, f'関節{k + 1}', fontsize=14, ha='center', va='center', color=INK)
    a.plot([2.2, 7.8], [hi_y, hi_y], color=NG, lw=2.5)
    a.text(8.0, hi_y, '上限', fontsize=16, color=NG, va='center', weight='bold')
    a.text(1.0, (lo_y + hi_y) / 2, 'ギア比', fontsize=15, color=INK, va='center', rotation=90)

    # ③ 関節の可動域のうち使った範囲
    a = axes[2]
    cx, cy, R = 5.0, 4.3, 2.9
    a.add_patch(Wedge((cx, cy), R, 0, 180, width=0.55, fc='#e5e7eb', ec=MUTED, lw=1.2))
    a.add_patch(Wedge((cx, cy), R, 85, 101, width=0.55, fc=NG, ec='none'))
    a.add_patch(Circle((cx, cy), 0.35, fc=INK))
    a.plot([cx, cx], [cy, cy + R - 0.6], color=INK, lw=5, solid_capstyle='round')
    a.text(cx - R - 0.1, cy - 0.45, '可動域', fontsize=14, color=MUTED, ha='left')
    a.text(cx + 1.45, cy + R + 0.25, '使った範囲', fontsize=15, color=NG, weight='bold')
    save(fig, 'three_layers.png')


# ---------------------------------------------------------------- 結果④ 閉ループ
def closed_loop():
    """修論 4.5.2 の手順 (i)〜(iv) を 1 列に並べて輪にし、4.5.3・4.5.4・5.2.3 の欠陥 3 つを、欠陥のあった工程の上に置く。"""
    fig = plt.figure(figsize=(14, 6.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.0)
    ax.axis('off')
    W, H, Y, G = 3.0, 1.2, 1.5, 0.45
    labels = ['① 診断', '② 助言を出す', '③ 助言どおりの\n形を作る', '④ 入れ直して\n学習・再生']
    xs = [0.3 + i * (W + G) for i in range(4)]
    for i, (x, t) in enumerate(zip(xs, labels)):
        tile(ax, x, Y, W, H, fc='white', ec=ACCENT, lw=2.2, r=0.15)
        ax.text(x + W / 2, Y + H / 2, t, fontsize=19, ha='center', va='center', color=INK, weight='bold')
        if i < 3:
            arrow(ax, (x + W + 0.02, Y + H / 2), (xs[i + 1] - 0.02, Y + H / 2), color=ACCENT)
    # 戻りの矢印（④ → ①）
    ax.plot([xs[3] + W / 2, xs[3] + W / 2, xs[0] + W / 2], [Y - 0.05, 0.6, 0.6], color=ACCENT, lw=2.5)
    arrow(ax, (xs[0] + W / 2, 0.6), (xs[0] + W / 2, Y - 0.05), color=ACCENT)
    ax.text((xs[0] + xs[3] + W) / 2, 0.33, '同じ診断に戻す', fontsize=15, color=MUTED, ha='center', va='center')

    items = [('1', '倍率の丸め', '四捨五入で 2 mm 足りず\n同じ診断が再び棄却\n→ 切り上げ（1.99 倍）'),
             ('2', '形を作る工程', '助言どおりの形を作る\n工程にも欠陥があった'),
             ('3', '押しの倍率', '押し込む距離を含めず\n（2.11 倍）奥の箱を逃す\n→ 含めて 2.36 倍')]
    w, h = W + 0.3, 2.35
    for (n, head, body), x in zip(items, xs[1:]):
        bx, by = x + W / 2 - w / 2, 3.25
        tile(ax, bx, by, w, h, fc='#fff7f5', ec=NG, lw=1.8, r=0.12)
        ax.text(bx + 0.18, by + h - 0.35, f'× {n}  {head}', fontsize=17, color=NG, weight='bold', va='center')
        ax.text(bx + 0.18, by + 0.95, body, fontsize=14, color=INK, va='center', linespacing=1.4)
        ax.plot([x + W / 2, x + W / 2], [by, Y + H + 0.03], color=NG, lw=1.5, ls=(0, (3, 2)))
    save(fig, 'closed_loop.png')


# ---------------------------------------------------------------- 今後の予定
def schedule():
    """スライド「今後の予定」の表と同じ中身（進捗.md の節目と D-10）。"""
    d = dt.date
    fig = plt.figure(figsize=(14, 5.2))
    ax = fig.add_axes([0.15, 0.15, 0.82, 0.60])
    lanes = {'実験': 1.0, '執筆・発表': 0.0}
    bars = [('実験', d(2026, 10, 25), d(2026, 11, 30), '核の実験は済み\n（評価の問い 5 つ）', '#9ca3af'),
            ('実験', d(2026, 12, 7), d(2027, 1, 8), '指摘への対応\n実物のアームで測る', ACCENT),
            ('執筆・発表', d(2026, 10, 25), d(2026, 11, 30), '修論の通し版\n想定問答', '#9ca3af'),
            ('執筆・発表', d(2026, 12, 7), d(2027, 1, 8), '指摘の反映・図・照合', ACCENT),
            ('執筆・発表', d(2027, 1, 9), d(2027, 2, 17), '仕上げ・最終発表の準備', ACCENT)]
    for lane, a, b, t, c in bars:
        y = lanes[lane]
        ax.barh(y, (b - a).days, left=mdates.date2num(a), height=0.62, color=c, alpha=0.9)
        ax.text(mdates.date2num(a) + (b - a).days / 2, y, t, fontsize=14, color='white',
                ha='center', va='center', weight='bold', linespacing=1.3)
    marks = [(d(2026, 12, 3), '予備審査\n（12 月頭）', NG), (d(2027, 1, 8), '実験の凍結\n（1 月上旬）', INK),
             (d(2027, 2, 18), '最終発表\n2/18', NG), (d(2027, 3, 5), '修論提出\n3/5', NG)]
    for m, t, c in marks:
        x = mdates.date2num(m)
        ax.axvline(x, color=c, lw=1.6, ls=(0, (4, 3)), zorder=0)
        ax.plot([x], [1.75], marker='D', ms=13, color=c, clip_on=False)
        ax.text(x, 1.98, t, fontsize=14, color=c, ha='center', va='bottom', weight='bold')
    today = mdates.date2num(d(2026, 11, 9))
    ax.axvline(today, color=OK, lw=2.2)
    ax.text(today + 2, 1.6, '今日 11/9', fontsize=14, color=OK, ha='left', va='center', weight='bold')
    ax.set_yticks(list(lanes.values()))
    ax.set_yticklabels(list(lanes.keys()), fontsize=17)
    ax.tick_params(axis='y', pad=10)
    ax.set_ylim(-0.5, 1.75)
    ax.set_xlim(mdates.date2num(d(2026, 10, 25)), mdates.date2num(d(2027, 3, 12)))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f'{mdates.num2date(v).month} 月'))
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', length=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='x', color='#e5e7eb')
    save(fig, 'schedule.png')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    crops()
    alloc_pair()
    eval_questions()
    three_layers()
    closed_loop()
    schedule()
