#!/usr/bin/env python3
"""修論の図を一次データ（single_run/*/log/log_train.txt）から生成する。

**方針（CLAUDE.md の4層モデルに従う）**:
- 学習曲線は必ず①原本（log_train.txt）から読む。台帳の転記値は使わない
- 形態パラメータ（ギア比・総リーチ）は morph dump 由来で原本に無いため、
  ②台帳（第4章 表4.5）の確定値をこのファイル内に明示的に書き、出典をコメントで残す
- 図を更新したら `python3 scripts/build_thesis_pdf.py` を走らせる

出力: figures/*.png
使い方: python3 scripts/make_thesis_figures.py [図名 ...]（省略時は全部）
"""
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, 'single_run')
OUT = os.path.join(ROOT, 'figures')

# matplotlib は fontconfig の別名（IPAexGothic 等）を見ないので、.ttc を直接登録する。
# LaTeX 側と同じ Noto CJK を使い、PDF 内で本文と図の書体を揃える。
_CJK = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(_CJK):
    matplotlib.font_manager.fontManager.addfont(_CJK)
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'
else:                                    # フォントが無い環境でも図だけは出す
    plt.rcParams['font.family'] = 'DejaVu Sans'
    print('[warn] Noto CJK が見つからない。日本語が豆腐になる')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
# 版面（A4・余白25mm）の textwidth は 160 mm = 6.30 in。図は width=1.0 で貼るので、
# 図の実寸 W [in] に対し文字は 6.30/W 倍に縮む。W を 8〜9.6 in に抑え、
# 図中の文字が刷り上がりで 7 pt を下回らないようにしてある。**W を広げるときは
# フォントも同じ比率で上げること**（W=11.2・8.4pt では 4.4 pt になり読めなかった）。
TEXTWIDTH_IN = 6.30
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['savefig.bbox'] = 'tight'

# 形態クラスの色は全図で固定する（読者が図を跨いで対応を取れるようにするため）
C_SHORT, C_MID, C_LONG = '#c0392b', '#2980b9', '#27ae60'


def curve(run, key='exec_R_eps'):
    """①原本から (epoch, 値) を読む。再開でログが分割されても最後の記録を採る。"""
    path = os.path.join(RUN, run, 'log', 'log_train.txt')
    per = {}
    with open(path, errors='ignore') as f:
        for line in f:
            m = re.search(r'(?:^|\]\s*)(\d+)\s+T_sample', line)
            if not m:
                continue
            v = re.search(rf'{key}\s+(-?[\d.]+)', line)
            if v:
                per[int(m.group(1))] = float(v.group(1))
    ep = sorted(per)
    return np.array(ep), np.array([per[e] for e in ep])


def running_best(y):
    return np.maximum.accumulate(y)


# ── 図1: マトリクス判定の順位反転（4.4.5） ──────────────────────────────────
def fig_matrix_reversal():
    labels = ['短腕 0.403 m', '中間 0.887 m', '長腕 1.451 m']
    cols = [C_SHORT, C_MID, C_LONG]
    runs_r = ['tripo_pj_short', 'tripo_pj_mid', 'tripo_pj_long']
    runs_p = ['tripo_pjp_short', 'tripo_pjp_mid', 'tripo_pjp_long']
    best_r = [running_best(curve(r)[1])[-1] for r in runs_r]
    best_p = [running_best(curve(r)[1])[-1] for r in runs_p]

    fig = plt.figure(figsize=(9.6, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.85], wspace=0.50)

    for k, (vals, title) in enumerate([(best_r, 'Reach（目標へ到達する）'),
                                       (best_p, 'Pusher（対象を押し出す）')]):
        ax = fig.add_subplot(gs[k])
        order = np.argsort(vals)[::-1]                  # 値が大きいほど良い
        rank = {i: r + 1 for r, i in enumerate(order)}
        ypos = np.arange(3)[::-1]                       # 短腕を上に置く
        ax.barh(ypos, vals, color=cols, height=0.58, zorder=3)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=11.5)
        ax.set_xscale('symlog', linthresh=1)
        ax.axvline(0, color='#555', lw=0.9, zorder=2)
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_xlabel('最良エピソード報酬', fontsize=11.5)
        lo = min(min(vals) * 8.0, -1.5)   # 値ラベルが軸ラベルと重ならない余白
        hi = max(max(vals) * 5.0, 1.5)
        ax.set_xlim(lo, hi)
        ax.grid(axis='y', alpha=0)
        for i, v in enumerate(vals):
            side = 1 if v >= 0 else -1
            ax.annotate(f'{v:.2f}', (v, ypos[i]), textcoords='offset points',
                        xytext=(7 * side, 8), ha='left' if side > 0 else 'right',
                        fontsize=11, color='#333')
            ax.annotate(f'{rank[i]} 位', (v, ypos[i]), textcoords='offset points',
                        xytext=(7 * side, -13), ha='left' if side > 0 else 'right',
                        fontsize=12, fontweight='bold', color=cols[i])

    ax = fig.add_subplot(gs[2])
    rr = {i: r + 1 for r, i in enumerate(np.argsort(best_r)[::-1])}
    rp = {i: r + 1 for r, i in enumerate(np.argsort(best_p)[::-1])}
    for i, lab in enumerate(['短腕', '中間', '長腕']):
        ax.plot([0, 1], [rr[i], rp[i]], '-o', color=cols[i], lw=3.0, ms=10, zorder=3)
        # ラベルは右側だけに置く。左にも置くと y 軸の「1 位」等と重なる
        ax.annotate(lab, (1, rp[i]), xytext=(11, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=12, color=cols[i])
    ax.set_xlim(-0.18, 1.62)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Reach', 'Pusher'], fontsize=12)
    ax.set_ylim(3.5, 0.5)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['1 位', '2 位', '3 位'], fontsize=11.5)
    ax.set_title('順位の入れ替わり', fontsize=13, pad=8)
    ax.grid(axis='x', alpha=0)

    fig.savefig(os.path.join(OUT, 'matrix_reversal.png'))
    plt.close(fig)
    return 'matrix_reversal.png'


# ── 図2: 転用成分の分解（4.4.1） ────────────────────────────────────────────
def fig_transfer_decomposition():
    spec = [('rrbot_arm_pusher_H1',    '転用元 H1',            '#7f8c8d', '--'),
            ('rrbot_arm_pusher_K1_v2', 'K1: 制御方策のみ転用', '#c0392b', '-'),
            ('rrbot_arm_pusher_K2_v2', 'K2: 形態のみ転用',     '#2980b9', '-'),
            ('rrbot_arm_pusher_I1_v2', 'I1: 全成分転用',       '#27ae60', '-')]

    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    h1_best = None
    for run, lab, c, ls in spec:
        ep, y = curve(run)
        rb = running_best(y)
        if run.endswith('H1'):
            h1_best = rb[-1]
        ax.plot(ep, rb, ls, color=c, lw=2.1, label=f'{lab}（best {rb[-1]:.2f}）', zorder=3)

    ax.axhline(h1_best, color='#7f8c8d', lw=1.0, ls=':', zorder=2)
    ax.annotate('転用元の到達点', (1000, h1_best), xytext=(-6, 6),
                textcoords='offset points', ha='right', fontsize=9.5, color='#555')
    ax.set_xlabel('エポック')
    ax.set_ylabel('最良エピソード報酬（running best）')
    ax.set_xlim(0, 1000)
    ax.legend(loc='lower right', fontsize=10.5, framealpha=0.95)
    ax.set_title('転用成分による差 — 形態を含む転用のみが転用元を上回る', fontsize=12.5, pad=9)
    fig.savefig(os.path.join(OUT, 'transfer_decomposition.png'))
    plt.close(fig)
    return 'transfer_decomposition.png'


# ── 図3: 診断 M7 の三層検証（3.12.4） ──────────────────────────────────────
def fig_diagnosis_validation():
    # 各セルは single_run/diag_validation/*.txt の実出力を要約したもの
    rows = [
        ('Reach / 短腕',  'bad',  '届かない\n2.19 倍推奨', 'warn', '下限張り付き\n判定不能', 'bad',  '397 mm 離れ', '達成不可', '3 位', '−396.92'),
        ('Reach / 中間',  'ok',   '余裕 10%',              'warn', '下限張り付き\n判定不能', 'ok',   '誤差 0.9 mm',      '障害なし', '1 位', '−3.42'),
        ('Reach / 長腕',  'warn', '余裕 45%\n長すぎ', 'ok',   '内部解',                'warn', '誤差 7 mm',        '障害なし', '2 位', '−12.48'),
        ('Pusher / 短腕', 'bad',  '触れない\n2.32 倍推奨', 'ok',   '内部解（誤）',          'warn', '対象が\n動かない',   '達成不可', '3 位', '−0.00'),
        ('Pusher / 中間', 'ok',   '届く\nめり込まない',    'ok',   '内部解',                'ok',   '1.48 m 移動',      '障害なし', '2 位', '34.32'),
        ('Pusher / 長腕', 'warn', 'めり込み警告',           'ok',   '内部解',                'ok',   '4.05 m 移動',      '障害なし', '1 位', '96.18'),
    ]
    FC = {'ok': '#d5f0dc', 'warn': '#fdf0d0', 'bad': '#f8d7da'}
    EC = {'ok': '#27ae60', 'warn': '#e0a800', 'bad': '#c0392b'}
    MK = {'ok': '✓', 'warn': '!', 'bad': '×'}

    fig, ax = plt.subplots(figsize=(9.6, 4.7))
    ax.set_axis_off()
    cols = ['第1層\n幾何（学習不要）', '第2層\n収束した設計', '第3層\n実際の行動', '総合判定', '実測順位']
    x = [0.0, 1.0, 2.0, 3.0, 3.78]
    w = [0.95, 0.95, 0.95, 0.72, 0.72]

    for j, (c, xi, wi) in enumerate(zip(cols, x, w)):
        ax.text(xi + wi / 2, len(rows) + 0.30, c, ha='center', va='bottom',
                fontsize=11.5, fontweight='bold')

    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        ax.text(-0.12, y + 0.42, r[0], ha='right', va='center', fontsize=11.5)
        for j in range(3):
            st, txt = r[1 + j * 2], r[2 + j * 2]
            # 誤判定のセルだけ赤い破線で囲う。矢印で指すと他のセルを跨いで読みにくい。
            wrong = '（誤）' in txt
            ax.add_patch(Rectangle((x[j], y), w[j], 0.85, fc=FC[st],
                                   ec=EC['bad'] if wrong else EC[st],
                                   ls='--' if wrong else '-',
                                   lw=1.9 if wrong else 1.3, zorder=2))
            ax.text(x[j] + 0.12, y + 0.42, MK[st], ha='center', va='center',
                    fontsize=15, color=EC[st], fontweight='bold', zorder=3)
            ax.text(x[j] + 0.55, y + 0.42, txt, ha='center', va='center',
                    fontsize=10, zorder=3)
        agree = '#d5f0dc'
        ax.add_patch(Rectangle((x[3], y), w[3], 0.85, fc=agree, ec='#27ae60', lw=1.3, zorder=2))
        ax.text(x[3] + w[3] / 2, y + 0.42, r[7], ha='center', va='center', fontsize=11, zorder=3)
        ax.add_patch(Rectangle((x[4], y), w[4], 0.85, fc='#eeeeee', ec='#999', lw=1.0, zorder=2))
        ax.text(x[4] + w[4] / 2, y + 0.55, r[8], ha='center', va='center',
                fontsize=11.5, fontweight='bold', zorder=3)
        ax.text(x[4] + w[4] / 2, y + 0.22, r[9], ha='center', va='center',
                fontsize=9.5, color='#444', zorder=3)

    ax.set_xlim(-1.30, x[4] + w[4] + 0.06)
    ax.set_ylim(-0.75, len(rows) + 0.95)
    ax.text(x[1] + w[1] / 2, -0.52, '破線 = 第2層の誤判定（この形態は対象に触れていない）',
            fontsize=10.5, color=EC['bad'], ha='center')
    ax.text(x[4] + w[4] / 2, -0.55, '総合判定は 6/6 一致', ha='center',
            fontsize=11, color='#1e6b3a', fontweight='bold')
    fig.savefig(os.path.join(OUT, 'diagnosis_validation.png'))
    plt.close(fig)
    return 'diagnosis_validation.png'


# ── 図4: タスク別の収束形態の分離（4.3） ───────────────────────────────────
def fig_task_separation():
    # ②台帳（第4章 表4.5）の確定値。morph dump 由来で①原本には無いため転記。
    # 値を直すときは第4章 表4.5 と 形態比較.md を同時に直すこと。
    pts = [  # (ラベル, 総リーチ m, 肩ギア, 肘ギア, タスク)
        ('L1',     0.905, 20,  207, 'Reach'),
        ('L1_s2',  1.103, 20,  20,  'Reach'),
        ('L2',     1.230, 400, 400, 'Pusher'),
        ('L2_s2',  1.722, 98,  400, 'Pusher'),
        ('TP2',    1.098, 400, 261, 'Target-Pusher'),
        ('TP2_s2', 1.502, 181, 400, 'Target-Pusher'),
    ]
    style = {'Reach': ('#2980b9', 'o'), 'Pusher': ('#c0392b', 's'),
             'Target-Pusher': ('#8e44ad', '^')}
    # ラベルの逃がし方向（点が近接する組でぶつからないよう個別指定）
    off = {'L1': (10, -4), 'L1_s2': (10, -4), 'L2': (10, -12),
           'L2_s2': (-10, -13), 'TP2': (-9, -15), 'TP2_s2': (9, 7)}

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), sharex=True, sharey=True)
    for ax, gi, name in [(axes[0], 2, '肩ギア比'), (axes[1], 3, '肘ギア比')]:
        seen = set()
        for row in pts:
            lab, reach, task = row[0], row[1], row[4]
            g = row[gi]
            c, m = style[task]
            ax.scatter(reach, g, s=160, c=c, marker=m, zorder=4,
                       edgecolor='white', lw=1.3,
                       label=task if task not in seen else None)
            seen.add(task)
            dx, dy = off[lab]
            ax.annotate(lab, (reach, g), xytext=(dx, dy), textcoords='offset points',
                        fontsize=10.5, ha='left' if dx > 0 else 'right', zorder=5)
        ax.axhline(20, color='#888', ls=':', lw=1, zorder=2)
        ax.axhline(400, color='#888', ls=':', lw=1, zorder=2)
        ax.set_xlabel('収束した総リーチ [m]')
        ax.set_title(name, fontsize=13, pad=8)
        ax.set_xlim(0.82, 1.86)
        ax.set_ylim(-52, 486)

    axes[0].set_ylabel('収束したギア比')
    axes[0].text(0.84, 466, '探索上限 400', fontsize=10, color='#666', va='top')
    axes[0].text(0.84, -36, '探索下限 20', fontsize=10, color='#666', va='bottom')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=12,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle('タスクのみを変えた統制比較における収束形態', fontsize=13.5, y=1.02)
    fig.savefig(os.path.join(OUT, 'task_separation.png'))
    plt.close(fig)
    return 'task_separation.png'


FIGS = {'matrix': fig_matrix_reversal,
        'transfer': fig_transfer_decomposition,
        'diagnosis': fig_diagnosis_validation,
        'separation': fig_task_separation}

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    want = sys.argv[1:] or list(FIGS)
    for k in want:
        if k not in FIGS:
            print(f'[skip] 未知の図: {k}（候補: {", ".join(FIGS)}）')
            continue
        name = FIGS[k]()
        print(f'[ok] figures/{name}')
