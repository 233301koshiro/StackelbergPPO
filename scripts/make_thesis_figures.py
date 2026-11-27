#!/usr/bin/env python3
"""修論の図を一次データ（single_run/*/log/log_train.txt）から生成する。

**方針（CLAUDE.md の4層モデルに従う）**:
- 学習曲線は必ず①原本（log_train.txt）から読む。台帳の転記値は使わない
- 形態パラメータ（ギア比・総リーチ）は morph dump 由来で原本に無いため、
  ②台帳（第4章 表4.5）の確定値をこのファイル内に明示的に書き、出典をコメントで残す
- 図を更新したら `python3 scripts/build_thesis_pdf.py` を走らせる

出力: docs/研究応用/執筆/使えそうな画像データ/修論/*.png（2026-11-19 までは figures/）
使い方: python3 scripts/make_thesis_figures.py [図名 ...]（省略時は全部）
"""
import os
import re
import sys

import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, 'single_run')
OUT = os.path.join(ROOT, 'docs', '研究応用', '執筆', '使えそうな画像データ', '修論')   # 2026-11-19 にルートの figures/ と平らな置き方をやめ、用途別のフォルダ（修論/・予備審査/・勉強/・過去/）にした

# matplotlib は fontconfig の別名（IPAexGothic 等）を見ないので、.ttc を直接登録する。
# LaTeX 側と同じ Noto CJK を使い、PDF 内で本文と図の書体を揃える。
_CJK = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(_CJK):
    matplotlib.font_manager.fontManager.addfont(_CJK)
    # 2026-11-17: 太字の .ttc も登録する（登録しないと fontweight='bold' が効かず、図 4.2 の順位が細字になる）
    _CJK_B = _CJK.replace('-Regular', '-Bold')
    if os.path.exists(_CJK_B):
        matplotlib.font_manager.fontManager.addfont(_CJK_B)
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


# ── 図1: マトリクス判定の順位反転（4.4.3.2） ──────────────────────────────────
def fig_matrix_reversal():
    labels = ['短腕 0.403 m', '中腕 0.887 m', '長腕 1.451 m']
    cols = [C_SHORT, C_MID, C_LONG]
    # 実在の物理のシード 0（系譜 9-246）。短腕は Bug 72 の門を直した取り直し（9-271）
    runs_r = ['tripo_pj_short_phys_cg', 'tripo_pj_mid_phys', 'tripo_pj_long_phys']
    # 押しの中腕・長腕は Bug 106・107 を直した物理の取り直し（系譜 9-351 の `_vp`）。短腕は箱に触れず直しが効かないので元の run
    runs_p = ['tripo_pjp_short_phys_cg', 'tripo_pjp_mid_phys_vp', 'tripo_pjp_long_phys_vp']
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
        # 2026-11-17: 形態の名前は左のパネルだけに書く（中央にも書くと左のパネルの目盛りと重なる）
        ax.set_yticklabels(labels if k == 0 else [''] * 3, fontsize=11.5)
        ax.set_xscale('symlog', linthresh=1)
        ax.axvline(0, color='#555', lw=0.9, zorder=2)
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_xlabel('最良エピソード報酬（右ほど良い）', fontsize=11.5)
        # 2026-11-17: 値と順位の文字が枠線をまたいでいた（図と表の指摘_20261117.md）。対数に近い軸なので余白は倍率で取る
        lo = min(min(vals) * 8.0, -1.5)
        hi = max(max(vals) * 40.0, 1.5)
        ax.set_xlim(lo, hi)
        ax.grid(axis='y', alpha=0)
        for i, v in enumerate(vals):
            side = 1 if v >= 0 else -1
            inside = v < -50          # 左へ長く伸びた棒は、文字を棒の中に白で書く（枠の外へ出さない）
            if inside:
                side = 1
            ax.annotate(f'{v:.2f}', (v, ypos[i]), textcoords='offset points',
                        xytext=(7 * side, 8), ha='left' if side > 0 else 'right',
                        fontsize=11, color='white' if inside else '#333', zorder=5)
            ax.annotate(f'{rank[i]} 位', (v, ypos[i]), textcoords='offset points',
                        xytext=(7 * side, -13), ha='left' if side > 0 else 'right',
                        fontsize=12, fontweight='bold', color='white' if inside else cols[i], zorder=5)

    ax = fig.add_subplot(gs[2])
    rr = {i: r + 1 for r, i in enumerate(np.argsort(best_r)[::-1])}
    rp = {i: r + 1 for r, i in enumerate(np.argsort(best_p)[::-1])}
    for i, lab in enumerate(['短腕', '中腕', '長腕']):
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


# ── 図1 の帯版（2026-11-25 試作。旧 matrix_reversal.png は before として残す） ──────────
def fig_matrix_bands():
    """図 4.4（11-26 に 4.2 から 4.3、11-27 に 4.4 へ繰り下げ）を、本文の主張（3 シードの範囲が重ならない）と同じ量で描く。

    旧図はシード 0 の棒だけで、本文の根拠（3 シードの範囲）を見せていなかった。
    各シードの最良エピソード報酬を点で、シード間の最小〜最大を帯で描く（11-25: すき間の矢印は外した。ユーザー「書かなくていい」）。
    11-27: 順位の入れ替わりのパネルを外した（ユーザー「2タスクしかやってないし、短腕はずっと最下位…ただ1つの事実を示すだけなのに
    わざわざグラフにする必要ある？」）。入れ替わりは本文で言い、2 つのドットプロットの並びで読める。左右のパネルは連続していないので、
    右のパネルにも縦軸の名前（短腕〔1 シード〕など）を付ける（ユーザー「横軸の情報を乗せ直すべき」）。
    """
    names = ['短腕', '中腕', '長腕']
    reach = [['tripo_pj_short_phys_cg'],
             ['tripo_pj_mid_phys', 'tripo_pj_mid_phys_s2_ws', 'tripo_pj_mid_phys_s3'],
             ['tripo_pj_long_phys', 'tripo_pj_long_phys_s2_ws', 'tripo_pj_long_phys_s3']]
    # 押しの中腕・長腕は Bug 106・107 を直した物理の取り直し（系譜 9-351 の `_vp`）。短腕は箱に触れず直しが効かないので元の run
    push = [['tripo_pjp_short_phys_cg'],
            ['tripo_pjp_mid_phys_vp', 'tripo_pjp_mid_phys_s2_ws_vp', 'tripo_pjp_mid_phys_s3_vp'],
            ['tripo_pjp_long_phys_vp', 'tripo_pjp_long_phys_s2_ws_vp', 'tripo_pjp_long_phys_s3_vp']]
    cols = [C_SHORT, C_MID, C_LONG]
    vals_r = [[running_best(curve(r)[1])[-1] for r in rs] for rs in reach]
    vals_p = [[running_best(curve(r)[1])[-1] for r in rs] for rs in push]
    for task, vv in (('Reach', vals_r), ('Pusher', vals_p)):
        print(f'[matrix_bands] {task}: ' + ' / '.join(f'{n} ' + ' '.join(f'{v:.2f}' for v in x) for n, x in zip(names, vv)))

    fig = plt.figure(figsize=(8.4, 3.6))
    gs = fig.add_gridspec(1, 2, wspace=0.42)
    # 短腕（1 シード）は帯を作れないので、点に何が起きたかを書く（目視記録: 押しの短腕は箱に近づかず 5 話とも 0.00 m）
    notes = ['目標に届かない', '箱に届かず動かせない']
    for k, (vv, title) in enumerate([(vals_r, 'Reach（目標へ到達する）'), (vals_p, 'Pusher（対象を押し出す）')]):
        ax = fig.add_subplot(gs[k])
        ypos = np.arange(3)[::-1]                       # 短腕を上に置く
        lo = [min(x) for x in vv]
        hi = [max(x) for x in vv]
        # 横軸は線形で、中腕と長腕（3 シードの帯で比べる 2 つ）が収まる範囲。短腕が外れたら端に値を書く
        span = max(hi[1:]) - min(lo[1:])
        xmin = min(min(lo[1:]) - 0.18 * span, -0.04 * span if min(lo[1:]) > 0 else min(lo[1:]) - 0.18 * span)
        xmax = max(max(hi[1:]) + 0.18 * span, 0 if max(hi[1:]) < 0 else max(hi[1:]) + 0.18 * span)
        for i in range(3):
            if len(vv[i]) > 1:
                ax.barh(ypos[i], hi[i] - lo[i], left=lo[i], height=0.40, color=cols[i], alpha=0.22,
                        edgecolor=cols[i], lw=1.2, zorder=2)
                above = ypos[i] >= 1                    # 中腕は帯の上、長腕は帯の下に書く（間を空ける）
                # 帯が右の枠の近くにあるとき（Reach の中腕）は、文字が枠にかからないよう帯の右端で右寄せにする
                near_right = (lo[i] + hi[i]) / 2 > xmax - 0.2 * (xmax - xmin)
                ax.text(hi[i] if near_right else (lo[i] + hi[i]) / 2, ypos[i] + (0.30 if above else -0.30),
                        f'[{lo[i]:.2f}, {hi[i]:.2f}]',
                        ha='right' if near_right else 'center', va='bottom' if above else 'top',
                        fontsize=9.5, color=cols[i])
            inside = [v for v in vv[i] if xmin <= v <= xmax]
            ax.scatter(inside, [ypos[i]] * len(inside), s=46, color=cols[i], edgecolor='white', lw=1.2, zorder=4)
            for v in vv[i]:
                if v < xmin:
                    ax.annotate(f'◀ {v:.2f}（軸の外）\n{notes[k]}', (xmin, ypos[i]), xytext=(4, 0), textcoords='offset points',
                                ha='left', va='center', fontsize=10, color=cols[i])
                elif len(vv[i]) == 1:
                    ax.annotate(f'{v:.2f}\n{notes[k]}', (v, ypos[i]), xytext=(9, 0), textcoords='offset points',
                                ha='left', va='center', fontsize=10, color=cols[i])
        ax.set_yticks(ypos)
        ax.set_yticklabels([f'{n}\n（{len(x)} シード）' for n, x in zip(names, vv)], fontsize=10.5)
        ax.set_xlim(xmin, xmax)
        if xmin < 0 < xmax:
            ax.axvline(0, color='#555', lw=0.9, zorder=1)
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_xlabel('最良エピソード報酬（右ほど良い）', fontsize=11.5)
        ax.set_ylim(-0.7, 2.7)
        ax.grid(axis='y', alpha=0)

    fig.savefig(os.path.join(OUT, 'matrix_bands.png'))
    plt.close(fig)
    return 'matrix_bands.png'


# ── 図2: 転用成分の分解（4.3.4.1） ────────────────────────────────────────────
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
    # 各セルは三層診断を今の物理の best（シード 0）にかけ直した出力の要約（系譜 9-292、2026-10-23）。
    # 順位・値はシード 0 の best exec_R_eps（log_train.txt）。2 シードの帯は修論 表 4.6（d6_bands.py 4.4.3.2）
    rows = [
        ('Reach / 短腕',  'bad',  '届かない\n1.99 倍推奨', 'ok',   '範囲の内側\n（誤）',   'bad',  '397 mm 離れ',      '達成不可', '3 位', '−397.20'),
        ('Reach / 中腕',  'ok',   '余裕 10%',              'ok',   '下限\n（到達では正常）', 'ok',   '誤差 0.5 mm\n16 step', '障害なし', '1 位', '−1.85'),
        ('Reach / 長腕',  'warn', '余裕 45%\n長すぎ',      'ok',   '下限\n（到達では正常）', 'ok',   '誤差 0.5 mm\n38 step', '障害なし', '2 位', '−9.68'),
        ('Pusher / 短腕', 'bad',  '触れない\n2.11 倍推奨', 'ok',   '範囲の内側\n（誤）',   'warn', '対象が\n動かない',   '達成不可', '3 位', '−0.00'),
        ('Pusher / 中腕', 'ok',   '届く',                  'warn', '上限の警告\n（出力不足）', 'ok', '0.47 m 移動',      '障害なし', '2 位', '10.42'),
        ('Pusher / 長腕', 'warn', 'めり込み警告',           'ok',   '範囲の内側',           'ok',   '1.09 m 移動',      '障害なし', '1 位', '24.92'),
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
    ax.text(x[1] + w[1] / 2, -0.52, '破線 = 第2層の見逃し（範囲の内側と報告したが届かない）',
            fontsize=10.5, color=EC['bad'], ha='center')
    ax.text(x[4] + w[4] / 2, -0.55, '達成不可の 2 条件はともに最下位', ha='center',
            fontsize=11, color='#1e6b3a', fontweight='bold')
    fig.savefig(os.path.join(OUT, 'diagnosis_validation.png'))
    plt.close(fig)
    return 'diagnosis_validation.png'


# ── 図4: タスク別の収束形態の分離（4.4.2） ───────────────────────────────────
def fig_task_separation():
    # ⛔ 旧い物理の rrbot の値（撤回、系譜 9-246）。修論からは外した図（削除・圧縮記録の旧 図 4.2）で、
    #   今の第4章 表4.5（縦型 4 関節）とは別物。転記元の形態比較.md は 2026-11-25 に消した（git 履歴にある）。
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



# ── 図5: 関節検出の色ドリフト許容幅（4.3.2 = 軸1） ──────────────────────────
def fig_marker_tolerance():
    """`eval_pipeline_robustness.py` の実測値をそのまま図にする。
    値を変えるときはスクリプトを走らせ直して両方を直すこと（表 4.4・4.5 と対）。"""
    tols = [40, 60, 70, 80, 90, 100, 120]
    # None = 検出失敗
    arm3 = [None, None, None, 12.8, 0.4, 0.3, 0.3]
    grip = [None, None, 9.8, 4.5, 1.5, 0.6, 0.2]

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    # 検出できない領域
    ax.axvspan(35, 65, color='#f8d7da', alpha=0.55, zorder=1)
    ax.text(50, 7.5, '両メッシュとも\n検出失敗', ha='center', va='center',
            fontsize=11, color='#8b2a33')
    ax.axhline(2.0, color='#888', ls='--', lw=1.2, zorder=2)
    ax.text(121, 2.25, '誤差 2 mm', fontsize=10, color='#555', ha='right')

    for vals, lab, c, m in [(arm3, '3 関節アーム', C_MID, 'o'),
                            (grip, 'グリッパ付きアーム', C_LONG, 's')]:
        xs = [t for t, v in zip(tols, vals) if v is not None]
        ys = [v for v in vals if v is not None]
        ax.plot(xs, ys, '-', color=c, lw=2.2, marker=m, ms=8, label=lab, zorder=4)
        # 検出失敗した点を × で軸下に置く
        fx = [t for t, v in zip(tols, vals) if v is None]
        ax.plot(fx, [-0.9] * len(fx), 'x', color=c, ms=10, mew=2.4, zorder=4)

    ax.plot([], [], 'x', color='#555', ms=10, mew=2.4, label='検出失敗')
    ax.axvline(40, color='#c0392b', lw=1.6, ls=':', zorder=3)
    ax.text(40, 13.9, '既定値 40', fontsize=10.5, color='#c0392b', ha='center', va='top')
    ax.set_xlabel('色検出の許容閾値 tolerance（公称マーカー色からの距離）')
    ax.set_ylabel('関節位置の誤差 [mm]')
    ax.set_xlim(35, 125)
    ax.set_ylim(-2.0, 14.5)
    ax.legend(loc='upper right', fontsize=10.5, framealpha=0.95)
    ax.set_title('固定した基準色では、検出の成立と位置精度を両立できない', fontsize=12.5, pad=9)
    fig.savefig(os.path.join(OUT, 'marker_tolerance.png'))
    plt.close(fig)
    return 'marker_tolerance.png'


def fig_margin_ladder():
    """余裕の梯子（9-11）＋制御コスト対照（9-15）。S8-(3) と第4章 4.5.3 の図。

    ⚠️ **①原本（log_train.txt の best）から読む。** 値を手で書かない。
    到達タスクは残存距離のほうが直感的なので、best を 1000 step で割って mm に直す。
    """
    # (ラベル, 余裕 %, seed=0 の run, seed=1 の run)
    ctrl02 = [('0 %', 0.0, 'tripo_pjr199', 'tripo_pjr199_s2'),
              ('5.3 %', 5.3, 'tripo_pjr209', 'tripo_pjr209_s2'),
              ('10 %', 10.0, 'tripo_pj_mid', 'tripo_pj_mid_s2')]
    ctrl00 = [('0 %', 0.0, 'tripo_pjr199_c0', 'tripo_pjr199_c0_s2'),
              ('10 %', 10.0, 'tripo_pj_mid_c0', 'tripo_pj_mid_c0_s2')]

    def resid(run):
        """best exec_R_eps → 平均残存距離 [mm]。Reach 報酬は -dist の 1000 step 積算なので
        best/1000 [m] がそのまま平均残存距離。mm にするので ×1000 と相殺して -best になる。"""
        try:
            _, y = curve(run)
        except FileNotFoundError:
            return None
        return None if len(y) == 0 else float(-np.max(y))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for pts, col, mk, lab in [(ctrl02, '#c0392b', 'o', '制御コスト 0.2（標準）'),
                              (ctrl00, '#2980b9', 's', '制御コスト 0（対照）')]:
        xs, lo, hi = [], [], []
        for _, m, r0, r1 in pts:
            v = [d for d in (resid(r0), resid(r1)) if d is not None]
            if not v:
                continue
            xs.append(m); lo.append(min(v)); hi.append(max(v))
        ax.plot(xs, lo, mk + '-', color=col, label=lab, zorder=3)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.18, zorder=2)
        for x, a, b in zip(xs, lo, hi):
            ax.plot([x, x], [a, b], color=col, lw=2.2, zorder=3)

    ax.set_xlabel('目標に対する余裕 [%]')
    ax.set_ylabel('平均残存距離 [mm]（小さいほど良い）')
    ax.set_title('余裕を広げるほど到達精度は単調に悪化する\n（帯は 2 シードの幅）', fontsize=11)
    ax.set_xticks([0, 5.3, 10]); ax.set_xticklabels(['0', '5.3', '10'])
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.savefig(os.path.join(OUT, 'margin_ladder.png'))
    plt.close(fig)
    return 'margin_ladder.png'


def fig_arm_forms():
    """修論 4.1.1 の図 4.1 ＝ 予備審査論文の図 3: 検証に用いた水平 3 関節アーム（2026-10-22、ユーザー指摘「肝心のロボットの画像の一つもない」）。

    左から (a) 手描きスケッチ（`data/tripo_arm_colorful2/tripo_arm_v2.png`）、(b) M1（Gemini）の出力（同 `tripo_arm_v2.jpg`）、
    (c) M2 が生成したメッシュ（同 GLB、系譜 第5段の v2c の元）を横から、(d) 基本形と比較に用いた 8 形態を上から。
    (a)(b) は 2026-11-19 にユーザー「手描きと gemini の画像を横並びで」で足した（最初に 2 段に積んだら「横並びになってないよ」）。
    (d) のリンク長と初期の太さは学習が読んだ XML（`*_phys_arm171`）から読む。
    (a)〜(c) と (d) は別の図に描いて横に継ぐ（(d) は形態名の幅に合わせて図の幅を変えるので、同じ gridspec に置くと (a)〜(c) まで伸び縮みする）。
    間に白い列を挟む（make_yobi_figures.py の crop_arm_forms は、いちばん右の白い列から右を (d) として切る）。
    以前は (a) メッシュと (b) 形態を横に並べており、(a) の題「（横から）」が (b) の上まではみ出していた（2026-11-19 ユーザー指摘）。
    """
    import io
    # 紙面の文字を約 8〜9 pt にする倍率（11-19、ユーザー「またここの文字が小さいよ」。チェックリスト 8 の基準:
    # 紙面の pt ＝ 図中の pt × 6.3 ÷ 図の幅〔インチ〕。倍率 1 では (a)(b) を足して横に長くなり、約 5.2 pt だった）
    # 枠（図の大きさ）は変えず文字だけを大きくする。枠も大きくすると図の幅が伸び、紙面では同じだけ縮むので効かない（11-19 に倍率 1.0〜1.4 で測った）。
    # 2.0 では (d) の行の間が足りず形態名と腕が重なった。11-19 は 1.6 で紙面の題 7.7 pt・形態名 7.0 pt だった。
    # 11-20（ユーザー「図 3 の文字を 8〜9 pt にする」）: (d) の形態名を短くし（図の幅が縮む）、長さの値を 1 列にそろえ、
    # 注記を 1 行にして、1.8 で題 8.9 pt・形態名と注記 8.1 pt
    FS = 1.8
    fs0 = plt.rcParams['font.size']
    plt.rcParams['font.size'] = 11 * FS
    import xml.etree.ElementTree as ET
    from PIL import Image
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import glb_to_links as G
    D = os.path.join(ROOT, 'data/tripo_arm_colorful2')
    mesh = G._apply_yup_zup(G._load_concat(os.path.join(D, 'colorful stacking rod 3d model.glb')))
    V = np.asarray(mesh.vertices); C = np.asarray(mesh.visual.vertex_colors[:, :3]) / 255
    idx = np.random.default_rng(0).choice(len(V), min(30000, len(V)), replace=False)
    k = 0 if np.ptp(V[:, 0]) >= np.ptp(V[:, 1]) else 1

    # 基本形（1.0 倍）は学習に使っていないので XML が無い。短腕（0.5 倍）の長さを 2 倍して描く
    # （太さは倍率で変えない。どの形も 0.045 / 0.0326 / 0.032。2026-11-13 ユーザーの指摘「図 4.1 に基本形を載せる」）。
    # 名前は短くする（11-20、ユーザー「図 3 の文字を 8〜9 pt にする」。名前の幅がそのまま (d) の幅になり、紙面で縮む）。
    # 「比較には不使用」と、修正指針の倍率をどのタスクに使ったか（1.99 倍は到達、2.11・2.36 倍は押し）は本文で言う（予 4.1・修論 4.1.1・4.1.2）。
    # 名前は修論と予備審査論文の表と同じ「修正指針の N 倍」（同じ形を 2 つの名前で呼ばない。判定 md の型 54）
    forms = [('基本形（1.0 倍）', 'base'), ('短腕（0.5 倍）', 'short'), ('中腕（1.1 倍）', 'mid'), ('長腕（1.8 倍）', 'long'),
             ('根元重（配分）', 'prox'), ('先端重（配分）', 'dist'),
             ('修正指針の 1.99 倍', 'rec199'), ('修正指針の 2.11 倍', 'rec211'), ('修正指針の 2.36 倍', 'rec236')]
    col = ['#888888', '#d94040', '#3070d0', '#30a050']        # 台座・1・2・3 本目

    def links(n):
        src, k = ('short', 2.0) if n == 'base' else (n, 1.0)
        body = ET.parse(os.path.join(ROOT, f'assets/mujoco_envs/tripo_arm_v2c_pj_{src}_phys_arm171.xml')).getroot().find('worldbody/body')
        out = []
        for g in body.iter('geom'):
            if g.get('fromto') is None:
                continue
            ft = list(map(float, g.get('fromto').split()))
            out.append((k * float(np.linalg.norm(np.subtract(ft[3:], ft[:3]))), float(g.get('size'))))
        return out

    def render(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert('RGB')

    # ── (d) 形態 ──
    X_VAL = 1.52   # 長さの値の列（長腕の先 1.451 m とカプセルの半径の右）
    fig, a = plt.subplots(figsize=(8.0, 5.6))
    for row, (lab, n) in enumerate(forms):
        y = -row * 0.3
        a.add_patch(plt.Circle((0, y), 0.1, color=col[0], alpha=0.6))
        x, j = 0.0, 1
        for L, r in links(n):
            a.add_patch(Rectangle((x, y - r), L, 2 * r, color=col[j], alpha=0.8, lw=0))
            for q in (x, x + L):
                a.add_patch(plt.Circle((q, y), r, color=col[j], alpha=0.8, lw=0))
            a.plot(x, y, 'k.', ms=4)
            x += L; j += 1
        a.text(-0.14, y, lab, ha='right', va='center', fontsize=10 * FS)
        a.text(X_VAL, y, f'{x:.3f} m', va='center', fontsize=10 * FS)   # 値は 1 列にそろえる（腕の先に置くと 0.80・0.85 の破線と交差した）
    for xv, lab in ((0.80, '目標 0.80 m'), (0.85, '箱 0.85 m')):   # 箱は腕に面した側面の位置（本文で言う）
        a.axvline(xv, color='k', ls='--' if xv == 0.80 else ':', lw=1)
        a.text(xv + (-0.02 if xv == 0.80 else 0.02), 0.2, lab, ha='right' if xv == 0.80 else 'left', va='center', fontsize=10 * FS)
    a.set_ylim(-(len(forms) - 1) * 0.3 - 0.2, 0.38); a.set_aspect('equal')   # 上は目標と箱の注記の 1 行ぶん
    a.set_yticks([]); a.set_xticks([0, 0.5, 1.0, 1.5]); a.set_xlabel('x [m]（根元の関節から。全関節 0°）')
    a.set_title('(d) 基本形と比較に用いた形態（上から）', fontsize=11 * FS)
    # 枠の左端は形態名の実際の幅から決める（11-19、ユーザー「説明が図の線からはみ出ている…文字を包み込める範囲の枠線の長さに」）。
    # 以前は −1.0 の決め打ちで、長い名前（基本形・修正指針の N 倍）が枠の外へ出ていた。
    # 縦横比を固定しているので、x の幅を広げた分だけ図の幅も広げ、腕と文字の大きさを保つ。
    x_right = 2.27   # 値の列「1.451 m」が枠に収まる右端（値の幅は約 0.66 m 分。2.20 では枠の線に触れた）
    a.set_xlim(-1.0, x_right)
    fig.tight_layout()
    labels = [t for t in a.texts if t.get_ha() == 'right' and t.get_position()[0] == -0.14]
    for _ in range(4):
        fig.canvas.draw()
        inv = a.transData.inverted()
        x_min = min(inv.transform((t.get_window_extent().x0, 0))[0] for t in labels)
        left = x_min - 0.06
        if abs(left - a.get_xlim()[0]) < 0.01:
            break
        grow = (x_right - left) / (x_right - a.get_xlim()[0])
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * grow, h)
        a.set_xlim(left, x_right)
        fig.tight_layout()
    right = render(fig)

    # ── (a) 手描き (b) M1 の出力 (c) メッシュ。(d) と同じ高さの図に、上端をそろえて置く ──
    sketch = Image.open(os.path.join(D, 'tripo_arm_v2.png')).convert('RGB')
    m1 = Image.open(os.path.join(D, 'tripo_arm_v2.jpg')).convert('RGB')
    g = np.asarray(m1).min(axis=2) < 235                      # M1 の出力は白地。周りの余白を落とす
    ys, xs = np.where(g); pad = 8
    m1 = m1.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, m1.width), min(ys.max() + pad, m1.height)))
    zr = (V[:, 2].min() - 0.03, V[:, 2].max() + 0.03); xr = (-0.13, 0.13)
    # (a)(b) を (c) と同じ高さにする幅（11-19、ユーザー「cd と同じサイズにまで拡大したら」）。(c) の枠は目盛りと題の幅だけ見込む。
    # 6.4 インチでは (a)(b) が (c) の 2/3 の高さだった。7.4 で (c) の枠とほぼ同じ高さ（6.8 では短くなる）
    ratios = [sketch.width / sketch.height, m1.width / m1.height, 0.40]
    fig, ax = plt.subplots(1, 3, figsize=(7.4, right.height / 200), gridspec_kw={'width_ratios': ratios})
    # 題は枠の幅に収まる長さで折る（11-19、ユーザー「（横から）という文字が (b) まで伸びている」）
    # (a) だけ題が 1 行だと、題と絵が (b)〜(d) より下にずれる（11-26、ユーザー「(a)と(b)~(d)の高さ位置がズレている」）。題はすべて 2 行にそろえる。
    # (b) は「M1」と書かない（予備審査論文は工程を ①〜④ の段の名前で呼び、M 番号を使わない。11-26 ユーザー）
    for a, im, t in ((ax[0], sketch, '(a) 手描き\nスケッチ'), (ax[1], m1, '(b) 画像整形の\n出力（Gemini）')):
        a.imshow(im); a.set_axis_off(); a.set_title(t, fontsize=11 * FS)
    a = ax[2]
    a.scatter(V[idx, k], V[idx, 2], c=C[idx], s=0.25)
    a.set_xlim(*xr); a.set_ylim(*zr); a.set_aspect('equal')
    a.set_xticks([-0.1, 0.1]); a.set_xlabel('横 [m]'); a.set_ylabel('高さ [m]')
    a.set_title('(c) 生成メッシュ\n（横から）', fontsize=11 * FS)
    for a in ax:
        a.set_anchor('N')                                     # 題の高さをそろえる
    fig.tight_layout(w_pad=2.0)
    left = render(fig)

    gap = 60
    out = Image.new('RGB', (left.width + gap + right.width, max(left.height, right.height)), 'white')
    out.paste(left, (0, 0))
    out.paste(right, (left.width + gap, 0))
    out.save(os.path.join(OUT, 'arm_forms.png'), dpi=(200, 200))
    plt.rcParams['font.size'] = fs0
    k_paper = 6.3 / (out.width / 200)                         # 本文の幅 1.0 で貼ったときの縮み
    print(f'  arm_forms: 幅 {out.width / 200:.1f} インチ → 紙面の文字 題 {11 * FS * k_paper:.1f} pt・形態名 {10 * FS * k_paper:.1f} pt・最小 {10 * FS * k_paper:.1f} pt')
    return 'arm_forms.png'


# ── 図: R4 の確定エポック（4.4.4、系譜 9-338） ──────────────────────────────
#   値は rank_settle.py の settle_epoch（running best・「以降一度も最終順位から外れない最小のエポック」）で
#   ①原本（log_train.txt）から数える。較正に使ったのは seed 0・1（seed 1 は新しい種の *_s2_ws）。
#   seed 2（*_s3）は較正に使っていない（9-338 で事前登録して確かめた）。
def fig_r4_settle():
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import rank_settle as rs
    rs.RUN_DIR = RUN

    def ws(r):
        return r + '_ws' if r.endswith('_s2') and os.path.isdir(os.path.join(RUN, r + '_ws')) else r

    def vp(r):
        """押しと縦型は Bug 106・107 を直した物理の取り直し（系譜 9-351 の `_vp`）に替える。短腕は箱に触れないので元の run"""
        if r.startswith(('tripo_pjp_', 'tripo_pjdp_', 'e2e_a1v_')) and 'short' not in r \
                and os.path.isdir(os.path.join(RUN, r + '_vp')):
            return r + '_vp'
        return r

    def m(runs):
        return [vp(ws(r)) for r in runs]

    def settle(runs):
        ep, _, _, _, _, _ = rs.settle_epoch(runs, runs)
        return ep

    def wrong_at(runs, e=30):
        """第 e エポックの running best の順位が最終順位と逆か（2 run の対だけ）。"""
        a, b = (rs.running_best(r) for r in runs)
        last = max(set(a) & set(b))
        return (a[e] > b[e]) != (a[last] > b[last])

    # 形が違う対（コホート B）: (行の名前, seed 0 の対, seed 1 の対, 較正に使っていない対の並び)
    #   押しの配分は seed 2〜4（*_s3・_s4・_s5）を持つ（系譜 9-351 の 57-8 の記録）
    B = [('到達: 中腕 対 長腕', ('tripo_pj_mid_phys', 'tripo_pj_long_phys'),
          ('tripo_pj_mid_phys_s2', 'tripo_pj_long_phys_s2'), [('tripo_pj_mid_phys_s3', 'tripo_pj_long_phys_s3')]),
         ('押し: 長腕 対 中腕', ('tripo_pjp_long_phys', 'tripo_pjp_mid_phys'),
          ('tripo_pjp_long_phys_s2', 'tripo_pjp_mid_phys_s2'), [('tripo_pjp_long_phys_s3', 'tripo_pjp_mid_phys_s3')]),
         ('到達: 根元重 対 先端重', ('tripo_pjd_prox_phys', 'tripo_pjd_dist_phys_cg'),
          ('tripo_pjd_prox_phys_s2', 'tripo_pjd_dist_phys_cg_s2'), [('tripo_pjd_prox_phys_s3', 'tripo_pjd_dist_phys_cg_s3')]),
         ('押し: 根元重 対 先端重', ('tripo_pjdp_prox_phys', 'tripo_pjdp_dist_phys_cg'),
          ('tripo_pjdp_prox_phys_s2', 'tripo_pjdp_dist_phys_s2'),
          [('tripo_pjdp_prox_phys_s3', 'tripo_pjdp_dist_phys_cg_s3'), ('tripo_pjdp_prox_phys_s4', 'tripo_pjdp_dist_phys_cg_s4'),
           ('tripo_pjdp_prox_phys_s5', 'tripo_pjdp_dist_phys_cg_s5')]),
         ('到達: A1 可動 対 関節 3 固定', ('e2e_a1_reach_phys', 'e2e_a1_fix3_reach_phys'),
          ('e2e_a1_reach_phys_s2', 'e2e_a1_fix3_reach_phys_s2'), [('e2e_a1_reach_phys_s3', 'e2e_a1_fix3_reach_phys_s3')])]
    SHORT = [('tripo_pj_long_phys', 'tripo_pj_short_phys_cg'), ('tripo_pj_mid_phys', 'tripo_pj_short_phys_cg'),
             ('tripo_pjp_long_phys', 'tripo_pjp_short_phys_cg'), ('tripo_pjp_mid_phys', 'tripo_pjp_short_phys_cg'),
             ('tripo_pjr199_phys', 'tripo_pj_short_phys_cg')]
    # 乱数だけ違う組（コホート A）: rank_settle の COHORTS のうち名前が「A」で始まるもの（seed 1 は *_s2_ws）
    A = [m(runs) for runs, _, name in rs.COHORTS if name.startswith('A')]

    rows = []   # (名前, [(エポック, 印の種類, 第30で逆か)])
    for name, p0, p1, held in B:
        pts = [(settle(m(p0)), 'cal', wrong_at(m(p0))), (settle(m(p1)), 'cal', wrong_at(m(p1)))]
        pts += [(settle(m(p)), 'held', wrong_at(m(p))) for p in held]
        rows.append((name, pts))
    rows.append(('短腕との対（5 対、seed 0）', [(settle(m(p)), 'cal', wrong_at(m(p))) for p in SHORT]))
    rows.append((f'乱数だけ違う組（{len(A)} 組）', [(settle(r), 'rand', False) for r in A]))
    for name, pts in rows:
        print(f'[r4] {name}: ' + ', '.join(f'{e}{"（第30で逆）" if w else ""}' for e, _, w in pts))

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ys = np.arange(len(rows))[::-1]
    jit = {'cal': -0.12, 'held': 0.16, 'rand': 0.0}
    for (name, pts), y in zip(rows, ys):
        for k, (e, kind, wrong) in enumerate(pts):
            dy = jit[kind] + (0.06 * (k % 2) if kind == 'cal' and len(pts) > 3 else 0)
            if kind == 'rand':
                ax.plot(e, y + (k % 3 - 1) * 0.12, 'o', ms=6.5, mfc='#bbbbbb', mec='#777777', zorder=3)
            elif kind == 'cal':
                ax.plot(e, y + dy, 'o', ms=7, mfc=C_MID, mec='#1d5f8c', zorder=3)
            else:
                ax.plot(e, y + dy, 'D', ms=7.5, mfc='white', mec='#c0392b', mew=1.8, zorder=4)
                if wrong:
                    ax.annotate('第 30 エポックでは\n順位が最終と逆', (e, y + dy), xytext=(9, -3), textcoords='offset points',
                                fontsize=9, color='#c0392b', va='top')
    ax.axvline(30, color='k', ls='--', lw=1.1)
    ax.text(30 * 0.95, ys[-1] - 0.45, '判定則の 30 エポック', fontsize=9.5, ha='right', va='center')
    ax.axhline(ys[-1] + 0.5, color='#999999', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlim(0.8, 260)
    ax.set_xticks([1, 3, 10, 30, 100, 200])
    ax.set_xticklabels(['1', '3', '10', '30', '100', '200'])
    ax.set_yticks(ys)
    ax.set_yticklabels([n for n, _ in rows], fontsize=10.5)
    ax.set_ylim(ys[-1] - 0.6, ys[0] + 0.9)
    ax.set_xlabel('確定エポック（以降、最終順位から一度も外れない最小のエポック、対数軸）')
    hs = [plt.Line2D([], [], ls='', marker='o', ms=7, mfc=C_MID, mec='#1d5f8c', label='形が違う対（較正に使った seed 0・1）'),
          plt.Line2D([], [], ls='', marker='D', ms=7.5, mfc='white', mec='#c0392b', mew=1.8, label='形が違う対（較正に使っていない seed 2〜4）'),
          plt.Line2D([], [], ls='', marker='o', ms=6.5, mfc='#bbbbbb', mec='#777777', label='乱数だけ違う組（同じ形・別の seed）')]
    ax.legend(handles=hs, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=9.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'r4_settle.png'))
    plt.close(fig)
    return 'r4_settle.png'


# ── 図: 軸6 の目標位置までの押し（修論 4.7.4、系譜 9-346） ──────────────────────
#   2026-11-13 ユーザー「修論に乗せていいよ」（11-06「印は映えを意識して星マークで」）。
#   値は各 run の best を 50 話再生した `trace/arm_trace_50.npz`（9-346 の再生。表 4.15 の d_end の中央値・0.03 m 以内の数を
#   この関数が出力に出すので、表と一致するかを見る）。上から見た図: 星＝目標（当たり判定なし）、灰の破線＝箱の初期位置、
#   青＝50 話の箱の軌跡と最終位置、赤＝d_end が中央値の話の手先の軌跡、腕＝その話で箱が初めて 2 mm 動く 1 step 前の当たり判定
#   （箱に触れない seed 0 は最後の step）。代表の話は d_end が中央値から 5 mm 以内の最初の話。
def fig_target_star():
    runs = [('シード 0', 'ra15_tp_rate_beta_ext400'), ('シード 1', 'ra15_tp_rate_beta_s2_ext400'),
            ('シード 2', 'ra15_tp_rate_beta_s3_ext400')]
    half = 0.03                                     # 箱の半辺（1 辺 60 mm、表 4.14）
    gold, blue, red, arm = '#e6b800', '#2f6db5', '#c0392b', '#555555'

    def stadium(p0, p1, r, n=24):
        d = p1 - p0; L = np.linalg.norm(d)
        u = d / L if L > 1e-9 else np.array([1.0, 0.0]); v = np.array([-u[1], u[0]])
        a = np.linspace(-np.pi / 2, np.pi / 2, n)
        e1 = p1 + r * (np.outer(np.cos(a), u) + np.outer(np.sin(a), v))
        e0 = p0 + r * (np.outer(-np.cos(a), u) + np.outer(-np.sin(a), v))
        return np.vstack([e1, e0, e1[:1]])

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.3), sharey=True)
    for ax, (lab, run) in zip(axes, runs):
        z = np.load(os.path.join(ROOT, 'single_run', run, 'trace', 'arm_trace_50.npz'), allow_pickle=True)
        cube = np.asarray(z['ep_cubes'], float); tgt = np.asarray(z['target'], float)
        xp = np.asarray(z['ep_xpos'], float); xm = np.asarray(z['ep_xmat'], float)
        bone = np.asarray(z['ep_bone'], float); cap = np.asarray(z['ep_cap'], float); rad = np.asarray(z['ep_geom'], float)
        dend = np.hypot(cube[:, -1, 0] - tgt[0], cube[:, -1, 1] - tgt[1])
        print(f'  {run}: 50 話 d_end 中央値 {np.median(dend):.3f} m・0.03 m 以内 {(dend < 0.03).sum()} / {len(dend)}'
              f'（表 4.15 と照らす）')
        # 代表の話: d_end が中央値から 5 mm 以内の最初の話（11-13: 中央値に最も近い話を選ぶと、seed 0 では少数側〔腕を +y へ倒す 12 / 50 話〕を引いた）
        e = int(np.where(np.abs(dend - np.median(dend)) <= 0.005)[0][0])
        side = np.sign(xp[:, -1, -1, 1]); print(f'    代表は話 {e}。最後の step で腕の先が +y 側 {(side > 0).sum()}・−y 側 {(side < 0).sum()} 話')
        moved = np.where(np.hypot(*(cube[e, :, :2] - cube[e, 0, :2]).T) > 0.002)[0]
        k = max(int(moved[0]) - 1, 0) if len(moved) else len(cube[e]) - 1
        tip = xp[e, :, -1] + np.einsum('tij,j->ti', xm[e, :, -1], bone[e, -1])
        # 箱の初期位置（50 話の中央）・箱の軌跡・最終位置
        c0 = np.median(cube[:, 0, :2], axis=0)
        ax.add_patch(Rectangle((c0[0] - half, c0[1] - half), 2 * half, 2 * half, fill=False, ec='#888888', ls='--', lw=1.2))
        for i in range(len(cube)):
            ax.plot(cube[i, :, 0], cube[i, :, 1], color=blue, lw=0.6, alpha=0.25)
        ax.plot(cube[:, -1, 0], cube[:, -1, 1], 'o', ms=3, color=blue, alpha=0.7, zorder=6)
        # 腕（当てる直前）と手先の軌跡
        for j in range(1, xp.shape[2]):
            p0 = xp[e, k, j] + xm[e, k, j] @ cap[e, j, :3]; p1 = xp[e, k, j] + xm[e, k, j] @ cap[e, j, 3:]
            if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
                p0, p1 = xp[e, k, j], xp[e, k, j] + xm[e, k, j] @ bone[e, j]
            s = stadium(p0[:2], p1[:2], rad[e, j])
            ax.fill(s[:, 0], s[:, 1], color=arm, alpha=0.35, lw=0)
            ax.plot(s[:, 0], s[:, 1], color=arm, lw=0.8)
        ax.plot(tip[:, 0], tip[:, 1], color=red, lw=1.2)
        ax.plot(0, 0, 'k+', ms=9, mew=1.5)
        # 目標（星）と 0.03 m の範囲
        ax.add_patch(plt.Circle(tgt[:2], 0.03, fill=False, ec='#8a6d00', ls=':', lw=1.2, zorder=5))
        # 2026-11-17: 星（ms=24 ≒ 0.068 m）が 0.03 m の円と最終位置の点を隠していた。円より小さくし、点を星の上に描く
        ax.plot(tgt[0], tgt[1], marker='*', ms=11, mfc=gold, mec='#8a6d00', mew=0.8, zorder=5)
        ax.set_title(lab, fontsize=11)
        ax.set_aspect('equal'); ax.set_xlim(-0.12, 0.48); ax.set_ylim(-0.36, 0.20)
        ax.set_xlabel('x [m]')
    axes[0].set_ylabel('y [m]')
    hs = [plt.Line2D([], [], ls='', marker='*', ms=11, mfc=gold, mec='#8a6d00', label='目標位置（点線の円は 0.03 m 以内）'),
          Rectangle((0, 0), 1, 1, fill=False, ec='#888888', ls='--', label='キューブの初期位置'),
          plt.Line2D([], [], color=blue, lw=1, marker='o', ms=3, label='キューブの軌跡と最終位置（50 試行）'),
          plt.Line2D([], [], color=red, lw=1.2, label='手先の軌跡（代表の 1 試行）'),
          Rectangle((0, 0), 1, 1, fc=arm, alpha=0.35, ec=arm, label='腕の姿勢（代表の 1 試行）')]
    fig.legend(handles=hs, loc='lower center', ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(OUT, 'ra_target_star.png'))
    plt.close(fig)
    return 'ra_target_star.png'


# ── 図: 学習曲線（シードの帯。修論第4章・予備審査の 2 文書、2026-11-24 ユーザー判断） ─────────────────
#   量は判定規則と同じ「そのエポックまでの最良エピソード報酬」（running best of exec_R_eps）。シードごとの細い線と、
#   シード間の最小〜最大の帯（判定規則「シード間の範囲が重ならない」と同じ。標準誤差の帯は判定規則と食い違うので使わない）。
#   第 30 エポックに縦線（判定に要する学習量、修論 4.4.4）。
#   ⛔ 押しの 2 枚は取り直し（系譜 9-351）の `_vp` を使うので、judge_reading57.py と同じ前提（47 本すべてが完走し、
#   目視記録に行がある）がそろうまで描かない（報酬を読み 57 の判定より前に見ないため）。そろう前は枠だけを出す。
C_PROX, C_DIST = '#8e44ad', '#e67e22'


def fig_learning_curves():
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import judge_reading57 as j57
    seen = j57.visual_names()
    push_ok, push_st, _ = j57.ready([n + '_vp' for n in sorted(j57.ALL_SRCS)], seen)
    S = lambda base, *suf: [base + s for s in suf]  # noqa: E731
    V = lambda names: [n + '_vp' for n in names]  # noqa: E731
    panels = [  # (題, 取り直しか, [(形の名前, 色, runs)])。シード数は凡例に書く
        ('到達: 中腕 対 長腕', False,
         [('中腕', C_MID, S('tripo_pj_mid_phys', '', '_s2_ws', '_s3')),
          ('長腕', C_LONG, S('tripo_pj_long_phys', '', '_s2_ws', '_s3'))]),
        ('押し: 中腕 対 長腕', True,
         [('中腕', C_MID, V(j57.MID)), ('長腕', C_LONG, V(j57.LONG))]),
        ('押し: 根元重 対 先端重', True,
         [('根元重', C_PROX, V(j57.PROX)), ('先端重', C_DIST, V(j57.DIST))]),
    ]
    # 2026-11-25: 13.2 in では版面（6.30 in）で文字が約 5 pt になったので 9.0 in に縮め、文字を上げた
    #   （刷り上がりで 8 pt 以上。予備審査論文の最終確認チェックリスト 20-2 の「図の文字は 8〜9 pt」）
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.7))
    for ax, (title, retake, groups) in zip(axes, panels):
        ax.set_title(title, fontsize=13, pad=6)
        ax.set_xlabel('エポック', fontsize=12)
        ax.tick_params(labelsize=11.5)
        if retake and not push_ok:
            ax.text(0.5, 0.5, f'読み 57 の判定の後に描く\n（{push_st}）', transform=ax.transAxes,
                    ha='center', va='center', fontsize=11, color='#777777')
            ax.set_xticks([])
            ax.set_yticks([])
            print(f'[curves] {title}: ⏳ 描かない（{push_st}）')
            continue
        for name, col, runs in groups:
            cs = [curve(r) for r in runs]
            last = min(int(e[-1]) for e, _ in cs)
            ep = np.arange(last + 1)
            ys = []
            for (e, y), r in zip(cs, runs):
                rb = running_best(y)
                yy = np.interp(ep, e, rb)
                ys.append(yy)
                ax.plot(ep, yy, color=col, lw=0.9, alpha=0.75)
            ys = np.array(ys)
            ax.fill_between(ep, ys.min(0), ys.max(0), color=col, alpha=0.22, lw=0, label=f'{name}（{len(runs)} シード）')
            print(f'[curves] {title} {name}: 第 30 エポック {ys[:, 30].min():.2f}〜{ys[:, 30].max():.2f}・'
                  f'第 {last} エポック {ys[:, -1].min():.2f}〜{ys[:, -1].max():.2f}')
        ax.axvline(30, color='k', ls='--', lw=1.0)
        # 「30」は文字で置くと曲線か凡例に重なるので、横軸の目盛りに入れる（下の set_xticks）
        ax.set_yscale('symlog', linthresh=1)
        # 縦軸はデータの範囲に合わせる（到達は全部負なので、正の側を空けない）。目盛りは 10^n でなく素の数で書く
        lo, hi = ax.dataLim.y0, ax.dataLim.y1
        ax.set_ylim(lo * 1.4 if lo < 0 else -1, hi * 0.6 if hi < -1 else hi * 1.6 + 1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.set_xlim(0, 200)
        ax.set_xticks([0, 30, 100, 200])
        ax.legend(loc='lower right', fontsize=11.5, frameon=False, handlelength=1.2, borderaxespad=0.2)
    axes[0].set_ylabel('その時点までの最良エピソード報酬', fontsize=12)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(OUT, 'learning_curves.png'))
    plt.close(fig)
    return 'learning_curves.png'


# ── 報酬の帯の幅と、収束した設計変数の seed 間の幅（2026-11-25、系譜 9-357・9-358。スライド・想定問答 Q85 用で修論には載せない） ──
def _var_spread_morph(run):
    """best の再生（trace/arm_trace.npz）から ギア比・半径・後方延長（関節 1〜3）を出す。後方延長はカプセルの始点の x の負の分で近似。"""
    z = np.load(os.path.join(ROOT, 'single_run', run, 'trace', 'arm_trace.npz'), allow_pickle=True)
    g = np.array(z['ep_gear'][0], float)[1:4]
    r = np.array(z['ep_geom'][0], float)[1:4]
    e = np.clip(-np.array(z['ep_cap'][0], float)[1:4, 0], 0, None)
    raw = np.concatenate([g, r, e])
    norm = np.concatenate([(g - 20) / 130, (r - 0.03) / 0.07, e / 0.2])
    return raw, norm


def fig_var_spread():
    """左に報酬（best、3 seed の点と最小〜最大の帯）、右に設計変数 9 個の 3 seed の位置（探索範囲を下限〜上限にそろえる）。
    結果の幅と変数の幅が対応しないこと（9-357）、到達の中腕の関節 1 は差し替えると両向きで下がること（9-358）を見せる。"""
    CONDS = [
        ('到達・中腕', ['tripo_pj_mid_phys', 'tripo_pj_mid_phys_s2_ws', 'tripo_pj_mid_phys_s3']),
        ('到達・長腕', ['tripo_pj_long_phys', 'tripo_pj_long_phys_s2_ws', 'tripo_pj_long_phys_s3']),
        ('押し・中腕', ['tripo_pjp_mid_phys_vp', 'tripo_pjp_mid_phys_s2_ws_vp', 'tripo_pjp_mid_phys_s3_vp']),
        ('押し・長腕', ['tripo_pjp_long_phys_vp', 'tripo_pjp_long_phys_s2_ws_vp', 'tripo_pjp_long_phys_s3_vp']),
    ]
    SEED_C = ['#0072B2', '#D55E00', '#009E73']
    SEED_M = ['o', 's', '^']
    VARS = ['関節1', '関節2', '関節3'] * 3
    GROUPS = [('ギア比\n20〜150 N·m', 20, 150), ('半径（太さ）\n0.03〜0.10 m', 0.03, 0.10), ('後方延長（近似）\n0〜0.2 m', 0.0, 0.2)]
    morph = _var_spread_morph
    fig = plt.figure(figsize=(11.5, 11.0))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.0, 2.5], wspace=0.18, hspace=0.95)
    rew = {}
    for k, (name, runs) in enumerate(CONDS):
        rew[name] = [running_best(curve(r)[1])[-1] for r in runs]
    xl = {}
    for t in ('到達', '押し'):
        v = sum([rew[n] for n in rew if n.startswith(t)], [])
        pad = 0.12 * (max(v) - min(v))
        xl[t] = (min(v) - pad, max(v) + pad)

    for k, (name, runs) in enumerate(CONDS):
        # 左: 報酬（best）。同じタスクの中腕と長腕は同じ横軸
        ax = fig.add_subplot(gs[k, 0])
        v = rew[name]
        ax.axvspan(min(v), max(v), color='0.5', alpha=0.18, lw=0)
        for s, x in enumerate(v):
            ax.plot(x, 0.35 - 0.35 * s, SEED_M[s], color=SEED_C[s], ms=9, mec='white', mew=1.0, zorder=3)  # 重なる値が見えるよう seed ごとに縦をずらす
        ax.set_xlim(*xl[name[:2]])
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.grid(axis='y', visible=False)
        ax.set_title(f'{name}　報酬（best）', loc='left', fontsize=11, fontweight='bold')
        ax.text(0.98, 0.80, f'幅 {max(v) - min(v):.2f}', transform=ax.transAxes, ha='right', fontsize=10, color='0.25')
        if name == '押し・長腕':
            ax.annotate('50 話の中央は 17.42\n（5 話だけ箱が 4〜19 m 滑る）', xy=(v[0], 0.35), xytext=(v[0] - 1, -0.76),
                        ha='right', va='center', fontsize=8.5, color='0.3',
                        arrowprops=dict(arrowstyle='-', color='0.5', lw=0.8))
        # 右: 設計変数（探索範囲を 0〜1 にそろえた位置）
        ax2 = fig.add_subplot(gs[k, 1])
        data = [morph(r) for r in runs]
        norm = np.array([d[1] for d in data])
        raw = np.array([d[0] for d in data])
        for j in range(9):
            xj = j + (j // 3) * 0.6
            lo, hi = norm[:, j].min(), norm[:, j].max()
            ax2.plot([xj, xj], [lo, hi], color='0.55', lw=5, alpha=0.35, solid_capstyle='round', zorder=1)
            for s in range(3):
                ax2.plot(xj + (s - 1) * 0.16, norm[s, j], SEED_M[s], color=SEED_C[s], ms=7, mec='white', mew=0.8, zorder=3)
            if hi - lo >= 0.45:
                ax2.text(xj, 1.13, f'{(hi - lo) * 100:.0f} %', ha='center', fontsize=8.5, color='0.2', fontweight='bold')
        ax2.set_ylim(-0.08, 1.25)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['下限', '上限'], fontsize=9)
        ticks = [j + (j // 3) * 0.6 for j in range(9)]
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(VARS, fontsize=9)  # どの行でも列の意味が分かるよう、関節名と変数名を全行に付ける
        for g, (lab, _, _) in enumerate(GROUPS):
            ax2.text(g * 3.6 + 1, -0.27, lab.split('\n')[0], ha='center', va='top', fontsize=9.5, fontweight='bold', color='0.25',
                     transform=ax2.get_xaxis_transform())
        ax2.set_xlim(-0.6, ticks[-1] + 0.6)
        ax2.grid(axis='x', visible=False)
        for g in range(1, 3):
            ax2.axvline(g * 3.6 - 0.9, color='0.8', lw=0.8)
        if k == 0:
            for g, (lab, _, _) in enumerate(GROUPS):
                ax2.text(g * 3.6 + 1, 1.55, lab, ha='center', va='bottom', fontsize=9.5)
            ax2.set_title('設計変数（3 seed の収束値。縦軸は探索範囲の中の位置）', loc='left', fontsize=11, fontweight='bold', pad=46)
        if name == '到達・中腕':
            ax2.annotate('差し替えると両向きで\n報酬が下がる（9-358）', xy=(0, 0.5), xytext=(0.45, 0.62),
                         fontsize=8.5, color='0.3', va='center',
                         arrowprops=dict(arrowstyle='-', color='0.5', lw=0.8))

    handles = [plt.Line2D([], [], marker=SEED_M[s], color=SEED_C[s], ls='', ms=8, label=f'seed {s}') for s in range(3)]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, -0.04, '灰色の帯・棒 ＝ 3 seed の最小〜最大。% は探索範囲に対する幅（45 % 以上だけ表示）。'
             '後方延長はカプセルの始点の x から出した近似。リンクの長さは設計変数ではない（スケッチで決まる）。',
             ha='center', fontsize=8.5, color='0.35')
    
    out = os.path.join(os.path.dirname(OUT), 'スライド用', 'var_spread.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out



def fig_r1r2_dots():
    """修論 4.4.1・4.4.2 の図: 形態最適化の効果（R1）と、タスクによる収束形態（R2）のドットプロット（2026-11-26、ユーザー「ドットプロットという名前があるなら入れよう」）。

    点は各シード、帯はシード間の最小〜最大（図 4.4 の帯版と同じ描き方。Cleveland のドットプロット）。
    値は系譜 9-351 の読み 57-1・57-5 の表から写す（ここで新しい値を計算しない）。
    (a) R1: 押し・中腕の最良エピソード報酬（形態を最適化 対 凍結、各 3 シード）
    (b)(c) R2: 縦型 4 関節アームの収束形態（到達 対 押し、各 5 シード）のリンク長の合計と、第 4 関節のギア比
    """
    C1, C2 = '#2a78d6', '#eb6834'
    panels = [
        ('(a) 形態最適化の効果（押し）', '最良エピソード報酬（右ほど良い）',
         [('形態を最適化', [8.63, 8.38, 8.46], C1), ('形態を凍結', [6.03, 5.93, 5.79], C2)]),
        ('(b) 総リーチ', '総リーチ [m]',
         [('到達', [1.066, 1.145, 1.831, 1.158, 1.158], C1), ('押し', [1.750, 1.688, 2.822, 3.112, 2.842], C2)]),
        ('(c) 第 4 関節のギア比', 'ギア比 [N·m]',
         [('到達', [140.0, 117.2, 103.3, 40.0, 41.8], C1), ('押し', [148.5, 149.9, 115.2, 148.1, 150.0], C2)]),
    ]
    # 紙面の字を 8 pt 以上に（チェックリスト 8: 紙面の pt ＝ 図中の pt × 6.3 ÷ 図の幅〔インチ〕）。幅 8.4 で 11 pt → 8.3 pt
    fs0 = plt.rcParams['font.size']
    plt.rcParams['font.size'] = 11
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.9))
    for ax, (title, xlabel, rows) in zip(axs, panels):
        allv = [v for _, vv, _ in rows for v in vv]
        span = max(allv) - min(allv)
        xmin, xmax = min(allv) - 0.15 * span, max(allv) + 0.15 * span
        for k, (lab, vv, c) in enumerate(rows):
            y = 1 - k
            lo, hi = min(vv), max(vv)
            ax.barh(y, hi - lo, left=lo, height=0.40, color=c, alpha=0.22, edgecolor=c, lw=1.2, zorder=2)
            ax.scatter(vv, [y] * len(vv), s=46, color=c, edgecolor='white', lw=1.2, zorder=4)
            nd = 2 if title.startswith('(a)') else 3 if title.startswith('(b)') else 1   # 表 4.5・本文と同じ桁
            # 帯が枠の端に近いときは、数字を帯の端にそろえて枠の内側に置く（枠の線をまたがせない）
            mid = (lo + hi) / 2
            if mid > xmax - 0.3 * (xmax - xmin):
                tx, ha = hi, 'right'
            elif mid < xmin + 0.3 * (xmax - xmin):
                tx, ha = lo, 'left'
            else:
                tx, ha = mid, 'center'
            ax.text(tx, y + (0.30 if k == 0 else -0.30), f'[{lo:.{nd}f}, {hi:.{nd}f}]', ha=ha,
                    va='bottom' if k == 0 else 'top', fontsize=11, color=c)
        ax.set_yticks([1, 0])
        ax.set_yticklabels([f'{lab}\n（{len(vv)} シード）' for lab, vv, _ in rows], fontsize=11)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.8, 1.8)
        ax.set_title(title, fontsize=11.5, pad=8)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.grid(axis='y', alpha=0)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(OUT, 'r1r2_dots.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    plt.rcParams['font.size'] = fs0
    return 'r1r2_dots.png'


def fig_a3_fail():
    """修論 4.6.1 の図: 斜め上方から描いた手描きスケッチ（A3）でモデル生成が止まる例（2026-11-26、ユーザー「失敗図が修論にないならいれよう」）。

    (a) 手描きスケッチ `data/test/A3/sketch/A3_hand.webp`、(b) 画像整形の出力 `A3_m1.jpeg`、
    (c)(d) 同じ入力から 2 回生成したメッシュ `data/test/A3/3D/A3.glb`・`A3_2.glb` を横から見たもの（図 4.1 (c) と同じ座標変換）。
    腕が水平に倒れて生成され、関節マーカー（マゼンタ）がほぼ同じ高さに並ぶため、関節の高さを境界とするリンク分割が成立しない（系譜 9-23）。
    """
    from PIL import Image
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import glb_to_links as G
    D = os.path.join(ROOT, 'data/test/A3')
    fs0 = plt.rcParams['font.size']
    plt.rcParams['font.size'] = 11      # 幅 8.4 インチで紙面 8.3 pt（チェックリスト 8）
    sketch = Image.open(os.path.join(D, 'sketch/A3_hand.webp')).convert('RGB')
    m1 = Image.open(os.path.join(D, 'sketch/A3_m1.jpeg')).convert('RGB')
    meshes = []
    for f in ('A3.glb', 'A3_2.glb'):
        m = G._apply_yup_zup(G._load_concat(os.path.join(D, '3D', f)))
        V = np.asarray(m.vertices); C = np.asarray(m.visual.vertex_colors[:, :3]) / 255
        idx = np.random.default_rng(0).choice(len(V), min(30000, len(V)), replace=False)
        k = 0 if np.ptp(V[:, 0]) >= np.ptp(V[:, 1]) else 1
        meshes.append((V, C, idx, k))
    ratios = [sketch.width / sketch.height, m1.width / m1.height, 1.0, 1.0]
    fig, ax = plt.subplots(1, 4, figsize=(8.4, 2.7), gridspec_kw={'width_ratios': ratios})
    for a, im, t in ((ax[0], sketch, '(a) 手描き\nスケッチ'), (ax[1], m1, '(b) 画像整形の\n出力（Gemini）')):
        a.imshow(im); a.set_axis_off(); a.set_title(t)
    for a, (V, C, idx, k), t in ((ax[2], meshes[0], '(c) 生成メッシュ\n（1 回目、横から）'), (ax[3], meshes[1], '(d) 生成メッシュ\n（2 回目、横から）')):
        a.scatter(V[idx, k], V[idx, 2], c=C[idx], s=0.25)
        a.set_aspect('equal'); a.set_xlabel('横 [m]'); a.set_ylabel('高さ [m]'); a.set_title(t)
    for a in ax:
        a.set_anchor('N')
    fig.tight_layout(w_pad=1.5)
    out = os.path.join(OUT, 'a3_fail.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    plt.rcParams['font.size'] = fs0
    return 'a3_fail.png'

def fig_joint_modes():
    """修論 図 4.2・新方針の予備審査資料 図 4: 関節軸の 2 つのモードと設計変数（2026-11-27、ユーザー「文字で言うと頭に入ってこないから…回転軸の円形の矢印がはいったやつ」
    →「後段で紹介するの設計変数も図の中に入れたら」「地面から15cmとかも」）。
    設計変数（ギア比 g・太さ r・カプセルの後方延長 e、縦型だけリンク長 l）を寸法線と引き出し線で描き込む。
    後方延長は 11-27 に足した（ユーザー「図4に後方延長が書いてないのはなんで？」）。両モードとも [0, 0.2] m（pusher_gearonly_real・
    pusher_tripo_v3_reallen の geom_params.ext_start）。カプセルを関節より後ろ（親リンクの側）へ延ばす部分を薄い紫で描く。リンク長を最適化するのは縦型アームだけ
    （修論 表 4.2 の「骨のベクトル（縦型アームのみ）」。平面の比較では長さはスケッチのまま）。11-27 に (a) のリンク長を外した。床からの 0.15 m は評価に用いた平面の腕の値
    （tripo_arm_v2c_pj_*_phys_arm171.xml・e2e_a1_phys_arm171.xml の body "0" の z、修論 4.1）。縦型（e2e_a1v）の根元は床から 0.02 m で、柱の長さはスケッチで決まるので数字を描かない。

    (a) 平面モード: 全関節がヨー軸（鉛直軸まわり）。腕は水平面内で動く。斜め上から見た図。
    (b) 縦型モード: 根元に旋回関節（ヨー）を 1 つ足し、それ以降の関節はピッチ軸（水平軸まわり）で鉛直面内に回る（4 関節）。横から見た図。
    関節の構成は修論 3.10（`mesh_to_params.py` の既定の軸の割り当て）に合わせた模式図で、長さ・角度は実際のモデルの値ではない。
    黄は鉛直軸まわり、橙は水平軸まわりの回転（ユーザーが示した見本の色に合わせた）。
    修論 4.1.1 の図 4.2 と新方針の資料の図 4 で同じ画像を使う（11-27 ユーザー「修論に入れてないの？」→ 表 4.2 の後）。出力は 修論/joint_modes.png。
    """
    from matplotlib.patches import FancyArrowPatch, Polygon
    YAW, PITCH, LINK, EDGE = '#f1c40f', '#e67e22', '#bdbdbd', '#555555'
    fs0 = plt.rcParams['font.size']
    plt.rcParams['font.size'] = 11      # 幅 8.4 インチで紙面 8.3 pt（チェックリスト 8）

    def link(a, p, q, w=11, z=3):
        a.plot([p[0], q[0]], [p[1], q[1]], color=EDGE, lw=w + 2.5, solid_capstyle='round', zorder=z)
        a.plot([p[0], q[0]], [p[1], q[1]], color=LINK, lw=w, solid_capstyle='round', zorder=z + 0.1)

    def box(a, x0, y0, w, h, z=2):
        a.add_patch(Rectangle((x0, y0), w, h, facecolor='#9e9e9e', edgecolor=EDGE, lw=1.5, zorder=z))

    def ring(a, c, rx, ry, color, t0, t1, z_front, z_back):
        """楕円の矢印。奥側（上半分）はリンクの後ろ、手前側（下半分）は前に描く。t は度で、t1 で矢じり。"""
        t = np.radians(np.linspace(t0, t1, 200))
        x, y = c[0] + rx * np.cos(t), c[1] + ry * np.sin(t)
        back = np.sin(t) > 0
        for m, z in ((back, z_back), (~back, z_front)):
            xs, ys = np.where(m, x, np.nan), np.where(m, y, np.nan)
            a.plot(xs, ys, color=color, lw=4, solid_capstyle='butt', zorder=z)
        a.add_patch(FancyArrowPatch((x[-6], y[-6]), (x[-1], y[-1]), arrowstyle='-|>', mutation_scale=22,
                                    color=color, lw=0, zorder=z_front if np.sin(t[-1]) <= 0 else z_back))

    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.1, wspace=0.05)
    for a in ax:
        a.set_aspect('equal'); a.set_axis_off()

    def dim(a, p, q, text, off, z=10, **kw):
        """寸法線: p〜q を off だけずらした位置に両矢印を引き、中央に文字。"""
        p2, q2 = (p[0] + off[0], p[1] + off[1]), (q[0] + off[0], q[1] + off[1])
        a.annotate('', xy=q2, xytext=p2, arrowprops=dict(arrowstyle='<->', color='#1f4e79', lw=1.3), zorder=z)
        m = ((p2[0] + q2[0]) / 2, (p2[1] + q2[1]) / 2)
        a.text(m[0] + kw.pop('dx', 0), m[1] + kw.pop('dy', 0), text, color='#1f4e79', zorder=z, **kw)

    def thick(a, p, q, half, xytext, text, **kw):
        """太さの両矢印: リンク p〜q の中点で、軸に垂直にリンクを横切る（half はリンクの見かけの半幅より少し大きく、データ座標）。
        矢印の外側の端から文字へ引き出し線を引く。"""
        m = np.array([(p[0] + q[0]) / 2, (p[1] + q[1]) / 2])
        d = np.array([q[0] - p[0], q[1] - p[1]]); d = d / np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        e0, e1 = m - n * half, m + n * half
        a.annotate('', xy=tuple(e1), xytext=tuple(e0), arrowprops=dict(arrowstyle='<->', color='#1f4e79', lw=1.4,
                   shrinkA=0, shrinkB=0, mutation_scale=10), zorder=12)
        end = e0 if np.dot(np.array(xytext) - m, -n) > 0 else e1
        callout(a, tuple(end), xytext, text, **kw)

    def tail(a, p, q, e, text, xytext, **kw):
        """後方延長: リンク p→q のカプセルを関節 p より後ろ（q と逆向き）へ e だけ延ばした部分を薄い色で描き、
        関節〜延長の端に両矢印を引いて文字へ引き出す（xml_robot の ext_start。到達半径は変えない）。"""
        d = np.array([q[0] - p[0], q[1] - p[1]]); d = d / np.linalg.norm(d)
        t = (p[0] - d[0] * e, p[1] - d[1] * e)
        a.plot([p[0], t[0]], [p[1], t[1]], color=EDGE, lw=13.5, solid_capstyle='round', zorder=2.6)
        a.plot([p[0], t[0]], [p[1], t[1]], color='#e9e3f3', lw=11, solid_capstyle='round', zorder=2.7)
        n = np.array([-d[1], d[0]])
        if n[1] < 0:
            n = -n                                  # 寸法線は上側に出す
        off = n * 0.2
        p2, t2 = (p[0] + off[0], p[1] + off[1]), (t[0] + off[0], t[1] + off[1])
        a.annotate('', xy=t2, xytext=p2, arrowprops=dict(arrowstyle='<->', color='#1f4e79', lw=1.3,
                   shrinkA=0, shrinkB=0, mutation_scale=10), zorder=12)
        for u in (p, t):
            a.plot([u[0] + n[0] * 0.08, u[0] + n[0] * 0.24], [u[1] + n[1] * 0.08, u[1] + n[1] * 0.24],
                   color='#1f4e79', lw=0.9, zorder=12)
        callout(a, ((p2[0] + t2[0]) / 2, (p2[1] + t2[1]) / 2), xytext, text, **kw)

    def callout(a, xy, xytext, text, **kw):
        a.annotate(text, xy=xy, xytext=xytext, color='#1f4e79', zorder=11,
                   arrowprops=dict(arrowstyle='-', color='#1f4e79', lw=1.0), **kw)

    # (a) 平面モード: 斜め上から。P(x, y) = (x + 0.55 y, 0.45 y) で水平面を潰して描く。腕の高さに薄い板を置いて水平面を示す
    a = ax[0]
    P = lambda x, y, z=0.0: (x + 0.55 * y, 0.45 * y + z)
    H = 0.55                                  # 腕の高さ（床から 0.15 m。XML の body "0" の z。模式図なので縮尺は合わせない）
    a.plot([-1.0, 4.4], [0, 0], color='#7f7f7f', lw=2, zorder=0)
    a.text(4.4, -0.06, '床', ha='right', va='top', color='#555555')
    plane = [P(-0.6, -0.7, H), P(3.4, -0.7, H), P(3.4, 2.2, H), P(-0.6, 2.2, H)]
    a.add_patch(Polygon(plane, closed=True, facecolor='#dfe9f5', edgecolor='#9fb6d0', lw=1, zorder=0))
    box(a, -0.25, 0.0, 0.5, H, z=0.5)
    dim(a, (-0.25, 0.0), (-0.25, H), '床から\n0.15 m', (-0.2, 0), ha='right', va='center', dx=-0.08)
    pts = [(0, 0), (1.3, -0.3), (2.0, 1.2), (3.25, 0.9)]
    J = [P(x, y, H) for x, y in pts]
    for i in range(3):
        link(a, J[i], J[i + 1])
    for i in range(3):
        a.plot([J[i][0], J[i][0]], [J[i][1] - 0.5, J[i][1] + 0.6], ls='--', color='#333333', lw=1, zorder=6)
        ring(a, J[i], 0.42, 0.17, YAW, 200, 520, 7, 2)
        a.text(J[i][0] + 0.12, J[i][1] + 0.62, f'j{i + 1}', ha='center', va='bottom')
    tail(a, J[2], J[3], 0.8, 'カプセルの後方延長 e\n（各リンク）', (1.3, 2.3), ha='center', va='bottom')
    callout(a, (J[0][0] - 0.38, J[0][1] + 0.1), (-1.15, 1.75), 'ギア比 g\n（各関節）', ha='left', va='center')
    m1 = ((J[0][0] + J[1][0]) / 2, (J[0][1] + J[1][1]) / 2)
    thick(a, J[0], J[1], 0.17, (m1[0] + 0.2, -0.45), '太さ r（各リンク）', ha='left', va='top')
    a.set_xlim(-1.3, 4.6); a.set_ylim(-0.75, 2.7)

    # (b) 縦型モード: 横から。根元の旋回（j1）＋鉛直面内の 3 関節（j2〜j4）
    a = ax[1]
    a.plot([-1.0, 3.1], [0, 0], color='#7f7f7f', lw=2, zorder=0)
    a.text(3.1, -0.06, '床', ha='right', va='top', color='#555555')
    box(a, -0.4, 0.0, 0.8, 0.25)
    J = [(0.0, 1.0), (0.85, 1.85), (1.95, 1.7), (2.65, 1.05)]
    link(a, (0.0, 0.25), J[0], w=16)
    for i in range(3):
        link(a, J[i], J[i + 1])
    a.plot([0, 0], [0.05, 2.2], ls='--', color='#333333', lw=1, zorder=6)
    ring(a, (0.0, 0.55), 0.5, 0.15, YAW, 200, 520, 7, 1)
    a.text(0.6, 0.4, 'j1（旋回）', ha='left', va='center')
    for i, c in enumerate(J[:3]):
        t = np.radians(np.linspace(40, 330, 200))
        r = 0.27
        x, y = c[0] + r * np.cos(t), c[1] + r * np.sin(t)
        a.plot(x, y, color=PITCH, lw=4, zorder=8)
        a.add_patch(FancyArrowPatch((x[-6], y[-6]), (x[-1], y[-1]), arrowstyle='-|>', mutation_scale=22, color=PITCH, lw=0, zorder=8))
        a.plot(*c, 'o', color='#333333', ms=4, zorder=9)
        if i == 2:                              # j4 の左上は後方延長の部分なので、左下に置く
            a.text(c[0] - 0.3, c[1] - 0.2, f'j{i + 2}', ha='right', va='top')
        else:
            a.text(c[0] - 0.36, c[1] + 0.2, f'j{i + 2}', ha='right', va='bottom')
    tail(a, J[2], J[3], 0.6, '後方延長 e', (2.2, 2.55), ha='left', va='center')
    callout(a, (J[1][0] - 0.1, J[1][1] + 0.27), (0.75, 2.6), 'ギア比 g', ha='right', va='center')
    m1 = ((J[2][0] + J[3][0]) / 2, (J[2][1] + J[3][1]) / 2)
    thick(a, J[2], J[3], 0.13, (2.55, 0.55), '太さ r', ha='left', va='center')
    dim(a, J[0], J[1], 'リンク長 l\n（縦型だけ）', (0.32, -0.32), ha='left', va='top', dx=0.12, dy=-0.05)
    a.set_xlim(-1.0, 3.3); a.set_ylim(-0.3, 2.7)

    fig.text(0.27, 0.97, '(a) 平面モード\n全関節が鉛直軸まわり', ha='center', va='top')
    fig.text(0.75, 0.97, '(b) 縦型モード\n根元は鉛直軸まわり、以降は水平軸まわり', ha='center', va='top')
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=YAW, lw=4, label='鉛直軸まわりの回転'),
                        Line2D([], [], color=PITCH, lw=4, label='水平軸まわりの回転')],
               loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(os.path.join(OUT, 'joint_modes.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    plt.rcParams['font.size'] = fs0
    return 'joint_modes.png'


FIGS = {'matrix': fig_matrix_reversal, 'a3fail': fig_a3_fail, 'r1r2': fig_r1r2_dots,
        'modes': fig_joint_modes,
        'matrix_bands': fig_matrix_bands,
        'var_spread': fig_var_spread,
        'curves': fig_learning_curves,
        'star': fig_target_star,
        'r4': fig_r4_settle,
        'arm': fig_arm_forms,
        'transfer': fig_transfer_decomposition,
        'diagnosis': fig_diagnosis_validation,
        'separation': fig_task_separation,
        'tolerance': fig_marker_tolerance,
        'ladder': fig_margin_ladder}

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    want = sys.argv[1:] or list(FIGS)
    for k in want:
        if k not in FIGS:
            print(f'[skip] 未知の図: {k}（候補: {", ".join(FIGS)}）')
            continue
        name = FIGS[k]()
        print(f'[ok] {os.path.join(OUT, name)}')
