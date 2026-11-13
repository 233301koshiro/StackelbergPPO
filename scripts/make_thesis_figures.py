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


# ── 図1: マトリクス判定の順位反転（4.4.3.2） ──────────────────────────────────
def fig_matrix_reversal():
    labels = ['短腕 0.403 m', '中間 0.887 m', '長腕 1.451 m']
    cols = [C_SHORT, C_MID, C_LONG]
    # 実在の物理のシード 0（系譜 9-246）。短腕は Bug 72 の門を直した取り直し（9-271）
    runs_r = ['tripo_pj_short_phys_cg', 'tripo_pj_mid_phys', 'tripo_pj_long_phys']
    runs_p = ['tripo_pjp_short_phys_cg', 'tripo_pjp_mid_phys', 'tripo_pjp_long_phys']
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
        ('Reach / 中間',  'ok',   '余裕 10%',              'ok',   '下限\n（到達では正常）', 'ok',   '誤差 0.5 mm\n16 step', '障害なし', '1 位', '−1.85'),
        ('Reach / 長腕',  'warn', '余裕 45%\n長すぎ',      'ok',   '下限\n（到達では正常）', 'ok',   '誤差 0.5 mm\n38 step', '障害なし', '2 位', '−9.68'),
        ('Pusher / 短腕', 'bad',  '触れない\n2.11 倍推奨', 'ok',   '範囲の内側\n（誤）',   'warn', '対象が\n動かない',   '達成不可', '3 位', '−0.00'),
        ('Pusher / 中間', 'ok',   '届く',                  'warn', '上限の警告\n（出力不足）', 'ok', '0.47 m 移動',      '障害なし', '2 位', '10.42'),
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
    """修論 4.1.1 の図: 検証に用いた水平 3 関節アーム（2026-10-22、ユーザー指摘「肝心のロボットの画像の一つもない」）。

    (a) M2 が生成したメッシュ（`data/tripo_arm_colorful2`、系譜 第5段の v2c の元）を横から。
    (b) 比較に用いた 7 形態を上から。リンク長と初期の太さは学習が読んだ XML（`*_phys_arm171`）から読む。
    """
    import xml.etree.ElementTree as ET
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import glb_to_links as G
    mesh = G._apply_yup_zup(G._load_concat(os.path.join(ROOT, 'data/tripo_arm_colorful2/colorful stacking rod 3d model.glb')))
    V = np.asarray(mesh.vertices); C = np.asarray(mesh.visual.vertex_colors[:, :3]) / 255
    idx = np.random.default_rng(0).choice(len(V), min(30000, len(V)), replace=False)
    k = 0 if np.ptp(V[:, 0]) >= np.ptp(V[:, 1]) else 1

    # 基本形（1.0 倍）は学習に使っていないので XML が無い。短腕（0.5 倍）の長さを 2 倍して描く
    # （太さは倍率で変えない。どの形も 0.045 / 0.0326 / 0.032。2026-11-13 ユーザーの指摘「図 4.1 に基本形を載せる」）。
    forms = [('基本形（1.0 倍）', 'base'), ('短腕（0.5 倍）', 'short'), ('中間（1.1 倍）', 'mid'), ('長腕（1.8 倍）', 'long'),
             ('根元重（配分）', 'prox'), ('先端重（配分）', 'dist'),
             ('助言の 1.99 倍', 'rec199'), ('助言の 2.11 倍', 'rec211'), ('助言の 2.36 倍', 'rec236')]
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

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 5.6), gridspec_kw={'width_ratios': [0.42, 2.6]})
    a = ax[0]
    a.scatter(V[idx, k], V[idx, 2], c=C[idx], s=0.4)
    a.set_aspect('equal'); a.set_xlabel('横 [m]'); a.set_ylabel('高さ [m]')
    a.set_title('(a) 生成したメッシュ（横から）', fontsize=11)
    a = ax[1]
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
        a.text(-0.14, y, lab, ha='right', va='center', fontsize=10)
        a.text(x + 0.06, y, f'{x:.3f} m', va='center', fontsize=9)
    for xv, lab in ((0.80, '到達の目標\n0.80 m'), (0.85, '押す箱の、腕に\n面した側面 0.85 m')):
        a.axvline(xv, color='k', ls='--' if xv == 0.80 else ':', lw=1)
        a.text(xv + (-0.02 if xv == 0.80 else 0.02), 0.2, lab, ha='right' if xv == 0.80 else 'left', fontsize=8.5)
    a.set_xlim(-1.0, 1.85); a.set_ylim(-(len(forms) - 1) * 0.3 - 0.2, 0.42); a.set_aspect('equal')
    a.set_yticks([]); a.set_xticks([0, 0.5, 1.0, 1.5]); a.set_xlabel('x [m]（根元の関節から。全関節 0°）')
    a.set_title('(b) 比較に用いた形態（上から）', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'arm_forms.png'))
    plt.close(fig)
    return 'arm_forms.png'


# ── 図: R4 の確定エポック（4.6.3、系譜 9-338） ──────────────────────────────
#   値は rank_settle.py の settle_epoch（running best・「以降一度も最終順位から外れない最小のエポック」）で
#   ①原本（log_train.txt）から数える。較正に使ったのは seed 0・1（seed 1 は新しい種の *_s2_ws）。
#   seed 2（*_s3）は較正に使っていない（9-338 で事前登録して確かめた）。
def fig_r4_settle():
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import rank_settle as rs
    rs.RUN_DIR = RUN

    def ws(r):
        return r + '_ws' if r.endswith('_s2') and os.path.isdir(os.path.join(RUN, r + '_ws')) else r

    def settle(runs):
        ep, _, _, _, _, _ = rs.settle_epoch(runs, runs)
        return ep

    def wrong_at(runs, e=30):
        """第 e エポックの running best の順位が最終順位と逆か（2 run の対だけ）。"""
        a, b = (rs.running_best(r) for r in runs)
        last = max(set(a) & set(b))
        return (a[e] > b[e]) != (a[last] > b[last])

    # 形が違う対（コホート B）: (行の名前, seed 0 の対, seed 1 の対, seed 2 の対)
    B = [('到達: 中間 対 長腕', ('tripo_pj_mid_phys', 'tripo_pj_long_phys'),
          ('tripo_pj_mid_phys_s2', 'tripo_pj_long_phys_s2'), ('tripo_pj_mid_phys_s3', 'tripo_pj_long_phys_s3')),
         ('押し: 長腕 対 中間', ('tripo_pjp_long_phys', 'tripo_pjp_mid_phys'),
          ('tripo_pjp_long_phys_s2', 'tripo_pjp_mid_phys_s2'), ('tripo_pjp_long_phys_s3', 'tripo_pjp_mid_phys_s3')),
         ('到達: 根元重 対 先端重', ('tripo_pjd_prox_phys', 'tripo_pjd_dist_phys_cg'),
          ('tripo_pjd_prox_phys_s2', 'tripo_pjd_dist_phys_cg_s2'), ('tripo_pjd_prox_phys_s3', 'tripo_pjd_dist_phys_cg_s3')),
         ('押し: 根元重 対 先端重', ('tripo_pjdp_prox_phys', 'tripo_pjdp_dist_phys_cg'),
          ('tripo_pjdp_prox_phys_s2', 'tripo_pjdp_dist_phys_s2'), ('tripo_pjdp_prox_phys_s3', 'tripo_pjdp_dist_phys_cg_s3')),
         ('到達: A1 可動 対 関節 3 固定', ('e2e_a1_reach_phys', 'e2e_a1_fix3_reach_phys'),
          ('e2e_a1_reach_phys_s2', 'e2e_a1_fix3_reach_phys_s2'), ('e2e_a1_reach_phys_s3', 'e2e_a1_fix3_reach_phys_s3'))]
    SHORT = [('tripo_pj_long_phys', 'tripo_pj_short_phys_cg'), ('tripo_pj_mid_phys', 'tripo_pj_short_phys_cg'),
             ('tripo_pjp_long_phys', 'tripo_pjp_short_phys_cg'), ('tripo_pjp_mid_phys', 'tripo_pjp_short_phys_cg'),
             ('tripo_pjr199_phys', 'tripo_pj_short_phys_cg')]
    # 乱数だけ違う組（コホート A）: rank_settle の COHORTS のうち名前が「A」で始まるもの（seed 1 は *_s2_ws）
    A = [[ws(r) for r in runs] for runs, _, name in rs.COHORTS if name.startswith('A')]

    rows = []   # (名前, [(エポック, 印の種類, 第30で逆か)])
    for name, p0, p1, p2 in B:
        pts = [(settle(list(p0)), 'cal', wrong_at(p0)),
               (settle([ws(r) for r in p1]), 'cal', wrong_at([ws(r) for r in p1])),
               (settle(list(p2)), 'held', wrong_at(p2))]
        rows.append((name, pts))
    rows.append(('短腕との対（5 対、seed 0）', [(settle(list(p)), 'cal', wrong_at(p)) for p in SHORT]))
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
          plt.Line2D([], [], ls='', marker='D', ms=7.5, mfc='white', mec='#c0392b', mew=1.8, label='形が違う対（較正に使っていない seed 2）'),
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
        ax.plot(cube[:, -1, 0], cube[:, -1, 1], 'o', ms=3, color=blue, alpha=0.7)
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
        ax.add_patch(plt.Circle(tgt[:2], 0.03, fill=False, ec=gold, ls=':', lw=1.2))
        ax.plot(tgt[0], tgt[1], marker='*', ms=24, mfc=gold, mec='#8a6d00', mew=1.0, zorder=5)
        ax.set_title(lab, fontsize=11)
        ax.set_aspect('equal'); ax.set_xlim(-0.12, 0.48); ax.set_ylim(-0.36, 0.20)
        ax.set_xlabel('x [m]')
    axes[0].set_ylabel('y [m]')
    hs = [plt.Line2D([], [], ls='', marker='*', ms=13, mfc=gold, mec='#8a6d00', label='目標位置'),
          Rectangle((0, 0), 1, 1, fill=False, ec='#888888', ls='--', label='キューブの初期位置'),
          plt.Line2D([], [], color=blue, lw=1, marker='o', ms=3, label='キューブの軌跡と最終位置（50 試行）'),
          plt.Line2D([], [], color=red, lw=1.2, label='手先の軌跡（代表の 1 試行）'),
          Rectangle((0, 0), 1, 1, fc=arm, alpha=0.35, ec=arm, label='腕の姿勢（代表の 1 試行）')]
    fig.legend(handles=hs, loc='lower center', ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(OUT, 'ra_target_star.png'))
    plt.close(fig)
    return 'ra_target_star.png'


FIGS = {'matrix': fig_matrix_reversal,
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
        print(f'[ok] figures/{name}')
