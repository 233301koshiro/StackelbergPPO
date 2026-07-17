#!/usr/bin/env python3
"""
前向き判定テスト（PJ実験）の学習曲線比較プロット。

使い方（部分完走中でも途中まで出力可）:
  python3 scripts/plot_pj_comparison.py
  python3 scripts/plot_pj_comparison.py --window 5 --out figures/pj_learning_curve.png

出力:
  - 図1: PJ_short vs PJ_long の学習曲線（exec_R_eps）
  - 標準出力: 最終スコア・順位確定エポック
"""
import re
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # display 不要
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─── 引数 ────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument('--window', type=int, default=5, help='移動平均ウィンドウ幅（デフォルト: 5ep）')
p.add_argument('--out', default='figures/pj_learning_curve.png', help='出力先')
p.add_argument('--base', default='.', help='single_run/ の親ディレクトリ')
args = p.parse_args()

BASE = args.base

# ─── ログ解析 ────────────────────────────────────────────────────────────────
PAT = re.compile(r'\[.+?\]\s+(\d+)\s+.*?exec_R_eps\s+(-?[\d.]+)')
BEST_PAT = re.compile(r'\[.+?\]\s+save best checkpoint with rewards\s+(-?[\d.]+)')

def parse_log(path):
    """(epoch, exec_R_eps) のリストと (epoch, best_R) のリストを返す"""
    if not os.path.exists(path):
        return [], []
    eps, best = [], []
    best_ep = 0
    with open(path) as f:
        for line in f:
            m = PAT.search(line)
            if m:
                best_ep = int(m.group(1))
                eps.append((int(m.group(1)), float(m.group(2))))
            bm = BEST_PAT.search(line)
            if bm:
                best.append((best_ep, float(bm.group(1))))
    return eps, best

def smooth(vals, w):
    if len(vals) < w:
        return vals
    return np.convolve(vals, np.ones(w)/w, mode='valid')

# 既存の参照ラインも読む（L1 Reach: 正しい co-design の天井）
runs = {
    'PJ_short（短腕 0.55 m）': f'{BASE}/single_run/rrbot_arm_reach_PJ_short/log/log_train.txt',
    'PJ_long（長腕 ~1.0 m）':  f'{BASE}/single_run/rrbot_arm_reach_PJ_long/log/log_train.txt',
    'L1（Reach co-design 参照）': f'{BASE}/single_run/rrbot_arm_reach_L1/log/log_train.txt',
}

COLORS = {
    'PJ_short（短腕 0.55 m）': '#d62728',   # 赤
    'PJ_long（長腕 ~1.0 m）':  '#1f77b4',   # 青
    'L1（Reach co-design 参照）': '#2ca02c', # 緑（参照、薄め）
}
STYLES = {
    'PJ_short（短腕 0.55 m）': '-',
    'PJ_long（長腕 ~1.0 m）':  '-',
    'L1（Reach co-design 参照）': '--',
}

# ─── プロット ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

data_by_run = {}
for label, path in runs.items():
    ep_data, _ = parse_log(path)
    if not ep_data:
        continue
    data_by_run[label] = ep_data
    epochs = [d[0] for d in ep_data]
    vals   = [d[1] for d in ep_data]
    w = args.window
    sm = smooth(vals, w)
    ep_sm = epochs[w-1:]
    ax.plot(epochs, vals, alpha=0.20, color=COLORS[label], linewidth=0.8)
    ax.plot(ep_sm, sm, label=f'{label}  (最終: {vals[-1]:.1f})',
            color=COLORS[label], linestyle=STYLES[label], linewidth=2)

# ─── 順位確定エポックを計算して垂線 ────────────────────────────────────────
short_data = data_by_run.get('PJ_short（短腕 0.55 m）', [])
long_data  = data_by_run.get('PJ_long（長腕 ~1.0 m）', [])

stable_ep = None
if short_data and long_data:
    # 共通エポックのみ比較
    short_d = {e: r for e, r in short_data}
    long_d  = {e: r for e, r in long_data}
    common_eps = sorted(set(short_d) & set(long_d))
    consecutive = 0
    for ep in common_eps:
        if long_d[ep] > short_d[ep]:
            consecutive += 1
            if consecutive >= 5 and stable_ep is None:
                stable_ep = ep - 4   # 最初の一致エポック
        else:
            consecutive = 0

    if stable_ep is not None:
        ax.axvline(stable_ep, color='gray', linestyle=':', linewidth=1.5,
                   label=f'順位確定 ep{stable_ep}（long > short が5ep連続）')

# ─── 装飾 ────────────────────────────────────────────────────────────────────
ax.set_xlabel('エポック', fontsize=12)
ax.set_ylabel('エピソード報酬 exec_R_eps', fontsize=12)
ax.set_title('前向き判定テスト: 短腕 vs 長腕 の学習曲線（Reach タスク、目標 x=0.8 m）', fontsize=11)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(20))

# 目標付近の参照線（L1 best = -0.014 だがスケール外なので-100付近を最低ラインに）
ax.set_ylim(top=min(50, ax.get_ylim()[1]))

plt.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
plt.savefig(args.out, dpi=150)
print(f'[plot_pj_comparison] saved → {args.out}')

# ─── テキスト出力 ────────────────────────────────────────────────────────────
print('\n=== PJ実験 学習曲線サマリー ===')
for label, ep_data in data_by_run.items():
    if not ep_data:
        continue
    vals = [d[1] for d in ep_data]
    print(f'{label}')
    for checkpoint in [0, 10, 50, 100, 150, 200]:
        idx = next((i for i, (e, _) in enumerate(ep_data) if e >= checkpoint), None)
        if idx is not None:
            print(f'  ep{ep_data[idx][0]:3d}: {ep_data[idx][1]:8.2f}')
    print(f'  最終 ep{ep_data[-1][0]}: {ep_data[-1][1]:.2f}')
    print()

if stable_ep is not None:
    print(f'順位確定エポック: ep{stable_ep}（long > short 5連続）')
else:
    print('順位確定エポック: 未確定（データ不足 or long が short を上回っていない）')
