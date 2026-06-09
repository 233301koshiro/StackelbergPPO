#!/usr/bin/env python3
"""
学習ログから報酬の推移グラフを生成する。

使い方:
  python3 scripts/plot_rewards.py single_run/pusher_cnoid
  python3 scripts/plot_rewards.py single_run/pusher_cnoid --window 50
"""

import re
import sys
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('restore_dir', help='学習ディレクトリ (例: single_run/pusher_cnoid)')
parser.add_argument('--window', type=int, default=30, help='移動平均ウィンドウ幅 (default: 30)')
parser.add_argument('--output', default=None, help='出力PNGパス (default: {restore_dir}/plots/reward_plot.png)')
args = parser.parse_args()

log_path = os.path.join(args.restore_dir, 'log', 'log_train.txt')
out_path = args.output or os.path.join(args.restore_dir, 'plots', 'reward_plot.png')
os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

# ── ログ解析 ──────────────────────────────────────────────────────────────────
pattern = re.compile(
    r'\[(.+?)\]\s+(\d+)\s+'
    r'T_sample\s+([\d.]+)\s+T_update\s+([\d.]+)\s+T_eval\s+([\d.]+)\s+'
    r'ETA\s+.+?\s+'
    r'train_R\s+([\d.]+)\s+train_R_eps\s+([\d.]+)\s+'
    r'exec_R\s+([\d.]+)\s+exec_R_eps\s+([\d.]+)'
)

epochs, train_R_eps, exec_R_eps = [], [], []
best_rewards = []  # (epoch, reward) for "save best" lines

best_pat = re.compile(r'\[(.+?)\]\s+save best checkpoint with rewards ([\d.]+)')

with open(log_path) as f:
    prev_epoch = None
    for line in f:
        m = pattern.search(line)
        if m:
            epoch = int(m.group(2))
            epochs.append(epoch)
            train_R_eps.append(float(m.group(7)))
            exec_R_eps.append(float(m.group(9)))
            prev_epoch = epoch
            continue
        bm = best_pat.search(line)
        if bm and prev_epoch is not None:
            best_rewards.append((prev_epoch, float(bm.group(2))))

print(f"エポック数: {len(epochs)}  ({epochs[0]} → {epochs[-1]})")
print(f"exec_R_eps  最大: {max(exec_R_eps):.1f}  最終: {exec_R_eps[-1]:.1f}")
print(f"train_R_eps 最大: {max(train_R_eps):.1f}  最終: {train_R_eps[-1]:.1f}")

# ── 移動平均 ──────────────────────────────────────────────────────────────────
def moving_avg(values, w):
    result = []
    for i in range(len(values)):
        lo = max(0, i - w // 2)
        hi = min(len(values), i + w // 2 + 1)
        result.append(sum(values[lo:hi]) / (hi - lo))
    return result

ma_exec  = moving_avg(exec_R_eps,  args.window)
ma_train = moving_avg(train_R_eps, args.window)

# ── プロット ──────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f'Reward history — {args.restore_dir}', fontsize=13)

for ax, raw, ma, label, color in [
    (axes[0], exec_R_eps,  ma_exec,  'exec_R_eps (execution reward / episode)',  '#2196F3'),
    (axes[1], train_R_eps, ma_train, 'train_R_eps (training reward / episode)',  '#FF9800'),
]:
    ax.plot(epochs, raw, color=color, alpha=0.25, linewidth=0.8, label='raw')
    ax.plot(epochs, ma,  color=color, linewidth=2.0, label=f'MA-{args.window}')
    ax.set_ylabel(label, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper left')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

# exec グラフに best checkpoint マーカーを追加
if best_rewards:
    bx = [e for e, _ in best_rewards]
    by = [r for _, r in best_rewards]
    axes[0].scatter(bx, by, color='red', s=25, zorder=5, label='best saved', marker='*')
    axes[0].legend(fontsize=8, loc='upper left')

axes[1].set_xlabel('Epoch', fontsize=10)

# epoch 960 の再開点を縦線で示す（ログが2セッションある場合）
if epochs[0] == 0 and 960 in epochs:
    for ax in axes:
        ax.axvline(x=960, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='resume')

plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"保存: {out_path}")
