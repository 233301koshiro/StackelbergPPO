#!/usr/bin/env python3
"""
比較レポート自動生成スクリプト
TensorBoard ログと eval_cross_env.py の出力 JSON から
学習曲線グラフと Markdown レポートを生成する。

使い方:
  cd /userdir/StackelbergPPO

  # ① 評価データが既にある場合
  python scripts/generate_comparison_report.py \
    --runs single_run/pusher_resume single_run/pusher_cnoid_v3 \
    --labels MuJoCo Choreonoid \
    --eval_json single_run/comparison/eval_results.json \
    --output single_run/comparison/

  # ② eval_cross_env.py を内部で自動実行する場合
  python scripts/generate_comparison_report.py \
    --runs single_run/pusher_resume single_run/pusher_cnoid_v3 \
    --labels MuJoCo Choreonoid \
    --run_eval --n_episodes 20 \
    --output single_run/comparison/

OK 判定基準 (--ok_reward_ratio / --ok_success_diff で変更可):
  ✅ OK: MuJoCo の最高 exec_R_eps が Choreonoid の 80% 以上
      かつ 成功率差 10% 以内
  ❌ NG: それ以外
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
from datetime import datetime

sys.path.insert(0, os.getcwd())

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ─── TensorBoard ユーティリティ ────────────────────────────────────────────────

def load_tb_scalar(tb_dir: str, tag: str):
    """TB ディレクトリからスカラー系列を (steps, values) で返す。"""
    ea = EventAccumulator(tb_dir, size_guidance={'scalars': 0})
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        return np.array([]), np.array([])
    events = ea.Scalars(tag)
    steps  = np.array([e.step  for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def find_tb_dir(restore_dir: str):
    tb = os.path.join(restore_dir, 'tb')
    return tb if os.path.isdir(tb) else None


# ─── グラフ生成 ────────────────────────────────────────────────────────────────

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

def plot_learning_curves(runs, labels, output_dir):
    """exec_R_eps_avg の学習曲線を重ね書きして PNG 保存。"""
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (run, label) in enumerate(zip(runs, labels)):
        tb_dir = find_tb_dir(run)
        if tb_dir is None:
            print(f"[warn] TB dir not found: {run}/tb")
            continue
        steps, vals = load_tb_scalar(tb_dir, 'exec_R_eps_avg')
        if len(steps) == 0:
            print(f"[warn] No exec_R_eps_avg in {tb_dir}")
            continue
        color = COLORS[i % len(COLORS)]
        ax.plot(steps, vals, color=color, label=label, linewidth=1.5, alpha=0.9)
        # 最高値にマーカー
        best_idx = np.argmax(vals)
        ax.scatter(steps[best_idx], vals[best_idx], color=color, s=80, zorder=5)
        ax.annotate(f"peak={vals[best_idx]:.0f}",
                    xy=(steps[best_idx], vals[best_idx]),
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=8, color=color)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('exec_R_eps (eval, per episode)', fontsize=12)
    ax.set_title('Learning curves: MuJoCo vs Choreonoid', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(output_dir, 'comparison_curves.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[plot] Saved: {out}")
    return out


def plot_eval_bars(eval_results, output_dir):
    """eval_cross_env の結果を棒グラフで表示。"""
    labels  = [r['label'] for r in eval_results]
    rewards = [r['mean_exec_reward'] for r in eval_results]
    disps   = [r['mean_cube_disp_m'] for r in eval_results]
    succs   = [r['success_rate'] * 100 for r in eval_results]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, vals, title, unit in zip(
        axes,
        [rewards, disps, succs],
        ['Mean exec reward', 'Mean cube disp', 'Success rate'],
        ['reward/ep', 'm', '%']
    ):
        bars = ax.bar(labels, vals, color=COLORS[:len(labels)], alpha=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(unit, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Checkpoint evaluation summary', fontsize=12)
    fig.tight_layout()

    out = os.path.join(output_dir, 'comparison_eval.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[plot] Saved: {out}")
    return out


# ─── OK 判定 ──────────────────────────────────────────────────────────────────

def judge(eval_results, ok_reward_ratio=0.80, ok_success_diff=0.10):
    """
    各 run はネイティブエンジンで評価済みとして比較する。
    runs[0] を target、runs[1] を reference とみなす。
    """
    if len(eval_results) < 2:
        return None, "比較対象が1つしかありません"

    target = eval_results[0]   # MuJoCo
    ref    = eval_results[1]   # Choreonoid

    ratio      = target['mean_exec_reward'] / (ref['mean_exec_reward'] + 1e-9)
    succ_diff  = abs(target['success_rate'] - ref['success_rate'])

    ok_reward  = ratio >= ok_reward_ratio
    ok_success = succ_diff <= ok_success_diff
    verdict    = ok_reward and ok_success

    details = (
        f"報酬比 {target['label']}/{ref['label']} = {ratio:.2f} "
        f"(基準 ≥ {ok_reward_ratio}) → {'✅' if ok_reward else '❌'}\n"
        f"成功率差 = {succ_diff*100:.1f}% (基準 ≤ {ok_success_diff*100:.0f}%) "
        f"→ {'✅' if ok_success else '❌'}"
    )
    return verdict, details


# ─── Markdown レポート生成 ────────────────────────────────────────────────────

def generate_markdown(eval_results, runs, labels, verdict, judge_details,
                      curves_png, eval_png, output_dir, ok_reward_ratio, ok_success_diff):
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    overall = "✅ OK" if verdict else ("❌ NG" if verdict is False else "⚠ 判定不能")

    # TB から各 run の最高 exec_R_eps を取得
    peak_rewards = {}
    final_rewards = {}
    for run, label in zip(runs, labels):
        tb_dir = find_tb_dir(run)
        if tb_dir:
            _, vals = load_tb_scalar(tb_dir, 'exec_R_eps_avg')
            if len(vals) > 0:
                peak_rewards[label]  = float(np.max(vals))
                final_rewards[label] = float(vals[-1])

    lines = [
        f"# 物理エンジン比較レポート",
        f"",
        f"生成日時: {now}  ",
        f"総合判定: **{overall}**",
        f"",
        f"---",
        f"",
        f"## 判定基準",
        f"",
        f"| 指標 | 基準値 |",
        f"|---|---|",
        f"| MuJoCo/Choreonoid 報酬比 | ≥ {ok_reward_ratio*100:.0f}% |",
        f"| 成功率差（cube +x ≥ 2m） | ≤ {ok_success_diff*100:.0f}% |",
        f"",
        f"---",
        f"",
        f"## 学習曲線",
        f"",
        f"![learning curves]({os.path.basename(curves_png)})",
        f"",
    ]

    # 学習曲線テーブル
    if peak_rewards:
        lines += [
            f"| 環境 | 最高 exec_R_eps | 最終 exec_R_eps | 総エポック |",
            f"|---|---|---|---|",
        ]
        for label in labels:
            tb_dir = find_tb_dir(runs[labels.index(label)])
            n_epochs = 0
            if tb_dir:
                steps, _ = load_tb_scalar(tb_dir, 'exec_R_eps_avg')
                n_epochs = int(steps[-1]) + 1 if len(steps) > 0 else 0
            lines.append(
                f"| {label} | {peak_rewards.get(label, 'N/A'):.1f} "
                f"| {final_rewards.get(label, 'N/A'):.1f} | {n_epochs} |"
            )
        lines.append("")

    lines += [
        f"---",
        f"",
        f"## チェックポイント評価結果",
        f"",
        f"![eval bars]({os.path.basename(eval_png)})",
        f"",
        f"| 環境 | epoch | mean exec_R | mean cube disp | success rate |",
        f"|---|---|---|---|---|",
    ]
    for r in eval_results:
        lines.append(
            f"| {r['label']} | {r['epoch']} "
            f"| {r['mean_exec_reward']:.1f} ± {r['std_exec_reward']:.1f} "
            f"| {r['mean_cube_disp_m']:.3f} m "
            f"| {r['success_rate']*100:.0f}% |"
        )

    engine_note = " / ".join(f"{r['label']}→{r['engine']}" for r in eval_results)
    lines += [
        f"",
        f"※ 成功 = 1 エピソードで cube が +x 方向に {eval_results[0]['success_threshold_m']}m 以上移動",
        f"※ 各ポリシーはその学習時と同じ物理エンジンで評価（{engine_note}）",
        f"※ n={eval_results[0]['n_episodes']} エピソード/run",
        f"",
        f"---",
        f"",
        f"## 判定詳細",
        f"",
        f"```",
        judge_details,
        f"```",
        f"",
        f"**総合: {overall}**",
        f"",
        f"---",
        f"",
        f"## 使用チェックポイント",
        f"",
    ]
    for r, run in zip(eval_results, runs):
        lines.append(f"- **{r['label']}**: `{run}/models/{r['epoch']}.p`")

    md = "\n".join(lines) + "\n"
    out = os.path.join(output_dir, 'comparison_report.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"[report] Saved: {out}")
    return out


# ─── メイン ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs',           nargs='+', required=True)
    parser.add_argument('--labels',         nargs='+', default=None)
    parser.add_argument('--eval_json',      type=str,  default=None,
                        help='eval_cross_env.py の出力 JSON（省略時は --run_eval が必要）')
    parser.add_argument('--run_eval',       action='store_true',
                        help='eval_cross_env.py を自動実行して評価データを生成する')
    parser.add_argument('--epochs',         nargs='+', default=None)
    parser.add_argument('--n_episodes',     type=int,  default=20)
    parser.add_argument('--threshold',      type=float, default=0.5)
    parser.add_argument('--output',         type=str,  default='single_run/comparison/')
    parser.add_argument('--ok_reward_ratio',  type=float, default=0.80,
                        help='MuJoCo/Choreonoid 報酬比の最低基準 (default: 0.80)')
    parser.add_argument('--ok_success_diff',  type=float, default=0.20,
                        help='成功率差の最大許容値 (default: 0.20)')
    args = parser.parse_args()

    labels = args.labels or args.runs
    epochs = args.epochs or ['best'] * len(args.runs)
    os.makedirs(args.output, exist_ok=True)

    # ── eval_cross_env の実行 ──────────────────────────────────────────────────
    eval_json = args.eval_json
    if eval_json is None or args.run_eval:
        eval_json = os.path.join(args.output, 'eval_results.json')
        cmd = [
            sys.executable, 'scripts/eval_cross_env.py',
            '--runs',       *args.runs,
            '--labels',     *labels,
            '--epochs',     *epochs,
            '--n_episodes', str(args.n_episodes),
            '--threshold',  str(args.threshold),
            '--output',     eval_json,
        ]
        print(f"[report] Running: {' '.join(cmd)}")
        ret = subprocess.run(cmd, cwd=os.getcwd())
        if ret.returncode != 0:
            sys.exit(f"eval_cross_env.py failed (code {ret.returncode})")

    with open(eval_json, encoding='utf-8') as f:
        eval_results = json.load(f)

    # ラベルが eval_results に入っていなければ補完
    for r, label in zip(eval_results, labels):
        r.setdefault('label', label)

    # ── グラフ生成 ──────────────────────────────────────────────────────────────
    curves_png = plot_learning_curves(args.runs, labels, args.output)
    eval_png   = plot_eval_bars(eval_results, args.output)

    # ── OK 判定 ─────────────────────────────────────────────────────────────────
    verdict, judge_details = judge(eval_results, args.ok_reward_ratio, args.ok_success_diff)

    # ── Markdown 生成 ──────────────────────────────────────────────────────────
    generate_markdown(
        eval_results, args.runs, labels,
        verdict, judge_details,
        curves_png, eval_png, args.output,
        args.ok_reward_ratio, args.ok_success_diff,
    )

    print("\n" + "=" * 60)
    print(f"総合判定: {'✅ OK' if verdict else ('❌ NG' if verdict is False else '⚠ 判定不能')}")
    print(judge_details)
    print("=" * 60)
    print(f"\nResults in: {args.output}")


if __name__ == '__main__':
    main()
