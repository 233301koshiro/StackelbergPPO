#!/usr/bin/env python3
"""
monitor_training.py: 学習プロセスの停滞・異常・前進を自動検知して通知する

使い方:
  # バックグラウンドで永続監視
  nohup python3 scripts/monitor_training.py \
      --runs single_run/rrbot_arm_cnoid_v5-5 single_run/rrbot_arm2_cnoid_v2 \
      > single_run/monitor.log 2>&1 &

オプション:
  --runs          監視対象の run ディレクトリ（複数指定可）
  --interval      チェック間隔（秒、デフォルト600=10分）
  --stall-epochs  best.p がこの epoch 数更新されなければ STALL 警告（デフォルト150）
  --zero-epochs   exec_R_eps がゼロ近傍でこの epoch 数続いたら STALL 警告（デフォルト200）
  --exec-thresh   exec_R_eps がこの値を超えたら PROGRESS 通知（デフォルト0.05）
  --zero-thresh   exec_R_eps がこの値未満をゼロ近傍と判定（デフォルト0.02）
  --auto-kill     STALL 検知時にプロセスを自動停止（デフォルト無効、危険）
"""

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log(msg: str, log_path: Path = None):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    if log_path:
        with open(log_path, 'a') as f:
            f.write(line + '\n')


def parse_last_epochs(run_dir: Path, n: int = 300):
    """log_train.txt の末尾 n 行をパースして epoch 情報を返す。"""
    log_file = run_dir / 'log' / 'log_train.txt'
    if not log_file.exists():
        return []
    lines = log_file.read_text().splitlines()
    results = []
    pattern = re.compile(
        r'\]\s+(\d+)\s+.*?train_R_eps\s+([\d.eE+\-]+).*?exec_R_eps\s+([\d.eE+\-]+)'
        r'.*?fwd_cube\s+([\d.eE+\-]+).*?fwd_contact\s+([\d.eE+\-]+)'
    )
    pattern_no_fwd = re.compile(
        r'\]\s+(\d+)\s+.*?train_R_eps\s+([\d.eE+\-]+).*?exec_R_eps\s+([\d.eE+\-]+)'
    )
    for line in lines[-n:]:
        m = pattern.search(line)
        if m:
            results.append({
                'epoch': int(m.group(1)),
                'train_R_eps': float(m.group(2)),
                'exec_R_eps': float(m.group(3)),
                'fwd_cube': float(m.group(4)),
                'fwd_contact': float(m.group(5)),
            })
            continue
        m = pattern_no_fwd.search(line)
        if m:
            results.append({
                'epoch': int(m.group(1)),
                'train_R_eps': float(m.group(2)),
                'exec_R_eps': float(m.group(3)),
                'fwd_cube': None,
                'fwd_contact': None,
            })
    return results


def get_best_epoch(run_dir: Path):
    """log_train.txt から最後の 'save best checkpoint' 行の epoch を返す。"""
    log_file = run_dir / 'log' / 'log_train.txt'
    if not log_file.exists():
        return None, None
    lines = log_file.read_text().splitlines()
    best_epoch = None
    best_reward = None
    for line in reversed(lines):
        if 'save best checkpoint with rewards' in line:
            m_ep = re.search(r'^\[.*?\] (\d+)\s+T_sample', '')
            # best行の直前のepoch行を探すより、前後から番号を逆引き
            m_rew = re.search(r'rewards\s+([\d.eE+\-]+)', line)
            if m_rew:
                best_reward = float(m_rew.group(1))
            break
    # 別アプローチ: 最後の best 行の前にある epoch 番号行を探す
    for i in range(len(lines) - 1, -1, -1):
        if 'save best checkpoint' in lines[i]:
            for j in range(i - 1, max(i - 5, -1), -1):
                m = re.search(r'\]\s+(\d+)\s+T_sample', lines[j])
                if m:
                    best_epoch = int(m.group(1))
                    break
            break
    return best_epoch, best_reward


def is_process_alive(run_dir: str) -> bool:
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    return str(run_dir) in result.stdout


def kill_run(run_dir: str):
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if str(run_dir) in line and 'choreonoid_train' in line:
            pid = int(line.split()[1])
            subprocess.run(['kill', '-9', str(pid)])


def check_run(run_dir: Path, prev_state: dict, args) -> dict:
    """1 run の状態をチェックし、アラートとアップデートされた状態を返す。"""
    epochs = parse_last_epochs(run_dir, n=500)
    if not epochs:
        return prev_state, ['❓ ログが空またはまだ実行フェーズに入っていない']

    latest = epochs[-1]
    current_epoch = latest['epoch']
    alerts = []

    # --- プロセス死活 ---
    alive = is_process_alive(str(run_dir))
    if not alive:
        alerts.append(f'💀 DEAD: プロセスが停止しています (最終 epoch={current_epoch})')
        return {**prev_state, 'alive': False}, alerts

    # --- exec_R_eps がゼロ近傍で長く続いているか ---
    recent = epochs[-args.zero_epochs:] if len(epochs) >= args.zero_epochs else epochs
    near_zero = [e for e in recent if abs(e['exec_R_eps']) < args.zero_thresh]
    if len(near_zero) == len(recent) and len(recent) >= args.zero_epochs:
        alerts.append(
            f'🔴 STALL: exec_R_eps が {args.zero_epochs} epoch 連続でゼロ近傍 '
            f'(最新 epoch={current_epoch}, exec_R_eps={latest["exec_R_eps"]:.5f})'
        )

    # --- best.p の停滞 ---
    best_epoch, best_reward = get_best_epoch(run_dir)
    if best_epoch is not None and current_epoch - best_epoch >= args.stall_epochs:
        alerts.append(
            f'🟡 STALL: best.p が {current_epoch - best_epoch} epoch 更新なし '
            f'(最終更新 epoch={best_epoch}, reward={best_reward})'
        )

    # --- exec_R_eps が閾値を超えた（前進！） ---
    if latest['exec_R_eps'] >= args.exec_thresh:
        if prev_state.get('exec_above_thresh') is None:
            alerts.append(
                f'🟢 PROGRESS: exec_R_eps が閾値 {args.exec_thresh} を超えました! '
                f'epoch={current_epoch}, exec_R_eps={latest["exec_R_eps"]:.5f}'
            )
        prev_state = {**prev_state, 'exec_above_thresh': current_epoch}
    else:
        prev_state = {**prev_state, 'exec_above_thresh': None}

    # --- fwd_cube が初めて非ゼロになった（cube が動き始めた！） ---
    if latest['fwd_cube'] is not None:
        first_nonzero_cube = next(
            (e for e in epochs[-50:] if e['fwd_cube'] is not None and abs(e['fwd_cube']) > 1e-5),
            None
        )
        if first_nonzero_cube and not prev_state.get('cube_moved'):
            alerts.append(
                f'🎉 CUBE MOVED: fwd_cube が非ゼロになりました! '
                f'epoch={first_nonzero_cube["epoch"]}, fwd_cube={first_nonzero_cube["fwd_cube"]:.6f}'
            )
            prev_state = {**prev_state, 'cube_moved': True}

    # --- 定期サマリー（アラートがなくても） ---
    fwd_info = ''
    if latest['fwd_cube'] is not None:
        fwd_info = f', fwd_cube={latest["fwd_cube"]:.4f}, fwd_contact={latest["fwd_contact"]:.4f}'
    alerts.append(
        f'📊 epoch={current_epoch}, train_R_eps={latest["train_R_eps"]:.3f}, '
        f'exec_R_eps={latest["exec_R_eps"]:.5f}{fwd_info}'
        + (f', best_epoch={best_epoch}' if best_epoch else '')
    )

    return {**prev_state, 'alive': True, 'last_epoch': current_epoch}, alerts


def main():
    parser = argparse.ArgumentParser(description='学習プロセス監視スクリプト')
    parser.add_argument('--runs', nargs='+', required=True, help='監視する run ディレクトリ')
    parser.add_argument('--interval', type=int, default=600, help='チェック間隔（秒）')
    parser.add_argument('--stall-epochs', type=int, default=150,
                        help='best.p がこの epoch 数更新されなければ STALL')
    parser.add_argument('--zero-epochs', type=int, default=200,
                        help='exec_R_eps がゼロ近傍でこの epoch 数続いたら STALL')
    parser.add_argument('--exec-thresh', type=float, default=0.05,
                        help='exec_R_eps がこの値を超えたら PROGRESS 通知')
    parser.add_argument('--zero-thresh', type=float, default=0.02,
                        help='exec_R_eps がこの値未満をゼロ近傍と判定')
    parser.add_argument('--auto-kill', action='store_true',
                        help='STALL 検知時にプロセスを自動停止（危険）')
    parser.add_argument('--log', default='single_run/monitor.log',
                        help='ログ出力先ファイル')
    args = parser.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    run_dirs = [Path(r) for r in args.runs]
    states = {str(d): {} for d in run_dirs}

    log(f'監視開始: {[str(d) for d in run_dirs]}', log_path)
    log(f'設定: interval={args.interval}s, stall_epochs={args.stall_epochs}, '
        f'zero_epochs={args.zero_epochs}, exec_thresh={args.exec_thresh}', log_path)

    while True:
        for run_dir in run_dirs:
            label = run_dir.name
            prev = states[str(run_dir)]
            new_state, alerts = check_run(run_dir, prev, args)
            states[str(run_dir)] = new_state
            for alert in alerts:
                log(f'[{label}] {alert}', log_path)
            # auto-kill
            if args.auto_kill and any('STALL' in a for a in alerts):
                if new_state.get('alive', True):
                    log(f'[{label}] ⚠️  auto-kill を実行します...', log_path)
                    kill_run(str(run_dir))

        time.sleep(args.interval)


if __name__ == '__main__':
    main()
