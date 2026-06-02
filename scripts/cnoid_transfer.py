#!/usr/bin/env python3
"""
Migrate a MuJoCo-trained StackelbergPPO checkpoint to Choreonoid.

Workflow
--------
1. Load MuJoCo checkpoint (morph_prior=true  → morphology weights only)
2. Retrain in Choreonoid from epoch 0         (reset_epoch=true)
3. Compare best_rewards; if below --threshold → optionally retrain from scratch

Usage examples
--------------
  # Try transfer; print recommendation but don't auto-scratch
  python3 scripts/cnoid_transfer.py --mujoco-dir single_run/pusher

  # Transfer + auto-scratch if result is poor
  python3 scripts/cnoid_transfer.py --mujoco-dir single_run/pusher --auto-scratch

  # Specify epoch and custom threshold
  python3 scripts/cnoid_transfer.py --mujoco-dir single_run/pusher \\
      --epoch 100 --threshold 0.4 --auto-scratch

Notes
-----
- Choreonoid and MuJoCo rewards are on the same task scale (distance-based) but
  physics differences mean direct numeric comparison has ~30% noise.
- The threshold therefore has an implicit ×0.7 allowance built in at default 0.5.
"""

import argparse
import os
import sys
import subprocess
import time

# ── Checkpoint helpers ───────────────────────────────────────────────────────

def _cp_path(run_dir: str, epoch) -> str:
    models = os.path.join(run_dir, 'models')
    if str(epoch).isnumeric():
        return os.path.join(models, f'epoch_{int(epoch):04d}.p')
    return os.path.join(models, f'{epoch}.p')


def read_best_rewards(run_dir: str, epoch='best'):
    path = _cp_path(run_dir, epoch)
    if not os.path.exists(path):
        print(f"  [WARN] checkpoint not found: {path}")
        return None
    try:
        import torch
        cp = torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        import pickle
        try:
            cp = pickle.load(open(path, 'rb'))
        except Exception as e:
            print(f"  [WARN] could not read {path}: {e}")
            return None
    return cp.get('best_rewards')


def get_cfg_from_run(run_dir: str):
    path = os.path.join(run_dir, '.hydra', 'overrides.yaml')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            line = line.strip().lstrip('- ')
            if line.startswith('cfg='):
                return line[4:]
    return None


# ── Training runner ──────────────────────────────────────────────────────────

CHOREONOID = '/choreonoid_ws/install/bin/choreonoid'
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_LAUNCHER = os.path.join(_SCRIPTS_DIR, 'choreonoid_train.py')


def run_train(cfg: str, overrides: list) -> bool:
    """Run training inside choreonoid --no-window. Logs to {run_dir}/train.log."""
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'USE_CHOREONOID': '1'}
    cmd = [CHOREONOID, '--no-window', '--python', TRAIN_LAUNCHER, f'cfg={cfg}'] + overrides
    print(f"  $ choreonoid --no-window --python choreonoid_train.py cfg={cfg} {' '.join(overrides)}")

    # Determine log file from hydra.run.dir override or default
    run_dir = f'single_run/{cfg}'
    for ov in overrides:
        if ov.startswith('hydra.run.dir='):
            run_dir = ov.split('=', 1)[1]
            break
    log_file = os.path.join(run_dir, 'train.log')
    os.makedirs(run_dir, exist_ok=True)
    print(f"  Logging to: {log_file}  (append)")

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with open(log_file, 'a') as lf:
        for line in iter(proc.stdout.readline, b''):
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
            lf.write(line.decode(errors='replace'))
            lf.flush()
    proc.wait()
    return proc.returncode == 0


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Migrate MuJoCo-trained StackelbergPPO to Choreonoid',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--mujoco-dir', required=True,
                        help='MuJoCo training run directory (e.g. single_run/pusher)')
    parser.add_argument('--epoch', default='best',
                        help='Checkpoint to load: "best" or an epoch number (default: best)')
    parser.add_argument('--cfg', default=None,
                        help='Config name; inferred from --mujoco-dir if omitted')
    parser.add_argument('--transfer-dir', default=None,
                        help='Output dir for transfer run (default: {mujoco-dir}_cnoid_transfer)')
    parser.add_argument('--scratch-dir', default=None,
                        help='Output dir for scratch run   (default: {mujoco-dir}_cnoid_scratch)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='If Choreonoid_reward / MuJoCo_reward < threshold, flag as poor '
                             '(default: 0.5). Accounts for ~30%% physics gap between simulators.')
    parser.add_argument('--auto-scratch', action='store_true',
                        help='Automatically retrain from scratch when transfer result is poor')
    args = parser.parse_args()

    # ── Resolve config ───────────────────────────────────────────────────────
    cfg = args.cfg or get_cfg_from_run(args.mujoco_dir)
    if not cfg:
        sys.exit(f"ERROR: cannot infer cfg from {args.mujoco_dir}/.hydra/overrides.yaml\n"
                 f"       Please pass --cfg explicitly.")

    transfer_dir = args.transfer_dir or f'{args.mujoco_dir}_cnoid_transfer'
    scratch_dir  = args.scratch_dir  or f'{args.mujoco_dir}_cnoid_scratch'

    print("=== Choreonoid Transfer Migration ===")
    print(f"  MuJoCo source  : {args.mujoco_dir}  (epoch={args.epoch})")
    print(f"  Config         : {cfg}")
    print(f"  Transfer output: {transfer_dir}")
    print(f"  Threshold      : {args.threshold}")
    print(f"  Auto-scratch   : {args.auto_scratch}")
    print()

    # ── Read MuJoCo baseline rewards ─────────────────────────────────────────
    mujoco_reward = read_best_rewards(args.mujoco_dir, args.epoch)
    print(f"  MuJoCo best_rewards: {mujoco_reward}")
    print()

    # ── Step 1: Transfer training ─────────────────────────────────────────────
    print("[Step 1/2] Transfer training  (morph_prior=true, reset_epoch=true)")
    run_train(cfg, [
        f'+restore_dir={args.mujoco_dir}',
        f'epoch={args.epoch}',
        'reset_epoch=true',
        'morph_prior=true',
        f'hydra.run.dir={transfer_dir}',
    ])

    # ── Step 2: Evaluate and decide ───────────────────────────────────────────
    print()
    print("[Step 2/2] Comparing rewards...")
    cnoid_reward = read_best_rewards(transfer_dir)

    print()
    print("=== Result Summary ===")
    print(f"  MuJoCo   best_rewards : {mujoco_reward}")
    print(f"  Choreonoid transfer   : {cnoid_reward}  (dir: {transfer_dir})")

    poor_result = False
    if cnoid_reward is None:
        print("  [WARN] Could not read Choreonoid rewards; transfer may have failed.")
        poor_result = True
    elif mujoco_reward is not None and mujoco_reward > 0:
        ratio = cnoid_reward / mujoco_reward
        print(f"  Ratio (Choreonoid/MuJoCo): {ratio:.2f}  [threshold: {args.threshold}]")
        if ratio < args.threshold:
            print(f"  [WARN] Ratio {ratio:.2f} is below threshold {args.threshold}.")
            poor_result = True
    else:
        print("  [INFO] MuJoCo reward ≤ 0; skipping ratio check.")

    if poor_result:
        if args.auto_scratch:
            print(f"\n  → Retraining from scratch in Choreonoid → {scratch_dir}")
            run_train(cfg, [f'hydra.run.dir={scratch_dir}'])
            scratch_reward = read_best_rewards(scratch_dir)

            print()
            print("=== Final Summary ===")
            print(f"  Transfer best_rewards : {cnoid_reward}")
            print(f"  Scratch  best_rewards : {scratch_reward}")
            if scratch_reward is not None and cnoid_reward is not None:
                winner = transfer_dir if cnoid_reward >= scratch_reward else scratch_dir
                print(f"  Winner: {winner}")
        else:
            print()
            print("  → Transfer result is below threshold.")
            print("  → Re-run with --auto-scratch, or manually retrain from scratch:")
            print()
            print(f"    choreonoid --no-window --python scripts/choreonoid_train.py \\")
            print(f"        cfg={cfg} hydra.run.dir={scratch_dir}")
    else:
        print(f"\n  [OK] Transfer looks good. Checkpoint: {transfer_dir}/models/best.p")


if __name__ == '__main__':
    main()
