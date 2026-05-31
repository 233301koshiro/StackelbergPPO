#!/usr/bin/env python3.9
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
  python3.9 scripts/cnoid_transfer.py --mujoco-dir single_run/pusher

  # Transfer + auto-scratch if result is poor
  python3.9 scripts/cnoid_transfer.py --mujoco-dir single_run/pusher --auto-scratch

  # Specify epoch and custom threshold
  python3.9 scripts/cnoid_transfer.py --mujoco-dir single_run/pusher \\
      --epoch 100 --threshold 0.4 --auto-scratch

Notes
-----
- Choreonoid and MuJoCo rewards are on the same task scale (distance-based) but
  physics differences mean direct numeric comparison has ~30% noise.
- The threshold therefore has an implicit ×0.7 allowance built in at default 0.5.
- Run this script inside the conda env that has torch + zmq installed.
"""

import argparse
import os
import sys
import subprocess
import signal
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


# ── Process management ───────────────────────────────────────────────────────

def _wait_for_heartbeat(cf_path: str, timeout=60):
    import zmq, json
    cf = json.load(open(cf_path))
    ctx = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    sock.connect(f"tcp://localhost:{cf['hb_port']}")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            sock.send(b'ping')
            if sock.recv() == b'ping':
                sock.close(); ctx.term()
                return True
        except zmq.Again:
            pass
        except Exception:
            break
        time.sleep(1)
    sock.close(); ctx.term()
    return False


def _write_cnoid_connection_file():
    """Write a Jupyter connection file with a proper HMAC key for xeus-python."""
    import json, uuid, tempfile, socket
    ports = {}
    for name in ('shell', 'iopub', 'stdin', 'control', 'hb'):
        s = socket.socket(); s.bind(('', 0))
        ports[name] = s.getsockname()[1]; s.close()
    data = {
        'shell_port': ports['shell'], 'iopub_port': ports['iopub'],
        'stdin_port': ports['stdin'], 'control_port': ports['control'],
        'hb_port': ports['hb'], 'ip': '127.0.0.1',
        'key': uuid.uuid4().hex, 'transport': 'tcp',
        'signature_scheme': 'hmac-sha256', 'kernel_name': 'choreonoid',
    }
    fd, path = tempfile.mkstemp(suffix='.json')
    with os.fdopen(fd, 'w') as f: json.dump(data, f)
    return path, data


def start_choreonoid(server_script: str, port: int = 5556):
    """
    Start Choreonoid ZMQ server via jupyter_process.sh (lab standard).
    Returns (proc, cf_path).
    """
    import jupyter_client, zmq

    cf_path, _ = _write_cnoid_connection_file()

    proc = subprocess.Popen(
        ['jupyter_process.sh', 'choreonoid', cf_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"  Choreonoid starting (pid={proc.pid})...")

    if not _wait_for_heartbeat(cf_path, timeout=60):
        proc.terminate()
        raise RuntimeError("Choreonoid kernel heartbeat did not respond within 60s")
    print(f"  Kernel heartbeat confirmed.")

    import jupyter_client as jc
    kc = jc.BlockingKernelClient(connection_file=cf_path)
    kc.load_connection_file()
    kc.start_channels()

    print(f"  Executing server script in Choreonoid kernel...")
    with open(server_script) as f:
        kc.execute(f.read())

    # Wait for ZMQ server to respond
    print(f"  Waiting for ZMQ server on port {port}...")
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    sock.connect(f"tcp://localhost:{port}")
    deadline = time.time() + 30
    ready = False
    while time.time() < deadline:
        try:
            sock.send_json({'cmd': 'ping'})
            if sock.recv_json().get('status') == 'ok':
                ready = True
                break
        except Exception:
            pass
        time.sleep(0.5)
    sock.close(); ctx.term()

    if ready:
        print(f"  ZMQ server ready on port {port}")
    else:
        print(f"  WARNING: ZMQ server did not respond on port {port}")
    print()
    return proc, cf_path


def stop_choreonoid(proc, cf_path):
    print("\nShutting down Choreonoid server...")
    proc.terminate()
    proc.wait(timeout=5)
    try:
        os.unlink(cf_path)
    except OSError:
        pass


# ── Training runner ──────────────────────────────────────────────────────────

def run_train(cfg: str, overrides: list, use_choreonoid: bool = True) -> bool:
    """Call design_opt.train. stdout/stderr appended to {run_dir}/train.log."""
    env = {**os.environ}
    if use_choreonoid:
        env['USE_CHOREONOID'] = '1'
    env['OMP_NUM_THREADS'] = '1'
    cmd = [sys.executable, '-m', 'design_opt.train', f'cfg={cfg}'] + overrides
    print(f"  $ OMP_NUM_THREADS=1 USE_CHOREONOID={int(use_choreonoid)} {' '.join(cmd[2:])}")

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
    parser.add_argument('--server-script',
                        default='khrylib/rl/envs/common/cnoid_sim_server.py',
                        help='Path to cnoid_sim_server.py')
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

    # ── Start Choreonoid server ───────────────────────────────────────────────
    print("[Setup] Starting Choreonoid server...")
    proc, cf_path = start_choreonoid(args.server_script)

    def _cleanup(*_):
        stop_choreonoid(proc, cf_path)
    signal.signal(signal.SIGINT,  _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    try:
        # ── Step 1: Transfer training ─────────────────────────────────────────
        print("[Step 1/2] Transfer training  (morph_prior=true, reset_epoch=true)")
        epoch_override = f'epoch={args.epoch}'
        run_train(cfg, [
            f'+restore_dir={args.mujoco_dir}',
            epoch_override,
            'reset_epoch=true',
            'morph_prior=true',
            f'hydra.run.dir={transfer_dir}',
        ])

        # ── Step 2: Evaluate and decide ───────────────────────────────────────
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
                run_train(cfg, [
                    f'hydra.run.dir={scratch_dir}',
                ])
                scratch_reward = read_best_rewards(scratch_dir)

                print()
                print("=== Final Summary ===")
                print(f"  Transfer best_rewards : {cnoid_reward}")
                print(f"  Scratch  best_rewards : {scratch_reward}")
                if scratch_reward is not None and cnoid_reward is not None:
                    if cnoid_reward >= scratch_reward:
                        print(f"  Winner: transfer  → use {transfer_dir}")
                    else:
                        print(f"  Winner: scratch   → use {scratch_dir}")
            else:
                print()
                print("  → Transfer result is below threshold.")
                print("  → Re-run with --auto-scratch, or manually retrain from scratch:")
                print()
                print(f"    USE_CHOREONOID=1 python3.9 -m design_opt.train cfg={cfg} \\")
                print(f"        hydra.run.dir={scratch_dir}")
        else:
            print(f"\n  [OK] Transfer looks good. Checkpoint: {transfer_dir}/models/best.p")

    finally:
        _cleanup()


if __name__ == '__main__':
    main()
