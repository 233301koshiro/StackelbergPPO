#!/usr/bin/env python3.9
"""
Start Choreonoid ZMQ simulation server via the lab's Jupyter kernel mechanism.

Replaces the 'xvfb-run choreonoid --python' approach with the standard
lab setup: jupyter_client starts choreonoid --jupyter-connection, which
uses the existing irsl_entryrc / VGL infrastructure.

Usage:
  python3.9 scripts/start_cnoid_server.py

Then in another terminal:
  USE_CHOREONOID=1 OMP_NUM_THREADS=1 python3.9 -m design_opt.train \\
      cfg=pusher hydra.run.dir=single_run/pusher_cnoid enable_wandb=false
"""

import os
import sys
import time
import signal
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CNOID_SERVER_SCRIPT = os.path.join(
    PROJECT_ROOT, 'khrylib/rl/envs/common/cnoid_sim_server.py'
)
PORT = 5556


def wait_for_zmq(port=PORT, timeout=30):
    import zmq
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    sock.connect(f"tcp://localhost:{port}")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            sock.send_json({'cmd': 'ping'})
            if sock.recv_json().get('status') == 'ok':
                sock.close()
                ctx.term()
                return True
        except Exception:
            pass
        time.sleep(0.5)
    sock.close()
    ctx.term()
    return False


def wait_for_heartbeat(cf_path: str, timeout=60):
    """Poll the kernel heartbeat port until it responds.

    Uses DEALER (not REQ) so the socket stays valid after a recv timeout.
    """
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


def _write_connection_file():
    """Write a Jupyter connection file with a proper HMAC key."""
    import json, uuid, tempfile
    ports = {}
    import socket
    for name in ('shell', 'iopub', 'stdin', 'control', 'hb'):
        s = socket.socket()
        s.bind(('', 0))
        ports[name] = s.getsockname()[1]
        s.close()
    data = {
        'shell_port': ports['shell'],
        'iopub_port': ports['iopub'],
        'stdin_port': ports['stdin'],
        'control_port': ports['control'],
        'hb_port': ports['hb'],
        'ip': '127.0.0.1',
        'key': uuid.uuid4().hex,
        'transport': 'tcp',
        'signature_scheme': 'hmac-sha256',
        'kernel_name': 'choreonoid',
    }
    fd, path = tempfile.mkstemp(suffix='.json')
    with os.fdopen(fd, 'w') as f:
        json.dump(data, f)
    return path, data


def start_server(server_script=CNOID_SERVER_SCRIPT, port=PORT):
    """
    Start Choreonoid via jupyter_process.sh (lab standard) and execute the
    ZMQ server script in the kernel.  Returns (proc, cf_path).

    Uses direct heartbeat polling instead of kc.wait_for_ready() because
    Choreonoid can take > 30 s to initialize the Python kernel.
    """
    import jupyter_client

    cf_path, _ = _write_connection_file()

    # jupyter_process.sh handles irsl_entryrc sourcing and optional VGL
    # Choreonoid stdout/stderr → cnoid_console.log (append) for debugging
    cnoid_log = open('/tmp/cnoid_console.log', 'a')
    proc = subprocess.Popen(
        ['jupyter_process.sh', 'choreonoid', cf_path],
        stdout=cnoid_log, stderr=cnoid_log,
    )
    print(f"[cnoid_server] Choreonoid starting (pid={proc.pid})...")

    if not wait_for_heartbeat(cf_path, timeout=60):
        proc.terminate()
        raise RuntimeError("Choreonoid kernel heartbeat did not respond within 60s")
    print("[cnoid_server] Kernel heartbeat confirmed.")

    kc = jupyter_client.BlockingKernelClient(connection_file=cf_path)
    kc.load_connection_file()
    kc.start_channels()

    print("[cnoid_server] Executing server script in Choreonoid kernel...")
    with open(server_script) as f:
        kc.execute(f.read())

    return proc, cf_path


def start_servers(num_servers=1, base_port=PORT):
    """
    Start num_servers Choreonoid ZMQ servers on consecutive ports.
    Worker i uses port base_port+i (i=0 is the main process, i=1..N are forked workers).
    Returns list of (proc, cf_path) tuples.
    """
    servers = []
    for i in range(num_servers):
        port = base_port + i
        # Inject port into server script via a wrapper that sets CNOID_ZMQ_PORT
        server_code = open(CNOID_SERVER_SCRIPT).read().replace(
            'PORT = 5556', f'PORT = {port}', 1
        )
        import tempfile
        fd, patched_script = tempfile.mkstemp(suffix='.py')
        with os.fdopen(fd, 'w') as f:
            f.write(server_code)

        proc, cf_path = start_server(server_script=patched_script, port=port)
        servers.append((proc, cf_path, patched_script))
        print(f"[cnoid_server] Server {i} ready on port {port}")
    return servers


def stop_servers(servers):
    for proc, cf_path, patched_script in servers:
        proc.terminate()
        proc.wait(timeout=5)
        for p in (cf_path, patched_script):
            try:
                os.unlink(p)
            except OSError:
                pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-servers', type=int, default=1,
                        help='Number of Choreonoid ZMQ servers to start (default: 1)')
    parser.add_argument('--base-port', type=int, default=PORT,
                        help=f'Base port (default: {PORT}). Servers use base, base+1, ...')
    args = parser.parse_args()

    servers = start_servers(args.num_servers, args.base_port)

    def _cleanup(signum=None, frame=None):
        print("\n[cnoid_server] Shutting down all servers...")
        stop_servers(servers)
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    print(f"[cnoid_server] {args.num_servers} server(s) running. Press Ctrl+C to stop.")
    while all(proc.poll() is None for proc, _, _ in servers):
        time.sleep(1)
    print("[cnoid_server] A Choreonoid process exited.")


if __name__ == '__main__':
    main()
