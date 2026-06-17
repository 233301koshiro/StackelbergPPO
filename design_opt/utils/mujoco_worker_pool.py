"""
MujocoWorkerPool: ChoreonoidWorkerPool と同じ永続ワーカー方式を MuJoCo 用に実装。

各ワーカーは独立した python3 サブプロセスとして起動し、
エポックごとに policy params を受け取ってサンプリングを実行する。
OMP_NUM_THREADS=1 を設定してスレッド競合を防ぐ。
"""

import os
import sys
import subprocess
import multiprocessing.connection as mc
import multiprocessing
import pickle
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_SCRIPTS_DIR  = os.path.join(_PROJECT_ROOT, 'scripts')
WORKER_SCRIPT = os.path.join(_SCRIPTS_DIR, 'mujoco_worker_sampler.py')


class MujocoWorkerPool:
    """
    N 個の永続 python3 ワーカープロセスを管理する。
    ChoreonoidWorkerPool と同じインターフェースを持つ。
    """

    def __init__(self, n_workers: int, cfg, project_path: str):
        self.n_workers = n_workers
        self._workers  = []
        self._launch_workers(cfg, project_path)

    # ------------------------------------------------------------------
    def _launch_workers(self, cfg, project_path: str):
        from omegaconf import OmegaConf
        flags    = cfg._flags if hasattr(cfg, '_flags') else cfg
        cfg_yaml = OmegaConf.to_yaml(flags)

        print(f'[MujocoWorkerPool] Launching {self.n_workers} workers...', flush=True)

        for i in range(self.n_workers):
            req_worker_r, req_main_w = multiprocessing.Pipe(duplex=False)
            res_main_r, res_worker_w = multiprocessing.Pipe(duplex=False)

            os.set_inheritable(req_worker_r.fileno(), True)
            os.set_inheritable(res_worker_w.fileno(), True)

            env = dict(os.environ)
            env.update({
                'MUJOCO_WORKER_ID': str(i),
                'MUJOCO_REQ_FD':    str(req_worker_r.fileno()),
                'MUJOCO_RES_FD':    str(res_worker_w.fileno()),
                'USE_CHOREONOID':   '0',
                'OMP_NUM_THREADS':  '1',
            })

            log_path = f'/tmp/mujoco_worker_{i}.log'
            log_file = open(log_path, 'w')
            proc = subprocess.Popen(
                [sys.executable, WORKER_SCRIPT],
                env=env,
                pass_fds=(req_worker_r.fileno(), res_worker_w.fileno()),
                stdout=log_file,
                stderr=log_file,
            )

            req_worker_r.close()
            res_worker_w.close()

            self._workers.append({
                'proc': proc,
                'req':  req_main_w,
                'res':  res_main_r,
                'id':   i,
            })

        # 初期化メッセージを JSON で送る
        import json
        init_bytes = json.dumps({
            'cmd':          'init',
            'cfg_yaml':     cfg_yaml,
            'project_path': project_path,
        }).encode()
        for w in self._workers:
            w['req'].send_bytes(init_bytes)

        # 全ワーカーの準備完了を待つ
        for w in self._workers:
            ack = w['res'].recv_bytes().decode()
            if ack != 'ready':
                raise RuntimeError(f"MuJoCo worker {w['id']} init failed: {ack}")
            print(f'  [MujocoWorkerPool] worker {w["id"]} ready', flush=True)

        print(f'[MujocoWorkerPool] All {self.n_workers} workers ready.', flush=True)

    # ------------------------------------------------------------------
    def sample(self, min_batch_size: int, mean_action: bool,
               policy_net, obs_norm, traj_cls, logger_cls, logger_kwargs: dict):
        per_worker = min_batch_size // self.n_workers
        remainder  = min_batch_size - per_worker * self.n_workers

        policy_state_np = {k: v.cpu().numpy() for k, v in policy_net.state_dict().items()}
        obs_norm_state_np = None
        if obs_norm is not None:
            obs_norm_state_np = {
                k: v.cpu().numpy() if hasattr(v, 'numpy') else v
                for k, v in obs_norm.state_dict().items()
            }

        # 全ワーカーに非同期でリクエスト送信
        for i, w in enumerate(self._workers):
            batch_size = per_worker + (remainder if i == self.n_workers - 1 else 0)
            w['req'].send_bytes(pickle.dumps({
                'cmd':            'sample',
                'batch_size':     batch_size,
                'mean_action':    mean_action,
                'policy_state':   policy_state_np,
                'obs_norm_state': obs_norm_state_np,
            }))

        # 全ワーカーから結果を収集
        memories = []
        loggers  = []
        for w in self._workers:
            result = pickle.loads(w['res'].recv_bytes())
            memories.append(result['memory'])
            loggers.append(result['logger'])

        batch  = traj_cls(memories)
        logger = logger_cls.merge(loggers, **logger_kwargs)
        return batch, logger

    # ------------------------------------------------------------------
    def close(self):
        for w in self._workers:
            try:
                w['req'].send_bytes(pickle.dumps({'cmd': 'quit'}))
            except Exception:
                pass
            w['proc'].terminate()
            try:
                w['proc'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                w['proc'].kill()
            w['req'].close()
            w['res'].close()
        self._workers.clear()
        print('[MujocoWorkerPool] All workers terminated.', flush=True)
