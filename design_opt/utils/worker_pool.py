"""
ChoreonoidWorkerPool: spawn + persistent worker 方式による並列サンプリング。

各ワーカーは独立した choreonoid プロセスとして起動し、
エポックごとに policy params を受け取ってサンプリングを実行する。
fork ではなく subprocess.Popen なので Qt / CUDA の状態が汚染されない。
"""

import os
import subprocess
import multiprocessing.connection as mc
import multiprocessing
import torch

CHOREONOID = '/choreonoid_ws/install/bin/choreonoid'
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'scripts')
WORKER_SCRIPT = os.path.join(_SCRIPTS_DIR, 'worker_sampler.py')


class ChoreonoidWorkerPool:
    """
    N 個の永続 choreonoid ワーカープロセスを管理する。

    ライフサイクル:
        pool = ChoreonoidWorkerPool(n_workers=4, cfg=cfg, project_path=...)
        # → 起動時に N 個の choreonoid を立ち上げ、env + policy を初期化
        # → 各エポックで pool.sample() を呼ぶ
        pool.close()  # 学習終了時
    """

    def __init__(self, n_workers: int, cfg, project_path: str):
        self.n_workers = n_workers
        self._workers = []
        self._launch_workers(cfg, project_path)

    # ------------------------------------------------------------------
    def _launch_workers(self, cfg, project_path: str):
        from omegaconf import OmegaConf
        # cfg は Config ラッパー; 元の OmegaConf DictConfig を取り出す
        flags = cfg._flags if hasattr(cfg, '_flags') else cfg
        cfg_yaml = OmegaConf.to_yaml(flags)

        print(f'[WorkerPool] Launching {self.n_workers} choreonoid workers...', flush=True)

        for i in range(self.n_workers):
            # duplex=False: conn1=読み取り専用, conn2=書き込み専用
            # main → worker: main が書き込み側、worker が読み取り側
            req_worker_r, req_main_w = multiprocessing.Pipe(duplex=False)
            # worker → main: worker が書き込み側、main が読み取り側
            res_main_r, res_worker_w = multiprocessing.Pipe(duplex=False)

            # worker に継承させる FD (worker 側の端点)
            os.set_inheritable(req_worker_r.fileno(), True)
            os.set_inheritable(res_worker_w.fileno(), True)

            env = dict(os.environ)
            env.update({
                'CNOID_WORKER_ID':  str(i),
                'CNOID_REQ_FD':     str(req_worker_r.fileno()),   # worker が読む
                'CNOID_RES_FD':     str(res_worker_w.fileno()),   # worker が書く
                'USE_CHOREONOID':   '1',
                'OMP_NUM_THREADS':  '1',
            })

            log_path = f'/tmp/cnoid_worker_{i}.log'
            log_file = open(log_path, 'w')
            proc = subprocess.Popen(
                [CHOREONOID, '--no-window', '--python', WORKER_SCRIPT],
                env=env,
                pass_fds=(req_worker_r.fileno(), res_worker_w.fileno()),
                stdout=log_file,
                stderr=log_file,
            )

            # main プロセス側では worker 側の端点を閉じる
            req_worker_r.close()
            res_worker_w.close()

            self._workers.append({
                'proc': proc,
                'req':  req_main_w,   # main が書き込む
                'res':  res_main_r,   # main が読み取る
                'id':   i,
            })

        # 初期化メッセージを JSON で送る（torch の pickle を避けるため）
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
                raise RuntimeError(f"Worker {w['id']} init failed: {ack}")
            print(f'  [WorkerPool] worker {w["id"]} ready', flush=True)

        print(f'[WorkerPool] All {self.n_workers} workers ready.', flush=True)

    # ------------------------------------------------------------------
    def sample(self, min_batch_size: int, mean_action: bool,
               policy_net, obs_norm, traj_cls, logger_cls, logger_kwargs: dict):
        """
        N ワーカーに並列サンプリングを依頼してマージ結果を返す。

        Returns:
            (TrajBatch, Logger)  ← genesis_agent.sample() と同じ型
        """
        per_worker = min_batch_size // self.n_workers
        remainder  = min_batch_size - per_worker * self.n_workers

        # torch テンソルを numpy に変換してから pickle
        # → torch の shared-memory pickle プロトコルを回避
        import pickle
        policy_state_np = {k: v.cpu().numpy() for k, v in policy_net.state_dict().items()}
        obs_norm_state_np = None
        if obs_norm is not None:
            obs_norm_state_np = {k: v.cpu().numpy() if hasattr(v, 'numpy') else v
                                 for k, v in obs_norm.state_dict().items()}

        # 全ワーカーにリクエスト送信（非同期・send_bytes で raw bytes）
        for i, w in enumerate(self._workers):
            batch_size = per_worker + (remainder if i == self.n_workers - 1 else 0)
            msg_bytes = pickle.dumps({
                'cmd':            'sample',
                'batch_size':     batch_size,
                'mean_action':    mean_action,
                'policy_state':   policy_state_np,
                'obs_norm_state': obs_norm_state_np,
            })
            w['req'].send_bytes(msg_bytes)

        # 全ワーカーから結果を収集（Memory/Logger は numpy のみなので通常 pickle で OK）
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
                w['req'].send({'cmd': 'quit'})
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
        print('[WorkerPool] All workers terminated.', flush=True)
