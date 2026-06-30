"""
MujocoWorkerPool のワーカープロセス。
python3 scripts/mujoco_worker_sampler.py として起動される。

環境変数:
  MUJOCO_WORKER_ID  ワーカー番号
  MUJOCO_REQ_FD     メインからのリクエストを受け取る Pipe の fd
  MUJOCO_RES_FD     メインへ結果を送る Pipe の fd
"""

import sys
import os

# MuJoCo OpenMP スレッドを 1 に制限（複数ワーカー間の競合を防ぐ）
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ['USE_CHOREONOID'] = '0'

import multiprocessing.connection as mc
import torch
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

WORKER_ID = int(os.environ['MUJOCO_WORKER_ID'])
req_conn = mc.Connection(int(os.environ['MUJOCO_REQ_FD']), readable=True,  writable=False)
res_conn = mc.Connection(int(os.environ['MUJOCO_RES_FD']), readable=False, writable=True)

import json, pickle, select

print(f'[mujoco_worker {WORKER_ID}] starting...', flush=True)

# ---- 初期化メッセージを JSON bytes で受け取る ----------------------------------------
rdy, _, _ = select.select([req_conn], [], [], 120)
if not rdy:
    print(f'[mujoco_worker {WORKER_ID}] ERROR: no init msg after 120s', flush=True)
    sys.exit(1)
init_msg = json.loads(req_conn.recv_bytes().decode())
assert init_msg['cmd'] == 'init'

from omegaconf import OmegaConf
from design_opt.utils.config import Config
from design_opt.envs import env_dict
from design_opt.models.bodygen_policy import BodyGenPolicy
from khrylib.rl.core.running_norm import RunningNorm
from design_opt.utils.logger import LoggerRLV1
from design_opt.utils.tools import TrajBatchDisc, set_global_seed
from khrylib.rl.agents.agent import Memory


def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) if i <= 1 or i == 4 or i >= 7 else x
                 for i, x in enumerate(y)] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]


FLAGS = OmegaConf.create(OmegaConf.to_object(OmegaConf.create(init_msg['cfg_yaml'])))
project_path = init_msg['project_path']

cfg = Config(FLAGS, project_path, f'/tmp/mujoco_worker_{WORKER_ID}')

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cpu')
set_global_seed(cfg.seed + WORKER_ID + 1)

# ---- 環境構築 -----------------------------------------------------------------------
env = env_dict[cfg.env_name](cfg, agent=None)
env.seed(cfg.seed + WORKER_ID + 1)

# ---- ポリシー（推論専用 CPU コピー）--------------------------------------------------
class _AgentProxy:
    def __init__(self, env, cfg):
        self.attr_fixed_dim     = env.attr_fixed_dim
        self.sim_obs_dim        = env.sim_obs_dim
        self.attr_design_dim    = env.attr_design_dim
        self.skel_num_action    = env.skel_num_action
        self.control_action_dim = env.control_action_dim
        self.cfg                = cfg

agent_proxy = _AgentProxy(env, cfg)
policy_net = BodyGenPolicy(cfg.policy_specs, agent_proxy)
policy_net.to(device)
policy_net.eval()

if cfg.uni_obs_norm:
    state_dim = env.attr_fixed_dim + env.sim_obs_dim + env.attr_design_dim
    obs_norm = RunningNorm(state_dim).to(device)
else:
    obs_norm = None

# ---- 準備完了を通知 ------------------------------------------------------------------
res_conn.send_bytes(b'ready')
print(f'[mujoco_worker {WORKER_ID}] ready', flush=True)

# ---- サンプリングループ ---------------------------------------------------------------
while True:
    msg = pickle.loads(req_conn.recv_bytes())
    cmd = msg['cmd']

    if cmd == 'quit':
        if hasattr(env, 'close'):
            env.close()
        break

    elif cmd == 'sample':
        # policy / obs_norm を numpy → tensor で更新
        policy_net.load_state_dict(
            {k: torch.from_numpy(v) for k, v in msg['policy_state'].items()})
        if obs_norm is not None and msg.get('obs_norm_state') is not None:
            obs_norm.load_state_dict(
                {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                 for k, v in msg['obs_norm_state'].items()})
            obs_norm.eval()

        batch_size  = msg['batch_size']
        mean_action = msg['mean_action']
        noise_rate  = msg.get('noise_rate', 1.0)

        memory = Memory()
        logger = LoggerRLV1()

        while logger.num_steps < batch_size:
            state = env.reset()
            logger.start_episode(env)

            while True:
                state_var = tensorfy([state])

                use_mean = mean_action or torch.bernoulli(
                    torch.tensor([1 - noise_rate])).item()

                with torch.no_grad():
                    action = policy_net.select_action(
                        state_var, use_mean).numpy().astype(np.float64)

                next_state, env_reward, termination, truncation, info = env.step(action)
                reward  = env_reward
                c_reward = info.get('reward_ctrl', 0)

                if info['stage'] == 'execution':
                    reward += cfg.reward_shift

                logger.step(env, env_reward, c_reward, info.get('reward_breakdown', np.array([0.0, 0.0])), info)

                done = termination or truncation
                exp  = 1 - use_mean
                memory.push(state, action, termination, done,
                            next_state, reward, exp, c_reward)

                if done:
                    break
                state = next_state

            logger.end_episode(env)

        logger.end_sampling()
        res_conn.send_bytes(pickle.dumps({'memory': memory, 'logger': logger}))

print(f'[mujoco_worker {WORKER_ID}] exiting.', flush=True)
