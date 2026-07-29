#!/usr/bin/env python3
"""
probe_k1_trajectory.py: 任意のrestore_dirについて、形態（bone_offset/gear）が
学習を通じてどう推移したかを、複数のepoch checkpointを順番に読み込んで確認する。
（K1_v2の調査用に作成したが、EVAL_RESTORE_DIRを変えれば任意のrunに使える）

使い方:
  EVAL_RESTORE_DIR=single_run/rrbot_arm_pusher_K1_v2 \
  USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
      --no-window --python scripts/probe_k1_trajectory.py

環境変数（省略時のデフォルトは元のK1調査時の挙動と同一）:
  EVAL_EPOCH_STEP  サンプリング間隔（epoch数、既定100）
  EVAL_DECIMALS    出力の小数点以下桁数（既定3）
  EVAL_MAX_EPOCH   これ以下のepochのみ対象（既定0=制限なし）
"""
import os, sys, glob, re
sys.path.append(os.getcwd())
os.environ['USE_CHOREONOID'] = '1'

import numpy as np
import yaml
import torch
from omegaconf import OmegaConf

from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent, tensorfy
from design_opt.utils.tools import set_global_seed

project_path = os.getcwd()
restore_dir = os.environ['EVAL_RESTORE_DIR']

FLAGS = OmegaConf.create(yaml.safe_load(open(f'{restore_dir}/.hydra/config.yaml')))
flags_dict = OmegaConf.to_container(FLAGS, resolve=True)
flags_dict.pop('restore_dir', None)
FLAGS = OmegaConf.create(flags_dict)
cfg = Config(FLAGS, project_path, restore_dir)
cfg.restore_dir = restore_dir
cfg.control_prior = False
cfg.morph_prior = False
torch.set_default_dtype(torch.float64)
set_global_seed(cfg.seed)

decimals = int(os.environ.get('EVAL_DECIMALS', '3'))
if os.environ.get('EVAL_CHECKPOINT'):
    sample_eps = [os.environ['EVAL_CHECKPOINT']]  # e.g. 'best' -> models/best.p
else:
    ckpts = sorted(glob.glob(f'{restore_dir}/models/epoch_*.p'))
    eps = [int(re.search(r'epoch_(\d+)\.p', c).group(1)) for c in ckpts]
    step = int(os.environ.get('EVAL_EPOCH_STEP', '100'))
    max_ep = int(os.environ.get('EVAL_MAX_EPOCH', '0')) or max(eps)
    sample_eps = [e for e in eps if ((e - min(eps)) % step == 0 or e == max(eps)) and e <= max_ep]

print(f'[k1_traj] sample_eps={sample_eps}', flush=True)
try:
    for ep in sample_eps:
        print(f'[k1_traj] loading epoch {ep}...', flush=True)
        agent = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                             seed=cfg.seed, num_threads=1, training=False, checkpoint=ep)
        env = agent.env
        # Leaderの出力（bone_offset/gear）はattribute_transformステージを実際に
        # ステップして初めてenv.robot.bodiesへ反映される（reset直後は初期XMLの
        # デフォルト値のまま）。eval_cnoid_visual.py等の既存プローブと同じ手順。
        state = env.reset()
        for _ in range(cfg.skel_transform_nsteps + 2):
            if env.stage == 'execution':
                break
            sv = tensorfy([state])
            if agent.obs_norm is not None:
                sv = agent.normalize_observation(sv)
            with torch.no_grad():
                action = agent.policy_net.select_action(sv, mean_action=True).numpy().astype(np.float64)
            state, reward, done, _, info = env.step(action)
        bodies = env.robot.bodies
        parts = []
        for b in bodies[1:]:
            off = np.asarray(b.bone_offset, dtype=float)
            length = float(np.linalg.norm(off))
            off_str = ','.join(f'{v:.{decimals}f}' for v in off)
            gear = None
            for joint in b.joints:
                if joint.actuator:
                    gear = float(joint.actuator.gear)
            gear_str = f'{gear:.{decimals}f}' if gear is not None else 'N/A'
            parts.append(f'{b.name}:off=[{off_str}],len={length:.{decimals}f},gear={gear_str}')
        print(f'[k1_traj] ep={ep}: {" ".join(parts)}', flush=True)
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
sys.stdout.flush()
os._exit(0)
