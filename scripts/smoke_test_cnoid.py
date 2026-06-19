#!/usr/bin/env python3
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
os.environ.setdefault('USE_CHOREONOID', '1')

import numpy as np
from omegaconf import OmegaConf
from design_opt.utils.config import Config
from design_opt.envs import env_dict

FLAG = OmegaConf.create({'cfg': 'pusher', 'xml_name': 'rrbot_arm', 'project_path': _ROOT, 'seed': 0, 'uni_obs_norm': False, 'enable_wandb': False, 'env_init_height': False})
cfg = Config(FLAG, _ROOT, '/tmp/smoke_test_cnoid')
env = env_dict[cfg.env_name](cfg, agent=None)
total_adim = env.control_action_dim + env.attr_design_dim + 1

for ep in range(3):
    obs = env.reset()
    stage_done = False
    while not stage_done:
        action = np.zeros((len(env.robot.bodies), total_adim), dtype=np.float32)
        # Apply random CONTROL actions only
        action[:, :env.control_action_dim] = np.random.uniform(-1, 1, (len(env.robot.bodies), env.control_action_dim))
        
        obs, reward, term, trunc, info = env.step(action)
        if not np.all(np.isfinite(np.concatenate([np.atleast_1d(o).ravel() for o in obs if o is not None]))):
            print(f"NaN generated at ep {ep}!")
            sys.exit(1)
        if term or trunc: stage_done = True
print("PASS RANDOM CONTROL")
