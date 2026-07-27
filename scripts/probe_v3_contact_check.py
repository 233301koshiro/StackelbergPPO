#!/usr/bin/env python3
"""
probe_v3_contact_check.py: Bug16修正後、_check_initial_contact()が実際に
何を計算しているかを直接確認する（reset直後の1点のみ、学習なし）。

使い方:
  EVAL_RESTORE_DIR=single_run/_smoke_bug16_v3reach EVAL_CHECKPOINT=best \
  USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
      --no-window --python scripts/probe_v3_contact_check.py
"""
import os, sys
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
checkpoint = os.environ.get('EVAL_CHECKPOINT', 'best')

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

ckpt_arg = int(checkpoint) if checkpoint != 'best' else 'best'
agent = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                     seed=cfg.seed, num_threads=1, training=False, checkpoint=ckpt_arg)
env = agent.env

state = env.reset()
# 実行フェーズ直前まで進める（skeleton_transform/attribute_transform を通過）
for _ in range(50):
    if env.stage == 'execution':
        break
    sv = tensorfy([state])
    if agent.obs_norm is not None:
        sv = agent.normalize_observation(sv)
    with torch.no_grad():
        action = agent.policy_net.select_action(sv, mean_action=True).numpy().astype(np.float64)
    state, reward, done, _, info = env.step(action)

print(f'[probe] stage = {env.stage}')
print(f'[probe] check_init_contact flag = {env.env_specs.get("check_init_contact", True)}')
tip = env._arm_tip_pos
cube = env.get_body_com('cube')
print(f'[probe] _arm_tip_pos = {tip}')
print(f'[probe] cube_com     = {cube}')
print(f'[probe] dist (x,y)   = {np.linalg.norm(np.asarray(tip[:2]) - np.asarray(cube[:2])):.4f}')
cube_half = env._get_cube_half_size()
arm_rad = env._get_max_arm_radius()
thresh = cube_half + arm_rad + 0.03
print(f'[probe] cube_half={cube_half:.4f}  arm_rad={arm_rad:.4f}  thresh={thresh:.4f}')
print(f'[probe] _check_initial_contact() = {env._check_initial_contact()}')
os._exit(0)
