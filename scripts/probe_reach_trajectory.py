#!/usr/bin/env python3
"""
probe_reach_trajectory.py: Reach 実行フェーズの先端位置・関節角度の推移を実測する。

用途: 「初期姿勢が target から遠い方向を向いている場合、動かないことが局所解に
なっていないか」を検証する（advisor 指摘への回答）。

使い方:
  EVAL_RESTORE_DIR=single_run/tripo_arm_v2c_reach EVAL_CHECKPOINT=best \
  USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
      --no-window --python scripts/probe_reach_trajectory.py
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

target = np.array([
    cfg.env_specs.get('target_x', 0.8),
    cfg.env_specs.get('target_y', 0.0),
    cfg.env_specs.get('target_z', 0.15),
])
print(f'[reach_trace] target = {target}')

ckpt_arg = int(checkpoint) if checkpoint != 'best' else 'best'
agent = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                     seed=cfg.seed, num_threads=1, training=False, checkpoint=ckpt_arg)
env = agent.env


def get_arm_tip(env):
    """env._body_names[-1] は cube を指すことがある（腕の body ではない）ため、
    env.robot.bodies（設計対象の腕のみのリスト）から末端リンクの tip 位置を実測する。"""
    bodies = env.robot.bodies
    b_last = bodies[-1]
    bo_last = getattr(b_last, 'bone_offset', None)
    if bo_last is None:
        bo_last = np.array([0.25, 0.0, 0.0])
    pos_last = np.array(env._body_xpos.get(b_last.name, np.zeros(3)))
    R_last = np.array(env._body_xmat.get(b_last.name, np.eye(3)))
    return pos_last + R_last @ np.asarray(bo_last, dtype=float)


state = env.reset()
n_joints = env.model.nu if hasattr(env, 'model') else None
trace = []
t_exec = 0
for _ in range(cfg.skel_transform_nsteps + 2 + 1100):
    in_exec = env.stage == 'execution'
    sv = tensorfy([state])
    if agent.obs_norm is not None:
        sv = agent.normalize_observation(sv)
    with torch.no_grad():
        action = agent.policy_net.select_action(sv, mean_action=True).numpy().astype(np.float64)
    state, reward, done, _, info = env.step(action)
    if in_exec:
        tip = get_arm_tip(env)
        dist = float(np.linalg.norm(tip - target))
        qpos = env.data.qpos.copy()
        ctrl = float(np.linalg.norm(action[:env.control_action_dim]))
        trace.append((t_exec, tip.copy(), dist, qpos.copy(), ctrl))
        t_exec += 1
    if done:
        break

print(f'[reach_trace] {restore_dir} ckpt={checkpoint} exec_steps={len(trace)}')
print(f'[reach_trace] body_names = {env._body_names}')
marks = [0, 1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 999]
for t, tip, dist, qpos, ctrl in trace:
    if t in marks:
        qpos_str = ' '.join(f'{q:+.3f}' for q in qpos[:3])
        print(f'  t={t:4d}: tip=({tip[0]:+.3f},{tip[1]:+.3f},{tip[2]:+.3f})  dist={dist:6.3f}  qpos=[{qpos_str}]  ctrl={ctrl:.3f}')

if trace:
    dists = [d for _, _, d, _, _ in trace]
    tips = [t for _, t, _, _, _ in trace]
    tip_travel = sum(np.linalg.norm(tips[i+1] - tips[i]) for i in range(len(tips)-1))
    print(f'[reach_trace] dist: start={dists[0]:.3f} min={min(dists):.3f} final={dists[-1]:.3f}')
    print(f'[reach_trace] tip total travel distance over episode = {tip_travel:.3f} m')
    print(f'[reach_trace] tip start = {tips[0]}, tip final = {tips[-1]}')

os._exit(0)
