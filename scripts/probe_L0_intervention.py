#!/usr/bin/env python3
"""
probe_L0_intervention.py: J1 崩壊原因の介入実験。

「L0（root bone_offset）は運動学的に不活性（no_root_offset）だが、設計パラメータとして
観測ベクトルに入るため、L0 の drift が学習済み方策を壊した」という仮説を検証する。

方法: 指定 checkpoint の方策で形態フェーズを通常どおり実行した直後、実行フェーズに入る前に
bodies[0].bone_offset のノルムだけを指定値に書き換え（方向は維持）、実行リターンを比較する。
物理は L0 に依存しないため、リターン差が出ればそれは観測経由の因果。

使い方:
  EVAL_RESTORE_DIR=single_run/rrbot_arm_reach_J1 EVAL_CHECKPOINT=360 \
  L0_TARGETS="natural,0.53,0.16" \
  USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
      --no-window --python scripts/probe_L0_intervention.py
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
targets = os.environ.get('L0_TARGETS', 'natural,0.53,0.16').split(',')

FLAGS = OmegaConf.create(yaml.safe_load(open(f'{restore_dir}/.hydra/config.yaml')))
flags_dict = OmegaConf.to_container(FLAGS, resolve=True)
flags_dict.pop('restore_dir', None)
FLAGS = OmegaConf.create(flags_dict)
cfg = Config(FLAGS, project_path, restore_dir)
cfg.restore_dir = restore_dir
torch.set_default_dtype(torch.float64)
set_global_seed(cfg.seed)

ckpt_arg = int(checkpoint) if checkpoint != 'best' else 'best'
agent = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                     seed=cfg.seed, num_threads=1, training=False, checkpoint=ckpt_arg)
env = agent.env

def run_episode(l0_target):
    state = env.reset()
    exec_R = 0.0
    n_exec = 0
    intervened = False
    for _ in range(cfg.skel_transform_nsteps + 2 + 1100):
        if (not intervened) and env.stage == 'execution':
            bo = np.asarray(env.robot.bodies[0].bone_offset, dtype=float)
            natural_norm = float(np.linalg.norm(bo))
            if l0_target != 'natural' and natural_norm > 1e-9:
                before = env.get_attr_design()[0].copy()
                env.robot.bodies[0].bone_offset = bo * (float(l0_target) / natural_norm)
                after = env.get_attr_design()[0].copy()
                delta = float(np.abs(after - before).max())
                print(f'    [intervene] L0 {natural_norm:.3f} -> {l0_target} '
                      f'(obs row0 max delta = {delta:.4f})')
                assert delta > 1e-6, 'obs did not change — intervention path is broken'
            else:
                print(f'    [natural] L0 = {natural_norm:.3f}')
            intervened = True
        sv = tensorfy([state])
        if agent.obs_norm is not None:
            sv = agent.normalize_observation(sv)
        with torch.no_grad():
            action = agent.policy_net.select_action(sv, mean_action=True).numpy().astype(np.float64)
        state, reward, done, _, info = env.step(action)
        if info.get('stage') == 'execution' or env.stage == 'execution':
            if intervened:
                exec_R += float(reward)
                n_exec += 1
        if done:
            break
    return exec_R, n_exec

print(f'[L0_probe] {restore_dir} checkpoint={checkpoint}')
for t in targets:
    R, n = run_episode(t)
    label = 'natural' if t == 'natural' else f'L0={t}'
    print(f'  {label:>12}: exec_R_eps = {R:9.2f}  (exec steps = {n})')
os._exit(0)
