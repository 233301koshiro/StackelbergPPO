#!/usr/bin/env python3
"""
probe_cube_trace.py: 実行フェーズの cube_x(t)・先端-cube 距離・ctrl ノルムを実測する。

用途: TP1「押して→減速→target 手前で静止」の確認、I1 vs L2 の bat/押し行動比較、
L2 vs L2_s2 の戦略差の定量化。

使い方:
  EVAL_RESTORE_DIR=single_run/rrbot_arm_targetpusher_TP1 EVAL_CHECKPOINT=best \
  USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
      --no-window --python scripts/probe_cube_trace.py
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
if os.environ.get('PROBE_DISABLE_INIT_CHECK') == '1':
    # 境界ぎりぎり形態の観察用: 初期接触の幾何チェックを無効化して実行フェーズを走らせる
    cfg.env_specs['check_init_contact'] = False
torch.set_default_dtype(torch.float64)
set_global_seed(cfg.seed)

ckpt_arg = int(checkpoint) if checkpoint != 'best' else 'best'
agent = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                     seed=cfg.seed, num_threads=1, training=False, checkpoint=ckpt_arg)
env = agent.env

state = env.reset()
trace = []  # (t, cube_x, ctrl_norm)
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
        cube_x = float(env.get_body_com("cube")[0])
        ctrl = float(np.linalg.norm(action[:env.control_action_dim]))
        trace.append((t_exec, cube_x, ctrl))
        t_exec += 1
    if done:
        break

print(f'[cube_trace] {restore_dir} ckpt={checkpoint} exec_steps={len(trace)}')
marks = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 999]
for t, x, c in trace:
    if t in marks:
        print(f'  t={t:4d}: cube_x={x:7.3f}  ctrl={c:6.3f}')
xs = [x for _, x, _ in trace]
if trace:
    peak_v = max(abs(xs[i+1]-xs[i]) for i in range(len(xs)-1)) / env.dt if len(xs) > 1 else 0
    print(f'  final cube_x = {xs[-1]:.3f} / max = {max(xs):.3f} / '
          f'peak |v| = {peak_v:.2f} m/s / 移動開始 t = '
          f'{next((t for t,x,_ in trace if abs(x-xs[0])>0.01), -1)}')
os._exit(0)
