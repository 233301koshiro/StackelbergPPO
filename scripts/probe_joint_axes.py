#!/usr/bin/env python3
"""関節軸の平面性プローブ: 実行フェーズ中の各関節の回転軸（ワールド系）と
全ボディ位置の Y 座標を出力し、アームが単一の縦平面内で動いているかを検証する。

使い方:
  EVAL_RESTORE_DIR=single_run/tripo_arm_v3_pusher_smoke USE_CHOREONOID=1 \
  /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/probe_joint_axes.py
"""
import os
import sys
sys.path.append(os.getcwd())
os.environ['USE_CHOREONOID'] = '1'

import numpy as np
import yaml
from omegaconf import OmegaConf

from khrylib.utils import *
from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent
from design_opt.utils.tools import set_global_seed

project_path = os.getcwd()
restore_dir = os.environ.get('EVAL_RESTORE_DIR')
assert restore_dir, 'EVAL_RESTORE_DIR is required'

FLAGS = OmegaConf.create(yaml.safe_load(open(os.path.join(project_path, restore_dir, '.hydra', 'config.yaml'))))
cfg = Config(FLAGS, project_path, restore_dir)
cfg.restore_dir = restore_dir
# 転用起動 run の再評価時に転用フィルタが残ると重みが読み込まれない（Bug 10）。
cfg.control_prior = False
cfg.morph_prior = False

torch.set_default_dtype(torch.float64)
set_global_seed(cfg.seed)
agent = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                     seed=cfg.seed, num_threads=1, training=False,
                     checkpoint=os.environ.get('EVAL_EPOCH', 'best'))
env = agent.env

# XML の関節ローカル軸（tripo_arm_v3: 根本=Z ヨー、他=Y ピッチ）
LOCAL_AXES = {'1': np.array([0., 0., 1.]), '11': np.array([0., 1., 0.]),
              '111': np.array([0., 1., 0.]), '1111': np.array([0., 1., 0.])}

state = env.reset()
done = False
t = 0
report_at = {0, 40, 120}
print('=== joint axis planarity probe ===')
while not done and t < 130:
    from design_opt.agents.genesis_agent import tensorfy
    state_var = tensorfy([state])
    if agent.obs_norm is not None:
        state_var = agent.normalize_observation(state_var)
    with torch.no_grad():
        action = agent.policy_net.select_action(state_var, mean_action=True)
    a = action.numpy().astype(np.float64)
    if np.isnan(a).any():
        a = np.zeros_like(a)
    next_state, reward, termination, truncation, info = env.step(a)
    if info.get('stage') == 'execution':
        if t in report_at:
            print(f'--- exec step {t} ---')
            qpos0 = float(env.data.qpos[0])
            print(f'  q0(yaw) = {qpos0:+.4f} rad')
            # ヨー角 q0 を Z 回転した Y 軸 = ピッチ軸の理論値
            expect = np.array([-np.sin(qpos0), np.cos(qpos0), 0.0])
            for name in ['1', '11', '111', '1111']:
                if name not in env._body_xmat:
                    continue
                R = np.array(env._body_xmat[name])
                axis_w = R @ LOCAL_AXES[name]
                pos = np.array(env._body_xpos[name])
                tag = 'yaw ' if name == '1' else 'pitch'
                aligned = ''
                if name != '1':
                    cosang = abs(float(np.dot(axis_w, expect)))
                    aligned = f'  |cos(理論ピッチ軸との角)|={cosang:.6f}'
                print(f'  body {name:>4} ({tag}) pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})'
                      f' axis_world=({axis_w[0]:+.4f},{axis_w[1]:+.4f},{axis_w[2]:+.4f}){aligned}')
        t += 1
    done = termination or truncation
    state = next_state
print('=== done ===')
os._exit(0)
