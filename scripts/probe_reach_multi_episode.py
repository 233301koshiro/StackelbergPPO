#!/usr/bin/env python3
"""
probe_reach_multi_episode.py: 同一方策で複数エピソード走らせ、初期位置によらず
同じ関節配置に収束する「固定アトラクタ」的な挙動になっていないかを確認する。

使い方:
  EVAL_RESTORE_DIR=single_run/tripo_v2c_reach EVAL_CHECKPOINT=best EVAL_N_EPISODES=6 \
  USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
      --no-window --python scripts/probe_reach_multi_episode.py
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
n_episodes = int(os.environ.get('EVAL_N_EPISODES', '6'))

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
print(f'[multi_ep] target = {target}')

ckpt_arg = int(checkpoint) if checkpoint != 'best' else 'best'
agent = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                     seed=cfg.seed, num_threads=1, training=False, checkpoint=ckpt_arg)
env = agent.env


def get_arm_tip(env):
    """env._body_names[-1] は cube を指すことがある（腕の body ではない）ため、
    env.robot.bodies（設計対象の腕のみのリスト）から末端リンクの tip 位置を
    live simulation データ（_body_xpos/_body_xmat）で計算する
    （eval_cnoid_visual.py の collect_arm_skeleton と同じロジック）。"""
    bodies = env.robot.bodies
    b_last = bodies[-1]
    bo_last = getattr(b_last, 'bone_offset', None)
    if bo_last is None:
        bo_last = np.array([0.25, 0.0, 0.0])
    pos_last = np.array(env._body_xpos.get(b_last.name, np.zeros(3)))
    R_last = np.array(env._body_xmat.get(b_last.name, np.eye(3)))
    return pos_last + R_last @ np.asarray(bo_last, dtype=float)


results = []
for ep in range(n_episodes):
    state = env.reset()
    tip0 = None
    tip_last = None
    qpos_last = None
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
            if tip0 is None:
                tip0 = tip.copy()
            tip_last = tip.copy()
            qpos_last = env.data.qpos.copy()
        if done:
            break
    dist0 = float(np.linalg.norm(tip0 - target))
    dist_last = float(np.linalg.norm(tip_last - target))
    qpos_str = ' '.join(f'{q:+.3f}' for q in qpos_last[:3])
    print(f'[multi_ep] ep{ep}: tip_start={np.round(tip0,3)} dist_start={dist0:.3f} '
          f'-> tip_final={np.round(tip_last,3)} dist_final={dist_last:.3f}  qpos_final=[{qpos_str}]')
    results.append((dist0, dist_last, qpos_last[:3].copy()))

print('[multi_ep] --- summary ---')
final_qposes = np.array([r[2] for r in results])
print(f'[multi_ep] final qpos across episodes (should differ if closed-loop, match if fixed-attractor):')
print(final_qposes)
print(f'[multi_ep] final qpos std per joint = {final_qposes.std(axis=0)}')

os._exit(0)
