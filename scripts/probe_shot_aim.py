#!/usr/bin/env python3
"""probe_shot_aim.py: 方策がパックの y に反応して狙いを変えているかを測る（9-33 の仮説検証）。

9-33 は「パックが左右に来るなら根元ヨーで向きを変える必要がある」と予測した。
関節使用率の比較では支持されなかった（ホッケー 25 % 対 正面固定 25〜74 %）が、
使用率は「動いた幅」しか見ないので、**y に応じて動きを変えているか**は測れない。

本プローブはパックの y を −0.4 / 0.0 / +0.4 に固定して決定論評価を回し、
**先端 y と根元ヨーの軌道が y ごとに変わるか**を直接見る。

  変わる   → 狙っている。設計は効いている
  変わらない → y を無視して同じ振りをしている。ノイズが学習を不安定にしているだけ

使い方（Choreonoid は引数を渡せないので環境変数。Bug 36）:
  EVAL_RESTORE_DIR=single_run/hockey_shot_d002smoke \
  USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
      --no-window --python scripts/probe_shot_aim.py
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
# Bug 10: 自身の checkpoint を再評価するときは転用フィルタを切る
cfg.control_prior = False
cfg.morph_prior = False
torch.set_default_dtype(torch.float64)
set_global_seed(cfg.seed)

ckpt_arg = int(checkpoint) if checkpoint != 'best' else 'best'
agent = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                     seed=cfg.seed, num_threads=1, training=False, checkpoint=ckpt_arg)
env = agent.env

# パックの y を固定する。reset_robot() → reload_sim_model() がモデルを作り直すので、
# reset() 前に init_qpos へ書いても消える。transit_execution() からも呼ばれる
# reset_state() 自体を包み、**呼ばれるたびに set_state で固定し直す**。
PIN = [0.0]
_orig_reset_state = env.reset_state


def reset_state_pinned(add_noise):
    _orig_reset_state(add_noise)
    qpos, qvel = env.data.qpos.copy(), env.data.qvel.copy()
    qpos[env.model.nq - 1] = PIN[0]           # cube_slide2 (y)
    env.set_state(qpos, qvel)


env.reset_state = reset_state_pinned
YAW_IDX = 0                                   # qpos[0] = 根元ヨー（arm_safe_init と同じ添字）


def rollout(force_y):
    """パックの y を force_y に固定して 1 エピソード回す。"""
    PIN[0] = force_y
    state = env.reset()
    yaw, tipy, cubex, cubey, dmin = [], [], [], [], []
    for _ in range(cfg.skel_transform_nsteps + 2 + 1100):
        in_exec = env.stage == 'execution'
        sv = tensorfy([state])
        if agent.obs_norm is not None:
            sv = agent.normalize_observation(sv)
        with torch.no_grad():
            a = agent.policy_net.select_action(sv, mean_action=True).numpy().astype(np.float64)
        state, _, done, _, _ = env.step(a)
        if in_exec:
            yaw.append(float(env.data.qpos[YAW_IDX]))
            tipy.append(float(env._arm_tip_pos[1]))
            com = env.get_body_com('cube')
            cubex.append(float(com[0]))
            cubey.append(float(com[1]))
            dmin.append(float(np.linalg.norm(com - env._arm_tip_pos)))
        if done:
            break
    return map(np.array, (yaw, tipy, cubex, cubey, dmin))


print(f'[shot_aim] {restore_dir} ckpt={checkpoint}', flush=True)
print(f'{"指定 y":>8}{"実測 y":>9}{"ヨー平均":>11}{"ヨー範囲":>11}'
      f'{"先端y 平均":>12}{"先端y 範囲":>12}{"パックx 変位":>13}{"最接近 距離":>12}', flush=True)
res = {}
for fy in (-0.4, 0.0, 0.4):
    yaw, tipy, cubex, cubey, dmin = rollout(fy)
    if len(yaw) == 0:
        print(f'{fy:>8.1f}  （実行フェーズに入らず）', flush=True)
        continue
    res[fy] = (yaw.mean(), tipy.mean())
    print(f'{fy:>8.1f}{cubey[0]:>9.3f}{yaw.mean():>11.4f}{yaw.max()-yaw.min():>11.4f}'
          f'{tipy.mean():>12.4f}{tipy.max()-tipy.min():>12.4f}'
          f'{cubex[-1]-cubex[0]:>13.4f}{dmin.min():>12.4f}', flush=True)

if len(res) == 3:
    ym = [res[k][0] for k in (-0.4, 0.0, 0.4)]
    tm = [res[k][1] for k in (-0.4, 0.0, 0.4)]
    d_yaw, d_tip = max(ym) - min(ym), max(tm) - min(tm)
    print(f'\nパックが 0.8 m 動いたとき: ヨー平均の差 {d_yaw:.4f} rad / '
          f'先端 y 平均の差 {d_tip:.4f} m', flush=True)
    print('判定: 先端 y の差が 0.1 m 未満なら、y を無視して同じ振りをしている'
          f' → {"❌ 狙っていない" if d_tip < 0.1 else "✅ 追従している"}', flush=True)

os._exit(0)      # Choreonoid は return では終了しない（前回 54 分ハングした原因）
