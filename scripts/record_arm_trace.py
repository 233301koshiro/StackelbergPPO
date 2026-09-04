#!/usr/bin/env python3
"""学習済み方策を1エピソード実行し、**実物メッシュで動画を作るための軌跡**を記録する。

**なぜ要るか**: 発表で「描いた絵がそのまま動く」ことを見せたい。
`diagnose_morphology.py` の第3層は数値でしか結果を返さず、
`save_morphology_urdf.py` は静止した形態しか出さない。
**動きと形態の両方**を1ファイルに落とす口が無かった。

出力（`<restore_dir>/trace/arm_trace.npz`）:
  body_names   リンク名（根元から順）
  xpos         (T, L, 3)  各リンクのワールド位置
  xmat         (T, L, 3, 3) 各リンクのワールド回転
  bone_offset  (L, 3)  **最適化後**のボーン（設計図の値ではない）
  cube         (T, 3)  Pusher の対象物（無い場合は形状 (0,)）
  target       (3,)    Reach の目標

⚠️ **最適化後の形態を使うこと。** co-design はリンク長も太さも変えるので、
XML の設計値でメッシュを並べると**学習結果と違う絵**になる。

使い方:
  EVAL_RESTORE_DIR=single_run/e2e_a1v_reach EVAL_CHECKPOINT=best \\
  USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \\
      --no-window --python scripts/record_arm_trace.py
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
max_steps = int(os.environ.get('TRACE_STEPS', '400'))

FLAGS = OmegaConf.create(yaml.safe_load(open(f'{restore_dir}/.hydra/config.yaml')))
d = OmegaConf.to_container(FLAGS, resolve=True)
d.pop('restore_dir', None)
FLAGS = OmegaConf.create(d)
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

target = np.array([cfg.env_specs.get('target_x', 0.8),
                   cfg.env_specs.get('target_y', 0.0),
                   cfg.env_specs.get('target_z', 0.15)])

state = env.reset()
xpos, xmat, cube = [], [], []
names = None
bone = None

for _ in range(cfg.skel_transform_nsteps + 2 + max_steps):
    in_exec = env.stage == 'execution'
    sv = tensorfy([state])
    if agent.obs_norm is not None:
        sv = agent.normalize_observation(sv)
    with torch.no_grad():
        action = agent.policy_net.select_action(sv, mean_action=True).numpy().astype(np.float64)
    state, reward, done, _, info = env.step(action)

    if in_exec:
        if names is None:
            # 設計フェーズが終わった時点の形態を確定させる（ここから先は変わらない）
            names = [b.name for b in env.robot.bodies]
            bone = np.array([np.asarray(getattr(b, 'bone_offset', [0, 0, 0]), dtype=float)
                             for b in env.robot.bodies])
        xpos.append([np.asarray(env._body_xpos[n], dtype=float) for n in names])
        xmat.append([np.asarray(env._body_xmat[n], dtype=float).reshape(3, 3) for n in names])
        # Pusher の対象物。⚠️ **`_body_xpos` には cube が入っていない**（腕の body だけ）。
        # 静止した初期値を読み続けて「動かない cube」を記録する事故を起こしたので、
        # `probe_cube_trace.py` と同じ `get_body_com()` を使う。
        try:
            cube.append(np.asarray(env.get_body_com('cube'), dtype=float))
        except Exception:
            cube.append(np.zeros(3))
    if done:
        break

out_dir = os.path.join(restore_dir, 'trace')
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'arm_trace.npz')
np.savez_compressed(out,
                    body_names=np.array(names),
                    xpos=np.array(xpos), xmat=np.array(xmat),
                    bone_offset=bone, cube=np.array(cube), target=target)

print(f'[trace] {restore_dir} ckpt={checkpoint}')
print(f'[trace] リンク: {names}')
print(f'[trace] 最適化後のボーン長: ' +
      ' / '.join(f'{np.linalg.norm(b):.4f}' for b in bone) +
      f'  合計 {sum(np.linalg.norm(b) for b in bone):.4f} m')
print(f'[trace] 実行ステップ {len(xpos)}  → {out}')
