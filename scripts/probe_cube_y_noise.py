#!/usr/bin/env python3
"""cube_y_noise（Shot タスク、9-33）が**実機で**効いているか確かめる。

`test_cube_y_noise.py` は式と実装の形しか見ない。こちらは実際に環境を作って
`reset_state()` を複数回呼び、**パックの y が本当に動くか**を測る。

使い方:
  CUBE_Y_NOISE=0.4 EVAL_RESTORE_DIR=single_run/tripo_pj_short \\
  USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \\
      --no-window --python scripts/probe_cube_y_noise.py

⚠️ Choreonoid 経由なのでスクリプトに引数は渡せない（Bug 36）。環境変数で指定する。
"""
import os
import sys

sys.path.append(os.getcwd())
os.environ['USE_CHOREONOID'] = '1'

import numpy as np
import yaml
import torch
from omegaconf import OmegaConf

from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent
from design_opt.utils.tools import set_global_seed

restore_dir = os.environ['EVAL_RESTORE_DIR']
noise = float(os.environ.get('CUBE_Y_NOISE', '0.4'))
n = int(os.environ.get('PROBE_N', '30'))

FLAGS = OmegaConf.create(yaml.safe_load(open(f'{restore_dir}/.hydra/config.yaml')))
d = OmegaConf.to_container(FLAGS, resolve=True)
d.pop('restore_dir', None)
FLAGS = OmegaConf.create(d)
cfg = Config(FLAGS, os.getcwd(), restore_dir)
cfg.restore_dir = restore_dir
cfg.control_prior = False
cfg.morph_prior = False
torch.set_default_dtype(torch.float64)
set_global_seed(cfg.seed)

agent = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                     seed=cfg.seed, num_threads=1, training=False, checkpoint='best')
env = agent.env


def sample(noise_val, add_noise):
    env.env_specs['cube_y_noise'] = noise_val
    ys = []
    for _ in range(n):
        env.reset_model()
        env.reset_state(add_noise)
        ys.append(float(env.get_body_com('cube')[1]))
    return np.array(ys)

print(f'[probe] {restore_dir}  n={n}  noise={noise}')
off = sample(0.0, True)
on = sample(noise, True)
ev = sample(noise, False)

for name, ys in (('noise=0（既存の挙動）', off), (f'noise={noise}（Shot）', on),
                 ('noise あり・評価時', ev)):
    print(f'  {name:26} y: 平均 {ys.mean():+.3f}  幅 [{ys.min():+.3f}, {ys.max():+.3f}]  '
          f'標準偏差 {ys.std():.3f}  異なる値 {len(np.unique(np.round(ys,6)))}/{len(ys)}')

ok = True
if on.std() <= off.std():
    print('  ❌ noise を入れてもばらつきが増えていない'); ok = False
if on.max() - on.min() < noise:          # ±noise なら幅は理屈上 2*noise に近づく
    print(f'  ⚠️ 幅 {on.max()-on.min():.3f} が noise {noise} より狭い（n が小さいだけかも）')
if abs(on).max() > noise * 1.6 + 0.15:   # 既存の ±0.1 ジッタぶんは許容
    print('  ❌ 指定した範囲を大きく超えている'); ok = False
print('✅ 実機でも効いている' if ok else '❌ 期待どおりでない')
