#!/usr/bin/env python3
"""**パックが床摩擦でどこまで進めるか**を測る（ホッケー Phase 2 / 9-173）。

⭐ **9-173 の計算**: パックが目標まで届くのに必要な初速は **0.63 m/s**、
9-135 の実測打撃は **2.88 m/s**（4.6 倍）。関節ダンピング（τ=m/b=1.5 s）だけなら
**4.32 m 進める**のに、実際は **0.87 m で止まる**。
⛔ **止めているのは床とパックの接触摩擦である。**

⭐ **摩擦は材質ごとに効く**（`AISTSimulatorItem.cpp:422` で materialTable が
ConstraintForceSolver へ渡り、`ConstraintForceSolver.cpp:628` で対を引く）。
⚠️ **反発は効かない**（AIST は材質の restitution を読まない。9-173）。

**測り方**: 腕を畳んでパックに +x の初速を与え、止まるまでの距離を測る。

    XML=e2e_hockey_mat CNOID_MATERIAL_TABLE=assets/choreonoid/materials_hockey.yaml \\
      USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \\
      --no-window --python scripts/probe_puck_coast.py > /tmp/coast.log 2>&1
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

XML = os.environ.get('XML', 'e2e_hockey_mat')
BASE = os.environ.get('BASE_RUN', 'single_run/hockey_easy')
V0 = float(os.environ.get('V0', '2.88'))    # 9-135 の実測打撃速度
STEPS = int(os.environ.get('STEPS', '400'))


def main():
    import torch
    import yaml
    from omegaconf import OmegaConf
    from design_opt.utils.config import Config
    from design_opt.agents.genesis_agent import BodyGenAgent

    d = OmegaConf.to_container(
        OmegaConf.create(yaml.safe_load(open(f'{BASE}/.hydra/config.yaml'))), resolve=True)
    d.pop('restore_dir', None)
    d['xml_name'] = XML
    cfg = Config(OmegaConf.create(d), os.getcwd(), '/tmp/_probe_coast')
    cfg.control_prior = cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)

    print(f'[coast] XML={XML} v0={V0} table={os.environ.get("CNOID_MATERIAL_TABLE", "(既定)")}',
          flush=True)
    ag = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                      seed=0, num_threads=1, training=False, checkpoint=0)
    env = ag.env
    env.reset()
    W = env.control_action_dim + np.shape(env.design_cur_params)[1] + 1

    def zero_action():
        return np.zeros((len(env.robot.bodies), W))

    info = {}
    for _ in range(64):
        *_, info = env.step(zero_action())
        if info.get('stage') == 'execution':
            break
    env.step(zero_action())          # ⚠️ 入った次の step まで進める（9-103）

    q = np.array(env.data.qpos, dtype=float)
    v = np.zeros_like(np.array(env.data.qvel, dtype=float))
    nj = len(q) - 2
    q[:nj] = 0.0                     # 腕は畳む（経路から外す）
    q[nj] = 0.0                      # パック x を可動域の中央寄りへ
    q[nj + 1] = 0.0
    v[nj] = V0                       # +x へ撃つ
    env.set_state(q, v)

    xs = []
    for _ in range(STEPS):
        env.step(zero_action())
        xs.append(float(np.asarray(env.get_body_com('cube'), dtype=float)[0]))
    xs = np.array(xs)
    travel = float(xs.max() - xs[0])
    # 止まった step（1 step あたり 0.1 mm を切ったところ）
    dx = np.abs(np.diff(xs))
    stopped = int(np.argmax(dx < 1e-4)) if (dx < 1e-4).any() else -1
    print(f'[coast] x: {xs[0]:.4f} → 最大 {xs.max():.4f} → 最終 {xs[-1]:.4f}')
    print(f'[coast] ⭐ **惰行距離 = {travel:.4f} m**   止まった step = {stopped}')
    print(f'[coast] 参考: 必要 0.94 m / 関節ダンピングだけなら 4.32 m')
    if travel < 0.94:
        print('[coast] ⛔ **必要距離に届かない。**')
    else:
        print('[coast] ⭐ **必要距離を超えた。**')


main()
