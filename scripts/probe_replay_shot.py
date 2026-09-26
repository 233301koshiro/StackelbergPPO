#!/usr/bin/env python3
"""**9-135 の実際の撃ち出しを、反射あり／なしで再現する**（9-179）。

⭐ 9-179 で 9-135 の失敗の機序が判明した:
  パックは **+49° へ撃たれ、側壁 `wall_side_p`（y≥0.450）に当たって滑り**、
  向きが −4° に変わったまま `wall_goal_p`（x≥1.500）へ当たって停止した。
  ⛔ **ゴール口（|y|<0.35）の 8.7 cm 外側で壁に当たっていた。**
  ⛔ **摩擦でも飛距離不足でもない**（9-135 の「摩擦で 6〜9 cm 手前」は誤り）。

⭐⭐ **反射があれば側壁で跳ね返るので、まったく別の軌道になる。**
本スクリプトは**同じ初期位置・同じ初速**で撃ち、入るかどうかを比べる。

    XML=e2e_hockey_easy [HOCKEY_WALL_RESTITUTION=0.75] USE_CHOREONOID=1 OMP_NUM_THREADS=1 \\
      /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/probe_replay_shot.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

XML = os.environ.get('XML', 'e2e_hockey_easy')
BASE = os.environ.get('BASE_RUN', 'single_run/hockey_easy')
# ⭐ 9-135 の軌跡から取った実測値
P0 = [float(v) for v in os.environ.get('P0', '0.5877,0.2170').split(',')]
V0 = [float(v) for v in os.environ.get('V0', '0.9216,1.0643').split(',')]
STEPS = int(os.environ.get('STEPS', '200'))
GOAL_X, GOAL_HALF = 1.55, 0.35


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
    cfg = Config(OmegaConf.create(d), os.getcwd(), '/tmp/_probe_replay')
    cfg.control_prior = cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)

    e = os.environ.get('HOCKEY_WALL_RESTITUTION', '(無効)')
    print(f'[replay] XML={XML}  P0={P0}  V0={V0}  反射 e={e}', flush=True)
    ag = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                      seed=0, num_threads=1, training=False, checkpoint=0)
    env = ag.env
    env.reset()
    W = env.control_action_dim + np.shape(env.design_cur_params)[1] + 1

    def za():
        return np.zeros((len(env.robot.bodies), W))

    info = {}
    for _ in range(64):
        *_, info = env.step(za())
        if info.get('stage') == 'execution':
            break
    env.step(za())                       # ⚠️ 入った次の step まで進める（9-103）

    q = np.array(env.data.qpos, dtype=float)
    v = np.zeros_like(np.array(env.data.qvel, dtype=float))
    nj = len(q) - 2
    q[:nj] = 0.0                         # 腕は畳む（再現の邪魔をしない）
    base = np.asarray(env.get_body_com('cube'), dtype=float)[:2]
    # cube の body pos を基準に、関節座標へ直す
    q[nj] = P0[0] - 0.55
    q[nj + 1] = P0[1]
    v[nj], v[nj + 1] = V0
    env.set_state(q, v)

    xs = []
    for _ in range(STEPS):
        env.step(za())
        xs.append(np.asarray(env.get_body_com('cube'), dtype=float)[:2].copy())
    xs = np.array(xs)
    # ゴール判定: x が GOAL_X を越えた瞬間の |y|
    entered = None
    for k in range(1, len(xs)):
        if xs[k - 1, 0] < GOAL_X <= xs[k, 0]:
            t = (GOAL_X - xs[k - 1, 0]) / max(1e-9, xs[k, 0] - xs[k - 1, 0])
            y = xs[k - 1, 1] + t * (xs[k, 1] - xs[k - 1, 1])
            entered = (k, y)
            break
    print(f'[replay] 軌跡 {xs[0].round(4)} → 最大x {xs[:,0].max():.4f} → 最終 {xs[-1].round(4)}')
    if entered:
        k, y = entered
        ok = abs(y) < GOAL_HALF
        print(f'[replay] x={GOAL_X} を step{k} で通過  そのとき y={y:+.4f}  '
              f'（口は |y|<{GOAL_HALF}）→ {"⭐⭐ ゴール" if ok else "⛔ 外す"}')
    else:
        print(f'[replay] ⛔ x={GOAL_X} に到達しなかった（最大 {xs[:,0].max():.4f}）')


main()
