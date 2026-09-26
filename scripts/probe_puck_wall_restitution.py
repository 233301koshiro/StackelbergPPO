#!/usr/bin/env python3
"""**パックが側壁で跳ね返るか**を Choreonoid の実走で測る（ホッケー Phase 2 / 9-173）。

⭐ 9-172 で「材質は足すだけなら既存を壊さない」ことは示した。
⛔ **足した材質が実際に効くことはまだ示していない。**本スクリプトがそれを測る。

**測り方**: パックに +y 方向の初速を与え、側壁（内面 y=+0.50）へ撃つ。
接触前の最大接近速度と接触後の最大離脱速度の比が反発係数。

| XML | 材質 | 期待 |
|---|---|---|
| `e2e_hockey_easy` | 無し（全リンク Default） | ⛔ **e ≈ 0.05**（`[Default, Default]` の restitution 0.0） |
| `e2e_hockey_mat` ＋ テーブル | Puck / Rink | ⭐ **e ≈ 0.75** なら材質が効いている |

⚠️ **テーブルを渡さずに `e2e_hockey_mat` を走らせると、材質名はあるが対の定義が無い**ので
既定にフォールバックするはず。⭐ **3 条件で測って切り分ける。**

    XML=e2e_hockey_mat CNOID_MATERIAL_TABLE=assets/choreonoid/materials_hockey.yaml \
      USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
      --no-window --python scripts/probe_puck_wall_restitution.py > /tmp/pw.log 2>&1
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

XML = os.environ.get('XML', 'e2e_hockey_mat')
BASE = os.environ.get('BASE_RUN', 'single_run/hockey_easy')
V0 = float(os.environ.get('V0', '3.0'))       # パックへ与える +y 速度 [m/s]
STEPS = int(os.environ.get('STEPS', '150'))


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
    cfg = Config(OmegaConf.create(d), os.getcwd(), '/tmp/_probe_pw')
    cfg.control_prior = cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)

    print(f'[pw] XML={XML}  v0={V0}  table={os.environ.get("CNOID_MATERIAL_TABLE", "(既定)")}',
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
    env.step(zero_action())      # ⚠️ 入った次の step まで進めてから設定する（9-103）

    q = np.array(env.data.qpos, dtype=float)
    v = np.zeros_like(np.array(env.data.qvel, dtype=float))
    nj = len(q) - 2              # 末尾 2 本が cube の slide（x, y）
    q[:nj] = 0.0                 # 腕は畳んでおく（パックの経路から外す）
    q[nj] = 0.0                  # パックの x（腕から離す）
    q[nj + 1] = 0.0              # パックの y
    v[nj + 1] = V0               # +y へ撃つ
    env.set_state(q, v)

    ys = []
    for _ in range(STEPS):
        env.step(zero_action())
        ys.append(float(np.asarray(env.get_body_com('cube'), dtype=float)[1]))
    ys = np.array(ys)
    dy = np.diff(ys)
    v_in = float(dy.max())       # +y へ最も速く近づいた区間
    v_out = float(dy.min())      # 反射後、−y へ最も速く離れた区間
    e = abs(v_out / v_in) if v_in != 0 else float('nan')
    print(f'[pw] パック y: {ys[0]:.4f} → 最大 {ys.max():.4f} → 最終 {ys[-1]:.4f}')
    print(f'[pw] 側壁の内面 y=+0.50（パック半幅 0.05 なので接触は y≈0.45）')
    print(f'[pw] 接近 {v_in*1000:8.3f} mm/step   反発 {v_out*1000:8.3f} mm/step')
    print(f'[pw] ⭐ **反発係数 e = {e:.4f}**')
    if ys.max() < 0.40:
        print('[pw] ⚠️ **壁まで届いていない。**初速か可動範囲を確認すること')


main()
