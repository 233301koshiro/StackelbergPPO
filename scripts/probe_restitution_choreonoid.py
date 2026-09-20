#!/usr/bin/env python3
"""**`solref` が Choreonoid の実走に効いているか**を測る（9-143）。

⛔⛔ **変換器 `mujoco_env_choreonoid.py` には `solref`/`solimp`/geom の `friction` を読む処理が
一つも無い**（`grep` で 0 件）。⚠️ **9-128 は MuJoCo で反発係数を測って `solref` を決めたが、
学習側に届いていない可能性がある**（9-98 と同じ型）。

⭐ **同じ形態・同じ初速で、`solref` だけが違う 2 つの XML を実走させて跳ね返りを比べる。**
**値が同じなら `solref` は無視されている。**

    XML=e2e_a1v_rho USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
      /choreonoid_ws/install/bin/choreonoid --no-window \
      --python scripts/probe_restitution_choreonoid.py > /tmp/e.log 2>&1
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

XML = os.environ.get('XML', 'e2e_a1v_rho')
BASE = os.environ.get('BASE_RUN', 'single_run/e2e_a1v_rho_pusher')
V0 = float(os.environ.get('V0', '-4.0'))      # cube に与える x 速度（腕へ向かう向き）
STEPS = int(os.environ.get('STEPS', '120'))


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
    cfg = Config(OmegaConf.create(d), os.getcwd(), '/tmp/_probe_e')
    cfg.control_prior = cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)

    print(f'[e] XML={XML}  v0={V0}', flush=True)
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
    # 腕を畳んで壁のように構え、cube をそこへ撃つ
    q[:nj] = 0.0
    q[nj] = 1.0                  # cube の x
    q[nj + 1] = 0.0
    v[nj] = V0
    env.set_state(q, v)

    xs = []
    for _ in range(STEPS):
        env.step(zero_action())
        xs.append(float(np.asarray(env.get_body_com('cube'), dtype=float)[0]))
    xs = np.array(xs)
    dx = np.diff(xs)
    # 接触前＝最も速く近づいている区間、接触後＝最も速く離れる区間
    v_in = float(dx.min())
    v_out = float(dx.max())
    e = abs(v_out / v_in) if v_in != 0 else float('nan')
    print(f'[e] cube x: {xs[0]:.4f} → 最小 {xs.min():.4f} → 最終 {xs[-1]:.4f}')
    print(f'[e] 接近 {v_in*1000:8.3f} mm/step   反発 {v_out*1000:8.3f} mm/step')
    print(f'[e] ⭐ **反発係数 e = {e:.4f}**')


main()
sys.stdout.flush()
os._exit(0)
