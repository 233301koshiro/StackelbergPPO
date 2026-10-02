#!/usr/bin/env python3
"""⭐⭐ 反発係数を **3 つの粒度**で同時に測り、どこで誤差が入るかを分離する。

⛔⛔ なぜ要るか（系譜 9-200）: 借用策 ③ を入れて実測 e が 0.0128 → 0.8299 になったが、
**設定 0.75 に対し +11 % 高い。**⚠️ **これが実装の誤差なのか測り方の粗さなのかが分からない。**

⭐ 既存の `probe_puck_wall_restitution.py` は **変位の差分**で測る:
    `e = |min(dy)| / max(dy)`、`dy` は **env step（frame_skip=4 tick）ごとの y 変位**
⛔ **接近の最大と反発の最大が壁から違う距離で取られうる。**

⭐ 本プローブが出すもの:

| 粒度 | 中身 |
|---|---|
| ⭐⭐ **① イベント** | 反射が起きた瞬間の `dq` 比（`HOCKEY_WALL_DEBUG` が出す値と同じ）。**理論値そのもの** |
| ⭐ **② 速度** | env step ごとの **関節速度 `qvel`** のピーク比。差分でなく実速度 |
| ⚠️ **③ 変位** | 既存プローブと同じ `dy` のピーク比。**比較のため再現する** |

⭐ **①②③ が一致しなければ、ずれているのは実装ではなく測り方である。**

    XML=e2e_hockey_easy USE_CHOREONOID=1 OMP_NUM_THREADS=1 \\
    HOCKEY_WALL_RESTITUTION=0.75 HOCKEY_WALL_DEBUG=1 \\
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/probe_wall_e_fine.py

⚠️ Bug 43: **パイプへ繋がず必ずファイルへリダイレクト。**Choreonoid は起動だけで約 105 秒。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

XML = os.environ.get('XML', 'e2e_hockey_easy')
BASE = os.environ.get('BASE_RUN', 'single_run/hockey_easy')
V0 = float(os.environ.get('V0', '3.0'))
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
    cfg = Config(OmegaConf.create(d), os.getcwd(), '/tmp/_probe_fine')
    cfg.control_prior = cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)

    e_set = float(os.environ.get('HOCKEY_WALL_RESTITUTION', '0') or 0)
    print(f'[fine] XML={XML}  v0={V0}  設定 e={e_set}', flush=True)

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
    nj = len(q) - 2
    q[:nj] = 0.0
    q[nj] = 0.0
    q[nj + 1] = 0.0
    v[nj + 1] = V0
    env.set_state(q, v)

    ys, vys = [], []
    for _ in range(STEPS):
        env.step(zero_action())
        ys.append(float(np.asarray(env.get_body_com('cube'), dtype=float)[1]))
        vys.append(float(np.array(env.data.qvel, dtype=float)[nj + 1]))
    ys, vys = np.array(ys), np.array(vys)

    # ── ③ 変位（既存プローブと同じ）──
    dy = np.diff(ys)
    v_in_d, v_out_d = float(dy.max()), float(dy.min())
    e_disp = abs(v_out_d / v_in_d) if v_in_d else float('nan')

    # ── ② 速度（関節速度のピーク）──
    v_in_v, v_out_v = float(vys.max()), float(vys.min())
    e_vel = abs(v_out_v / v_in_v) if v_in_v else float('nan')

    # ── ② ' 反転の直前直後（壁に最も近づいた step の前後）──
    k = int(np.argmax(ys))
    lo, hi = max(0, k - 2), min(len(vys), k + 3)
    near = ' '.join(f'{x:+.4f}' for x in vys[lo:hi])

    print(f'\n[fine] パック y: {ys[0]:+.4f} → 最大 {ys.max():+.4f} → 最終 {ys[-1]:+.4f}')
    print(f'[fine] 壁の内面 y=+0.50（半幅 0.05 なので接触は y≈0.45）')
    print(f'[fine] 最接近 step {k} の前後の関節速度: {near}')
    print()
    print(f'[fine] ⚠️ ③ 変位ベース e = {e_disp:.4f}   '
          f'（接近 {v_in_d*1000:7.3f} / 反発 {v_out_d*1000:7.3f} mm/step）')
    print(f'[fine] ⭐ ② 速度ベース e = {e_vel:.4f}   '
          f'（接近 {v_in_v:+.4f} / 反発 {v_out_v:+.4f} m/s）')
    print(f'[fine] ⭐⭐ ① イベントの比は上の [wall] 行を見る（期待 {e_set}）')
    print()
    if e_set:
        print(f'[fine] 設定 {e_set} との差: ③ {e_disp - e_set:+.4f} / ② {e_vel - e_set:+.4f}')
        print('[fine] ⭐ ② が設定に一致し ③ だけずれるなら、**ずれているのは測り方**である。')
    if ys.max() < 0.40:
        print('[fine] ⚠️ **壁まで届いていない。**初速か可動範囲を確認すること')
    return 0


main()      # ⚠️ Bug 43: Choreonoid は sys.exit(0) を受け付けない
