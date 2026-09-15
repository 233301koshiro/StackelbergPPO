#!/usr/bin/env python3
"""**Choreonoid の実走で**、パックの初期 y ごとにゴールへ入る撃ち方があるかを測る（実験系譜 9-103）。

⚠️⚠️ **9-65 は同じことを MuJoCo で測っていた。** だが学習は Choreonoid で走り、
**MuJoCo にしか壁が存在しなかった**（9-98）。9-99 で壁を `<body>` に移して初めて
Choreonoid にも壁が実在するようになったので（9-100 で実測）、**測り直す。**

腕は使わない。**パックへ直接初速を与え**、ゴール口（x=1.55・|y|<=0.15）を通るかを数える。
壁で跳ね返って入ったかどうかも記録する（反射が必須かの判定）。

⚠️ **行動の形は `record_arm_trace.py`（動いている実例）に合わせる。**
自分でゼロ行動を組もうとすると形が合わず落ちる（9-99 で 4 回失敗した）。

    USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
        --no-window --python scripts/probe_shot_choreonoid.py > out.log 2>&1
    # ⚠️ **パイプへ繋がない**（Bug 43。出力が緩衝されて消える）

環境変数: XML（既定 e2e_hockey_wall3）・YS（初期 y のリスト）・ANGLES（撃ち出し角の刻み）・V（初速）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

XML = os.environ.get('XML', 'e2e_hockey_wall3')
V = float(os.environ.get('V', '3.5'))
YS = [float(v) for v in os.environ.get('YS', '0.0,0.10,0.20,0.30,0.40').split(',')]
STEP_DEG = float(os.environ.get('STEP_DEG', '6'))
SPAN_DEG = float(os.environ.get('SPAN_DEG', '60'))
STEPS = int(os.environ.get('STEPS', '160'))
GOAL_X, GOAL_HALF = 1.55, 0.15
WALL_FACE_Y = 0.45          # パック中心が壁に触れ始める |y|


def main() -> int:
    import torch
    import yaml
    from omegaconf import OmegaConf
    from design_opt.utils.config import Config
    from design_opt.agents.genesis_agent import BodyGenAgent, tensorfy

    base = os.environ.get('BASE_RUN', 'single_run/hockey_goal4')
    d = OmegaConf.to_container(
        OmegaConf.create(yaml.safe_load(open(f'{base}/.hydra/config.yaml'))), resolve=True)
    d.pop('restore_dir', None)
    d['xml_name'] = XML
    # ⚠️ **初期 y をこちらで決めるので、環境側のランダム化を切る。**
    #   cube_y_noise=0.4 が効いたままだと reset のたびに y が上書きされ、
    #   qpos への代入が無意味になる（9-103 で実測。y が指定値と全く違った）
    d.setdefault('env_specs', {})['cube_y_noise'] = 0.0
    cfg = Config(OmegaConf.create(d), os.getcwd(), '/tmp/_probe_shot')
    cfg.control_prior = cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)
    print('[probe] agent 構築', flush=True)
    ag = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                      seed=0, num_threads=1, training=False, checkpoint=0)
    env = ag.env
    angles = np.arange(-SPAN_DEG, SPAN_DEG + 1e-9, STEP_DEG)
    print(f'[probe] XML={XML} v={V} 初期y={YS} 角度={len(angles)} 通り', flush=True)

    rows = []
    for y0 in YS:
        n_in = n_reflect = 0
        for deg in angles:
            state = env.reset()
            # ⭐ 設計フェーズを方策に通す（record_arm_trace.py と同じ形）
            # ⚠️ **`transit_execution()` が execution へ移る瞬間に `reset_state(True)` を
            #   呼んで状態を上書きする**（`pusher.py` 465-469 行）。
            #   ⭐ **execution に入った「次の step」まで進めてから初速を与える。**
            #   ここを間違えると、上書きされて初速が消える試行が混ざる（9-103 で実測）。
            for _ in range(cfg.skel_transform_nsteps + 4):
                sv = tensorfy([state])
                if ag.obs_norm is not None:
                    sv = ag.normalize_observation(sv)
                with torch.no_grad():
                    a = ag.policy_net.select_action(sv, mean_action=True).numpy().astype(np.float64)
                state, _, _, _, info = env.step(a)
                if env.stage == 'execution' and env.control_nsteps >= 1:
                    break
            assert env.stage == 'execution', 'execution へ入れていない'
            # パックを初期位置へ置き、初速を与える（腕は以降ゼロ方策で放置）
            th = np.deg2rad(deg)
            # ⚠️ **`env.data.qpos[...] = x` は Choreonoid へ届かない**（9-103）。
            #   `set_state` が唯一の入口（`mujoco_env_choreonoid.py` 1237 行）。
            qp = np.array(env.data.qpos, dtype=float).copy()
            qv = np.zeros_like(np.array(env.data.qvel, dtype=float))
            qp[-1] = y0
            qv[-2] = V * np.cos(th)
            qv[-1] = V * np.sin(th)
            env.set_state(qp, qv)
            touched = False
            scored = False
            trace_x = []
            for _ in range(STEPS):
                sv = tensorfy([state])
                if ag.obs_norm is not None:
                    sv = ag.normalize_observation(sv)
                with torch.no_grad():
                    a = ag.policy_net.select_action(sv, mean_action=True).numpy().astype(np.float64)
                state, _, done, _, _ = env.step(a)
                c = np.asarray(env.get_body_com('cube')).reshape(-1)
                trace_x.append((c[0], c[1]))
                if abs(c[1]) >= WALL_FACE_Y - 0.01:
                    touched = True
                if c[0] >= GOAL_X and abs(c[1]) <= GOAL_HALF:
                    scored = True
                    break
                if done:
                    break
            if scored:
                n_in += 1
                if touched:
                    n_reflect += 1
            if abs(deg + 12) < 0.1 or abs(deg + 18) < 0.1:
                xs = [p[0] for p in trace_x]; ys = [p[1] for p in trace_x]
                print(f'    [dbg] y0={y0:+.2f} deg={deg:+.0f} '
                      f'x {xs[0]:.2f}->{max(xs):.2f}  y {ys[0]:+.2f}->{ys[-1]:+.2f} '
                      f'scored={scored} touched={touched}', flush=True)
        rows.append((y0, n_in, n_reflect))
        print(f'[probe] y0={y0:+.2f}  入った {n_in}/{len(angles)}  '
              f'うち壁に触れてから {n_reflect}', flush=True)

    print('\nRESULT ' + ' | '.join(f'y0={y:+.2f}:{i}/{len(angles)}(反射{r})' for y, i, r in rows),
          flush=True)
    return 0


if __name__ == '__main__':
    # ⚠️ Choreonoid の python 環境は `sys.exit(0)` を受け付けない
    #   （`TypeError: incompatible function arguments ... Invoked with: 0`）。
    #   ⭐ 測定自体は完走しているのに終了コード 137 で「失敗」に見えるので、
    #   **exit を呼ばずに返す**（9-103 で 2 回この誤解を生んだ）。
    main()
