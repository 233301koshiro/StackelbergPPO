#!/usr/bin/env python3
"""**1 回の読み込みで複数エピソードを走らせ、頭の話が当てにならないかを測る**（9-142）。

⚠️⚠️ **`record_arm_trace.py` は 1 エピソードしか記録しない。しかもそれは「読み込み直後の 1 話目」である。**
⭐ 9-66 は「**読み込み直後の 1〜2 話は当てにならず、3 話目から log の値に落ち着く**」と述べている。

⛔⛔ **反復して同じ値が出ても、この偏りは検出できない。**
**再生し直すたびに「1 話目」をやり直しているだけだからである**（9-138 の反復検証の盲点）。

    EVAL_RESTORE_DIR=single_run/<run> N=6 USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
      /choreonoid_ws/install/bin/choreonoid --no-window \
      --python scripts/probe_episode_bias.py > /tmp/bias.log 2>&1
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

RUN = os.environ['EVAL_RESTORE_DIR']
N = int(os.environ.get('N', '6'))
MAXSTEP = int(os.environ.get('MAXSTEP', '1100'))


def main():
    import torch
    import yaml
    from omegaconf import OmegaConf
    from design_opt.utils.config import Config
    from design_opt.agents.genesis_agent import BodyGenAgent, tensorfy
    from design_opt.utils.tools import set_global_seed

    d = OmegaConf.to_container(
        OmegaConf.create(yaml.safe_load(open(f'{RUN}/.hydra/config.yaml'))), resolve=True)
    d.pop('restore_dir', None)
    cfg = Config(OmegaConf.create(d), os.getcwd(), RUN)
    cfg.restore_dir = RUN
    cfg.control_prior = cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)
    set_global_seed(cfg.seed)

    ag = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                      seed=cfg.seed, num_threads=1, training=False, checkpoint='best')
    env = ag.env
    print(f'[bias] {RUN}  {N} エピソード（1 回の読み込みのまま連続で回す）', flush=True)
    print(f'{"話":>3} {"cube 移動[m]":>13} {"報酬合計":>10} {"step":>6}')
    res = []
    for k in range(N):
        state = env.reset()
        c0 = None
        tot = 0.0
        n = 0
        for _ in range(cfg.skel_transform_nsteps + 2 + MAXSTEP):
            sv = tensorfy([state])
            if ag.obs_norm is not None:
                sv = ag.normalize_observation(sv)
            with torch.no_grad():
                a = ag.policy_net.select_action(sv, mean_action=True).numpy().astype(np.float64)
            in_exec = env.stage == 'execution'
            state, rew, done, _, info = env.step(a)
            if in_exec:
                if c0 is None:
                    c0 = np.asarray(env.get_body_com('cube'), dtype=float).copy()
                tot += float(rew)
                n += 1
            if done:
                break
        c1 = np.asarray(env.get_body_com('cube'), dtype=float)
        mv = float(c1[0] - c0[0]) if c0 is not None else float('nan')
        res.append((mv, tot, n))
        print(f'{k+1:>3} {mv:>13.3f} {tot:>10.2f} {n:>6}', flush=True)

    a = np.array([r[0] for r in res])
    print(f'\n⭐ 1 話目 {a[0]:.3f} m   3 話目以降の平均 {a[2:].mean():.3f} m'
          f'   差 {(a[2:].mean()-a[0]):+.3f} m（{(a[2:].mean()/max(a[0],1e-9)-1)*100:+.0f} %）')
    print(f'⭐ 3 話目以降の幅: {a[2:].min():.3f} 〜 {a[2:].max():.3f} m')


main()

# ⚠️⚠️ Choreonoid は完走しても終了しない（Bug 43 の親戚）。`sys.exit(0)` も受け付けない。
#   ⭐ `os._exit(0)` で強制終了する。出力は既に flush 済みなので取りこぼさない。
#   ⚠️ これをしないと、キューで連続実行したとき 2 本目以降が始まらない（2026-09-20 に踏んだ）。
sys.stdout.flush()
os._exit(0)
