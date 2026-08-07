#!/usr/bin/env python3
"""指定 run の checkpoint から、収束した gear・リンク太さ・リンク長を取り出す。

`boundary_compare.py` は `env.design_cur_params`（正規化済みの設計変数）を見るが、
そこには **gear が含まれていない**（2026-08-07 実測: 出力は offset_x/offset_y/size/ext_start の
5ボディ×4種=20個のみ）。nsteps の検証は gear の境界張り付きが主題なので、
実際に組み上がったロボットから物理量として読み出す必要がある。

`diagnose_morphology.py` の第2層と同じ読み方だが、**行動トレース（1100ステップ）を行わない**ため
数分で終わる。属性変換フェーズを回すのに必要な最小ステップ数だけ進める。

使い方（Choreonoid 経由が必須）:
    BC_RUNS="single_run/a:single_run/b" BC_CKPT=epoch_0200 \
      USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
      /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/extract_gear.py
"""
import os
import sys

sys.path.insert(0, os.getcwd())
os.environ.setdefault('USE_CHOREONOID', '1')

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from design_opt.utils.config import Config
from design_opt.utils.tools import set_global_seed
from design_opt.agents.genesis_agent import BodyGenAgent, tensorfy

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEAR_LB, GEAR_UB = 20.0, 400.0
SIZE_LB, SIZE_UB = 0.03, 0.10
TOL = 0.05          # 探索幅の 5% 以内なら「境界に張り付き」とみなす


def extract(run_dir, ckpt):
    d = OmegaConf.to_container(
        OmegaConf.create(yaml.safe_load(open(f'{run_dir}/.hydra/config.yaml'))), resolve=True)
    d.pop('restore_dir', None)
    cfg = Config(OmegaConf.create(d), PROJECT, run_dir)
    cfg.restore_dir = run_dir
    # Bug 10: 完走 run 自身の再評価では転用フィルタを無効化しないと重みが読まれない
    cfg.control_prior = False
    cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)
    set_global_seed(cfg.seed)
    cfg.num_threads = 1

    ag = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                      seed=cfg.seed, num_threads=1, training=False, checkpoint=ckpt)
    env = ag.env
    st = env.reset()
    # 属性変換フェーズを終えるまでだけ進める（行動トレースはしない＝速い）
    for _ in range(cfg.skel_transform_nsteps + 2):
        if env.stage == 'execution':
            break
        sv = tensorfy([st])
        if ag.obs_norm is not None:
            sv = ag.normalize_observation(sv)
        with torch.no_grad():
            a = ag.policy_net.select_action(sv, mean_action=True).numpy().astype(np.float64)
        st, _, _, _, _ = env.step(a)

    gears, sizes, lens = [], [], []
    for b in env.robot.bodies[1:]:
        g = None
        for j in b.joints:
            if j.actuator is not None:
                g = float(j.actuator.gear)
        if g is not None:
            gears.append(g)
        if b.geoms:
            sizes.append(float(np.asarray(b.geoms[0].size, dtype=float).ravel()[0]))
        if b.bone_offset is not None:
            lens.append(float(np.linalg.norm(np.asarray(b.bone_offset, dtype=float))))
    return gears, sizes, lens


def at_bound(v, lo, hi):
    span = hi - lo
    return abs(v - lo) <= span * TOL or abs(v - hi) <= span * TOL


def main():
    runs = [r.strip() for r in os.environ.get('BC_RUNS', '').split(':') if r.strip()]
    labels = [l.strip() for l in os.environ.get('BC_LABELS', '').split(':') if l.strip()]
    ckpt = os.environ.get('BC_CKPT', 'best')
    if not runs:
        print('BC_RUNS を指定してください'); os._exit(1)
    if len(labels) != len(runs):
        labels = [os.path.basename(r) for r in runs]

    print(f'\n{"="*72}\n  収束した物理量（ckpt={ckpt}）\n{"="*72}')
    for r, lab in zip(runs, labels):
        rp = r if os.path.isabs(r) else os.path.join(PROJECT, r)
        gears, sizes, lens = extract(rp, ckpt)
        gb = sum(at_bound(g, GEAR_LB, GEAR_UB) for g in gears)
        sb = sum(at_bound(s, SIZE_LB, SIZE_UB) for s in sizes)
        print(f'\n  {lab}')
        print(f'    gear      : {", ".join(f"{g:7.1f}" for g in gears)}   境界 {gb}/{len(gears)}'
              f'   （探索範囲 [{GEAR_LB:.0f}, {GEAR_UB:.0f}]）')
        print(f'    太さ      : {", ".join(f"{s:7.3f}" for s in sizes)}   境界 {sb}/{len(sizes)}'
              f'   （探索範囲 [{SIZE_LB}, {SIZE_UB}]）')
        print(f'    リンク長  : {", ".join(f"{l:7.3f}" for l in lens)}   合計 {sum(lens):.3f} m')
    print()
    os._exit(0)


if __name__ == '__main__':
    main()
