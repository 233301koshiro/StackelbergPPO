#!/usr/bin/env python3
"""⭐⭐ `env.step` の実時間を条件別に測る（系譜 9-210）。

⛔⛔ なぜ要るか: 初期姿勢にピッチ探索を入れたら **`T_sample` が 235 → 965 s（4.1 倍）**、
200 epoch の見積もりが **21.4 h → 約 70 h** になった。
⭐ **FK 探索は 20 ms なので原因ではない**（9-209 でベクトル化済み）。
⭐⭐ **腕が実際にリンク内へ下りたことで、腕ブロック（毎 tick に `_ARM_SWEEP` 点の掃過判定）が
常時発火しているのではないか**、という仮説を測る。

⚠️ **epoch を削る前に原因を測る**（CLAUDE.md §5-2 ① 局所対症療法の禁止）。

    EVAL_RESTORE_DIR=single_run/hockey_bank7 USE_CHOREONOID=1 OMP_NUM_THREADS=1 \\
    HOCKEY_WALL_RESTITUTION=0.75 [HOCKEY_ARM_BLOCK=1] [ARM_SWEEP_OFF=1] \\
    timeout -k 30 1200 /choreonoid_ws/install/bin/choreonoid --no-window \\
    --python scripts/probe_step_cost.py > out.txt 2>&1

⚠️ Bug 43: **パイプへ繋がず必ずファイルへ。**
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())

RESTORE = os.environ.get('EVAL_RESTORE_DIR', 'single_run/hockey_bank7')
STEPS = int(os.environ.get('PROBE_STEPS', '200'))


def main():
    import torch
    import yaml
    from omegaconf import OmegaConf
    from design_opt.utils.config import Config
    from design_opt.agents.genesis_agent import BodyGenAgent

    d = OmegaConf.to_container(
        OmegaConf.create(yaml.safe_load(open(f'{RESTORE}/.hydra/config.yaml'))), resolve=True)
    d.pop('restore_dir', None)
    cfg = Config(OmegaConf.create(d), os.getcwd(), RESTORE)
    cfg.restore_dir = RESTORE
    cfg.control_prior = cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)

    blk = os.environ.get('HOCKEY_ARM_BLOCK')
    swp = '2（ARM_SWEEP_OFF）' if os.environ.get('ARM_SWEEP_OFF') else '8'
    print(f'[cost] ⭐ 条件: 腕ブロック={"有効" if blk else "⛔ 無効"} / 掃過={swp if blk else "—"}',
          flush=True)

    ag = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                      seed=0, num_threads=1, training=False, checkpoint='best')
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
    env.step(zero_action())

    # ⭐ 計測（execution のみ）
    t0 = time.time()
    n = 0
    for _ in range(STEPS):
        *_, info = env.step(zero_action())
        if info.get('stage') != 'execution':
            break
        n += 1
    dt = time.time() - t0

    inner = None
    stack, seen = [env], []
    for _ in range(6):
        nxt = []
        for o in stack:
            if o is None or id(o) in seen:
                continue
            seen.append(id(o))
            if hasattr(o, '_arm_caps'):
                inner = o; break
            nxt += [getattr(o, a, None) for a in ('env', '_world', 'unwrapped', 'wrapped')]
        if inner is not None:
            break
        stack = nxt
    fires = getattr(inner, '_arm_blocks', 0) if inner else 0
    hits = getattr(inner, '_wall_hits', 0) if inner else 0

    print(f'\n[cost] ⭐ step 数 {n}  所要 {dt:.2f} s')
    print(f'[cost] ⭐⭐ 1 step あたり {dt / max(n,1) * 1000:.2f} ms')
    print(f'[cost] ⭐ 腕ブロックの発火 {fires} / 壁反射 {hits}')
    return 0


main()      # ⚠️ Bug 43: Choreonoid は sys.exit(0) を受け付けない
