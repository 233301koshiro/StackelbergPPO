#!/usr/bin/env python3
"""壁が **Choreonoid の実走で** 効いているかを測る（実験系譜 9-99）。

⚠️⚠️ **このリポジトリのホッケー用プローブは全部 MuJoCo で書かれていた。**
そして **MuJoCo では壁が効き、Choreonoid では壁が存在しなかった**（9-98）。
8 回の修正すべてが空振りした原因がこれである。**壁を測るなら Choreonoid で測る。**

パックへ初速を与え、壁を越えるかを見る。学習は不要（腕は動かさない）。

    XML=e2e_hockey_wall3 python3 scripts/probe_wall_choreonoid.py     # ← choreonoid 経由で起動する
    USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
        --no-window --python scripts/probe_wall_choreonoid.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

XML = os.environ.get('XML', 'e2e_hockey_wall3')
V = float(os.environ.get('V', '2.0'))
AXIS = os.environ.get('AXIS', 'y')          # y=側壁へ / x=ゴール端へ
STEPS = int(os.environ.get('STEPS', '150'))
# 側壁の内面 0.50・パック半幅 0.05 → 中心が 0.45 を大きく超えたら貫通
LIMIT = {'y': 0.45, 'x': 1.50}[AXIS]


def main() -> int:
    import torch
    import yaml
    from omegaconf import OmegaConf
    from design_opt.utils.config import Config
    from design_opt.agents.genesis_agent import BodyGenAgent

    base = os.environ.get('BASE_RUN', 'single_run/hockey_goal3')
    d = OmegaConf.to_container(OmegaConf.create(yaml.safe_load(open(f'{base}/.hydra/config.yaml'))),
                               resolve=True)
    d.pop('restore_dir', None)
    d['xml_name'] = XML
    cfg = Config(OmegaConf.create(d), os.getcwd(), '/tmp/_probe_wall_cnoid')
    cfg.control_prior = cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)
    print('[probe] agent 構築開始', flush=True)
    # ⚠️ 学習済みの重みは要らない（腕を動かさないので）。restore_dir を渡さないまま
    #   checkpoint='best' にすると load_checkpoint が None を join して落ちる。
    ag = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                      seed=0, num_threads=1, training=False, checkpoint=0)
    env = ag.env
    print('[probe] env 取得', flush=True)
    env.reset()
    print('[probe] reset 完了', flush=True)

    # ⚠️ この環境は execution の前に **形態を決める 2 段階**（skeleton_transform →
    #   attribute_transform）を通る。そこでの行動は **(リンク数, 次元) の 2 次元**であり、
    #   1 次元のゼロを渡すと `pusher.py:151` で IndexError になる（9-99 の検証で踏んだ）。
    #   ⭐ `info['stage']` が 'execution' になるまで、形態は既定のまま（ゼロ行動）で進める。
    print('[probe] 形態決定の段階を通す', flush=True)
    # ⚠️ 行動は **(体の数, control_action_dim + attr_design_dim) の 2 次元**
    #   （`pusher.py` 151・188-189 行）。1 次元を渡すと IndexError になる（9-99 で 3 回踏んだ）。
    def zero_action():
        n = len(env.robot.bodies)
        return np.zeros((n, env.control_action_dim + env.attr_design_dim))

    for _ in range(64):
        _, _, _, _, info = env.step(zero_action())
        if info.get('stage') == 'execution':
            break
    print(f"[probe] stage={info.get('stage')}", flush=True)

    # cube の slide 関節は qvel の末尾 2 つ（x, y の順）
    i = -2 if AXIS == 'x' else -1
    env.data.qvel[i] = V
    peak = 0.0
    zero = zero_action()
    print('[probe] step 開始', flush=True)
    for _k in range(STEPS):
        env.step(zero)
        if _k % 30 == 0: print(f'[probe] step {_k} peak={peak:.3f}', flush=True)
        com = np.asarray(env.get_body_com('cube')).reshape(-1)
        peak = max(peak, abs(float(com[0 if AXIS == 'x' else 1])))
    ok = peak <= LIMIT + 0.02
    print(f"RESULT xml={XML} axis={AXIS} v={V} peak={peak:.3f} limit={LIMIT} "
          f"{'✅ 壁が効いた' if ok else '❌ 壁を越えた（= 壁が存在しない）'}")
    return 0


if __name__ == '__main__':
    # ⚠️ Choreonoid の python 環境は `sys.exit(0)` を受け付けない（9-103）
    main()
