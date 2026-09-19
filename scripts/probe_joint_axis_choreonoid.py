#!/usr/bin/env python3
"""関節軸が **Choreonoid の実走に届いているか** を測る（実験系譜 9-131）。

⚠️⚠️ **9-98 の再発を防ぐためのプローブである。**
ホッケーの壁は MuJoCo に存在して Choreonoid に存在せず、**8 回の修正が空振りした。**
⭐ `mesh_to_params.py --axes` で軸を変えられるようにしたが、
**MuJoCo で通ることは Choreonoid で通ることを意味しない。**

やること: 各関節を 1 つずつ曲げ、**先端がどちらへ動くか**を見る。
- Y 軸ヒンジ → 先端は **XZ 面**内を動く（y はほぼ 0 のまま）
- X 軸ヒンジ → 先端は **YZ 面**内を動く（**y が動く**）

⭐ **この違いが出れば、軸は確かに Choreonoid まで届いている。**

    # ⚠️ パイプへ繋がない。ファイルへリダイレクトする（Bug 43）
    XML=e2e_a1v_axtest USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
      /choreonoid_ws/install/bin/choreonoid --no-window \
      --python scripts/probe_joint_axis_choreonoid.py > /tmp/axis.log 2>&1
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

XML  = os.environ.get('XML', 'e2e_a1v_axtest')
BASE = os.environ.get('BASE_RUN', 'single_run/e2e_a1v_pusher')
ANG  = float(os.environ.get('ANG', '0.6'))       # rad


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
    cfg = Config(OmegaConf.create(d), os.getcwd(), '/tmp/_probe_axis')
    cfg.control_prior = cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)

    print(f'[probe] XML={XML}  agent 構築開始', flush=True)
    ag = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                      seed=0, num_threads=1, training=False, checkpoint=0)
    env = ag.env
    env.reset()

    # ⚠️ 形態決定の 2 段階を **ゼロ行動**で通す。行動は (体の数, 制御+属性) の 2 次元
    #   （1 次元を渡すと IndexError。9-99 で 3 回踏んだ）
    # ⚠️ 行動の幅は **control_action_dim + 属性の次元 + 1（skel）**。
    #   `control_action_dim + attr_design_dim` で組むと pusher.py:158 で
    #   broadcast エラーになる（9-131 で踏んだ。CLAUDE.md「行動を自分で組む」の罠）。
    print(f'[probe] control_action_dim={env.control_action_dim} '
          f'attr_design_dim={env.attr_design_dim} '
          f'design_cur_params={np.shape(env.design_cur_params)}', flush=True)
    W = env.control_action_dim + np.shape(env.design_cur_params)[1] + 1

    def zero_action():
        return np.zeros((len(env.robot.bodies), W))

    info = {}
    for _ in range(64):
        *_, info = env.step(zero_action())
        if info.get('stage') == 'execution':
            break
    # ⚠️ execution へ入る瞬間に reset_state(True) が状態を上書きする。
    #   **入った次の step まで進めてから**設定する（CLAUDE.md §5-2 ⑤-3）
    env.step(zero_action())
    print(f"[probe] stage={info.get('stage')}", flush=True)

    names = [b.name for b in env.robot.bodies]
    tip   = names[-1]
    q0    = np.array(env.data.qpos, dtype=float).copy()
    v0    = np.zeros_like(np.array(env.data.qvel, dtype=float))
    print(f'[probe] リンク {names}  先端={tip}  qpos 次元={len(q0)}', flush=True)

    def tip_at(q):
        # ⚠️ env.data.qpos[...] = x は届かない。set_state が唯一の入口（9-103）
        env.set_state(q, v0)
        env.step(zero_action())
        return np.asarray(env._body_xpos[tip], dtype=float)

    base = tip_at(np.zeros_like(q0))
    print(f'\n=== 各 qpos を {ANG} rad にしたときの先端位置')
    print(f'{"idx":>4} {"tip x":>9} {"tip y":>9} {"tip z":>9}   基準からの変位')
    print(f'{"—":>4} {base[0]:9.4f} {base[1]:9.4f} {base[2]:9.4f}   （基準・全関節 0）')
    for i in range(len(q0)):
        q = np.zeros_like(q0); q[i] = ANG
        p = tip_at(q)
        d = p - base
        mv = ' '.join(f'{a}{v*1000:+.0f}mm' for a, v in zip('xyz', d) if abs(v) > 2e-3)
        print(f'{i:>4} {p[0]:9.4f} {p[1]:9.4f} {p[2]:9.4f}   {mv or "—"}')

    print('\n⭐ **Y 軸ヒンジなら y は動かない。X 軸ヒンジなら y が動く。**')
    print('⚠️ これが XML の axis と一致していれば、軸は Choreonoid まで届いている。')


# ⚠️ sys.exit(main()) にしない。Choreonoid は sys.exit(0) を受け付けず exit 137 になる（Bug 43）
main()
