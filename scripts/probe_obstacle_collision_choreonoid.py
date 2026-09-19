#!/usr/bin/env python3
"""**静的な追加 body（柱）が腕と当たるか**を Choreonoid の実走で測る（Bug 45 / 9-133）。

⛔ **9-132 の症状**: 柱は `box` として正しく存在する（`elements` 確認済み）のに、
腕のリンク111 が 394〜396 / 401 step 柱の内部を通る。

⭐⭐ **測り方**: **同じ初期姿勢・同じゼロ制御**で
  ① 柱がある XML（`e2e_a1v_obs_tall2`）
  ② 柱が無い XML（`e2e_a1v`）
を走らせ、**軌跡が一致するかを見る。**

| 結果 | 読み |
|---|---|
| ⭐ **軌跡が完全に一致** | ⛔ **柱は腕に何の力も及ぼしていない**（衝突判定が働いていない） |
| 軌跡がずれる | ⭐ **柱は効いている。**9-132 の貫通は別の理由（制御が押し切った等） |

⚠️ **あわせてパックでも同じことを測る**（`--cube`）。
⭐ **パックが柱で止まって腕が止まらないなら、「腕だけが当たらない」が確定する**
（9-93 のホッケーと同一パターン）。

    # ⚠️ パイプへ繋がない。ファイルへリダイレクトする（Bug 43）
    XML=e2e_a1v_obs_tall2 OUT=/tmp/obs.npz USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
      /choreonoid_ws/install/bin/choreonoid --no-window \
      --python scripts/probe_obstacle_collision_choreonoid.py > /tmp/obs.log 2>&1
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

XML   = os.environ.get('XML', 'e2e_a1v_obs_tall2')
BASE  = os.environ.get('BASE_RUN', 'single_run/e2e_a1v_obs_tall2_reach')
OUT   = os.environ.get('OUT', '/tmp/obs_probe.npz')
STEPS = int(os.environ.get('STEPS', '60'))

# ⭐ 柱は (0.45, 0, 0〜0.85)、半幅 0.07。腕を +x 側へ倒して柱へ向かわせる姿勢を並べる。
#   （根元ヨー, ピッチ1, ピッチ2, ピッチ3）[rad]
POSES = [
    (0.0,  1.2,  0.0,  0.0),
    (0.0,  1.4, -0.3,  0.0),
    (0.0,  1.0,  0.3,  0.0),
    (0.0,  1.5, -0.6,  0.3),
]


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
    cfg = Config(OmegaConf.create(d), os.getcwd(), '/tmp/_probe_obs')
    cfg.control_prior = cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)

    print(f'[probe] XML={XML}  構築開始', flush=True)
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
    # ⚠️ execution へ入る瞬間に reset_state(True) が上書きする。次の step まで進める（9-103）
    env.step(zero_action())
    print(f"[probe] stage={info.get('stage')}", flush=True)

    names = [b.name for b in env.robot.bodies]
    q0 = np.array(env.data.qpos, dtype=float)
    v0 = np.zeros_like(np.array(env.data.qvel, dtype=float))
    nj = min(4, len(q0))
    print(f'[probe] リンク {names}  qpos 次元={len(q0)}', flush=True)

    traj = []
    for pi, pose in enumerate(POSES):
        q = np.zeros_like(q0)
        q[:nj] = pose[:nj]
        # ⚠️ env.data.qpos[...] = x は届かない。set_state が唯一の入口（9-103）
        env.set_state(q, v0)
        rec = []
        for _ in range(STEPS):
            env.step(zero_action())
            rec.append([np.asarray(env._body_xpos[n], dtype=float) for n in names])
        traj.append(rec)
        p = np.array(rec)
        print(f'  姿勢{pi} {np.round(pose,2)} → 最終リンク位置 '
              f'{np.round(p[-1,-1],4)}  経路長 {np.abs(np.diff(p,axis=0)).sum():.4f}', flush=True)

    np.savez(OUT, xml=XML, body_names=np.array(names),
             poses=np.array(POSES), traj=np.array(traj))
    print(f'\n[probe] → {OUT}')


# ⚠️ sys.exit(main()) にしない。Choreonoid は sys.exit(0) を受け付けず exit 137 になる（Bug 43）
main()
