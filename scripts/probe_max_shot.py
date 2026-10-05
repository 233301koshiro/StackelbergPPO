#!/usr/bin/env python3
"""⭐⭐ **この形態でパックに出せる最大の速さを、台本の全力打撃で測る**（系譜 9-214）。

⭐ 9-213: 学習した方策は「打つ」ことはできているが、最良 0.82 m/s で 0.54 m しか進まず、
  壁経由でゴールまでの約 1.2 m に**初速が 1.5〜2.2 倍足りない**（約 1.2〜1.8 m/s 要る）。
⭐⭐ **出せるなら学習の問題、出せないなら「この形態ではバンクショットは無理」が判定器の答え**（§5-2 ⑤）。

⭐ 方策は使わない。**FK でパックのすぐ脇に先端を置き、全関節に全力トルク（ctrl=±1）を
  符号 16 通りで掛けて**、パックの最大速度を測る。
⚠️ **速度を直接書き込むのではなくトルクで振る**ので、アクチュエータの能力込みの値になる
  （`set_state` で速度を与えると、出せない速さで打ててしまう）。
⚠️ 構えは `PROBE_POSES` 個（既定 12）を乱択。**全探索ではないので「出せない」は下限付きの主張**になる。

    EVAL_RESTORE_DIR=single_run/hockey_bank7_s2 USE_CHOREONOID=1 OMP_NUM_THREADS=1 \\
    HOCKEY_WALL_RESTITUTION=0.75 HOCKEY_ARM_BLOCK=1 \\
    timeout -k 30 2400 /choreonoid_ws/install/bin/choreonoid --no-window \\
    --python scripts/probe_max_shot.py > out.txt 2>&1

⚠️ Bug 43: **パイプへ繋がず必ずファイルへ。**起動だけで約 105 秒。
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

RESTORE = os.environ.get('EVAL_RESTORE_DIR', 'single_run/hockey_bank7_s2')
N_POSES = int(os.environ.get('PROBE_POSES', '12'))
STEPS = int(os.environ.get('PROBE_STEPS', '25'))
DT = 0.04                      # timestep 0.01 × frame_skip 4
NEED = (1.2, 1.8)              # 9-213 の見積もり [m/s]（クーロン / 粘性）
# ⭐ 構えの窓。⚠️ 既定では seed0 の形態で候補 0 件だった（2026-10-05）。緩めて確かめられるようにする
DZ = float(os.environ.get('PROBE_DZ', '0.04'))   # 先端とパックの高さの差の許容 [m]
DH = float(os.environ.get('PROBE_DH', '0.15'))   # 離隔から先の水平距離の幅 [m]


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
    ag = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                      seed=0, num_threads=1, training=False, checkpoint='best')
    env = ag.env
    env.reset()
    W = env.control_action_dim + np.shape(env.design_cur_params)[1] + 1
    nb = len(env.robot.bodies)

    # ── 学習済みの設計で execution へ入り、次の step まで進める（9-103）──
    # ⭐ 設計フェーズは方策の行動で形態を決める。⛔ ゼロ行動だと形態が学習結果と違う
    from design_opt.agents.genesis_agent import tensorfy   # ⭐ record_arm_trace.py と同じ入口
    state = env.reset()
    for _ in range(64):
        if env.stage == 'execution':
            break
        sv = tensorfy([state])
        if ag.obs_norm is not None:
            sv = ag.normalize_observation(sv)
        with torch.no_grad():
            a = ag.policy_net.select_action(sv, mean_action=True).numpy().astype(np.float64)
        state, *_ = env.step(a)
    if env.stage != 'execution':
        raise RuntimeError('⛔⛔ execution へ入れない（Bug 53 を疑う）')
    env.step(np.zeros((nb, W)))

    xml = getattr(env, 'cur_xml_str', None) or env.init_xml_str
    chain = env._parse_arm_chain(xml)
    nj = len(chain)
    q0 = np.array(env.data.qpos, dtype=float)
    iq = np.asarray(env.init_qpos, dtype=float)
    cube = np.array([0.55 + iq[nj], iq[nj + 1], 0.2125])   # ⭐ y は既定位置（ノイズ無し）
    gap = env._get_cube_half_size() + env._get_max_arm_radius() + 0.01
    print(f'[shot] {RESTORE}  関節 {nj} 本 {[c["name"] for c in chain]}  パック {cube.round(3)}  '
          f'離隔 {gap:.3f} m', flush=True)

    # ── 構え: 先端がパックの高さで、パックから gap〜gap+0.15 m の所 ─────────
    rng = np.random.default_rng(0)
    lim = np.radians([180] + [90] * (nj - 1))
    A = rng.uniform(-lim, lim, size=(200000, nj))
    P, Q = env._fk_points_batch(chain, A)
    tip = Q[:, -1, :]
    dh = np.linalg.norm(tip[:, :2] - cube[:2], axis=1)
    ok = (np.abs(tip[:, 2] - cube[2]) < DZ) & (dh > gap) & (dh < gap + DH)
    ok &= np.minimum(P[:, :, 2], Q[:, :, 2]).min(1) >= 0.02          # 床より上
    ok &= np.abs(np.concatenate([P[:, :, 1], Q[:, :, 1]], 1)).max(1) <= 0.45   # 壁の内側
    ok &= env._seg_point_dist_batch(P, Q, cube).min(1) >= gap            # どのリンクも触れていない
    idx = np.flatnonzero(ok)
    # ⭐ どの条件が候補を消しているか（0 件のときに「打てない」と読む前に必ず見る）
    c_h = np.abs(tip[:, 2] - cube[2]) < DZ
    c_d = (dh > gap) & (dh < gap + DH)
    c_f = np.minimum(P[:, :, 2], Q[:, :, 2]).min(1) >= 0.02
    c_w = np.abs(np.concatenate([P[:, :, 1], Q[:, :, 1]], 1)).max(1) <= 0.45
    print(f'[shot] 条件ごとの通過数: 高さ {c_h.sum()} / 距離 {c_d.sum()} / 高さ&距離 {(c_h & c_d).sum()} / '
          f'+床 {(c_h & c_d & c_f).sum()} / +壁 {(c_h & c_d & c_f & c_w).sum()} / 最終 {len(idx)}', flush=True)
    print(f'[shot] 先端の高さの範囲（全乱択）: {tip[:, 2].min():.3f}〜{tip[:, 2].max():.3f} m / '
          f'先端の水平距離（原点から）: {np.linalg.norm(tip[:, :2], axis=1).min():.3f}〜'
          f'{np.linalg.norm(tip[:, :2], axis=1).max():.3f} m', flush=True)
    print(f'[shot] 構えの候補 {len(idx)} / {len(A)}（高さ ±{DZ} m・距離の幅 {DH} m）', flush=True)
    if len(idx) == 0:
        raise RuntimeError('⛔⛔ パックの脇に先端を置ける構えが無い')
    poses = A[rng.choice(idx, size=min(N_POSES, len(idx)), replace=False)]

    # ── 全力トルク × 符号 16 通り ──────────────────────────
    results = []
    for pi, pose in enumerate(poses):
        for signs in itertools.product([-1.0, 1.0], repeat=nj):
            q = q0.copy(); q[:nj] = pose
            q[nj], q[nj + 1] = iq[nj], iq[nj + 1]
            env.set_state(q, np.zeros_like(np.array(env.data.qvel, dtype=float)))
            act = np.zeros((nb, W))
            act[1:1 + nj, 0] = signs                # ⭐ bodies[1:] が関節の順（action_to_control）
            prev = np.asarray(env.get_body_com('cube'), dtype=float)[:2]
            vmax, vdir = 0.0, None
            for _ in range(STEPS):
                *_, info = env.step(act)
                cur = np.asarray(env.get_body_com('cube'), dtype=float)[:2]
                v = (cur - prev) / DT
                if np.linalg.norm(v) > vmax:
                    vmax, vdir = float(np.linalg.norm(v)), v
                prev = cur
                if info.get('stage') != 'execution':
                    break
            ang = float(np.degrees(np.arctan2(vdir[1], vdir[0]))) if vdir is not None else float('nan')
            results.append((vmax, ang, pi, signs))
        best = max(r[0] for r in results if r[2] == pi)
        print(f'[shot] 構え {pi + 1}/{len(poses)}  この構えの最大 {best:.2f} m/s', flush=True)

    results.sort(key=lambda r: -r[0])
    v_all = np.array([r[0] for r in results])
    fwd = [r for r in results if -90 < r[1] < 90]                     # +x 成分がある打撃
    print(f'\n[shot] ⭐ 試行 {len(results)}（構え {len(poses)} × 符号 {2 ** nj}）')
    print(f'[shot] ⭐⭐ パックの最大速度: 全方向 {v_all.max():.2f} m/s / '
          f'+x 成分あり {max((r[0] for r in fwd), default=0.0):.2f} m/s')
    print(f'[shot]    上位 5: ' + ' / '.join(f'{r[0]:.2f} m/s @{r[1]:+.0f}°' for r in results[:5]))
    print(f'[shot]    学習した方策の最良（9-213）: 0.82 m/s ・必要: {NEED[0]}〜{NEED[1]} m/s')
    top = max((r[0] for r in fwd), default=0.0)
    if top >= NEED[1]:
        print('[shot] ⭐⭐⭐ **必要な速さを出せる形態。**ゴールに入らないのは学習の問題')
    elif top >= NEED[0]:
        print('[shot] ⚠️⚠️ **見積もりの帯の中。**減速のモデル次第で届く／届かないが分かれる')
    else:
        print('[shot] ⛔⛔ **全力でも必要な速さに届かない。**「この形態ではバンクショットは無理」が答えの候補'
              f'（⚠️ 構えは {len(poses)} 個の乱択なので下限付き）')


main()
