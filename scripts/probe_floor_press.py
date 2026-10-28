#!/usr/bin/env python3
"""**腕を台本で床へ押し付けて、止まる深さを測る**（系譜 9-288 の追い調べ、2026-10-22）。

なぜ: ホッケーの再生で、先端のカプセルが床に 30〜48 mm 入ったまま残る話があった。半径・Capsule の長さ・分割・
反復の上限・STRICT・可動域のどれでも説明できず（9-288）、方策の再生では条件を変えると動きも変わって比べにくい。
方策を使わず、同じ姿勢から同じ入力で押し、止まる深さだけを条件ごとに比べる。

やること: run の best の形を読み込み、実行の段階へ入れた次の step で、腕を −x 向き（パックと壁の無い側）に伸ばして
動くリンクの表面が床の PROBE_CLEAR [m] 上に来る姿勢を 1 回だけ set_state で置く（以後は物理に任せる。毎 step の
set_state はしない）。ピッチの関節 1 つに一定の入力を PROBE_STEPS step 与え続け、動くリンクの表面の最低の高さを毎 step 測る。

    EVAL_RESTORE_DIR=single_run/hockey_bank9_phys_g10_wfix USE_CHOREONOID=1 OMP_NUM_THREADS=1 \\
    [HOCKEY_ARM_BLOCK= / CNOID_MAX_ITER=1000 / CNOID_JOINT_MAX_VEL_STRICT=] \\
    timeout -k 10 600 /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/probe_floor_press.py > out.txt 2>&1

⭐ 結果（2026-10-22、`hockey_bank9_phys_g10_wfix` の best、先端の半径 0.0997 m）: 床の 6.5 mm 上から押すと、どの関節に ±1 を掛けても最深 6.8 mm で止まり最後は 0 mm 付近。`PROBE_CLEAR=-0.030`（30 mm 入れて置く）でも押し戻され、最も強く押す条件で最後は −0.2 mm。＝床の接触は効いている（系譜 9-288）
⚠️ 学習時の物理の環境変数は launch_env.json から入れる（手で渡した値が優先、record_arm_trace.py と同じ）。
⚠️ 末尾は os._exit(0)（Choreonoid は自分では終わらない、Bug 43）。
"""
import json
import os
import sys
import traceback

sys.path.insert(0, os.getcwd())
RESTORE = os.environ['EVAL_RESTORE_DIR']
_le = os.path.join(RESTORE, 'launch_env.json')
if os.path.exists(_le):
    for k, v in json.load(open(_le)).items():
        os.environ.setdefault(k, v)
STEPS = int(os.environ.get('PROBE_STEPS', '60'))
CLEAR = float(os.environ.get('PROBE_CLEAR', '0.010'))

import numpy as np


def main():
    import torch
    import yaml
    from omegaconf import OmegaConf
    from design_opt.utils.config import Config
    from design_opt.agents.genesis_agent import BodyGenAgent

    print('[press] 物理の環境変数:', {k: os.environ.get(k) for k in (
        'CNOID_SELF_COLLISION', 'CNOID_CULLING_DEPTH', 'CNOID_TICK_SYNC', 'CNOID_JOINT_MAX_VEL',
        'CNOID_JOINT_MAX_VEL_STRICT', 'HOCKEY_ARM_BLOCK', 'HOCKEY_WALL_RESTITUTION', 'CNOID_MAX_ITER')}, flush=True)
    d = OmegaConf.to_container(OmegaConf.create(yaml.safe_load(open(f'{RESTORE}/.hydra/config.yaml'))), resolve=True)
    d.pop('restore_dir', None)
    cfg = Config(OmegaConf.create(d), os.getcwd(), RESTORE)
    cfg.restore_dir = RESTORE
    cfg.control_prior = cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)
    ag = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                      seed=0, num_threads=1, training=False, checkpoint='best')
    env = ag.env
    state = env.reset()
    nb = len(env.robot.bodies)
    W = env.control_action_dim + np.shape(env.design_cur_params)[1] + 1

    def act(u=None):
        a = np.zeros((nb, W))
        if u is not None:
            a[u[0], 0] = u[1]
        return a

    # 実行の段階へ入れ、その次の step まで進める（9-103）。設計の段階は方策の平均の行動で（best の形にする。record_arm_trace.py と同じ）
    from design_opt.agents.genesis_agent import tensorfy
    info = {}
    for _ in range(64):
        if env.stage == 'execution':
            break
        sv = tensorfy([state])
        if ag.obs_norm is not None:
            sv = ag.normalize_observation(sv)
        with torch.no_grad():
            a = ag.policy_net.select_action(sv, mean_action=True).numpy().astype(np.float64)
        state, _r, _t, _tr, info = env.step(a)
    env.step(act())
    names = [b.name for b in env.robot.bodies]
    rads = [float(np.asarray(b.geoms[0].size, float).flatten()[0]) for b in env.robot.bodies]
    print(f'[press] ⭐ 形: リンク {names} / 半径 {np.round(rads, 4).tolist()}', flush=True)

    xml = getattr(env, 'cur_xml_str', None) or env.init_xml_str
    chain = env._parse_arm_chain(xml)
    print(f'[press] ⭐ 連鎖 {[lk["name"] for lk in chain]} / 半径 {[round(lk["rad"], 4) for lk in chain]}', flush=True)

    def surf_low(segs):
        """動くリンク（連鎖の 2 本目以降）の表面の最低の高さ"""
        return min(min(p[2], q[2]) - lk['rad'] for (p, q), lk in list(zip(segs, chain))[1:])

    # −x 向き（ヨー π）で、表面が床の CLEAR 上に来るピッチの組を探す（先端が最も遠くへ伸びるもの）
    best = None
    for a1 in np.linspace(-1.5, 1.5, 61):
        for a2 in np.linspace(-1.5, 1.5, 31):
            for a3 in np.linspace(-1.5, 1.5, 31):
                segs = env._fk_points(chain, [np.pi, a1, a2, a3])
                s = surf_low(segs)
                if abs(s - CLEAR) > 0.004:
                    continue
                reach = -segs[-1][1][0]
                if best is None or reach > best[0]:
                    best = (reach, a1, a2, a3, s)
    if best is None:
        raise RuntimeError('⛔ 表面が床の CLEAR 上に来る姿勢が無い')
    _, a1, a2, a3, s0 = best
    print(f'[press] ⭐ 姿勢: ヨー 180° / ピッチ {np.degrees([a1, a2, a3]).round(1).tolist()}° → FK の表面の最低 {s0*1000:.1f} mm', flush=True)

    def measure_low():
        bx, bm = env._body_xpos, env._body_xmat
        lows = []
        for b, r in zip(env.robot.bodies[2:], rads[2:]):     # 根元の球と縦の柱は除く
            p = np.asarray(bx[b.name], float)
            q = p + np.asarray(bm[b.name], float).reshape(3, 3) @ np.asarray(b.bone_offset, float)
            lows.append(min(p[2], q[2]) - r)
        return min(lows)

    q0 = np.array(env.data.qpos, dtype=float)
    q0[0], q0[1], q0[2], q0[3] = np.pi, a1, a2, a3
    v0 = np.zeros_like(np.array(env.data.qvel, dtype=float))
    trials = [('入力なし（重力だけ）', None)]
    for j in (2, 3, 4):                 # 11・111・1111 の関節（robot.bodies の番号）
        for u in (1.0, -1.0):
            trials.append((f'{names[j]} に {u:+.0f}', (j, u)))
    print(f'\n{"条件":18s} {"置いた直後":>9s} {"最低":>8s} {"最後の 20 step の平均":>18s}  （表面の高さ mm、負＝床の下）', flush=True)
    for label, u in trials:
        env.set_state(q0, v0)
        env.step(act())                 # 置いた姿勢を反映させる 1 step（入力なし）
        start = measure_low()
        lows = []
        for _ in range(STEPS):
            *_, info = env.step(act(u))
            if info.get('stage') != 'execution':
                print('⚠️ 実行の段階が終わった', flush=True)
                break
            lows.append(measure_low())
        lows = np.array(lows)
        print(f'{label:18s} {start*1000:9.1f} {lows.min()*1000:8.1f} {lows[-20:].mean()*1000:18.1f}', flush=True)


try:
    main()
except Exception:
    traceback.print_exc()
finally:
    sys.stdout.flush()
    os._exit(0)
