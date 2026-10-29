#!/usr/bin/env python3
"""**部品で組んだ 1/5 の A1v（`e2e_a1v_ra15.xml`）が、小さいスケールで変な動きをしないか**を台本で確かめる（軸6 L-3b、系譜 9-326）。

なぜ: 1/5 では質量が 1/20〜1/40（腕 約 0.26 kg・箱 22 g）、gear 1.4、armature 0.008、減衰 0.235 と、今までの run と桁が違う。
接触の固定値（接触点の間引き 5 mm・めり込みを戻し始める 0.25 mm）は m 単位のまま（スケール表 S2）。学習の前に、方策を使わず同じ入力で確かめる。

やること（方策は使わない。形は XML のまま。cfg の設計変数は空にする＝ α と同じ）:
  0. 読み込まれた値: シミュレータの中のリンクの質量・重心・armature・可動域・gear（⑤-3-2: 何に効いたか）
  1. 重力だけ: まっすぐ上から肩を 2° 前へ倒して手を離す → 落ち方・止まる角・床・NaN
  2. 速さの上限: 旋回に +1 → 定常の角速度（期待 min(5.97, 1.4 / 0.235 = 5.96)）と立ち上がりの時定数
  3. 可動域: 肘・肩・手先に ±1 → 止まる角（スケール表 A5）と行き過ぎ
  4. 床: 前へお辞儀して手先を床へ押し付ける（肩に +1）→ めり込みの深さ
  5. 箱の滑り: 腕を後ろへ逃がし、箱に 0.1 m/s を与える → 止まる時間 τ（期待 m / b = 21.6 g / 0.016 = 1.35 s）
  6. 押し: 手先を箱の手前に置き、肩に +1 → 箱が動くか・めり込み（カプセルと箱の表面の距離）・箱の速さ

    cd /userdir/StackelbergPPO_ra && USE_CHOREONOID=1 OMP_NUM_THREADS=1 \\
    timeout -k 10 600 /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/probe_ra15_physics.py > out.txt 2>&1

⚠️ 物理の環境変数は `e2e_a1v_pusher_phys` の launch_env.json に、速さの上限だけ 5.97 rad/s（スケール表 P5）を上書きする（手で渡した値が優先）。
⚠️ 末尾は os._exit(0)（Choreonoid は自分では終わらない、Bug 43）。
"""
import json
import os
import sys
import traceback

sys.path.insert(0, os.getcwd())
REF = os.environ.get('PROBE_REF', '/userdir/StackelbergPPO/single_run/e2e_a1v_pusher_phys')
XML = os.environ.get('PROBE_XML', 'e2e_a1v_ra15')
JOB = os.environ.get('PROBE_JOB', '/tmp/probe_ra15_job')
os.environ.setdefault('CNOID_JOINT_MAX_VEL', '5.97')
for k, v in json.load(open(os.path.join(REF, 'launch_env.json'))).items():
    os.environ.setdefault(k, v)

import numpy as np

DT = 0.04          # 1 step = 0.01 s × frame_skip 4（スケール表 S3）
TESTS = os.environ.get('PROBE_TESTS', '0123456')      # 回す試験の番号
DETAIL = os.environ.get('PROBE_DETAIL', '')            # 例 '3:2+' = 試験 3 の肩 +1 を step ごとにリンク別の床からの高さで出す


def seg_box_gap(p, q, r, c, h, n=21):
    """線分 p→q・半径 r のカプセルと、中心 c・半辺 h の軸平行の箱の表面の距離（負 = めり込み）"""
    best = 1e9
    for t in np.linspace(0, 1, n):
        x = p + t * (q - p)
        d = np.abs(x - c) - h
        out = np.linalg.norm(np.maximum(d, 0))
        ins = min(np.max(d), 0)
        best = min(best, out + ins - r)
    return best


def main():
    import torch
    import yaml
    from omegaconf import OmegaConf
    from design_opt.utils.config import Config
    from design_opt.agents.genesis_agent import BodyGenAgent

    print('[ra15] 物理の環境変数:', {k: os.environ.get(k) for k in (
        'CNOID_SELF_COLLISION', 'CNOID_CULLING_DEPTH', 'CNOID_CTRL_CLIP', 'CNOID_HINGE_DAMP_TORQUE', 'CNOID_TICK_SYNC',
        'CNOID_JOINT_MAX_VEL', 'CNOID_JOINT_MAX_VEL_STRICT')}, flush=True)
    d = OmegaConf.to_container(OmegaConf.create(yaml.safe_load(open(f'{REF}/.hydra/config.yaml'))), resolve=True)
    d.pop('restore_dir', None)
    d['xml_name'] = XML
    d['cfg'] = os.environ.get('PROBE_CFG', 'pusher_ra15')   # ⭐ 9-328: β の cfg（capsule_lateral・part_mass_file・上腕と前腕の長さ）。設計の行動は 0 なので形は XML のまま
    cfg = Config(OmegaConf.create(d), os.getcwd(), JOB)
    cfg.env_specs['arm_safe_init'] = False          # 初期姿勢は台本で置く
    # ⚠️ 骨（関節から関節）で測ると、台の球を関節 1 のホーンに置いたものとして扱い、肩のリンクと 1 mm 重なると判定して
    #   1 step 目で終わらせる（実物の台のサーボはホーンより下）。本物のカプセル・球の位置で測る（Bug 72 の init_self_contact_caps）
    cfg.env_specs['init_self_contact_caps'] = True
    if os.environ.get('PROBE_CUBE_NOISE'):
        cfg.env_specs['cube_init_noise'] = float(os.environ['PROBE_CUBE_NOISE'])   # スケール表 T7（× k なら 0.02）
    cfg.control_prior = cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)
    ag = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                      seed=0, num_threads=1, training=False, checkpoint=0)
    env = ag.env
    env.reset()
    nb = len(env.robot.bodies)
    W = env.control_action_dim + np.shape(env.design_cur_params)[1] + 1
    names = [b.name for b in env.robot.bodies]

    def act(u=()):
        a = np.zeros((nb, W))
        for j, v in u:
            a[j, 0] = v
        return a

    info = {}
    for _ in range(64):
        if env.stage == 'execution':
            break
        _s, _r, _t, _tr, info = env.step(act())
    env.step(act())
    print(f'[ra15] 段階 {env.stage} / リンク {names} / 実行の開始の箱の qpos（XML の位置からのずれ）'
          f'{np.round(np.array(env.data.qpos, float)[-2:], 4).tolist()} m（cube_init_noise {cfg.env_specs.get("cube_init_noise")}）', flush=True)

    if os.environ.get('PROBE_BENCH'):               # 1 step の所要を測るだけ（刻みの分割のコスト、系譜 9-327）
        import time
        n = int(os.environ['PROBE_BENCH'])
        rng = np.random.default_rng(0)
        t0 = time.perf_counter()
        for _ in range(n):
            u = [(j, float(rng.uniform(-1, 1))) for j in range(1, 5)]
            *_, info = env.step(act(u))
            if info.get('stage') != 'execution':
                env.reset()
                for _ in range(64):
                    if env.stage == 'execution':
                        break
                    env.step(act())
        dt = time.perf_counter() - t0
        print(f'[ra15] ⭐ BENCH {n} step（乱数の入力）: {dt:.2f} s、1 step {dt / n * 1000:.2f} ms（CNOID_SUBSTEPS={os.environ.get("CNOID_SUBSTEPS")}）', flush=True)
        return

    # ---- 0. 読み込まれた値 ----
    xml = env.cur_xml_str if isinstance(env.cur_xml_str, str) else env.cur_xml_str.decode()
    n_inert = xml.count('<inertial')
    print(f'[ra15] ⭐ 書き出し後の XML の <inertial>: {n_inert} 個（期待 4。0 なら xml_robot が消した）', flush=True)
    eb = None
    for o in (env, getattr(env, '_world', None), getattr(env, 'sim', None), getattr(getattr(env, 'sim', None), '_world', None)):
        if o is not None and getattr(o, '_arm_body', None) is not None:
            eb = o._arm_body
            world = o
            break
    if eb is None:
        raise RuntimeError(f'腕の Choreonoid の body が見つからない: env の属性 {[k for k in vars(env) if "sim" in k or "world" in k]}')
    for k in range(eb.numLinks):
        lk = eb.link(k)
        jm = None
        for attr in ('equivalentRotorInertia', 'Jm2'):
            if hasattr(lk, attr):
                v = getattr(lk, attr)
                jm = v() if callable(v) else v
                break
        print(f'[ra15] ⭐ リンク {lk.name}: 質量 {lk.m*1000:.1f} g / 重心 {np.round(np.asarray(lk.c)*1000, 1).tolist()} mm / '
              f'armature {jm} / 可動域 {[round(float(np.degrees(x)), 1) for x in (lk.q_lower, lk.q_upper)] if hasattr(lk, "q_lower") else "?"}',
              flush=True)
    import re
    print(f'[ra15] ⭐ gear（書き出し後の XML）: {re.findall(r"gear=\"([^\"]+)\"", xml)} / 減衰・armature（同）: '
          f'{re.findall(r"<joint armature=\"([^\"]+)\" damping=\"([^\"]+)\"", xml)}'
          f' / 腕の総質量 {sum(eb.link(k).m for k in range(eb.numLinks))*1000:.1f} g', flush=True)
    caps = {}
    for b in env.robot.bodies:
        g = b.geoms[0].node
        if g.get('fromto'):
            ft = np.array(g.get('fromto').split(), float)
            caps[b.name] = (ft[:3], ft[3:], float(g.get('size').split()[0]))
    cube_h = np.array([0.03, 0.03, 0.03])

    def world_caps():
        out = []
        for nm, (a, bb, r) in caps.items():
            p = np.asarray(env._body_xpos[nm], float)
            R = np.asarray(env._body_xmat[nm], float).reshape(3, 3)
            out.append((nm, p + R @ a, p + R @ bb, r))
        return out

    def floor_lows():
        return {nm: min(p[2], q[2]) - r for nm, p, q, r in world_caps() if nm != '1'}

    def floor_low():
        return min(floor_lows().values())

    def cube_gap():
        c = np.asarray(env.get_body_com('cube'), float)
        return min(seg_box_gap(p, q, r, c, cube_h) for _, p, q, r in world_caps())

    nq = len(env.data.qpos)
    print(f'[ra15] qpos の長さ {nq}（関節 4 ＋ 箱 2）', flush=True)

    def place(q4, cube=(0.0, 0.0), qv=None):
        q = np.array(env.data.qpos, float)
        q[:4] = q4
        q[4:6] = cube
        v = np.zeros(nq) if qv is None else np.asarray(qv, float)
        env.set_state(q, v)
        env.step(act())

    def run(u, steps, every=0, detail=False):
        rows = []
        for t in range(steps):
            if detail and t < 30:
                fl = floor_lows()
                q = np.array(env.data.qpos, float)
                v = np.array(env.data.qvel, float)
                print(f'      step {t:2d} 角 {np.degrees(q[:4]).round(1).tolist()} 角速度 {v[:4].round(2).tolist()} 床から '
                      + ' '.join(f'{k}:{x*1000:6.1f}' for k, x in fl.items()), flush=True)
            *_, info = env.step(act(u))
            if info.get('stage') != 'execution':
                print('⚠️ 実行の段階が終わった', flush=True)
                break
            q = np.array(env.data.qpos, float)
            v = np.array(env.data.qvel, float)
            rows.append((q.copy(), v.copy(), floor_low(), cube_gap()))
            if not np.all(np.isfinite(q)):
                print('⛔⛔ NaN', flush=True)
                break
            if every and t % every == 0:
                print(f'    t={t*DT:5.2f}s 角 {np.degrees(q[:4]).round(1).tolist()}° 角速度 {v[:4].round(2).tolist()} '
                      f'床からの最低 {rows[-1][2]*1000:6.1f} mm 箱との隙間 {rows[-1][3]*1000:6.1f} mm', flush=True)
        return rows

    deg = np.radians

    # ---- 1. 重力だけ ----
    print('\n[ra15] === 1. 重力だけ（肩を 2° 前へ倒して離す）', flush=True)
    place([0, deg(2), 0, 0], cube=(0.5, 0))         # 箱は退避（中心 x 0.70）
    rows = run((), 100, every=10)
    Q = np.array([r[0][:4] for r in rows])
    print(f'  ⭐ 角の範囲 {[(round(float(a), 1), round(float(b), 1)) for a, b in zip(np.degrees(Q.min(0)), np.degrees(Q.max(0)))]}° / '
          f'最大角速度 {np.abs([r[1][:4] for r in rows]).max(0).round(2).tolist()} rad/s / 床からの最低 {min(r[2] for r in rows)*1000:.1f} mm', flush=True)

    # ---- 2. 速さの上限 ----
    print('\n[ra15] === 2. 旋回に +1（腕はまっすぐ上）', flush=True)
    place([0, 0, 0, 0], cube=(0.5, 0))
    rows = run(((1, 1.0),), 25)
    w = np.array([r[1][0] for r in rows])
    print(f'  ⭐ 旋回の角速度 {w[:8].round(3).tolist()} … 最後 {w[-1]:.3f} rad/s（期待 5.96〜5.97）', flush=True)
    i63 = int(np.argmax(w >= 0.632 * w[-1]))
    print(f'  ⭐ 63 % に達する step {i63 + 1}（{(i63 + 1)*DT:.2f} s。armature ÷ b = {0.008/0.235:.3f} s ＋ 腕の慣性の分）', flush=True)

    # ---- 3. 可動域 ----
    print('\n[ra15] === 3. 可動域（各関節に ±1、他は 0。2 秒）', flush=True)
    for j, nm in ((2, '肩'), (3, '肘'), (4, '手先')):
        for s in (+1.0, -1.0):
            base = [0, 0, 0, 0]
            if nm == '手先':
                base = [0, 0, deg(60), 0]            # 手先はまっすぐ上だと肘から先が軽く倒れるので、肘を少し曲げて重力の向きを変えない
            place(base, cube=(0.5, 0))
            rows = run(((j, s),), 50, detail=(DETAIL == f'3:{j}{"+" if s > 0 else "-"}'))
            qj = np.degrees([r[0][j - 1] for r in rows])
            ext = qj.max() if s > 0 else qj.min()
            ov = getattr(world, '_limit_overshoot', None)
            print(f'  ⭐ {nm} {s:+.0f}: 行き着いた角 {ext:7.1f}° 最後 {qj[-1]:7.1f}°  床からの最低 {min(r[2] for r in rows)*1000:6.1f} mm'
                  + (f'  端を越えた最大 {np.degrees(ov):.2f}°' if ov is not None else ''), flush=True)
            if ov is not None:
                world._limit_overshoot = 0.0

    # ---- 4. 床 ----
    print('\n[ra15] === 4. 前へお辞儀して床へ押し付ける（肩 +1・肘 +1）', flush=True)
    place([0, deg(30), deg(60), 0], cube=(0.5, 0))
    rows = run(((2, 1.0), (3, 1.0)), 60, every=10)
    lows = np.array([r[2] for r in rows])
    print(f'  ⭐ 床からの最低 {lows.min()*1000:.1f} mm / 最後の 20 step の平均 {lows[-20:].mean()*1000:.1f} mm（負 = 床の下）', flush=True)

    # ---- 5. 箱の滑り ----
    print('\n[ra15] === 5. 箱に 0.1 m/s を与える（腕は後ろへ逃がす）', flush=True)
    qv = np.zeros(nq)
    qv[4] = 0.1
    place([0, deg(-60), 0, 0], cube=(0.0, 0.0), qv=qv)
    x0 = float(np.array(env.data.qpos)[4])
    rows = run((), 75)
    cv = np.array([r[1][4] for r in rows])
    cx = np.array([r[0][4] for r in rows])
    t_e = (int(np.argmax(cv <= cv[0] / np.e)) + 1) * DT if np.any(cv <= cv[0] / np.e) else None
    print(f'  ⭐ 箱の速さ {cv[:5].round(4).tolist()} … 最後 {cv[-1]:.4f} m/s / 1/e になる時間 {t_e} s（期待 1.35 s）/ '
          f'進んだ距離 {(cx[-1]-x0)*1000:.1f} mm（期待 0.1 × 1.35 = 135 mm 弱）', flush=True)

    # ---- 6. 押し ----
    print('\n[ra15] === 6. 押し: 手先を箱の手前に置き、肩 +1', flush=True)
    # 箱の手前の面 x = 0.17、中心の高さ 0.04。お辞儀の姿勢を粗く探す（手先のカプセルの表面が面の 5 mm 手前、高さは箱の中）
    best = None
    for a1 in np.radians(np.arange(0, 91, 3)):
        for a2 in np.radians(np.arange(0, 148, 3)):
            for a3 in np.radians(np.arange(-90, 91, 6)):
                # FK は XML の関節の位置から（まっすぐの姿勢で z 方向の関節の間）
                p = np.array([0.0, 0.039 + 0.0422])
                th = 0.0
                pts = []
                for L, a in ((0.080399, a1), (0.065892, a2), (0.056322, a3)):
                    th += a
                    p = p + L * np.array([np.sin(th), np.cos(th)])
                    pts.append(p.copy())
                tip = pts[-1]
                if abs(tip[1] - 0.04) > 0.01 or min(pp[1] for pp in pts) < 0.03:
                    continue
                gap = 0.17 - (tip[0] + 0.02)
                if 0.003 < gap < 0.012 and (best is None or gap < best[0]):
                    best = (gap, a1, a2, a3)
    if best is None:
        print('  ⛔ 手前に置ける姿勢が見つからない', flush=True)
        return
    _, a1, a2, a3 = best
    print(f'  姿勢: 肩 {np.degrees(a1):.0f}° 肘 {np.degrees(a2):.0f}° 手先 {np.degrees(a3):.0f}°（FK の隙間 {best[0]*1000:.1f} mm）', flush=True)
    place([0, a1, a2, a3], cube=(0.0, 0.0))
    print(f'  置いた直後: 箱との隙間 {cube_gap()*1000:.1f} mm / 床からの最低 {floor_low()*1000:.1f} mm', flush=True)
    if getattr(world, '_limit_overshoot', None) is not None:
        world._limit_overshoot = 0.0
    rows = run(((2, 1.0),), 50, every=5, detail=(DETAIL == '6'))
    gaps = np.array([r[3] for r in rows])
    cx = np.array([r[0][4] for r in rows])
    cv = np.array([r[1][4] for r in rows])
    cy = np.array([r[0][5] for r in rows])
    print(f'  ⭐ 箱の x の変位 最後 {cx[-1]*1000:.1f} mm / 最大の速さ {cv.max():.3f} m/s / y のずれ {np.abs(cy).max()*1000:.1f} mm / '
          f'箱との隙間の最小 {gaps.min()*1000:.1f} mm（負 = めり込み）/ 床からの最低 {min(r[2] for r in rows)*1000:.1f} mm'
          + (f' / 端を越えた最大 {np.degrees(world._limit_overshoot):.2f}°' if getattr(world, '_limit_overshoot', None) is not None else ''), flush=True)

    # ---- 7. 横から振って押す（旋回 +1）----
    print('\n[ra15] === 7. 横から振って押す: 手先を箱の高さ・距離に置き、旋回 −40° から +1', flush=True)
    # 縦の面（旋回 0）での運動学。カプセルの横のずれ（fromto の x）も含める（⚠️ 試験 6 はこれを無視して箱に届かなかった）
    order = ['1', '11', '111', '1111']
    z1 = float(np.asarray(env._body_xpos['1'], float)[2])
    def fk2(a):
        p = np.array([0.0, z1])
        th = 0.0
        segs = []
        for nm, q in zip(order, [0.0] + list(a)):
            a0, a1_, r = caps[nm]
            th += q
            R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])   # +y 回りの回転を xz 面で
            segs.append((nm, p + R @ a0[[0, 2]], p + R @ a1_[[0, 2]], r))
            L = {'1': 0.0422, '11': 0.080399, '111': 0.065892, '1111': 0.0}[nm]
            p = p + R @ np.array([0.0, L])
        return segs
    best = None
    for a1 in np.radians(np.arange(0, 121, 4)):
        for a2 in np.radians(np.arange(-10, 147, 4)):
            for a3 in np.radians(np.arange(-90, 136, 6)):
                segs = fk2([a1, a2, a3])
                if min(min(p[1], q[1]) - r for nm, p, q, r in segs if nm != '1') < 0.006:
                    continue
                _, p, q, r = segs[-1]
                ts = np.linspace(0, 1, 11)
                pts = p[None] + ts[:, None] * (q - p)[None]
                ok = (np.abs(pts[:, 0] - 0.20) < 0.015) & (np.abs(pts[:, 1] - 0.04) < 0.015)
                if ok.any():
                    sc = abs(pts[ok][:, 1].mean() - 0.04)
                    if best is None or sc < best[0]:
                        best = (sc, a1, a2, a3)
    if best is None:
        print('  ⛔ 手先が箱を通る姿勢が見つからない', flush=True)
        return
    _, a1, a2, a3 = best
    print(f'  姿勢: 肩 {np.degrees(a1):.0f}° 肘 {np.degrees(a2):.0f}° 手先 {np.degrees(a3):.0f}°（旋回 0 で手先のカプセルが箱の中心を通る）', flush=True)
    place([np.radians(-40), a1, a2, a3], cube=(0.0, 0.0))
    print(f'  置いた直後: 箱との隙間 {cube_gap()*1000:.1f} mm / 床からの最低 {floor_low()*1000:.1f} mm / 姿勢 {np.degrees(np.array(env.data.qpos, float)[:4]).round(1).tolist()}°', flush=True)
    if getattr(world, '_limit_overshoot', None) is not None:
        world._limit_overshoot = 0.0
    rows = run(((1, 1.0), (2, 0.15)), 40, every=4)    # 旋回 +1、肩は自重を少し支える
    gaps = np.array([r[3] for r in rows])
    cx = np.array([r[0][4] for r in rows]); cy = np.array([r[0][5] for r in rows])
    cv = np.hypot([r[1][4] for r in rows], [r[1][5] for r in rows])
    print(f'  ⭐ 箱の変位 x {cx[-1]*1000:.1f} mm・y {cy[-1]*1000:.1f} mm / 最大の速さ {cv.max():.3f} m/s / 箱との隙間の最小 {gaps.min()*1000:.1f} mm（負 = めり込み）'
          f' / 床からの最低 {min(r[2] for r in rows)*1000:.1f} mm', flush=True)


try:
    main()
except Exception:
    traceback.print_exc()
finally:
    sys.stdout.flush()
    os._exit(0)
