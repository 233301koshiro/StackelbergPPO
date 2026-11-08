#!/usr/bin/env python3
"""押しの「学習を使わない参照の制御」— 腕を伸ばしたまま根元を最大トルクで箱の方へ振り下ろす決まった打ち方（系譜 9-336）。

**なぜ要るか**: 想定問答 Q59 の表で、押しだけ「学習の外の比較」が無かった（到達は 9-299 の運動学の参照がある）。
学習した方策の型は「振り下ろして一度当てる」（目視記録: 中間・根元重）なので、その型を決まった規則で打たせ、
同じ判定（どの形が上か）が学習なしで出るかを見る。学習はしない。

規則（事前登録 9-336 と同じ。結果を見てから変えない）:
  - 形: 凍結（ギア比 150 / 100 / 80、先端の太さ 0.032 の既定値）。利用者の絵が決めるのは長さだけなので、既定値で比べる。
    凍結の run が無い形は `REF_PARAM_SCALE`（既定 0.0001）で凍結にする。使った形（ギア比・リンク長）を出力に出す。
  - 実行フェーズの最初の根元の角 q0 の符号の逆へ、根元の指令を最大（±1）にし、q0 が 0 を越えたら全関節の指令を 0 にする。
  - 振っている間、肘と手先は伸ばしたまま保つ: u = clip(−KP·q − KD·q̇, −1, 1)、KP = 1.0・KD = 0.1（指令は ±1 に正規化）。
  - 開始のくじ（関節と箱に ±0.1）は学習と同じ（env.reset()）。話数は `REF_EPISODES`（既定 10）。

`REF_RULE=place_push`（系譜 9-337。11-05 にユーザーへ説明した規則）:
  - 寄せる: 先端を P =（箱の手前の面 − 0.10 m, 0）へ、作業空間の PD で寄せる。F = 2000·(P − 先端) − 200·v、
    指令 u = clip(Jᵀ F / ギア比, −1, 1)。J は各関節（z 軸）の位置と先端から幾何で作る。
    |P − 先端| < 0.03 m かつ |v| < 0.1 m/s、または 150 step で押しへ移る。
  - 押す: 方向 Jᵀ [1, −5·先端の y] を、指令の最大の絶対値が 1 になるように割って 25 step（1 秒）与え、その後は全関節 0。

出力: `REF_OUT`（既定 single_run/_probe_push_reference/<run 名>.json）に話ごとの実行フェーズの報酬・箱の移動・形。

使い方（コンテナで。物理の環境変数は run の launch_env.json から読む）:
  EVAL_RESTORE_DIR=single_run/tripo_pjdp_prox_frozen_phys_fx \\
  USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \\
      --no-window --python scripts/probe_push_reference.py > <ログ> 2>&1
"""
import os, sys, json, traceback
sys.path.append(os.getcwd())
os.environ['USE_CHOREONOID'] = '1'

try:
    restore_dir = os.environ['EVAL_RESTORE_DIR']
    _le = os.path.join(restore_dir, 'launch_env.json')
    LAUNCH_ENV = json.load(open(_le))
    for _k, _v in LAUNCH_ENV.items():
        os.environ.setdefault(_k, _v)
    print(f'[ref] 学習時の環境で回す: {LAUNCH_ENV}', flush=True)

    import numpy as np
    import yaml
    import torch
    from omegaconf import OmegaConf
    from design_opt.utils.config import Config, replay_robot_param_scale
    from design_opt.agents.genesis_agent import BodyGenAgent, tensorfy
    from design_opt.utils.tools import set_global_seed

    KP, KD = 1.0, 0.1
    RULE = os.environ.get('REF_RULE', 'swing')
    assert RULE in ('swing', 'place_push'), RULE
    N_EP = int(os.environ.get('REF_EPISODES', '10'))
    SCALE = float(os.environ.get('REF_PARAM_SCALE', '0.0001'))
    run = os.path.basename(os.path.normpath(restore_dir))
    out = os.environ.get('REF_OUT', f'single_run/_probe_push_reference/{run}' + ('' if RULE == 'swing' else '_' + RULE) + '.json')

    FLAGS = OmegaConf.create(yaml.safe_load(open(f'{restore_dir}/.hydra/config.yaml')))
    d = OmegaConf.to_container(FLAGS, resolve=True)
    d.pop('restore_dir', None)
    cfg = Config(OmegaConf.create(d), os.getcwd(), restore_dir)
    cfg.restore_dir = restore_dir
    replay_robot_param_scale(cfg, restore_dir)
    if float(cfg.robot_param_scale) != SCALE:
        print(f'[ref] ⭐ robot_param_scale {cfg.robot_param_scale} → {SCALE}（凍結にする）', flush=True)
        cfg.robot_param_scale = SCALE
    cfg.control_prior = False
    cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)
    set_global_seed(cfg.seed)
    agent = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                         seed=cfg.seed, num_threads=1, training=False, checkpoint='best')
    env = agent.env

    eps = []
    for k in range(N_EP):
        state = env.reset()
        r_exec = r_ctrl = 0.0
        q0 = None; swinging = True; n_swing = 0; shape = None
        cube0 = None; vmax = 0.0
        phase = None; place_steps = None; place_err = None
        for _ in range(cfg.skel_transform_nsteps + 2 + 1200):
            sv = tensorfy([state])
            if agent.obs_norm is not None:
                sv = agent.normalize_observation(sv)
            with torch.no_grad():
                action = agent.policy_net.select_action(sv, mean_action=True).numpy().astype(np.float64)
            in_exec = env.stage == 'execution'
            if in_exec:
                nj = len(env.robot.bodies) - 1
                q = np.asarray(env._qpos[:nj], dtype=float)
                qd = np.asarray(env._qvel[:nj], dtype=float)
                if q0 is None:
                    q0 = float(q[0])
                    cube0 = np.asarray(env.get_body_com('cube'), dtype=float).copy()
                    shape = {'gear': [float(b.joints[0].actuator.gear) for b in env.robot.bodies[1:]],
                             'bone': [np.asarray(b.bone_offset, dtype=float).round(4).tolist() for b in env.robot.bodies[1:]],
                             'tip_radius': float(np.asarray(env.robot.bodies[-1].geoms[0].size, dtype=float).flatten()[0])}
                u = np.zeros(nj)
                if RULE == 'swing':
                    if swinging and np.sign(q[0]) != np.sign(q0):
                        swinging = False
                    if swinging:
                        n_swing += 1
                        u[0] = -np.sign(q0)
                        u[1:] = np.clip(-KP * q[1:] - KD * qd[1:], -1, 1)
                else:
                    tip = np.asarray(env._arm_tip_pos, dtype=float)
                    if phase is None:
                        P = np.array([cube0[0] - env._get_cube_half_size() - 0.10, 0.0])
                        phase, n_phase, tip_prev = 'place', 0, tip.copy()
                    v = (tip[:2] - tip_prev[:2]) / env.dt
                    tip_prev = tip.copy()
                    jp = [np.asarray(env._body_xpos[b.name], dtype=float) for b in env.robot.bodies[1:]]
                    J = np.array([[-(tip[1] - p_[1]), tip[0] - p_[0]] for p_ in jp]).T   # 2 × nj（z 軸の回転）
                    gear = np.array(shape['gear'])
                    n_phase += 1
                    if phase == 'place':
                        F = 2000.0 * (P - tip[:2]) - 200.0 * v
                        u = np.clip(J.T @ F / gear, -1, 1)
                        if (np.linalg.norm(P - tip[:2]) < 0.03 and np.linalg.norm(v) < 0.1) or n_phase >= 150:
                            place_steps, place_err = n_phase, float(np.linalg.norm(P - tip[:2]))
                            phase, n_phase = 'push', 0
                    elif phase == 'push':
                        dvec = J.T @ np.array([1.0, -5.0 * tip[1]]) / gear
                        u = dvec / max(np.abs(dvec).max(), 1e-9)
                        n_swing += 1
                        if n_phase >= 25:
                            phase = 'done'
                action[:, :env.control_action_dim] = 0.0
                action[1:, 0] = u
                vmax = max(vmax, float(np.linalg.norm(qd[0])))
            state, reward, term, trunc, info = env.step(action)
            if in_exec:
                r_exec += float(reward); r_ctrl += float(info.get('reward_ctrl', 0.0))
            if term or trunc:
                break
        cube1 = np.asarray(env.get_body_com('cube'), dtype=float)
        e = {'ep': k, 'reward_exec': r_exec, 'reward_ctrl': r_ctrl, 'q0_deg': np.degrees(q0) if q0 is not None else None,
             'swing_steps': n_swing, 'cube_dx': float(cube1[0] - cube0[0]) if cube0 is not None else None,
             'cube_dy': float(cube1[1] - cube0[1]) if cube0 is not None else None, 'base_qd_max': vmax,
             'place_steps': place_steps, 'place_err': place_err,
             'cube0': cube0.round(4).tolist() if cube0 is not None else None, 'cube1': cube1.round(4).tolist()}
        eps.append(e)
        print(f'[ref] 話 {k}: 報酬 {r_exec:.3f}（制御 {r_ctrl:.3f}）  q0 {e["q0_deg"]:.1f}°  振った {n_swing} step  '
              f'箱 dx {e["cube_dx"]:+.3f} dy {e["cube_dy"]:+.3f} m  根元の最大角速度 {vmax:.2f} rad/s'
              + (f'  寄せ {place_steps} step・残り {place_err:.3f} m' if place_steps is not None else ''), flush=True)
    rs = np.array([e['reward_exec'] for e in eps])
    res = {'run': run, 'rule': {'name': RULE, 'KP': KP, 'KD': KD, 'param_scale': SCALE, 'episodes': N_EP},
           'shape': shape, 'replay_env': LAUNCH_ENV, 'episodes': eps,
           'reward_median': float(np.median(rs)), 'reward_min': float(rs.min()), 'reward_max': float(rs.max())}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)
    print(f'[ref] ⭐ 形: ギア比 {shape["gear"]}・ボーン {shape["bone"]}・先端の太さ {shape["tip_radius"]}', flush=True)
    print(f'[ref] ⭐ {run}: 報酬 中央 {np.median(rs):.3f} [{rs.min():.3f}, {rs.max():.3f}]（{N_EP} 話）→ {out}', flush=True)
except Exception:
    traceback.print_exc()
    sys.stdout.flush()
    os._exit(1)
# Choreonoid（Qt）のイベントループが残るので os._exit で落とす（Bug 43、record_arm_trace.py と同じ）
os._exit(0)
