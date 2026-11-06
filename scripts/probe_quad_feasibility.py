#!/usr/bin/env python3
"""**四脚（脚 4 本 × 各脚 2 関節）を Choreonoid の経路に通し、床に立つか・脚を振ると進むかを見る**（系譜 9-342、2026-11-06）。

なぜ: ユーザー「今学習回ってる間にすることないからちょっと小規模な4脚の実験をしてみたい」。この repo の Choreonoid の経路は
根元を固定した腕でしか使っておらず、根元が自由な体（free joint）は BodyGen の ant を移植したときの名残のコードだけがある
（`design_opt/envs/ant.py`、この repo で run は 0 本）。学習（GPU）の前に、方策を使わない台本だけで経路が通るかを確かめる。

やること（学習しない・GPU を使わない）: `quad_gearonly` の cfg（env は ant、設計変数はギア比だけ）と `assets/mujoco_envs/quad_4x2.xml`
（胴は原点の球、各脚は股〔鉛直軸〕と膝）で env を作り、
設計の段階を行動 0 で抜けて実行の段階に入れ、次の 2 つを PROBE_STEPS step（既定 250 step ＝ 10 s）ずつ回す。
  A. 入力 0: 胴の高さと傾きを毎 step 測る（床に立つか、床を突き抜けないか、跳ねて飛ばないか）
  B. 台本の速歩（トロット）: 股 u = A sin(2π f t + φ)、膝 u = A sin(2π f t + φ + π/2)、対角の脚どうしを同じ位相（φ = 0 と π）。
     A = PROBE_AMP（既定 0.8）、f = PROBE_FREQ（既定 1.5 Hz）。胴の x の進みと高さ・傾きを測る。B は A の後に env を reset して始める。

読み（回す前に決めた。⑤-2 のとおりプローブは本番より弱い証拠で、言えるのは「経路が通るか」まで）:
  ① 全 step で状態が有限・胴の中心の高さが PROBE_MIN_Z（既定 0.15 m＝quad_4x2 の胴の球の半径。quad_tripo は箱の半分の高さ 0.10）より上・傾きが 60° 未満 → 床に立てる（A）・振っても転ばない（B）
     （⚠️ 最初の版は胴がカプセル〔半径 0.06〕で閾値 0.06 だった。形を作り直したので、回す前に胴の半径へ合わせた。9-342）
  ② B で胴の x の進みの絶対値が 0.10 m 以上 → 足と床の接触と摩擦で前後に進める（進む向きは問わない。台本は最適な歩き方ではない）
  ③ どちらも満たす → 学習のスモーク（GPU、数エポック）へ進んでよい。満たさなければ、どこで崩れたかを出力から読んで止まる
⚠️ 物理の値は ant.xml の既定のまま（実在の部品から導いていない）。物理の環境変数は TICK_SYNC（Bug 64）と JOINT_LIMIT_TORQUE
（可動域を効かせる）だけ。腕の速度・トルクの値は使わない。自己衝突は切る（脚が全部胴の中心から出るので、根元どうしが重なる）。

    USE_CHOREONOID=1 OMP_NUM_THREADS=1 CNOID_TICK_SYNC=1 CNOID_JOINT_LIMIT_TORQUE=1 \\
    timeout -k 10 600 /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/probe_quad_feasibility.py > out.txt 2>&1

⚠️ 末尾は os._exit(0)（Choreonoid は自分では終わらない、Bug 43）。
"""
import os
import sys
import traceback

sys.path.insert(0, os.getcwd())
os.environ.setdefault('USE_CHOREONOID', '1')
STEPS = int(os.environ.get('PROBE_STEPS', '250'))
AMP = float(os.environ.get('PROBE_AMP', '0.8'))
FREQ = float(os.environ.get('PROBE_FREQ', '1.5'))
XML = os.environ.get('PROBE_XML', 'quad_4x2')
MIN_Z = float(os.environ.get('PROBE_MIN_Z', '0.15'))   # ① の胴の中心の高さの下限（胴の半分の高さ。quad_4x2 は球の半径 0.15、quad_tripo は箱の半分の高さ 0.10）

import numpy as np


def main():
    import yaml
    from omegaconf import OmegaConf
    from design_opt.utils.config import Config
    from design_opt.envs import env_dict
    from khrylib.utils.transformation import quaternion_matrix

    print('[quad] 物理の環境変数:', {k: os.environ.get(k) for k in (
        'CNOID_TICK_SYNC', 'CNOID_SELF_COLLISION', 'CNOID_JOINT_LIMIT_TORQUE', 'CNOID_CTRL_CLIP', 'CNOID_JOINT_MAX_VEL')}, flush=True)
    d = yaml.safe_load(open('design_opt/conf/config.yaml'))
    d.pop('hydra', None)
    d.update(cfg=os.environ.get('PROBE_CFG', 'quad_gearonly'), xml_name=XML, fix_skeleton=True, enable_wandb=False, num_threads=1)
    out_dir = '/tmp/probe_quad_feasibility'
    cfg = Config(OmegaConf.create(d), os.getcwd(), out_dir)
    env = env_dict[cfg.env_name](cfg, None)
    names = [b.name for b in env.robot.bodies]
    print(f'[quad] env {type(env).__name__}・XML {XML}・体 {len(names)} 個 {names}', flush=True)
    nb = len(names)
    W = env.control_action_dim + np.shape(env.design_cur_params)[1] + 1
    act_names = [b.get_actuator_name() for b in env.robot.bodies]
    print(f'[quad] 体ごとのアクチュエータ: {dict(zip(names, act_names))}', flush=True)

    def to_exec():
        env.reset()
        for _ in range(20):
            if env.stage == 'execution':
                break
            env.step(np.zeros((nb, W)))
        assert env.stage == 'execution', f'実行の段階に入らない（{env.stage}）'
        env.step(np.zeros((nb, W)))   # 入った「次の」step まで進める（9-103）

    def measure(s):
        z = float(s[2])
        zdir = quaternion_matrix(s[3:7])[:3, 2]
        return z, float(np.degrees(np.arccos(np.clip(zdir[2], -1, 1))))

    def run(label, ctrl_fn):
        to_exec()
        x0 = float(env.get_body_com('0')[0])
        zs, angs, fin, ends = [], [], True, None
        for t in range(STEPS):
            a = np.zeros((nb, W))
            a[:, 0] = ctrl_fn(t)
            _ob, _r, term, trunc, _info = env.step(a)
            s = env.state_vector()
            fin = fin and bool(np.isfinite(s).all())
            z, ang = measure(s)
            zs.append(z); angs.append(ang)
            if (term or trunc) and ends is None:
                ends = (t, 'termination' if term else 'truncation')
                x_end = float(env.get_body_com('0')[0])
                break   # ⚠️ 終わった後も回すと、転がった距離まで「進み」に入る（最初の版でそうなった。9-342）
        dx = (x_end if ends else float(env.get_body_com('0')[0])) - x0
        zs, angs = np.array(zs), np.array(angs)
        print(f'[quad] {label}: 有限 {fin}・胴の高さ 最小 {zs.min():.3f} / 最後 {zs[-1]:.3f} / 最大 {zs.max():.3f} m・'
              f'傾き 最大 {angs.max():.1f}°・x の進み {dx:+.3f} m・終わり {ends or "なし"}（{STEPS} step）', flush=True)
        ok1 = fin and zs.min() > MIN_Z and angs.max() < 60
        return ok1, dx

    # 股＝親にアクチュエータが無い駆動の体、膝＝親にアクチュエータがある駆動の体（quad_4x2 と、関節の無い取り付けを挟む quad_tripo の両方で成り立つ）。
    # 位相は脚の股の位置の象限で決め、対角（前左＋後右・前右＋後左）を同じ位相にする
    phase = np.zeros(nb)
    is_knee = np.zeros(nb, bool)
    for i, bd in enumerate(env.robot.bodies):
        if bd.get_actuator_name() is None:
            continue
        hip = bd
        is_knee[i] = bd.parent is not None and bd.parent.get_actuator_name() is not None
        if is_knee[i]:
            hip = bd.parent
        phase[i] = 0.0 if hip.pos[0] * hip.pos[1] > 0 else np.pi
    print(f'[quad] 位相 {dict(zip(names, np.round(phase, 2)))}・膝 {[n for n, k in zip(names, is_knee) if k]}', flush=True)
    # ⚠️ Bug 94: CNOID_SUBSTEPS=K のとき env.dt は 1 step の本当の長さの 1/K を返す（物理の刻みを割った値 × frame_skip）。
    #   1 step で進む時間は timestep × frame_skip のまま（tick を K 倍回す）なので、台本の時刻はその長さで刻む
    dt = env.dt * max(1, int(os.environ.get('CNOID_SUBSTEPS') or 1))
    print(f'[quad] 1 step の長さ {dt:.3f} s（env.dt {env.dt:.4f} s・刻みの分割 {os.environ.get("CNOID_SUBSTEPS") or 1}）', flush=True)

    okA, dxA = run('A 入力 0', lambda t: np.zeros(nb))
    trot = lambda t: np.where(np.array(names) == '0', 0.0,
                              AMP * np.sin(2 * np.pi * FREQ * t * dt + phase + np.where(is_knee, np.pi / 2, 0.0)))
    okB, dxB = run(f'B 速歩 A={AMP} f={FREQ} Hz', trot)
    print(f'[quad] 読み ① A {"✅" if okA else "⛔"}・B {"✅" if okB else "⛔"}／② |Δx| {abs(dxB):.3f} m {"✅" if abs(dxB) >= 0.10 else "⛔"}'
          f'／③ {"✅ スモークへ進んでよい" if (okA and okB and abs(dxB) >= 0.10) else "⛔ 止まる"}', flush=True)


try:
    main()
except Exception:
    traceback.print_exc()
sys.stdout.flush()
os._exit(0)
