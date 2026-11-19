#!/usr/bin/env python3
"""箱のばね（`CNOID_CUBE_SPRING_K`、系譜 9-354）が実際の Choreonoid で効いているかを確かめる。学習はしない。

箱を前（+x）へ `PROBE_DX` [m] ずらして静かに離し、腕への指令 0 のまま動かして、箱の x の時系列を取る。
- ばねあり: 減衰つきの振動 x'' = −(k/m)x − (b/m)x' になるはず。ゼロを横切る間隔から測った周期を、
  理論の周期 2π/ω_d（ω_d = √(k/m − (b/2m)²)）と並べる。b は XML の箱の減衰（⚠️ 横型 10.0・縦型 2.0 と XML ごとに違う。
  11-19 に 2.0 と思い込んで「減衰が 5 倍」と読み違えた）、m は箱の body の総質量（減衰・ばねの処理と同じ）。
- `PROBE_V0=v`（ばね無し）: 箱に初速 v を与え、止まるまでの距離から減衰の速さ b/m を測り、XML の値と並べる。
- ばね無し（`CNOID_CUBE_SPRING_K` を指定しない）: 箱は離した位置に止まったまま（今の振る舞い）。
  ⇒ 戻る動きがばねの処理から来ていることの対照（CLAUDE.md ⑤-3「既知の振る舞いを再現できるか」）。

    # ばねあり（k は N/m）
    CNOID_CUBE_SPRING_K=50 EVAL_RESTORE_DIR=single_run/tripo_pjp_mid_phys \\
    USE_CHOREONOID=1 OMP_NUM_THREADS=1 timeout -k 10 300 /choreonoid_ws/install/bin/choreonoid \\
        --no-window --python scripts/probe_cube_spring.py > /tmp/probe_spring.log 2>&1
    # ばね無しは CNOID_CUBE_SPRING_K を外して同じ行

⚠️ 物理の環境変数は run の `launch_env.json` から読む（学習と同じ物理。上で指定した変数は上書きしない）。
⚠️ Choreonoid 経由なので引数は渡せない（Bug 36）。出力はファイルへリダイレクトする（Bug 43）。末尾は os._exit(0)。
"""
import json
import math
import os
import sys
import traceback

sys.path.append(os.getcwd())
os.environ['USE_CHOREONOID'] = '1'
restore_dir = os.environ['EVAL_RESTORE_DIR']
_le = os.path.join(restore_dir, 'launch_env.json')
if os.path.exists(_le):
    for _k, _v in json.load(open(_le)).items():
        os.environ.setdefault(_k, _v)


def main():
    import numpy as np
    import yaml
    import torch
    from omegaconf import OmegaConf
    from design_opt.utils.config import Config
    from design_opt.agents.genesis_agent import BodyGenAgent
    from design_opt.utils.tools import set_global_seed

    dx = float(os.environ.get('PROBE_DX', '0.10'))
    nstep = int(os.environ.get('PROBE_STEPS', '400'))
    k = float(os.environ.get('CNOID_CUBE_SPRING_K') or 0.0)

    FLAGS = OmegaConf.create(yaml.safe_load(open(f'{restore_dir}/.hydra/config.yaml')))
    d = OmegaConf.to_container(FLAGS, resolve=True)
    d.pop('restore_dir', None)
    cfg = Config(OmegaConf.create(d), os.getcwd(), restore_dir)
    cfg.restore_dir = restore_dir
    cfg.control_prior = False
    cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)
    set_global_seed(cfg.seed)
    agent = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                         seed=cfg.seed, num_threads=1, training=False, checkpoint='best')
    env = agent.env
    env.reset_model()
    env.reset_state(False)

    # 箱の x のスライド（cube_slide）は変換器が作る仮のリンク `cube_virt0` の関節、y（cube_slide2）は `cube` の関節
    #   （mujoco_env_choreonoid.py の Multi-joint body）。ずらす向きはばねの向き（CNOID_CUBE_SPRING_AXIS、既定 x）
    axis = (os.environ.get('CNOID_CUBE_SPRING_AXIS') or 'x').strip().lower()
    link = {'x': 'cube_virt0', 'y': 'cube'}[axis]
    bnames = list(env.model.body_names)
    bid = bnames.index(link)
    qadr = env.model.jnt_qposadr[env.model.body_jntadr[bid]]
    print(f'[probe] 箱の {axis} のスライド: リンク {link} → qpos[{qadr}]', flush=True)
    qpos = env.data.qpos.copy()
    qvel = np.zeros_like(env.data.qvel)
    v0 = float(os.environ.get('PROBE_V0', '0'))   # 箱の x の初速 [m/s]。ばね無しで減衰そのものを測るとき（止まるまでの距離 = v0·m/b）
    q0 = float(qpos[qadr])
    qpos[qadr] = q0 + dx
    ci = {'x': 0, 'y': 1}[axis]
    xb = float(env.get_body_com('cube')[ci])
    qvel[qadr] = v0
    env.set_state(qpos, qvel)
    xa = float(env.get_body_com('cube')[ci])
    print(f'[probe] ずらす前後の箱の {axis}: {xb:+.4f} → {xa:+.4f}（差 {xa - xb:+.4f}、期待 {dx:+.4f}）', flush=True)
    W0 = env.model._env._world
    if W0._cube_spring:
        # ⭐ 自然長は set_state のたびに「その時の位置」へ取り直す（各話の初期位置）。ずらした位置が自然長に
        #   なっていること（＝取り直しが効いていること）を確かめてから、ずらす前の位置へ戻してばねを効かせる
        rest = W0._cube_spring[0][2]
        print(f'[probe] set_state 後の自然長 {rest:+.4f}（期待: ずらした位置 {q0 + dx:+.4f}）'
              + (' ✅ 各話の初期位置で取り直している' if abs(rest - (q0 + dx)) < 1e-6 else ' ❌ 取り直されていない'), flush=True)
        W0._cube_spring[0][2] = q0
    dt = float(env.dt)
    print(f'[probe] {restore_dir}  k={k} N/m  ずらし {dx} m（関節 {q0:+.4f} → {q0 + dx:+.4f}）  env の dt={dt}  {nstep} step', flush=True)

    ctrl = np.zeros(env.model.nu)
    xs = []
    for _ in range(nstep):
        env.do_simulation(ctrl, env.frame_skip)
        xs.append(float(env.data.qpos[qadr]))
    xs = np.array(xs)
    W = env.model._env._world
    b = float(W._joint_damping.get(('extra_0', {'x': 'cube_slide', 'y': 'cube_slide2'}[axis]), 0.0))   # XML の減衰（横型 10.0・縦型 2.0 など、XML ごとに違う）
    eb = next(sb.body() for sb, _ in W._sim_body_entries[1:]
              if 'cube_slide' in [sb.body().joint(i).jointName for i in range(sb.body().numJoints)])
    m = sum(eb.link(i).m for i in range(eb.numLinks))   # 減衰・ばねの処理と同じ m（箱の body の総質量）
    print(f'[probe] 箱: m={m:.4f} kg・XML の減衰 b={b} N·s/m → 減衰の速さ b/m={b / m:.4f} /s', flush=True)
    t = dt * (np.arange(nstep) + 1)
    print(f'[probe] 箱の関節 q（ずらす前 ＝ ばねの自然長）: 最初 {xs[0]:+.4f}  最小 {xs.min():+.4f}  最大 {xs.max():+.4f}  最後 {xs[-1]:+.4f}', flush=True)
    for i in range(0, nstep, max(1, nstep // 20)):
        print(f'   t={t[i]:6.3f} s  q={xs[i]:+.5f}', flush=True)

    if k <= 0:
        moved = abs(xs[-1] - (q0 + dx))
        print(f'[probe] ばね無し: 離した位置からの動き {moved * 1000:.2f} mm（0 に近ければ今の振る舞いどおり）', flush=True)
        if v0:
            lam = v0 / max(moved, 1e-9)
            err = abs(lam / (b / m) - 1)
            print(f'[probe] 初速 {v0} m/s で止まるまでに {moved:.4f} m → 測った減衰の速さ {lam:.4f} /s（期待 b/m={b / m:.4f}、ずれ {err * 100:.1f} %）', flush=True)
            print('✅ 減衰は XML の値どおり' if err < 0.05 else '❌ 減衰が XML の値と合わない', flush=True)
            return
        print('✅ ばね無しでは止まったまま' if moved < 1e-3 else '⚠️ ばね無しでも動いた（腕に触れている・重力など。値を見る）', flush=True)
        return

    xs = xs - q0   # 自然長（ずらす前の位置）からのずれ
    cross = [t[i] - (t[i] - t[i - 1]) * xs[i] / (xs[i] - xs[i - 1])
             for i in range(1, nstep) if xs[i - 1] * xs[i] < 0]
    print(f'[probe] ゼロを横切った時刻: {[round(c, 3) for c in cross[:8]]}', flush=True)
    if len(cross) >= 3:
        T = 2 * float(np.mean(np.diff(cross)))
        w2 = k / m - (b / (2 * m)) ** 2
        if w2 <= 0:
            print(f'⚠️ 減衰が強すぎて振動しない条件（k/m={k / m:.3f} ≤ (b/2m)²={(b / (2 * m)) ** 2:.3f}）なのにゼロを横切った', flush=True)
            return
        Tth = 2 * math.pi / math.sqrt(w2)
        err = abs(T / Tth - 1)
        print(f'[probe] 測った周期 {T:.4f} s（ゼロ交差 {len(cross)} 回から）・理論 2π/√(k/m−(b/2m)²) = {Tth:.4f} s（減衰なしなら {2 * math.pi * math.sqrt(m / k):.4f} s）・ずれ {err * 100:.1f} %', flush=True)
        print('✅ ばねは計算どおりに効いている' if err < 0.05 else '❌ 周期が理論と合わない', flush=True)
    else:
        print('⚠️ ゼロ交差が 3 回未満（減衰が強いか、時間が短い）。周期は測れない', flush=True)


try:
    main()
except Exception:
    traceback.print_exc()
    sys.stdout.flush()
finally:
    sys.stdout.flush()
    os._exit(0)
