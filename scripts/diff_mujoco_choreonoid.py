#!/usr/bin/env python3
"""**同じ XML・同じ台本の制御で MuJoCo と Choreonoid を動かし、物体の位置を比べる**（差分テスト、2026-10-12）。学習はしない。

[異常を能動的に見つける方法.md](../docs/リポジトリ説明/異常を能動的に見つける方法.md) の方法 4（そこでは「方法 1」）。
この研究の物理の穴の最多の型は「**MuJoCo（原論文）が暗黙にやっていることを、Choreonoid への移植が落とす**」だった
（Bug 1・2 減衰、Bug 44 円柱、9-98 壁、9-198 反発係数、Bug 57 深い接触、Bug 60 制御入力の上限）。

台本ごとに、**差が出るはずか（期待）を先に書いてから**比べる。接触のある台本はモデルが違うので量は合わなくてよい（向きと有無だけ見る）。

    # ① Choreonoid 側だけ回して軌跡を保存（⚠️ パイプへ繋がない。Bug 43。Choreonoid の中では mujoco を import できない）
    XML=tripo_arm_v2c_pj_mid OUT=/tmp/cn.npz USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \\
      --no-window --python scripts/diff_mujoco_choreonoid.py > /tmp/diff.log 2>&1
    # ② 普通の python3 で MuJoCo を回して比べる
    XML=tripo_arm_v2c_pj_mid OUT=/tmp/cn.npz python3 scripts/diff_mujoco_choreonoid.py
    # 物理の環境変数（CNOID_CTRL_CLIP など）を ① に渡すと、その設定の Choreonoid と比べる

出力: 台本ごとに、先端（最後の body）と対象の位置の最大差・最終差と、期待との一致（✅／⛔）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

XML = os.environ.get('XML', 'tripo_arm_v2c_pj_mid')
STEPS = int(os.environ.get('DIFF_STEPS', '150'))
FS = 4            # frame_skip（学習と同じ）
TOL = 0.02        # 一致とみなす位置の差 [m]


# ⭐ 自由運動の台本は対象を x+100 m へ逃がす（XML の初期姿勢では先端が箱に 11 cm めり込んでいた＝台本の欠陥。
#   MuJoCo は押し出し、Choreonoid は深い接触を捨てて幽霊にした＝Bug 57 を偶然再現した）。
#   接触の台本は根元の関節を −1.2 rad にして、横から振って当てる。
CUBE_FAR = {'S1', 'S2', 'S3', 'S4', 'S6', 'S7'}
Q1_INIT = {'S5': -1.2}
# ⭐ 接触の台本は対象を x 0.6 m へ寄せる（元の 1.0 m では伸びきった先端が 5 cm かすめるだけで、結果が姿勢の差に敏感すぎた。9-246）
CUBE_SHIFT = {'S5': -0.4}
# ⭐ 9-250: 縦型の接触の台本 S8。根元が鉛直軸の腕は S5（根元を回す）では動かないので、2 番目の関節（水平軸）を
#   +1 で押し続けて腕を +x へ倒し、上から対象に当てる。⭐ 期待（回す前に決めた）: 両方の物理で対象が動く（> 0.01 m）。
#   ⚠️ 対象は**最初のリンクの真下**（中心 x ＝ 最初のリンクの長さ）へ寄せる（`s8_shift`）。⛔ 最初は S5 と同じ −0.4 m にしたが、
#   Choreonoid では可動域の端（硬い止め）で腕が L 字に止まり、ホッケー・tripo_arm_v3 で対象に一度も触れなかった（接触を比べていない）。
#   平面の腕（2 番目の関節も鉛直軸）では比べない


def scenarios(nu):
    """(名前, step→ctrl の関数, 初期の関節速度（全関節に足す）, 期待)。期待は 'same'（一致するはず）か 'contact'（有無だけ）。"""
    return [
        ('S1 弱い一定の入力 0.3', lambda t: np.full(nu, 0.3), 0.0, 'same'),
        ('S2 範囲外の入力 3.0（MuJoCo は ±1 に切る）', lambda t: np.full(nu, 3.0), 0.0, 'same'),
        ('S3 入力 0 で惰行（初速 2 rad/s）', lambda t: np.zeros(nu), 2.0, 'same'),
        ('S4 可動域の端まで押し続ける 1.0', lambda t: np.full(nu, 1.0) if t < STEPS else np.zeros(nu), 0.0, 'same'),
        ('S6 入力 0 で惰行（初速 0.3 rad/s、可動域に当たらない。⚠️ 縦型は真上に立った腕の鉛直軸で先端が動かず比較にならない、9-253）', lambda t: np.zeros(nu), 0.3, 'same'),
        ('S7 全力で正負を切り替える（10 step ごと、関節ごとに逆向き）', lambda t: np.array([1.0 if (t // 10 + k) % 2 == 0 else -1.0 for k in range(nu)]), 0.0, 'same'),
        ('S5 振って対象に当てる（接触）', lambda t: np.array([1.0] + [0.0] * (nu - 1)) * (1 if t < 40 else -1), 0.0, 'contact'),
        ('S8 腕を前へ倒して上から対象に当てる（縦型の接触）', lambda t: np.array([0.0, 1.0] + [0.0] * (nu - 2)), 0.0, 'contact'),
    ]


def far_ok(xml):
    """対象のスライドの範囲が 100 m 先まで許すか（ホッケーのパックは −0.35〜1.25 m で逃がせない）。"""
    import re as _re
    m = _re.search(r'<joint name="cube_slide"[^>]*range="([-\d.]+) ([-\d.]+)"', xml)
    return (m is None) or float(m.group(2)) > 100


def is_vertical(xml):
    """2 番目の関節（`11_joint`）が水平軸か＝縦型の腕か（9-250。S8 は縦型でだけ比べる）。"""
    import re as _re
    m = _re.search(r'<joint name="11_joint"[^>]*axis="([-\d.]+) ([-\d.]+) ([-\d.]+)"', xml)
    return bool(m) and abs(float(m.group(3))) < 0.5


def s8_shift(xml):
    """S8: 対象の中心を最初のリンクの長さ（body 111 の位置）へ寄せる量。"""
    import re as _re
    l1 = _re.search(r'<body name="111" pos="[-\d.]+ [-\d.]+ ([-\d.]+)"', xml)
    c0 = _re.search(r'<body name="cube" pos="([-\d.]+) ', xml)
    return (float(l1.group(1)) - float(c0.group(1))) if (l1 and c0) else 0.0


def shift_of(tag, xml):
    return s8_shift(xml) if tag == 'S8' else CUBE_SHIFT.get(tag, 0.0)


def shift_ok(xml, shift):
    """ずらした対象がスライドの範囲の内側に収まるか（9-249）。
    ⛔ ホッケーのパックは範囲 −0.35〜1.25 m で、−0.4 ずらすと範囲の外に出る。MuJoCo は範囲の拘束で押し戻して
    パックが打ち出され（腕に触れずに 0.6 m 動く）、Choreonoid は動かさない＝台本の欠陥で差が出ていた。"""
    import re as _re
    m = _re.search(r'<joint name="cube_slide"[^>]*range="([-\d.]+) ([-\d.]+)"', xml)
    return (m is None) or float(m.group(1)) <= shift <= float(m.group(2))


def run_mujoco(xml, ctrl_fn, v0, name):
    import mujoco
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    robot_dofs = [m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
    tag = name.split()[0]
    if tag in Q1_INIT:
        d.qpos[m.jnt_qposadr[m.actuator_trnid[0, 0]]] = Q1_INIT[tag]
    if tag in CUBE_FAR and far_ok(xml):
        d.qpos[m.jnt_qposadr[m.joint('cube_slide').id]] += 100.0
    if shift_of(tag, xml) and shift_ok(xml, shift_of(tag, xml)):
        d.qpos[m.jnt_qposadr[m.joint('cube_slide').id]] += shift_of(tag, xml)
    mujoco.mj_forward(m, d)
    d.qvel[robot_dofs] = v0
    names = [m.body(i).name for i in range(m.nbody)]
    traj = []
    for t in range(STEPS):
        d.ctrl[:] = ctrl_fn(t)
        for _ in range(FS):
            mujoco.mj_step(m, d)
        traj.append({n: d.xpos[i].copy() for i, n in enumerate(names)})
    return traj


def run_choreonoid(xml, ctrl_fn, v0, name):
    from khrylib.rl.envs.common.mujoco_env_choreonoid import ChoreonoidSimWorld
    w = ChoreonoidSimWorld()
    info = w.load_model(xml, FS)
    nu = len(info['actuator_names'])
    q0 = np.array(info['init_qpos'], float)
    v = np.array(info['init_qvel'], float)
    # ⚠️ 関節の速度の並びは actuators_map の順（ロボット本体の関節）。対象のスライドは後ろ
    v[:nu] = v0
    tag = name.split()[0]
    if tag in Q1_INIT:
        q0[0] = Q1_INIT[tag]
    if tag in CUBE_FAR and far_ok(xml):
        q0[nu] += 100.0          # 対象のスライド x（ロボットの関節の後ろ）
    if shift_of(tag, xml) and shift_ok(xml, shift_of(tag, xml)):
        q0[nu] += shift_of(tag, xml)
    w.set_state_cmd(q0.tolist(), v.tolist())
    traj = []
    for t in range(STEPS):
        resp = w.step(list(ctrl_fn(t)), FS)
        traj.append({k: np.asarray(p, float) for k, p in resp['body_xpos'].items()})
    return traj, nu


def scen_nu(xml):
    import re
    return len(re.findall(r'<motor ', xml))


def main_choreonoid():
    import json
    xml = open(os.path.join('assets', 'mujoco_envs', f'{XML}.xml')).read()
    nu = scen_nu(xml)
    env = {k: os.environ[k] for k in ('CNOID_CTRL_CLIP', 'CNOID_HINGE_DAMP_TORQUE', 'CNOID_JOINT_MAX_VEL', 'CNOID_JOINT_MAX_VEL_STRICT', 'CNOID_TICK_SYNC', 'CNOID_CULLING_DEPTH', 'CNOID_SELF_COLLISION', 'HOCKEY_ARM_BLOCK', 'HOCKEY_WALL_RESTITUTION') if k in os.environ}
    out = {}
    for k, (name, fn, v0, expect) in enumerate(scenarios(nu)):
        traj, _ = run_choreonoid(xml, fn, v0, name)
        names = sorted(traj[0])
        out[f's{k}'] = np.array([[t[n] for n in names] for t in traj])
        out[f's{k}_names'] = np.array(names)
    np.savez(os.environ['OUT'], env=json.dumps(env), **out)
    print(f'[diff] Choreonoid 側を保存: {os.environ["OUT"]}  環境 {env}', flush=True)
    os._exit(0)   # Choreonoid は終わらないことがある（choreonoid_train.py と同じ）


def main():
    import json
    import mujoco
    path = os.path.join('assets', 'mujoco_envs', f'{XML}.xml')
    xml = open(path).read()
    nu = mujoco.MjModel.from_xml_string(xml).nu
    z = np.load(os.environ['OUT'])
    env = json.loads(str(z['env']))
    print(f'[diff] XML={XML}  関節 {nu}  {STEPS} step × frame_skip {FS}  Choreonoid の物理の環境変数 {env}', flush=True)
    for k, (name, fn, v0, expect) in enumerate(scenarios(nu)):
        mj = run_mujoco(xml, fn, v0, name)
        nm = [str(n) for n in z[f's{k}_names']]
        cn = [{n: z[f's{k}'][t, i] for i, n in enumerate(nm)} for t in range(z[f's{k}'].shape[0])]
        common = [n for n in mj[0] if n in cn[0] and n not in ('world',)]
        robot = [n for n in common if n[:1].isdigit()]   # 腕の body は数字の名前（壁・板・対象を除く）
        tip = robot[-1]
        dt = np.array([np.linalg.norm(a[tip] - b[tip]) for a, b in zip(mj, cn)])
        line = f'  先端 {tip}: 最大差 {dt.max()*1000:6.0f} mm・最後 {dt[-1]*1000:6.0f} mm'
        if 'cube' in common:
            mc = max(np.linalg.norm(t['cube'][:2] - mj[0]['cube'][:2]) for t in mj)
            cc = max(np.linalg.norm(t['cube'][:2] - cn[0]['cube'][:2]) for t in cn)
            line += f'  ／ 対象の移動 MuJoCo {mc:.3f} m・Choreonoid {cc:.3f} m'
        if name.startswith('S8') and not is_vertical(xml):
            print(f"➖ {name}（縦型用。この腕は 2 番目の関節も鉛直軸）", flush=True)
            continue
        if expect == 'contact' and 'cube' in common and shift_ok(xml, shift_of(name.split()[0], xml)) is False:
            print(f"➖ {name}（⚠️ 対象をずらすと範囲の外に出るので、ずらさずに回した）", flush=True)
        # ⭐ 9-249: 接触の台本で腕がどちらでも動かないなら、何も比べていない（縦に立った腕は根元を回しても動かない）
        moved_mj = max(np.linalg.norm(t[tip] - mj[0][tip]) for t in mj)
        moved_cn = max(np.linalg.norm(t[tip] - cn[0][tip]) for t in cn)
        if expect == 'contact' and max(moved_mj, moved_cn) < 0.001:
            print(f"➖ {name}（比べられない: 腕の先端がどちらでも動かない。縦に立った腕は根元の鉛直軸を回しても動かない）\n{line}", flush=True)
            continue
        if expect == 'same':
            ok = dt.max() < TOL
        else:
            ok = (mc > 0.01) == (cc > 0.01)
        print(f"{'✅' if ok else '⛔'} {name}（期待: {'一致' if expect == 'same' else '対象が動くかどうかが一致'}）\n{line}", flush=True)
    print('[diff] 完了', flush=True)


if os.environ.get('USE_CHOREONOID'):
    main_choreonoid()
else:
    main()
