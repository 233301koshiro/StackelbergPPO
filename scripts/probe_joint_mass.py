#!/usr/bin/env python3
"""probe_joint_mass.py: 段1b「トルクに質量を払わせる」の投入前確認（実験系譜 9-83 → 9-97）。

段1b は `actuator_params.gear.torque_density` を置いた cfg でだけ有効になり、
関節ごとに **gear / トルク密度 [kg]** を body の <inertial> として足す（xml_robot.Body._sync_inertial）。

3 つのことを機械で確かめる。

  ① --check-cfgs : design_opt/cfg/*.yml 全部で gear=400 にして XML を出し、
                   <inertial> と armature が**書かれない**ことを確認する（既存 run の物理を変えない）
  ② 既定        : トルク密度 × gear の格子で、腕を水平に伸ばした最悪姿勢の
                   自重トルク（MuJoCo の qfrc_bias）と対象を押すトルク（Jᵀ·μmg）を測り、
                   **余裕 = gear / 自重トルク** と **残り = gear − 自重 − 押し** を表にする。
                   9-83 の解析値「余裕 = 密度 / 18.7」を実測で置き換える
  ③ --write-xml : gear ごとの XML を書き出す（probe_shot_speed.py に渡して先端加速度を測る）

    python3 scripts/probe_joint_mass.py --check-cfgs
    python3 scripts/probe_joint_mass.py                       # 感度表
    python3 scripts/probe_joint_mass.py --write-xml /tmp/x    # → /tmp/x/gear{20,100,400}.xml
    PROBE_XML=/tmp/x/gear400.xml python3 scripts/probe_shot_speed.py
"""
import argparse
import copy
import glob
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from khrylib.robot.xml_robot import Robot   # noqa: E402

XML = os.environ.get('PROBE_XML_BASE', 'assets/mujoco_envs/e2e_a1v.xml')
CFG = os.environ.get('PROBE_CFG', 'design_opt/cfg/pusher_tripo_v3_actuator_mass.yml')
MU = 0.5          # cube の摩擦（XML の friction="0.5 ..."）


def build(robot_cfg, gear, density=None):
    """cfg・gear・トルク密度から XML を作る。返り値 (xml, 先端ボディの局所先端座標)。"""
    cfg = copy.deepcopy(robot_cfg)
    if density is not None:
        cfg['actuator_params']['gear']['torque_density'] = density
    r = Robot(cfg, XML)
    for b in r.bodies:
        for j in b.joints:
            if j.actuator is not None:
                j.actuator.gear = float(gear)
    r.sync_node()
    tip_body = r.bodies[-1]                       # Robot は最初の worldbody/body（腕）だけを読む
    tip_off = tip_body.geoms[0].end - tip_body.pos
    return r.export_xml_string().decode(), tip_off


def build_xml(robot_cfg, gear, density=None):
    return build(robot_cfg, gear, density)[0]


def check_cfgs():
    bad = 0
    for p in sorted(glob.glob('design_opt/cfg/*.yml')):
        cfg = yaml.safe_load(open(p)).get('robot')
        if not cfg or 'actuator_params' not in cfg:
            print(f'  -   {os.path.basename(p):40s} robot/actuator_params 無し'); continue
        try:
            xml = build_xml(cfg, 400)
        except Exception as e:   # 他環境用の cfg は e2e_a1v と噛み合わないことがある
            print(f'  ?   {os.path.basename(p):40s} 生成不可 ({type(e).__name__})'); continue
        body = xml.split('<worldbody>')[1]
        has_m, has_a = '<inertial' in body, 'armature=' in body
        on = cfg['actuator_params'].get('gear', {})
        expect_m, expect_a = bool(on.get('torque_density')), bool(on.get('armature_ref_gear'))
        ok = (has_m == expect_m) and (has_a == expect_a)
        bad += not ok
        print(f'  {"✅" if ok else "❌"}  {os.path.basename(p):40s} gear=400 → '
              f'inertial {"有" if has_m else "無"} / armature {"有" if has_a else "無"}')
    print(f'\n→ 期待と食い違う cfg: {bad} 件')
    return bad


def measure(robot_cfg, gear, density, grid=13):
    """(腕質量, 最悪姿勢の自重τ, 押し姿勢の最良残り, 押せる姿勢数/届く姿勢数) を返す。

    最悪姿勢: 根元ヨー 0、最初のピッチ 90°（腕を水平に伸ばす）、以降 0。ここで自重τ が最大。
    押し姿勢: 対象の手前の面（x = cube_x − half, y = 0, z = cube_z）に先端が届く姿勢を格子で集め、
              各姿勢で 残り_j = gear − |自重τ_j| − |(Jᵀ·μmg)_j| の最小関節を取り、姿勢の最良を報告する
              （方策は楽な姿勢を選べる）。⚠️ 水平に伸ばした 1 姿勢では押し力が腕の軸に乗って
              トルクが 0 になるので、押しは格子でしか測れない。
    """
    import mujoco
    xml, tip_off = build(robot_cfg, gear, density)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    jn = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    arm = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and not jn(j).startswith('cube')]
    dof = [m.jnt_dofadr[j] for j in arm]
    qadr = [m.jnt_qposadr[j] for j in arm]
    tip_body = m.jnt_bodyid[arm[-1]]
    arm_mass = float(sum(m.body_mass[m.jnt_bodyid[j]] for j in arm))
    cube = m.body(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'cube'))
    cube_geom = m.geom(cube.geomadr[0])
    target = np.array([cube.pos[0] - cube_geom.size[0], 0.0, cube.pos[2]])
    F = np.array([-MU * float(m.body_mass[cube.id]) * 9.81, 0, 0])   # 押しの反作用
    jacp = np.zeros((3, m.nv))

    def at(q):
        d.qpos[:] = 0
        for a, v in zip(qadr, q):
            d.qpos[a] = v
        mujoco.mj_forward(m, d)
        return np.abs(d.qfrc_bias[dof])

    g_worst = at([0, np.pi / 2] + [0] * (len(arm) - 2)).max()
    rng = [m.jnt_range[j] for j in arm[1:]]
    axes = [np.linspace(r[0], r[1], grid) for r in rng]
    best, n_ok, n_reach = -np.inf, 0, 0
    for q in np.array(np.meshgrid(*axes, indexing='ij')).reshape(len(arm) - 1, -1).T:
        g_tau = at([0.0, *q])
        tip = d.xpos[tip_body] + d.xmat[tip_body].reshape(3, 3) @ tip_off
        if np.linalg.norm(tip - target) > 0.03:
            continue
        n_reach += 1
        mujoco.mj_jac(m, d, jacp, None, tip, tip_body)
        rest = float((gear - g_tau - np.abs(jacp[:, dof].T @ F)).min())
        n_ok += rest > 0
        best = max(best, rest)
    return arm_mass, g_worst, best, n_ok, n_reach


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-cfgs', action='store_true')
    ap.add_argument('--write-xml', metavar='DIR')
    ap.add_argument('--gears', default='20,30,50,100,200,400')
    ap.add_argument('--densities', default='17,18.7,20,25,34')
    ap.add_argument('--grid', type=int, default=13, help='押し姿勢の格子（関節あたり）')
    a = ap.parse_args()
    if a.check_cfgs:
        return 1 if check_cfgs() else 0
    robot_cfg = yaml.safe_load(open(CFG))['robot']
    gears = [float(x) for x in a.gears.split(',')]
    if a.write_xml:
        os.makedirs(a.write_xml, exist_ok=True)
        for g in gears:
            p = os.path.join(a.write_xml, f'gear{g:g}.xml')
            open(p, 'w').write(build_xml(robot_cfg, g))
            print('wrote', p)
        return 0
    dens = [float(x) for x in a.densities.split(',')]
    print(f'[joint_mass] {XML}  cfg={CFG}  トルク上限 = gear（ctrlrange ±1）')
    print(f'{"密度":>6} {"gear":>5} {"腕質量":>7} {"自重τ(水平)":>11} {"余裕":>6} {"押し残り(最良)":>13} {"押せる/届く":>10}  判定')
    for rho in dens:
        for g in gears:
            mass, g_worst, best, n_ok, n_reach = measure(robot_cfg, g, rho, a.grid)
            margin = g / g_worst
            verdict = ('❌ 自重で潰れる' if margin < 1 else
                       '⚠️ 自重は支えるが押せない' if best < 0 else '✅')
            print(f'{rho:>6g} {g:>5g} {mass:>7.2f} {g_worst:>11.2f} {margin:>6.2f} {best:>13.2f} '
                  f'{n_ok:>4d}/{n_reach:<5d}  {verdict}')
        print()
    print('読み方: 余裕 < 1 なら水平に伸ばした姿勢を保持できない（密度の崖。gear に依らない）。'
          '押し残り < 0 なら届く姿勢のどれでも 2.7 kg の対象を押せない（低 gear 側の床）。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
