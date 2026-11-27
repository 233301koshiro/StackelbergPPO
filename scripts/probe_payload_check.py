#!/usr/bin/env python3
"""先端の荷物（`robot.payload_mass`、系譜 9-366）が効いたことを、何に対して効いたかまで出す（CLAUDE.md ⑤-3-2）。数秒。

  ① 生成した XML: 荷物 0 なら <inertial> を書かない（既存の run の物理を変えない）。荷物 m なら先端の body（子を持たない body）だけに
     書かれ、質量 = カプセル + m、重心がカプセルの先端側へ寄る。太さを変えるとカプセルの質量が追随する
  ② MuJoCo: 腕を水平に倒した姿勢の重力のトルク（qfrc_bias）の増え方が、m × g × 先端までの水平距離（関節ごと）と一致する
  ③ Choreonoid（環境変数 `PAYLOAD_CNOID=1`、コンテナで `choreonoid --no-window --python` から。Choreonoid は `--cnoid` を自分の引数として読んで落ちる）: 同じ XML を変換器に通し、先端のリンクの質量を読む

    python3 scripts/probe_payload_check.py
    PAYLOAD_CNOID=1 CNOID_CAPSULE_INERTIA_ROT=1 /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/probe_payload_check.py > ログ 2>&1
"""
import copy
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from khrylib.robot.xml_robot import Robot   # noqa: E402

CFG = 'design_opt/cfg/pusher_tripo_v3_reallen.yml'
XMLS = ['assets/mujoco_envs/e2e_a1v_phys_arm171.xml', 'assets/mujoco_envs/e2e_a1v_phys_arm171_vprox.xml',
        'assets/mujoco_envs/e2e_a1v_phys_arm171_vdist.xml']


def build(xml, payload, size_scale=None):
    cfg = copy.deepcopy(yaml.safe_load(open(CFG))['robot'])
    if payload:
        cfg['payload_mass'] = payload
    r = Robot(cfg, xml)
    if size_scale:
        for b in r.bodies:
            for g in b.geoms:
                g.size = g.size * size_scale
    r.sync_node()
    return r, r.export_xml_string().decode()


def inertials(xml_str):
    from lxml import etree
    root = etree.fromstring(xml_str.encode())
    out = {}
    for b in root.find('worldbody').iter('body'):
        el = b.find('inertial')
        if el is not None:
            out[b.get('name')] = (float(el.get('mass')), [float(x) for x in el.get('pos').split()])
    return out


def main_mujoco():
    import mujoco
    ok_all = True
    for xml in XMLS:
        name = os.path.basename(xml)
        _, x0 = build(xml, 0.0)
        r10, x10 = build(xml, 10.0)
        i0, i10 = inertials(x0), inertials(x10)
        tip = r10.bodies[-1]
        cap = tip.geoms[0]; L = float(np.linalg.norm(cap.end - cap.start))
        rho = float(cap.node.attrib.get('density', r10.geom_density)); rr = float(cap.size[0])
        m_cap = rho * (np.pi * rr ** 2 * L + 4 / 3 * np.pi * rr ** 3)
        ok1 = (len(i0) == 0) and (list(i10) == [tip.name]) and abs(i10[tip.name][0] - (m_cap + 10)) < 1e-3
        print(f'{name}: ① 荷物 0 の <inertial> {len(i0)} 個（期待 0）/ 荷物 10 kg: body {list(i10)}（期待 [{tip.name}]）'
              f' 質量 {i10[tip.name][0]:.4f}（期待 カプセル {m_cap:.4f} + 10）重心 z {i10[tip.name][1][2]:.4f}（カプセルの長さ {L:.4f}）'
              f' {"✅" if ok1 else "❌"}')
        _, x10b = build(xml, 10.0, size_scale=1.5)
        mb = inertials(x10b)[tip.name][0]
        print(f'    太さ ×1.5 で質量 {mb:.4f}（カプセルの質量が追随すれば 10 kg との差が増える: {mb - 10:.4f} > {m_cap:.4f}）'
              f' {"✅" if mb - 10 > m_cap * 1.5 else "❌"}')
        # ② 水平に倒した姿勢（ピッチ関節 1 を +90°、残り 0）の重力のトルク
        tq = []
        for xs in (x0, x10):
            m = mujoco.MjModel.from_xml_string(xs); d = mujoco.MjData(m)
            jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ('11_joint', '111_joint', '1111_joint')]
            d.qpos[m.jnt_qposadr[jid[0]]] = np.pi / 2
            mujoco.mj_forward(m, d)
            tq.append(np.array([d.qfrc_bias[m.jnt_dofadr[j]] for j in jid]))
            if xs is x10:
                tipx = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, tip.name)].copy()
                Rm = d.xmat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, tip.name)].reshape(3, 3)
                tip_world = tipx + Rm @ (cap.end - tip.pos)
                jx = [d.xanchor[j][0] for j in jid]
        dtq = tq[1] - tq[0]; expect = np.array([10 * 9.81 * (tip_world[0] - x) for x in jx])
        ok2 = np.allclose(np.abs(dtq), np.abs(expect), rtol=1e-3)
        print(f'    ② 水平の姿勢の重力のトルクの増え [N·m] {np.round(dtq, 3)}  期待 ±{np.round(expect, 3)} {"✅" if ok2 else "❌"}')
        ok_all &= ok1 and ok2
    print('⭐ すべて一致' if ok_all else '⛔ 一致しないものがある')


def main_cnoid():
    """③ 変換器（mujoco_xml_to_body）で .body にし、Choreonoid の BodyLoader で読み込んだ体のリンクの質量を、荷物 0 と 10 kg で並べる。"""
    import tempfile
    from cnoid.Body import BodyLoader
    from khrylib.rl.envs.common import mujoco_env_choreonoid as mc
    ok_all = True
    try:
        for xml in XMLS:
            ms = {}
            for pay in (0.0, 10.0):
                _, xs = build(xml, pay)
                defs = mc.mujoco_xml_to_body(xs)[0]
                yaml_str = dict(defs)[defs[0][0]]              # 腕（最初の worldbody の子）
                with tempfile.NamedTemporaryFile(suffix='.body', mode='w', delete=False) as f:
                    f.write(yaml_str); path = f.name
                b = BodyLoader().load(path); os.unlink(path)
                ms[pay] = {b.link(k).name: (b.link(k).m, np.asarray(b.link(k).c, float)) for k in range(b.numLinks)}
            tipname = [n for n in ms[10.0] if n.startswith('1111')][0]
            d = {n: ms[10.0][n][0] - ms[0.0][n][0] for n in ms[0.0]}
            ok = abs(d[tipname] - 10.0) < 1e-3 and all(abs(v) < 1e-6 for n, v in d.items() if n != tipname)
            ok_all &= ok
            print(f'{os.path.basename(xml)}: Choreonoid のリンクの質量 荷物 0 → 10 kg の差 '
                  + ', '.join(f'{n} {v:+.4f}' for n, v in d.items())
                  + f'  先端 {tipname} の重心 {np.round(ms[0.0][tipname][1], 4).tolist()} → {np.round(ms[10.0][tipname][1], 4).tolist()} {"✅" if ok else "❌"}', flush=True)
        print('⭐ Choreonoid でも先端のリンクだけが +10 kg' if ok_all else '⛔ 一致しないものがある', flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f'⛔ 例外 {e!r}', flush=True)
    os._exit(0)


if __name__ == '__main__':
    if '--cnoid' in sys.argv or os.environ.get('PAYLOAD_CNOID'):
        main_cnoid()
    else:
        main_mujoco()
