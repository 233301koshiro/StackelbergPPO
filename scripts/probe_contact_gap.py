"""台座の球 `0` と 2 本目のリンク `11` の**幾何の隙間**を変えて、AIST が接触を作り始める隙間を測る（2026-10-20、Bug 72）。

⭐ なぜ要るか: 配分 dist（1 本目 0.137 m）は球と `11` の隙間が幾何では 6.8 mm なのに、AIST が深さ 32〜76 mm・
  最大 2×10⁹ N の接触を作っていた（`probe_contact_log.py`）。どの隙間から作り物が出るかが分かれば、
  どの形態がこれを踏むかを幾何で言える。

1 回の起動で 1 設定だけ測る（シミュレーションの再開を挟むと malloc で落ちることがあるため、Bug 72 のプローブの限界）。
  PROBE_GAP_MM   球の表面と `11` のカプセルの表面の隙間（mm）。1 本目の長さ L1 = 0.1 + r11 + gap
  PROBE_FOLD_DEG `11` の関節角（度。0 でまっすぐ、大きいほど球の側へ折り返す）
  PROBE_R11      `11` の半径（既定 0.03、dist seed0 の最適化後の値）
  PROBE_TICKS    止めたまま進める tick 数（既定 40）

    for g in 30 10 6.8 0 -5; do PROBE_GAP_MM=$g CNOID_SELF_COLLISION=1 CNOID_CULLING_DEPTH=1.0 CNOID_TICK_SYNC=1 \\
      USE_CHOREONOID=1 OMP_NUM_THREADS=1 timeout -k 5 60 /choreonoid_ws/install/bin/choreonoid --no-window \\
      --python scripts/probe_contact_gap.py >> out.txt 2>&1; done
"""
import os, sys, re, traceback
sys.path.append(os.getcwd())
os.environ['CNOID_CONTACT_LOG'] = '1'
import numpy as np


def main():
    from khrylib.rl.envs.common.mujoco_env_choreonoid import ChoreonoidSimWorld
    gap = float(os.environ.get('PROBE_GAP_MM', '6.8')) / 1000
    fold = np.deg2rad(float(os.environ.get('PROBE_FOLD_DEG', '0')))
    r11 = float(os.environ.get('PROBE_R11', '0.03'))
    ticks = int(os.environ.get('PROBE_TICKS', '40'))
    L1 = 0.1 + r11 + gap
    xml = open('assets/mujoco_envs/tripo_arm_v2c_pj_dist_phys_arm171.xml').read()
    xml = xml.replace('fromto="0.000000 0.000000 0.000000  0.136800 0.000000 0.000000"',
                      f'fromto="0 0 0  {L1:.6f} 0 0"')
    xml = xml.replace('<body name="11" pos="0.136800 0.000000 0.000000">', f'<body name="11" pos="{L1:.6f} 0 0">')
    xml = xml.replace('0.250000 0.000000 0.000000" size="0.0326"', f'0.250000 0.000000 0.000000" size="{r11}"')
    xml = re.sub(r'<body name="cube" pos="[^"]*"', '<body name="cube" pos="50 0 0.2"', xml)   # 対象は遠くへ
    assert f'pos="{L1:.6f} 0 0"' in xml and 'pos="50 0 0.2"' in xml, '置換が効いていない'

    w = ChoreonoidSimWorld()
    info = w.load_model(xml, 1)
    nu = len(info['actuator_names'])
    q0 = np.array(info['init_qpos'], float)
    q0[1] = fold
    w.set_state_cmd(q0.tolist(), [0.0] * len(info['init_qvel']))
    b = w.sim_bodies['robot'].body()
    nm = {b.link(k).name: k for k in range(b.numLinks)}
    p11 = np.asarray(b.link(nm['11']).p, float)
    hits, dmax, fmax, nrm = 0, 0.0, 0.0, None
    for _ in range(ticks):
        w.step([0.0] * nu, 1)
        cps = b.link(nm['11']).contactPoints
        if cps:
            hits += 1
            i = int(np.argmax([c.depth for c in cps]))
            dmax = max(dmax, cps[i].depth)
            fmax = max(fmax, max(float(np.linalg.norm(np.asarray(c.force, float))) for c in cps))
            nrm = np.round(np.asarray(cps[i].normal, float), 2).tolist()
    # ⭐ 置いた姿勢から動いたか（作り物の力で腕が弾かれるか）
    moved = float(np.linalg.norm(np.asarray(b.link(nm['11']).p, float) - p11))
    o = np.asarray(b.link(nm['0']).p, float)
    a = np.asarray(b.link(nm['11']).p, float)
    print(f'[gap] 隙間 {gap*1000:6.1f} mm  折り {np.rad2deg(fold):5.0f}°  r11 {r11}  L1 {L1:.4f}  '
          f'球の中心↔11 の根元 {np.linalg.norm(a-o)*1000:.1f} mm  接触 {hits}/{ticks} tick  '
          f'めり込み最大 {dmax*1000:.1f} mm  力の最大 {fmax:.3g} N  法線 {nrm}  11 の根元の移動 {moved*1000:.2f} mm',
          flush=True)


try:
    main()
except Exception:
    traceback.print_exc()
    sys.stdout.flush()
    os._exit(1)
sys.stdout.flush()
os._exit(0)
