#!/usr/bin/env python3
"""probe_shot_speed.py: 「届く」とは別に「**そこで +x に速度を出せるか**」を測る（9-46 の残る問い）。

第1層は「届くか」しか判定しない。だが打つ・弾く系のタスク（Shot）が要求するのは
**対象に触れた姿勢から、狙う方向へ速度を出せること**である。
伸びきった姿勢では先端は腕に垂直な方向にしか動けないので、
+x へ出せる速度は姿勢に依存する。**それを姿勢ごとに測る。**

測り方（学習不要・シミュレーション不要）:
  1. 腕の関節角を格子で網羅し、MuJoCo の順運動学で先端位置を出す
  2. 先端が「対象の手前の面」に触れる姿勢だけを残す
  3. その姿勢でのヤコビアンから、単位関節速度あたり先端が **+x 方向へ出せる速度**を求める
     （|q̇| ≤ 1 のときの max ẋ = ヤコビアンの x 行のノルム）

自前の順運動学は Bug 23・Bug 24 で二度誤ったので、**MuJoCo 自身に計算させる**。

  docker exec <container> python3 scripts/probe_shot_speed.py
"""
import os
import sys

import numpy as np
import mujoco

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.diagnose_morphology import parse_arm_xml, ASSET_DIR   # noqa: E402

XML = os.environ.get('PROBE_XML', 'e2e_hockey_court')
GRID = int(os.environ.get('PROBE_GRID', '41'))
TOL = float(os.environ.get('PROBE_TOL', '0.02'))

path = os.path.join(ASSET_DIR, f'{XML}.xml')
geo = parse_arm_xml(path)
model = mujoco.MjModel.from_xml_path(path)
data = mujoco.MjData(model)
if os.environ.get('PROBE_ARMATURE'):        # アーマチュアの寄与を切り分けるため
    model.dof_armature[:] = float(os.environ['PROBE_ARMATURE'])

# 腕の関節 = cube_slide 系を除いたヒンジ
arm_j = [j for j in range(model.njnt)
         if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
         and not mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j).startswith('cube')]
qadr = [model.jnt_qposadr[j] for j in arm_j]
dadr = [model.jnt_dofadr[j] for j in arm_j]
rng = [model.jnt_range[j] if model.jnt_limited[j] else np.array([-np.pi, np.pi]) for j in arm_j]

tip_body = model.nbody - 1
for b in range(model.nbody - 1, 0, -1):
    if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) != 'cube':
        tip_body = b
        break
tip_off = np.asarray(geo['dirs'][-1], float) * float(geo['lengths'][-1])   # 最終リンクの先端

print(f'[shot_speed] {XML}')
print(f'  腕の関節 {len(arm_j)} 個: ' + ', '.join(
    f'{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)}'
    f'[{r[0]:.2f},{r[1]:.2f}]' for j, r in zip(arm_j, rng)))
print(f'  先端ボディ: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, tip_body)} + 局所 {tip_off}')

cube = geo['cube']
base = np.asarray(geo['base_pos'], float)
half = float(cube['half'])
cx, cz = float(cube['pos'][0]), float(cube['pos'][2])

# --- 姿勢の網羅 ---
# ⚠️ 根元ヨーを格子に含めると刻みが粗すぎて目標に当たらない（±π を 13 分割 = 30° 刻みに対し、
#    狙う方位角は 0〜33°）。ヨーは**方位角そのもの**なので解析的に与え、残りだけ格子にする。
jacp = np.zeros((3, model.nv))
fullM = np.zeros((model.nv, model.nv))
# アクチュエータの最大トルク（gear × ctrl の上限）。関節ごとに対応づける。
tau_max = np.zeros(model.nv)
for a in range(model.nu):
    if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
        tau_max[model.jnt_dofadr[model.actuator_trnid[a, 0]]] = abs(float(model.actuator_gear[a, 0]))


def scan(yaw):
    """根元ヨーを yaw に固定し、残りの関節を格子で網羅する。"""
    axes = [np.linspace(r[0], r[1], GRID) for r in rng[1:]]
    tips, jxs, jall, axm = [], [], [], []
    for q in np.array(np.meshgrid(*axes, indexing='ij')).reshape(len(arm_j) - 1, -1).T:
        data.qpos[:] = 0.0
        data.qpos[qadr[0]] = yaw
        for a, v in zip(qadr[1:], q):
            data.qpos[a] = v
        mujoco.mj_forward(model, data)      # qM も要るので forward（crb だけでは埋まらない）
        tip = data.xpos[tip_body] + data.xmat[tip_body].reshape(3, 3) @ tip_off
        mujoco.mj_jac(model, data, jacp, None, tip, tip_body)
        tips.append(tip.copy())
        jxs.append(np.linalg.norm(jacp[0, dadr]))      # |q̇|≤1 での max ẋ
        jall.append(np.linalg.norm(jacp[:, dadr]))     # 向きを問わない速度の目安
        # 動力学版: トルク上限のもとで先端を +x へ加速できる最大値
        # ẍ = J M⁻¹ τ なので、|τ_i| ≤ tau_max_i での最大は Σ_i |(J M⁻¹)_{x,i}| · tau_max_i
        mujoco.mj_fullM(model, fullM, data.qM)
        sub = fullM[np.ix_(dadr, dadr)]
        w = np.linalg.solve(sub, jacp[0, dadr])
        axm.append(float(np.abs(w) @ tau_max[dadr]))
    return np.array(tips), np.array(jxs), np.array(jall), np.array(axm)


t0, _, _, _ = scan(0.0)
r_max = float(np.max(np.linalg.norm(t0[:, :2] - base[:2], axis=1)))
print(f'  ヨー1値あたり {len(t0)} 姿勢（格子 {GRID}^{len(arm_j)-1}）'
      f' / この格子での最大水平到達: {r_max:.3f} m\n')

print(f'{"パック y":>9}{"接触点まで":>11}{"伸展率":>9}{"届く姿勢":>10}'
      f'{"max ẋ (+x)":>12}{"ẋ/|v|":>8}{"max ẍ (+x)":>12}')
rows = []
for y in np.linspace(0.0, 0.45, 10):
    p = np.array([cx, y])
    d = float(np.linalg.norm(p - base[:2]))
    contact = base[:2] + (p - base[:2]) / d * (d - half)
    tgt = np.array([contact[0], contact[1], cz])
    tips, jxs, jall, axm = scan(float(np.arctan2(p[1] - base[1], p[0] - base[0])))
    ok = np.linalg.norm(tips - tgt, axis=1) < TOL
    n = int(ok.sum())
    if n == 0:
        print(f'{y:>9.2f}{d-half:>11.3f}{(d-half)/r_max*100:>8.0f}%{0:>10}   （届く姿勢が無い）')
        continue
    vx, va, ax = float(jxs[ok].max()), float(jall[ok].max()), float(axm[ok].max())
    rows.append((y, (d - half) / r_max, vx, ax))
    print(f'{y:>9.2f}{d-half:>11.3f}{(d-half)/r_max*100:>8.0f}%{n:>10}'
          f'{vx:>12.3f}{vx/va:>8.2f}{ax:>12.2f}')

if len(rows) >= 2:
    print()
    print(f'伸展率 {rows[0][1]*100:.0f}% → {rows[-1][1]*100:.0f}% のとき')
    print(f'  運動学 ẋ : {rows[0][2]:.3f} → {rows[-1][2]:.3f}（{rows[-1][2]/rows[0][2]*100:.0f} %）')
    print(f'  動力学 ẍ : {rows[0][3]:.2f} → {rows[-1][3]:.2f}（{rows[-1][3]/rows[0][3]*100:.0f} %）'
          f'  ← トルク上限と慣性を含む。**こちらが打力に近い**')
    print('判定: 端で ẋ が大きく落ちるなら「届くが打てない」。'
          '落ちないなら、失敗の原因は形状ではなく制御側にある。')
