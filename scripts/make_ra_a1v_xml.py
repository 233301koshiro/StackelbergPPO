"""お辞儀型の A1v を研究室の部品で組んだ腕（`build_arm3.py` の出力の JSON）から、学習に使う MuJoCo XML を作る（軸6 L-3b、系譜 9-326）。

⭐ 雛形は今の A1v（`e2e_a1v_phys_arm171.xml`）。体の並び（0 台 / 1 旋回 / 11 肩 / 111 肘 / 1111 手先）と名前は変えず、値だけを置き換える。
⭐ 関節の位置・質量・重心・慣性は部品の実物から: 腕を `joint_range_a1v.py` と同じくまっすぐ上へ伸ばし、
   関節 k の体 = サーボ k のホーンより先、サーボ k+1 の本体まで（サーボの本体は根元側の体に乗る）。部品ごとの mass-param（g・mm）を合成して `<inertial>` に書く
   （変換器は `<inertial>` を geom の密度より優先する、9-83）。
⭐ シミュレータの座標: z = 上、+y 回りの正の回転で手先が +x（箱の側）へ倒れる。可動域はスケール表 A5（＋ = 前）。
   ピッチの関節（2〜4）の原点は、軸の線上で旋回の軸を含む面（y = 0）に取る（軸が平行なので運動学は変わらない）。
⭐ 当たり判定は雛形と同じカプセル（体ごとに 1 つ。変換器は最初の geom しか読まない）。半径は部品の広がりから（RADIUS_Q の分位）。
⭐ その他の値はスケール表の決定: 箱は × k（中心 x 0.20・半辺 0.03・密度そのまま・減衰は τ を保つ）、gear 1.4、armature 0.008、減衰 0.235。
   角速度の上限 5.97 rad/s は XML ではなく起動時の環境変数 `CNOID_JOINT_MAX_VEL`（P5）。

    IN=data/axis6/family_range/Hinge_Short_Plug+Frame_Long_Socket+connect_point_right-side.json \
    OUT=assets/mujoco_envs/e2e_a1v_ra15.xml python3 scripts/make_ra_a1v_xml.py
"""
import json
import os

import numpy as np
import yaml

from joint_range_a1v import PROJ, Shape, rot_about

IN = os.environ.get('IN', '/userdir/StackelbergPPO/data/axis6/family_range/'
                    'Hinge_Short_Plug+Frame_Long_Socket+connect_point_right-side.json')
OUT = os.environ.get('OUT', 'assets/mujoco_envs/e2e_a1v_ra15.xml')
K = float(os.environ.get('K', '0.2'))              # 縮尺（箱・目標に掛ける）
RADIUS_Q = float(os.environ.get('RADIUS_Q', '99'))   # カプセル = 部品の前後（x）の (100−Q)〜Q % 点の範囲
GEAR = float(os.environ.get('GEAR', '1.4'))
ARMATURE = float(os.environ.get('ARMATURE', '0.008'))
DAMPING = float(os.environ.get('DAMPING', '0.235'))
RANGES = [(-180, 180), (-115, 131), (-11, 147), (-148, 136)]   # スケール表 A5（＋ = 前）


def main():
    cfg = yaml.safe_load(open(PROJ + '/irsl_assembler_config.yaml'))
    P = {p['type']: p for p in cfg['PartsSettings']}
    d = json.load(open(IN))
    parts = [(p['name'], p['type'], np.array(p['T'])) for p in d['parts']]
    S = [Shape(t, P) for _, t, _ in parts]
    sv = [i for i, (_, t, _) in enumerate(parts) if t == 'XL-430']
    T = [M.copy() for _, _, M in parts]
    up = np.array([0, 0, 1.0])

    def horn(i):
        return T[sv[i]][:3, 3].copy(), T[sv[i]][:3, 2].copy()

    def world_pts(i):
        return S[i].pts @ T[i][:3, :3].T + T[i][:3, 3]

    def tip():
        o, a = horn(len(sv) - 1)
        w = np.vstack([world_pts(i) for i in range(sv[-1] + 1, len(T))])
        v = w - o
        return w[np.argmax(np.linalg.norm(v - np.outer(v @ a, a), axis=1))]

    def signed(u, w, a):
        u = u - (u @ a) * a
        w = w - (w @ a) * a
        return np.degrees(np.arctan2(np.cross(u, w) @ a, u @ w))

    for k in (1, 2, 3):                      # まっすぐ上へ（joint_range_a1v.py と同じ）
        o, a = horn(k)
        b = horn(k + 1)[0] if k + 1 < len(sv) else tip()
        M = rot_about(o, a, signed(b - o, up, a))
        for i in range(sv[k] + 1, len(T)):
            T[i] = M @ T[i]

    a_ref = horn(1)[1]
    a1 = horn(0)[1]
    assert abs(abs(a1 @ up) - 1) < 1e-3, f'関節 1 の軸が鉛直でない {a1}'
    assert all(abs(abs(horn(k)[1] @ a_ref) - 1) < 1e-3 for k in (1, 2, 3)), '関節 2〜4 の軸が平行でない'
    f = np.cross(a_ref, up)
    R = np.vstack([-f, -a_ref, up])          # 行 = シミュレータの x, y, z を部品の座標で
    o1 = horn(0)[0]

    def sim(p):
        return (np.asarray(p) - o1) @ R.T

    J = [sim(horn(k)[0]) for k in range(4)]
    tip_s = sim(tip())
    print('関節の位置（シミュレータの座標、mm）:', [np.round(j, 1).tolist() for j in J], '手先', np.round(tip_s, 1).tolist())
    print('ピッチの関節の y（軸方向のずれ、mm）:', [round(float(J[k][1]), 1) for k in (1, 2, 3)], '→ y = 0 に取る')
    Jp = [J[0].copy()] + [np.array([j[0], 0.0, j[2]]) for j in J[1:]]
    tip_p = np.array([tip_s[0], 0.0, tip_s[2]])

    # 体ごとの部品: 0 = 台（サーボ 1 まで）、k = サーボ k のホーンより先、サーボ k+1 の本体まで
    groups = [range(0, sv[0] + 1)] + [range(sv[k] + 1, (sv[k + 1] + 1) if k + 1 < 4 else len(T)) for k in range(4)]
    bodies = []
    for g, rng in enumerate(groups):
        origin = np.zeros(3) if g == 0 else Jp[g - 1]
        m_tot, c_acc, items = 0.0, np.zeros(3), []
        for i in rng:
            mp = P[parts[i][1]]['mass-param']
            m = mp['mass'] / 1000.0
            Rw = R @ T[i][:3, :3]
            c = (sim(T[i][:3, :3] @ np.array(mp['center-of-mass']) + T[i][:3, 3]) - origin) / 1000.0
            I = Rw @ np.array(mp['inertia-tensor']).reshape(3, 3) @ Rw.T * 1e-9
            items.append((m, c, I))
            m_tot += m
            c_acc += m * c
        def compose(its):
            mt = sum(m for m, _, _ in its)
            cm = sum(m * c for m, c, _ in its) / mt
            Ic = np.zeros((3, 3))
            for m, c, I in its:
                dv = c - cm
                Ic += I + m * ((dv @ dv) * np.eye(3) - np.outer(dv, dv))
            return mt, cm, Ic
        _m, com, Ic = compose(items)
        is_sv = [parts[i][1] == 'XL-430' for i in rng]
        split = None
        if g > 0 and any(is_sv) and not all(is_sv):       # 構造（サーボ以外）と先のサーボに分ける（β の質量の模型、系譜 9-328）
            split = (compose([it for it, sv in zip(items, is_sv) if not sv]), compose([it for it, sv in zip(items, is_sv) if sv]))
        pts = np.vstack([sim(world_pts(i)) for i in rng]) - origin
        bodies.append(dict(names=[parts[i][0] for i in rng], mass=m_tot, com=com, I=Ic, pts=pts / 1000.0, split=split))

    seg_end = [Jp[1], Jp[2], Jp[3], tip_p]          # 体 1〜4 のカプセルの終点（体の原点からの相対は下で）
    lens = []
    for g in range(1, 5):
        b = bodies[g]
        L = (seg_end[g - 1] - Jp[g - 1]) / 1000.0
        lens.append(float(np.linalg.norm(L)))
        u = L / np.linalg.norm(L)
        x = b['pts'][:, 0] - np.outer(b['pts'] @ u, u)[:, 0]
        lo, hi = np.percentile(x, [100 - RADIUS_Q, RADIUS_Q])
        b['radius'] = float(hi - lo) / 2.0          # 前後（x）の幅の半分
        b['xoff'] = float(hi + lo) / 2.0            # 部品は関節の線から前後にずれる（サーボの本体はホーンより後ろ）ので、カプセルを部品の中央へ
        ylo, yhi = np.percentile(b['pts'][:, 1], [100 - RADIUS_Q, RADIUS_Q])
        b['radius_y'] = float(yhi - ylo) / 2.0
        b['ymid'] = float(yhi + ylo) / 2.0
        b['L'] = L
    b0 = bodies[0]
    z_floor = -float(b0['pts'][:, 2].min())          # 台（サーボ 1）の下面を床に置くと関節 1 のホーンはこの高さ
    c0 = b0['pts'].mean(0)
    b0['radius'] = min(float(np.percentile(np.linalg.norm(b0['pts'] - c0, axis=1), 90)), z_floor + float(c0[2]))   # 台の球は床に埋めない
    for g, b in enumerate(bodies):
        print(f'体 {g}: 質量 {b["mass"]*1000:.1f} g、重心 {np.round(b["com"]*1000, 1).tolist()} mm、'
              f'慣性の対角 {np.round(np.diag(b["I"])*1e9, 0).tolist()} g·mm²、半径 {b["radius"]*1000:.1f} mm'
              + (f'（前後の中央 x {b["xoff"]*1000:+.1f} mm）・左右の幅の半分 {b["radius_y"]*1000:.1f} mm（中央 y {b["ymid"]*1000:+.1f}）' if g else '')
              + f'、部品 {b["names"]}')
    print(f'関節の間（まっすぐの姿勢、mm）: {[round(x * 1000, 1) for x in lens]}、関節 1 のホーンの高さ {z_floor*1000:.1f} mm')
    for g, (L, b) in enumerate(zip(lens[1:3], bodies[2:4])):
        s = (1.338 * L * 1000 - 60.3) + 57.0
        print(f'  P1 の式との照合 リンク {g+1}: 部品 {b["mass"]*1000:.1f} g / 式（構造 1.338·L − 60.3 ＋ サーボ 57.0）{s:.1f} g')

    def v3(x):
        return ' '.join(f'{float(t):.6f}' for t in x)

    def inertial(b):
        I = b['I']
        return (f'<inertial pos="{v3(b["com"])}" mass="{b["mass"]:.6f}" '
                f'fullinertia="{I[0,0]:.4e} {I[1,1]:.4e} {I[2,2]:.4e} {I[0,1]:.4e} {I[0,2]:.4e} {I[1,2]:.4e}"/>')

    rel = [Jp[0] / 1000.0] + [(Jp[g] - Jp[g - 1]) / 1000.0 for g in range(1, 4)]
    names = ['1', '11', '111', '1111']
    axes = ['0 0 1', '0 1 0', '0 1 0', '0 1 0']
    ind = '    '
    xml_bodies = ''
    close = ''
    for g in range(4):
        b = bodies[g + 1]
        pad = ind * (g + 3)
        pos = rel[g] if g else np.zeros(3)
        xml_bodies += (f'{pad}<body name="{names[g]}" pos="{v3(pos)}">\n'
                       f'{pad}  {inertial(b)}\n'
                       f'{pad}  <joint name="{names[g]}_joint" type="hinge" pos="0 0 0" axis="{axes[g]}" '
                       f'range="{RANGES[g][0]:.1f} {RANGES[g][1]:.1f}"/>\n'
                       f'{pad}  <geom type="capsule" fromto="{v3([b["xoff"], 0, 0])}  {v3(b["L"] + [b["xoff"], 0, 0])}" size="{b["radius"]:.4f}"/>\n')
        close = f'{pad}</body>\n' + close
    cube_h = 0.15 * K
    # 箱の減衰はスケール表 T5「τ を保つ」。⚠️ 変換器は滑りの関節の前に 1 g のダミーのリンクを 2 つ置き、減衰を b ÷ 総質量で掛ける
    #   （mujoco_env_choreonoid.py の _joint_decay）。全寸では 2 g / 2.7 kg で無視できるが、1/5 では 2 g / 21.6 g ＝ 9 % で、
    #   b = 2.0 × k³ のままだと τ が 1.35 → 1.48 s に延びた（系譜 9-326 のプローブで 1.52 s）。ダミーを含めた総質量で τ をそろえる
    DUMMY = 0.002
    tau = (100 * 0.3 ** 3 + DUMMY) / 2.0
    cube_damp = (100 * (2 * cube_h) ** 3 + DUMMY) / tau
    xml = f'''<mujoco model="a1v_ra_k{K:g}">
  <!-- 生成物: scripts/make_ra_a1v_xml.py（ブランチ feat/layer4-robotassembler、系譜 9-326）。手で直さない。
       入力 {os.path.basename(IN)}。関節の間 {[round(x*1000, 1) for x in lens]} mm。質量・慣性は部品の mass-param の合成 -->
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1=".2 .3 .4" rgb2="0 0 0" width="800" height="800" mark="random" markrgb="1 1 1"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .1 .2" rgb2=".3 .2 .3" width="300" height="300" mark="edge" markrgb=".3 .2 .3"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
    <material name="self" rgba=".7 .5 .3 1"/>
  </asset>
  <visual>
    <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular="0 0 0"/>
    <map znear=".001"/>
    <quality shadowsize="16384"/>
  </visual>
  <compiler angle="degree"/>
  <option integrator="RK4" timestep="0.01"/>
  <default>
    <joint armature="{ARMATURE:g}" damping="{DAMPING:g}" limited="true"/>
    <geom conaffinity="0" condim="3" density="189.9" friction="1.0 0.5 0.5" margin="{0.01*K:g}" rgba=".7 .5 .3 1"/>
  </default>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="0.3 0 -1.3" directional="true" exponent="1" pos="-30 0 130" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="1 1 1 1" size="200 200 .125" type="plane" material="grid"/>
    <body name="0" pos="0.000000 0.000000 {z_floor:.6f}">
      <camera name="track" mode="trackcom" pos="0 {-3*K:g} {0.3*K:g}" xyaxes="1 0 0 0 0 1"/>
      <geom pos="{v3(c0)}" size="{b0['radius']:.4f}" type="sphere"/>
{xml_bodies}{close}    </body>
    <body name="cube" pos="{1.0*K:.6f} 0.000000 {0.2*K:.6f}">
      <joint name="cube_slide" armature="0" type="slide" axis="1 0 0" damping="{cube_damp:.6g}" range="-2.0 200.0"/>
      <joint name="cube_slide2" armature="0" type="slide" axis="0 1 0" damping="{cube_damp:.6g}" range="-100.0 100.0"/>
      <geom name="cube_geom" type="box" size="{cube_h:g} {cube_h:g} {cube_h:g}" rgba="0.8 0.2 0.2 1.0" friction="0.5 0.1 0.1" density="100" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
  <actuator>
''' + ''.join(f'    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="{n}_joint" gear="{GEAR:g}" name="{n}_joint"/>\n' for n in names) + '''  </actuator>
</mujoco>
'''
    open(OUT, 'w').write(xml)
    print('書き出した:', OUT, flush=True)
    # β の質量の模型（xml_robot の part_mass_file、系譜 9-328）: 体ごとに 構造 と 先のサーボ を分けて書く。座標は体の原点（関節）から、骨は +z
    pm = {'slope_kg_per_m': 1.338, 'note': '構造の質量 = 組んだ質量 + slope × (L − L0)。slope はスケール表 P1（1.338 g/mm）', 'bodies': {}}
    for g in range(1, 5):
        b = bodies[g]
        if b['split'] is None:
            continue
        (ms, cs, Is), (mv, cv, Iv) = b['split']
        L0 = float(np.linalg.norm(b['L']))
        pm['bodies'][names[g - 1]] = dict(L0=L0, struct=dict(mass=ms, com=cs.tolist(), inertia=Is.tolist()),
                                          servo=dict(mass=mv, com_from_distal=(cv - np.array([0, 0, L0])).tolist(), inertia=Iv.tolist()))
        print(f'  模型 体 {names[g-1]}: L0 {L0*1000:.1f} mm・構造 {ms*1000:.1f} g（重心 z {cs[2]*1000:.1f} mm）・サーボ {mv*1000:.1f} g（先の関節から {np.round((cv - [0, 0, L0])*1000, 1).tolist()} mm）', flush=True)
    pm_out = OUT.replace('.xml', '_mass.json')
    json.dump(pm, open(pm_out, 'w'), indent=1, ensure_ascii=False)
    print('書き出した:', pm_out, flush=True)


if __name__ == '__main__':
    main()
