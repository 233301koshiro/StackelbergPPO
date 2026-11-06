"""研究室の部品で組んだ四脚（`build_quad_ra.py` の JSON）から、プローブに通す MuJoCo XML を作る（系譜 9-342 の追記、2026-11-06）。

⭐ 雛形は近似の XML（main の `quad_ra_approx.py` が作った `quad_<QUAD>_ra.xml`）。体の並び・名前（0 胴 / k 取り付け / 1k 腿 / 11k 脛）・
   関節の名前・軸（+y）・可動域（±30°）・モータの並びとギア比 1.4・減衰・回転子の慣性は変えず、部品から決まる値だけを置き換える。
   ⇒ 近似との違いは「関節の位置・長さ・質量の分布・当たり判定の太さ」だけになる。
⭐ 関節 0° の姿勢: 組んだ腿・脛はまっすぐなので、腿を股の軸のまわりに、脛を膝の軸のまわりに回して、近似の腿（股 → 膝）と
   脛（膝 → 足先）の向きにそろえる（ホーンへ部品を付ける角度を変えたのと同じで、部品どうしの付け方は変わらない）。
⭐ 質量・重心・慣性: 部品ごとの mass-param（g・mm）を体ごとに合成して `<inertial>` に書く（`make_ra_a1v_xml.py` と同じ）。
   体 0 ＝ 胴のプレート、体 k ＝ 脚 k の股のフレームとサーボ、体 1k ＝ 腿（Hinge・LEGO・Frame・膝のサーボ）、体 11k ＝ 脛（Hinge・LEGO）。
⭐ 当たり判定（変換器は体ごとに最初の geom だけ読む）:
   胴 ＝ プレートの箱。取り付け ＝ 近似と同じ胴の中心 → 股のカプセル（半径 10 mm）。
   腿 ＝ 股 → 膝のカプセル、脛 ＝ 膝 → 先端のカプセル。半径は部品の、脚の面の中で線に垂直な向きの幅の半分（RADIUS_Q の分位）、
   中心線は部品の幅の中央へずらす。脛のカプセルの終点は「先端 − 半径」に置き、カプセルの外形が部品の先端に届くようにする。
⚠️ 当たり判定はカプセルの近似で、部品の形そのものではない。胴に制御基板・電池は載せていない（部品の質量だけ）。

    QUAD=tripo IN=data/axis6/quad/quad_tripo_parts.json OUT=assets/mujoco_envs/quad_tripo_rab.xml python3 scripts/make_quad_ra_xml.py
"""
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_range_a1v import PROJ, Shape, rot_about   # noqa: E402

QUAD = os.environ.get('QUAD', 'tripo')
IN = os.environ.get('IN', f'/userdir/StackelbergPPO/data/axis6/quad/quad_{QUAD}_parts.json')
TPL = os.environ.get('TPL', f'/userdir/StackelbergPPO/assets/mujoco_envs/quad_{QUAD}_ra.xml')
OUT = os.environ.get('OUT', f'/userdir/StackelbergPPO/assets/mujoco_envs/quad_{QUAD}_rab.xml')
RADIUS_Q = float(os.environ.get('RADIUS_Q', '99'))
CLEAR = float(os.environ.get('CLEAR', '0.005'))
R_MOUNT = 0.010


def signed(u, w, a):
    u = u - (u @ a) * a
    w = w - (w @ a) * a
    return float(np.degrees(np.arctan2(np.cross(u, w) @ a, u @ w)))


def lever(d, ah, ak):
    """股の軸（ah）と膝の軸（ak）の 2 本の直線の間の、股の軸に垂直なてこの向き（build_quad_ra.py と同じ）。軸が平行なら d の ah に垂直な成分"""
    c = np.cross(ah, ak)
    if np.linalg.norm(c) < 0.01:          # 平行（約 0.6° 以内。数値の誤差で外積は 1e-6 を超える）
        return d - (d @ ah) * ah
    n = c / np.linalg.norm(c)
    return n if d @ n > 0 else -n


def compose(its):
    mt = sum(m for m, _, _ in its)
    cm = sum(m * c for m, c, _ in its) / mt
    Ic = np.zeros((3, 3))
    for m, c, I in its:
        dv = c - cm
        Ic += I + m * ((dv @ dv) * np.eye(3) - np.outer(dv, dv))
    return mt, cm, Ic


def main():
    cfg = yaml.safe_load(open(PROJ + '/irsl_assembler_config.yaml'))
    P = {p['type']: p for p in cfg['PartsSettings']}
    d = json.load(open(IN))
    parts = d['parts']
    T = [np.array(p['T']) for p in parts]
    center = np.array(d['center_mm'])
    hz = np.array([0, 0, 1.0])
    tpl = ET.parse(TPL).getroot()
    tb = {b.get('name'): b for b in tpl.iter('body')}
    # 雛形の名前: quad_tripo_ra は 取り付け k / 腿 1k / 脛 11k、quad_4x2_ra は取り付けが無く 腿 k / 脛 1k。出力は前者にそろえる
    if '111' in tb:
        TN = dict(shin=lambda k: f'11{k}', hipj=lambda k: f'1{k}_joint', kneej=lambda k: f'11{k}_joint')
    else:
        TN = dict(shin=lambda k: f'1{k}', hipj=lambda k: f'{k}_joint', kneej=lambda k: f'1{k}_joint')

    def vec(s):
        return np.array([float(x) for x in s.split()])

    def world_pts(i):
        return Shape(parts[i]['type'], P).pts @ T[i][:3, :3].T + T[i][:3, 3]

    legs = {}
    for j in d['joints']:
        k = j['leg']
        Mh, Mk = np.array(j['hip']), np.array(j['knee'])
        thigh = [i for i, p in enumerate(parts) if p['leg'] == k and p['seg'] == 'thigh']
        shin = [i for i, p in enumerate(parts) if p['leg'] == k and p['seg'] == 'shin']
        oh, ah = Mh[:3, 3], Mh[:3, :3] @ hz
        ok_ = Mk[:3, 3]
        # 近似の腿の向き（股 → 膝 ＝ 脛の体の pos）と脛の向き（脛のカプセルの終点）
        v_thigh = vec(tb[TN['shin'](k)].get('pos')) * 1000
        v_shin = vec(tb[TN['shin'](k)].find('geom').get('fromto'))[3:] * 1000
        th = signed(lever(ok_ - oh, ah, Mk[:3, :3] @ hz), v_thigh, ah)   # 股の軸と膝の軸の間のてこの向きを近似の腿へ
        M = rot_about(oh, ah, th)
        for i in thigh + shin:
            T[i] = M @ T[i]
        Mk = M @ Mk
        ok_, ak = Mk[:3, 3], Mk[:3, :3] @ hz
        w = np.vstack([world_pts(i) for i in shin])
        v = w - ok_
        rad = np.linalg.norm(v - np.outer(v @ ak, ak), axis=1)
        tip = w[rad > rad.max() - 5.0].mean(0)        # 先端の面の中央（最も遠い 1 点はブロックの角で、左右の角のどちらかで ±9° 振れる）
        sh = signed(tip - ok_, v_shin, ak)
        M2 = rot_about(ok_, ak, sh)
        for i in shin:
            T[i] = M2 @ T[i]
        legs[k] = dict(oh=oh, ah=ah, ok=ok_, ak=ak, thigh=thigh, shin=shin, th=th, sh=sh)
        print(f'脚 {k}: 腿を股の軸のまわりに {th:+.1f}°、脛を膝の軸のまわりに {sh:+.1f}° 回して近似の向きにそろえた', flush=True)

    def sim(p):
        return (np.asarray(p) - center) / 1000.0

    def body_inertial(idx, origin):
        its = []
        for i in idx:
            mp = P[parts[i]['type']]['mass-param']
            Rw = T[i][:3, :3]
            c = sim(Rw @ np.array(mp['center-of-mass']) + T[i][:3, 3]) - origin
            I = Rw @ np.array(mp['inertia-tensor']).reshape(3, 3) @ Rw.T * 1e-9
            its.append((mp['mass'] / 1000.0, c, I))
        return compose(its)

    def cap_fit(idx, o, a, end):
        """o（関節）から end へのカプセル。部品の点を脚の面の中の垂直な向き（n）と軸の向き（a）で測り、中心線を幅の中央へ。"""
        pts_ = sim(np.vstack([world_pts(i) for i in idx])) - o
        L = end - o
        L = L - (L @ a) * a
        u = L / np.linalg.norm(L)
        n = np.cross(a, u)
        t = pts_ @ n
        lo, hi = np.percentile(t, [100 - RADIUS_Q, RADIUS_Q])
        wlo, whi = np.percentile(pts_ @ a, [100 - RADIUS_Q, RADIUS_Q])
        r = float(hi - lo) / 2.0
        off = (hi + lo) / 2.0 * n + (whi + wlo) / 2.0 * a
        return r, off, u, float((pts_ @ u).max())

    torso = [i for i, p in enumerate(parts) if p['seg'] == 'torso']
    mt0, c0, I0 = body_inertial(torso, np.zeros(3))
    tp = np.vstack([world_pts(i) for i in torso]) / 1000.0 - center / 1000.0
    half = (np.percentile(tp, 100, axis=0) - np.percentile(tp, 0, axis=0)) / 2.0
    boxc = (np.percentile(tp, 100, axis=0) + np.percentile(tp, 0, axis=0)) / 2.0
    geoms_low = [float(boxc[2] - half[2])]
    xml_legs = []
    report = [f'胴（プレート）: {mt0*1000:.1f} g・箱の半分 {np.round(half*1000, 1).tolist()} mm']
    for k, L in sorted(legs.items()):
        oh, ok_ = sim(L['oh']), sim(L['ok'])
        mount = [i for i, p in enumerate(parts) if p['leg'] == k and p['seg'] == 'mount']
        mm, cm_, Im = body_inertial(mount, np.zeros(3))
        mtg, ctg, Itg = body_inertial(L['thigh'], oh)
        msh, csh, Ish = body_inertial(L['shin'], ok_)
        rt, offt, ut, _ = cap_fit(L['thigh'], oh, L['ah'], ok_)
        wsh = sim(np.vstack([world_pts(i) for i in L['shin']]))
        rad = np.linalg.norm((wsh - ok_) - np.outer((wsh - ok_) @ L['ak'], L['ak']), axis=1)
        rs, offs, us, smax = cap_fit(L['shin'], ok_, L['ak'], wsh[rad > rad.max() - 0.005].mean(0))   # 先端の面の中央へ向ける
        t0, t1 = offt, (ok_ - oh) + offt
        s0, s1 = offs, offs + us * (smax - rs)
        geoms_low += [float(oh[2] + min(t0[2], t1[2]) - rt), float(ok_[2] + min(s0[2], s1[2]) - rs)]
        xml_legs.append(dict(k=k, oh=oh, ok=ok_, mount=(mm, cm_, Im), thigh=(mtg, ctg, Itg), shin=(msh, csh, Ish),
                             rt=rt, t0=t0, t1=t1, rs=rs, s0=s0, s1=s1))
        report.append(f'脚 {k}: 股 {np.round(oh*1000, 1).tolist()} mm・膝 {np.round(ok_*1000, 1).tolist()} mm・'
                      f'取り付け {mm*1000:.1f} g・腿 {mtg*1000:.1f} g（半径 {rt*1000:.1f} mm）・脛 {msh*1000:.1f} g（半径 {rs*1000:.1f} mm、'
                      f'先端まで {smax*1000:.1f} mm）')
    z0 = CLEAR - min(geoms_low)
    mtot = mt0 + sum(L['mount'][0] + L['thigh'][0] + L['shin'][0] for L in xml_legs)
    for line in report:
        print(line, flush=True)
    print(f'質量の合計 {mtot*1000:.1f} g・胴の中心の高さ {z0*1000:.1f} mm（当たり判定の最も低い点を床から {CLEAR*1000:.0f} mm）', flush=True)

    def v3(x):
        return ' '.join(f'{float(t):.6g}' for t in x)

    def inertial(m, c, I):
        return (f'<inertial pos="{v3(c)}" mass="{m:.6g}" '
                f'fullinertia="{I[0,0]:.6g} {I[1,1]:.6g} {I[2,2]:.6g} {I[0,1]:.6g} {I[0,2]:.6g} {I[1,2]:.6g}" />')

    rng = {j.get('name'): j.get('range') for j in tpl.iter('joint') if j.get('range')}
    ax = {j.get('name'): j.get('axis') for j in tpl.iter('joint') if j.get('axis')}
    body = ''
    for L in xml_legs:
        k = L['k']
        body += f'''      <body name="{k}" pos="0 0 0">
        {inertial(*L['mount'])}
        <geom fromto="0 0 0 {v3(L['oh'])}" size="{R_MOUNT:.4f}" type="capsule" />
        <body name="1{k}" pos="{v3(L['oh'])}">
          {inertial(*L['thigh'])}
          <joint axis="{ax[TN['hipj'](k)]}" name="1{k}_joint" pos="0 0 0" range="{rng[TN['hipj'](k)]}" type="hinge" />
          <geom fromto="{v3(L['t0'])} {v3(L['t1'])}" size="{L['rt']:.4f}" type="capsule" />
          <body name="11{k}" pos="{v3(L['ok'] - L['oh'])}">
            {inertial(*L['shin'])}
            <joint axis="{ax[TN['kneej'](k)]}" name="11{k}_joint" pos="0 0 0" range="{rng[TN['kneej'](k)]}" type="hinge" />
            <geom fromto="{v3(L['s0'])} {v3(L['s1'])}" size="{L['rs']:.4f}" type="capsule" />
          </body>
        </body>
      </body>
'''
    head = open(TPL).read()
    head = head[head.index('<mujoco'):head.index('<worldbody>')]
    head = re.sub(r'model="[^"]+"', f'model="quad_{QUAD}_rab"', head, count=1)
    # モータ: 雛形の並び（脚ごとに股・膝）とギア比・ctrlrange をそのまま、名前だけ出力の関節へ
    ren = {TN[j](k): f'{n}{k}_joint' for k in legs for j, n in (('hipj', '1'), ('kneej', '11'))}
    act_el = tpl.find('actuator')
    for m in act_el:
        m.set('joint', ren[m.get('joint')])
        m.set('name', ren[m.get('name')])
    act = ET.tostring(act_el, encoding='unicode')
    xml = f'''<!-- scripts/make_quad_ra_xml.py（ブランチ feat/layer4-robotassembler）が {os.path.basename(IN)} から作った（系譜 9-342 の追記、2026-11-06）。手で直さない。
     研究室の部品を RobotAssembler で組んだ四脚。雛形 {os.path.basename(TPL)} の名前・関節の軸・可動域・ギア比はそのままで、関節の位置・質量・当たり判定を部品から。
     質量の合計 {mtot*1000:.1f} g。胴に制御基板・電池は載せていない。 -->
{head}<worldbody>
    <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="1 1 1 1" size="200 200 .125" type="plane" />
    <body name="0" pos="0 0 {z0:.6g}">
      {inertial(mt0, c0, I0)}
      <geom pos="{v3(boxc)}" size="{v3(half)}" type="box" />
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free" />
{body}    </body>
  </worldbody>
  {act}
</mujoco>
'''
    open(OUT, 'w').write(xml)
    print('書き出した:', OUT, flush=True)
    # 0° の姿勢にそろえた後の部品の姿勢（mm、胴の中心が原点）と関節の軸。当たりの確かめと描画に使う
    po = os.path.join(os.path.dirname(IN), os.path.basename(OUT).replace('.xml', '_parts_posed.json'))
    json.dump(dict(parts=[dict(p, T=(np.vstack([np.hstack([T[i][:3, :3], (T[i][:3, 3:] - center[:, None])]), [0, 0, 0, 1]])).tolist())
                          for i, p in enumerate(parts)],
                   joints=[dict(leg=k, hip=((L['oh'] - center) ).tolist(), hip_axis=L['ah'].tolist(), knee=((L['ok'] - center)).tolist(),
                                knee_axis=L['ak'].tolist()) for k, L in sorted(legs.items())]),
              open(po, 'w'), ensure_ascii=False, indent=1)
    print('書き出した:', po, flush=True)


if __name__ == '__main__':
    main()
