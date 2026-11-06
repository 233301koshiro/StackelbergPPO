#!/usr/bin/env python3
"""**四脚の XML を、研究室の部品（RobotAssembler の部品・XL430）で組んだと見なした形へ近似する**（系譜 9-342 の追記、2026-11-06）。

なぜ: ユーザー「robot assemblerでそのふたつを近似したらどう変わるかてみて」（興味本位）。手で書いた `quad_4x2` と、立体から作った
`quad_tripo` は、どちらも ant の既定の物理（密度 5・ギア比 150、質量の合計 0.1〜0.4 kg）で、台本の速歩で跳ねて倒れた。
軸6（修論 4.7）と同じ部品の値に置き換えると、立ち方と速歩がどう変わるかを見る。

近似の決まり（どれも根拠の無い所は宣言する）:
  - 全体を SCALE 倍（既定 0.3）に縮める。腿・脛の長さが部品で組める範囲（57.6〜186.7 mm）に入るように選んだ値
  - 駆動する体（腿・脛）の長さ（関節から子の関節、子が無ければカプセルの先まで）を、組み方の表（`data/axis6/irsl_link_all_v2.csv`、
    9-325 の系統 Hinge_Short_Plug＋Frame_Long_Socket・connect_point ↔ right-side）で組める最も近い長さへ丸める。先の体はその分だけずらす
  - 質量: 腿・脛の構造は表の線形の当てはめ（1.335 g/mm × L − 59.8 g、57.6 mm で 20.1 g を下限）、関節ごとにサーボ 57.2 g（XL430）。
    サーボは、その関節を動かす側の親の体の先（＝関節の位置）に置く。股のサーボ 4 個は胴（取り付けがあれば取り付け）に入る。
    胴の構造は FRAME_G（既定 100 g）。⚠️ 胴の構造の質量には根拠が無い（Text2Robot は胴に Raspberry Pi と電池を積む。宣言して使う）
  - 慣性: 構造は軸に沿った一様な棒、サーボは点の質量として合成し `<inertial>` に書く（xml_robot.py は `<inertial>` を残し、変換器も読む）
  - 当たり判定の太さ: 腿 23.2 mm・脛 20.0 mm（軸6 の上腕・手先のカプセル、修論 4.7 の表 4.27）。取り付けは 10 mm
  - 関節: XL430 の値（ギア比 1.4 N·m・減衰 0.235・回転子の慣性 0.008、修論 4.7 の表 4.27）。可動域は元の XML のまま（±30°）
  - 足先の最も低い点が床から CLEAR の高さに来るよう上下にずらす

    python3 scripts/quad_ra_approx.py --src quad_tripo [--scale 0.3] [--frame-g 100]   # → assets/mujoco_envs/quad_tripo_ra.xml
"""
import argparse
import csv
import xml.etree.ElementTree as ET

import numpy as np

SERVO_KG = 0.0572
CLEAR = 0.005
R_THIGH, R_SHIN, R_MOUNT = 0.0232, 0.0200, 0.010


def vec(s):
    return np.array([float(x) for x in s.split()])


def fmt(v):
    return ' '.join(f'{x:.6g}' if abs(x) > 1e-9 else '0' for x in np.atleast_1d(v))


def buildable_lengths(path='data/axis6/irsl_link_all_v2.csv'):
    L = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            if (r['Hinge'] == 'Hinge_Short_Plug' and r['Frame'] == 'Frame_Long_Socket'
                    and r['Frame→サーボ'] == 'connect_point' and r['サーボの点'] == 'right-side'):
                L.add(float(r['軸間距離_mm']))
    return np.array(sorted(L)) / 1000.0


def struct_kg(L):
    return max(1.335 * L * 1000 - 59.8, 20.1) / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--scale', type=float, default=0.3)
    ap.add_argument('--frame-g', type=float, default=100.0)
    a = ap.parse_args()
    LB = buildable_lengths()
    tree = ET.parse(f'assets/mujoco_envs/{a.src}.xml')
    root = tree.getroot()
    root.set('model', f'{a.src}_ra')
    d = root.find('default')
    d.find('joint').attrib.update(armature='0.008', damping='0.235')
    d.find('geom').attrib.update(density='1', margin='0.002')   # 質量は <inertial> で与える（カプセルの密度は使わない）
    for m in root.find('actuator'):
        m.set('gear', '1.4')
    wb = root.find('worldbody')
    torso = wb.find('body')
    act = lambda b: b.find('joint[@type="hinge"]') is not None
    rows = []

    # 1. 縮める（体の位置・カプセルの端・箱と球の大きさ）
    for b in torso.iter('body'):
        if b is not torso:
            b.set('pos', fmt(vec(b.get('pos')) * a.scale))
        for g in b.findall('geom'):
            if g.get('fromto'):
                g.set('fromto', fmt(vec(g.get('fromto')) * a.scale))
            elif g.get('size'):
                g.set('size', fmt(vec(g.get('size')) * a.scale))
            if g.get('pos'):
                g.set('pos', fmt(vec(g.get('pos')) * a.scale))
    torso.set('pos', fmt(vec(torso.get('pos')) * a.scale))

    # 2. 駆動する体の長さを丸め、質量と慣性を書く
    def link_vec(b):
        ch = b.find('body')
        if ch is not None:
            return vec(ch.get('pos'))
        ft = vec(b.find('geom').get('fromto'))
        return ft[3:] - ft[:3]

    servo_on = {}   # 体 → その体に載るサーボの数（子の関節のサーボ）
    for b in torso.iter('body'):
        for ch in b.findall('body'):
            if act(ch):
                servo_on[b] = servo_on.get(b, 0) + 1
    for b in torso.iter('body'):
        if b is torso or not act(b):
            continue
        v = link_vec(b)
        L0 = np.linalg.norm(v)
        L1 = float(LB[np.argmin(abs(LB - L0))])
        u = v / L0
        ch = b.find('body')
        g = b.find('geom')
        leaf = ch is None
        r = R_SHIN if leaf else R_THIGH
        if leaf:
            g.set('fromto', fmt(np.concatenate([np.zeros(3), u * (L1 - r)])))   # 先の面が丸めた長さに来る
        else:
            ch.set('pos', fmt(u * L1))
            g.set('fromto', fmt(np.concatenate([np.zeros(3), u * L1])))
        g.set('size', f'{r:.4f}')
        ms, nserv = struct_kg(L1), servo_on.get(b, 0)
        M = ms + nserv * SERVO_KG
        com = (ms * u * L1 / 2 + nserv * SERVO_KG * u * L1) / M
        I = np.zeros((3, 3))
        I_rod = ms * L1 ** 2 / 12 * (np.eye(3) - np.outer(u, u))
        for m_, c_ in ((ms, u * L1 / 2), (nserv * SERVO_KG, u * L1)):
            dd = c_ - com
            I += m_ * (dd @ dd * np.eye(3) - np.outer(dd, dd))
        I += I_rod
        I += 1e-7 * np.eye(3)
        inr = ET.Element('inertial', pos=fmt(com), mass=f'{M:.6g}',
                         fullinertia=fmt([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]))
        b.insert(0, inr)
        rows.append((b.get('name'), 'leaf' if leaf else 'mid', L0, L1, ms, nserv, M))
    # 取り付け（関節の無い体）: 細く、質量はほぼ 0。股のサーボは取り付けに載る
    for b in torso.iter('body'):
        if b is torso or act(b):
            continue
        g = b.find('geom'); g.set('size', f'{R_MOUNT:.4f}')
        ns = servo_on.get(b, 0)
        tip = vec(b.find('body').get('pos')) if b.find('body') is not None else np.zeros(3)
        M = max(ns * SERVO_KG, 1e-4)
        b.insert(0, ET.Element('inertial', pos=fmt(tip), mass=f'{M:.6g}', diaginertia='1e-6 1e-6 1e-6'))
        rows.append((b.get('name'), 'mount', np.linalg.norm(tip), np.linalg.norm(tip), 0.0, ns, M))
    # 胴: 構造 FRAME_G ＋ 胴に直接載るサーボ（取り付けが無い形では股のサーボ 4 個）
    nt = servo_on.get(torso, 0)
    Mt = a.frame_g / 1000 + nt * SERVO_KG
    g = torso.find('geom')
    if g.get('type') == 'box':
        hx, hy, hz = vec(g.get('size'))
        It = Mt / 3 * np.array([hy ** 2 + hz ** 2, hx ** 2 + hz ** 2, hx ** 2 + hy ** 2])
    else:
        rr = vec(g.get('size'))[0]
        It = 0.4 * Mt * rr ** 2 * np.ones(3)
    torso.insert(0, ET.Element('inertial', pos='0 0 0', mass=f'{Mt:.6g}', diaginertia=fmt(It)))

    # 3. 足先が床から CLEAR に来るよう上下にずらす
    def lowest(b, p0):
        p = p0 + (vec(b.get('pos')) if b is not torso else vec(torso.get('pos')))
        zs = []
        for g in b.findall('geom'):
            if g.get('fromto'):
                ft = vec(g.get('fromto')); r = vec(g.get('size'))[0]
                zs += [p[2] + ft[2] - r, p[2] + ft[5] - r]
        for ch in b.findall('body'):
            zs.append(lowest(ch, p))
        return min(zs) if zs else 1e9
    z = lowest(torso, np.zeros(3))
    tp = vec(torso.get('pos')); tp[2] += CLEAR - z
    torso.set('pos', fmt(tp))

    out = f'assets/mujoco_envs/{a.src}_ra.xml'
    ET.indent(tree)
    body = ET.tostring(root, encoding='unicode')
    # ⚠️ XML のコメントの中に「--」を書けない（lxml が読めない。最初の版で起動の引数をそのまま書いて落ちた）
    head = (f'<!-- scripts/quad_ra_approx.py（src={a.src}・scale={a.scale}・frame_g={a.frame_g}）が作った（系譜 9-342 の追記、2026-11-06）。手で直さない。\n'
            f'     研究室の部品（XL430・組み方の表 9-325 の系統）で組んだと見なした近似。胴の構造 {a.frame_g:.0f} g には根拠が無い。 -->\n')
    open(out, 'w').write(head + body + '\n')
    tot = Mt + sum(rw[6] for rw in rows)
    print(f'→ {out}  縮め {a.scale}・胴 {Mt * 1000:.1f} g（構造 {a.frame_g:.0f} g ＋ サーボ {nt} 個）・質量の合計 {tot * 1000:.1f} g・胴の中心の高さ {tp[2]:.3f} m')
    for n, kind, L0, L1, ms, ns, M in rows:
        print(f'  体 {n:>4s}（{kind}）: 長さ {L0 * 1000:6.1f} → {L1 * 1000:6.1f} mm・構造 {ms * 1000:5.1f} g・サーボ {ns} 個・計 {M * 1000:6.1f} g')


if __name__ == '__main__':
    main()
