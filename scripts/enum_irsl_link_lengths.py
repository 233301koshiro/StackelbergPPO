"""研究室の部品（IRSL の RobotAssembler 部品ライブラリ）で、XL-430 から次の XL-430 までの 1 本のリンクに作れる長さを数え上げる（軸6 L-3a、系譜 9-277）。

⭐ 1 本のリンク = A の XL-430 のホーン → Hinge（Short/Long の Plug/Socket）→（LEGO）→ Frame（Short/Long/Bottom の Socket/Plug）→ B の XL-430 の本体。
   RobotAssembler 自身の `attach` で組み、A と B のホーン（回転軸）の世界座標から、2 本の軸の平行度・軸間距離・軸方向のずれを出す。
   ホーンの局所座標は設定ファイル（`irsl_assembler_config.yaml` の XL-430 の actuators）から読む。
⭐ LEGO の挟み方は 3 通り: なし／長い棒（block_1_16・plate_1_12）の k ポッチ目に Frame／ブロック m 段＋プレート n 段の重ね（同じポッチ）。
⭐ 確かめ: 研究室の腕（`/userdir/TA_assembler/arm_history.roboasm`）の Hinge_Short_Plug → Frame_Short_Socket のリンクで、
   Body の関節位置と同じ 76.80 mm・軸方向 4.0 mm を出す（起動時に自己検査し、合わなければ止める）。

⚠️ RobotAssembler は部品どうしのめり込みを調べないので、数えた組み合わせには実物では組めないもの（部品が重なる）が混ざりうる。
⚠️ 1×1 のブロックを高く重ねた柱は実物では弱い。強度・トルクは見ていない。
⚠️ リンクの先端（最後の関節から手先まで）はサーボで終わらないので、この表の対象外。

    OUT=data/axis6/irsl_link_lengths.csv \\
      /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/enum_irsl_link_lengths.py > /tmp/enum.log 2>&1
"""
import csv
import os
import re
import sys
import time

import numpy as np
import yaml

PROJ = '/choreonoid_ws/install/share/choreonoid-2.3/robot_assembler/irsl'
OUT = os.environ.get('OUT', 'data/axis6/irsl_link_lengths.csv')
CF = ['fixed', 'invert', 'rotate90', 'rotate-90']
MATCH = {'FRAME_ASIDE_X': 'SERVO_ASIDE_X', 'FRAME_BSIDE_X': 'SERVO_BSIDE_X', 'FRAME_CSIDE_X': 'SERVO_CSIDE_X'}


def rotmat(r):
    if r is None:
        return np.eye(3)
    ax = np.array(r[:3], float)
    ax /= np.linalg.norm(ax)
    ang = np.radians(r[3])
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def main():
    import cnoid.RobotAssembler as cra
    cfg = yaml.safe_load(open(PROJ + '/irsl_assembler_config.yaml'))
    P = {p['type']: p for p in cfg['PartsSettings']}
    ru = cra.RoboasmUtil(PROJ + '/irsl_assembler_config.yaml')
    act = P['XL-430']['actuators'][0]
    horn_t = np.array(act['translation'], float) / 1000.0      # 設定は mm、座標は m
    horn_z = rotmat(act.get('rotation'))[:, 2]

    def horn_world(parts):
        wc = parts.worldcoords
        wc = wc() if callable(wc) else wc
        cp = wc.cnoidPosition
        T = np.array(cp() if callable(cp) else cp, float)
        return T[:3, :3] @ horn_t + T[:3, 3], T[:3, :3] @ horn_z

    def gap(A, B):
        pa, za = horn_world(A.rootParts())
        pb, zb = horn_world(B)
        d = pb - pa
        ax = float(d @ za)
        return abs(float(za @ zb)), float(np.linalg.norm(d - ax * za)) * 1000, ax * 1000

    def types(t, n):
        return next(c.get('types', []) for c in P[t]['connecting-points'] if c['name'] == n)

    def face(t):
        return [c['name'] for c in P[t]['connecting-points'] if set(c.get('types', [])) & {'PC1', 'SC1'}]

    def corners(names):
        ij = [(n, tuple(map(int, re.findall(r'(\d+)_(\d+)$', n)[0]))) for n in names]
        I = [x[1][0] for x in ij]
        J = [x[1][1] for x in ij]
        return [n for n, (i, j) in ij if i in (min(I), max(I)) and j in (min(J), max(J))]

    servo = {c['name']: c.get('types', []) for c in P['XL-430']['connecting-points']}

    def start(h):
        A = ru.makeRobotFromKey('XL-430')
        H = ru.makeParts(h)
        A.attach(H, H.name + '/side_connect_point', A.rootParts().name + '/horn', 'fixed')
        return A, H

    def finish(A, last, lastpt, f, fpt, fc, fs, bs):
        F = ru.makeParts(f)
        if not A.attach(F, F.name + '/' + fpt, last.name + '/' + lastpt, fc):
            return None
        B = ru.makeParts('XL-430')
        if not A.attach(B, B.name + '/' + bs, F.name + '/' + fs, 'fixed'):
            return None
        return gap(A, B)

    # ⭐ 自己検査: 研究室の腕のリンクを再現する
    A, H = start('Hinge_Short_Plug')
    g = finish(A, H, 'plug_0_0', 'Frame_Short_Socket', 'socket_3_3', 'invert', 'side_connect_point1', 'bottom-side3')
    print(f'[self-check] 研究室の腕のリンク: 平行 {g[0]:.4f}・軸間 {g[1]:.2f} mm・軸方向 {g[2]:.2f} mm（期待 1・76.80・4.00）', flush=True)
    if not (g[0] > 0.9999 and abs(g[1] - 76.80) < 0.05 and abs(g[2] - 4.0) < 0.05):
        print('⛔ 自己検査に失敗。止める', flush=True)
        return

    hinges = ['Hinge_Short_Plug', 'Hinge_Long_Plug', 'Hinge_Short_Socket', 'Hinge_Long_Socket']
    frames = ['Frame_Short_Socket', 'Frame_Long_Socket', 'Frame_Bottom_Socket',
              'Frame_Short_Plug', 'Frame_Long_Plug', 'Frame_Bottom_Plug']
    rows = []
    t0 = time.time()
    for h in hinges:
        hpol = 'P' if 'PC1' in types(h, face(h)[0]) else 'S'
        for f in frames:
            fpol = 'P' if 'PC1' in types(f, face(f)[0]) else 'S'
            if hpol == fpol:
                continue
            sp = [c['name'] for c in P[f]['connecting-points'] if set(c.get('types', [])) & set(MATCH)]
            fsbs = [(fs, bs) for fs in sp for bs, v in servo.items()
                    if MATCH[[x for x in types(f, fs) if x in MATCH][0]] in v]
            # なし（面のポッチは全部）
            for fs, bs in fsbs:
                for a in face(h):
                    for b in face(f):
                        for fc in CF:
                            A, H = start(h)
                            g = finish(A, H, a, f, b, fc, fs, bs)
                            if g:
                                rows.append(['なし', h, a, f, b, fc, fs, bs, '', '', *g])
            # LEGO（面のポッチは四隅）。Plug の Hinge は LEGO の下面（SK）に、Socket の Hinge は上面（PL）につく
            hin, hout = ('SK', 'PL') if hpol == 'P' else ('PL', 'SK')
            for fs, bs in fsbs:
                for a in corners(face(h)):
                    for b in corners(face(f)):
                        for beam, N in [('block_1_16', 16), ('plate_1_12', 12)]:
                            for bc in CF:
                                for k in range(N):
                                    for fc in CF:
                                        A, H = start(h)
                                        Bm = ru.makeParts(beam)
                                        if not A.attach(Bm, f'{Bm.name}/{hin}1_0_0', f'{H.name}/{a}', bc):
                                            continue
                                        g = finish(A, Bm, f'{hout}1_0_{k}', f, b, fc, fs, bs)
                                        if g:
                                            rows.append([f'棒 {beam}', h, a, f, b, fc, fs, bs, bc, k, *g])
                        for nb in range(0, 13):
                            for npl in range(0, 3):
                                if nb + npl == 0:
                                    continue
                                for fc in CF:
                                    A, H = start(h)
                                    last, lp, ok = H, a, True
                                    for blk in ['block_1_1'] * nb + ['plate_1_1'] * npl:
                                        Bm = ru.makeParts(blk)
                                        if not A.attach(Bm, f'{Bm.name}/{hin}1_0_0', f'{last.name}/{lp}', 'fixed'):
                                            ok = False
                                            break
                                        last, lp = Bm, f'{hout}1_0_0'
                                    if not ok:
                                        continue
                                    g = finish(A, last, lp, f, b, fc, fs, bs)
                                    if g:
                                        rows.append(['重ね', h, a, f, b, fc, fs, bs, f'block×{nb}+plate×{npl}', nb * 9.6 + npl * 3.2, *g])
        print(f'[enum] {h} まで {len(rows)} 通り（{time.time() - t0:.0f} 秒）', flush=True)

    par = [r for r in rows if r[10] > 0.999]
    # 軸間距離（0.1 mm に丸め）ごとに 1 つの組み方を残す。軸方向のずれが小さいものを優先
    best = {}
    for r in par:
        key = round(r[11], 1)
        if key not in best or abs(r[12]) < abs(best[key][12]):
            best[key] = r
    cnt = {}
    for r in par:
        cnt[round(r[11], 1)] = cnt.get(round(r[11], 1), 0) + 1
    os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
    with open(OUT, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['軸間距離_mm', '組み方の数', '軸方向のずれ_mm', 'LEGO', 'Hinge', 'Hinge の点', 'Frame', 'Frame の点',
                    'つなぎ方', 'Frame→サーボ', 'サーボの点', 'LEGO の詳細', 'LEGO の段・ポッチ'])
        for key in sorted(best):
            r = best[key]
            w.writerow([key, cnt[key], round(r[12], 1), r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]])
    L = sorted(best)
    gaps = np.diff(L)
    print(f'[done] 組んだ {len(rows)}・軸が平行 {len(par)}・軸間距離 {len(L)} 種類 {L[0]}〜{L[-1]} mm・'
          f'隣どうしの隙間 中央値 {np.median(gaps):.2f} mm・最大 {gaps.max():.2f} mm → {OUT}', flush=True)


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
# ⚠️ Choreonoid は自分では終わらない。os._exit(0) で終える（Bug 43）
os._exit(0)
