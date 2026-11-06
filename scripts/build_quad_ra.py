"""四脚を研究室の部品（RobotAssembler の IRSL 部品・XL430）で組む（系譜 9-342 の追記、2026-11-06、ユーザー「これらをそれぞれrobot assemblerにしたものは？」
「近似と比べて動きがどう変わるかまで見る」）。

なぜ: main の `quad_ra_approx.py` は、四脚の XML のカプセルに部品の質量と XL430 の値を当てただけの近似で、部品は組んでいない。
組むと、股のサーボの置き場所（4 個が胴の中心に重ならない）・関節の間の実際の長さ・質量の分布が変わる。それで動きがどう変わるかを見る。

組み方（どれも `build_arm3.py` の部品の付け方と同じ API。`attach` が通らない組み方は使わない）:
  QUAD=tripo（股・膝とも水平の軸で平行。`quad_tripo_ra` を組む）
    - 胴: `plate_16_16`（128 mm 角、質量 67.6 g）。股のサーボを `Frame_Long_Plug` で胴の下に吊る（ホーンの軸がポッチの面と平行＝水平）。
      ホーンが外向き（±y）になる付け方のうち、ホーンの xy が近似の股の位置にいちばん近いポッチの位置を脚ごとに選ぶ
    - 腿: 股のホーン → 組み方の表（`data/axis6/irsl_link_all_v2.csv`、9-325 の系統 Hinge_Short_Plug＋Frame_Long_Socket・
      connect_point ↔ right-side）の、近似の関節の間（THIGH_MM、既定 108.8 mm）にいちばん近い行 → 膝のサーボ
    - 脛: 膝のホーン → TIP_HINGE ＋ ブロックの段（継ぎ目を直交）。先端までの長さが FOOT_MM（既定 102.4 mm ＝ 近似の脛 82.4 ＋ 半径 20.0）に
      いちばん近い段数（0〜8）を選ぶ
  QUAD=walk（股・膝とも水平の軸。`quad_walk_ra235` を組む。11-06 深夜、ユーザー「おすすめで」。四脚の形のノウハウ.md の K4「股は胴の側面の外」）
    - 股の位置は雛形 TPL の取り付けのカプセルの先から読む（tripo のように手で書かない）。腿・脛の長さは THIGH_MM・FOOT_MM で渡す
    - RAIL（例 plate_4_8）を付けると、股のサーボを胴の板の下に直接吊らず、板の下に RAIL を付けて側面の外へはみ出させ、
      そのはみ出した所に Frame_Long_Plug でサーボを吊る（サーボを板の下に直接吊ると、ホーンは板の縁から 5 mm 外の y = 69 mm が限り）。
      RAIL のポッチのうち胴の板の下に入るものはすべて板の穴に入り、入るのは 2 列以上
    - 狙うホーンの位置: x は雛形の股、y は雛形の足先の y ＋ HINGE_IN（19 mm。Hinge は脚の中心線をホーンより内側へ寄せる、K4）
出力: OUT の JSON（部品ごとの型・姿勢 T〔mm〕・脚の番号・部分〔torso / mount / thigh / shin〕、関節のホーンの姿勢）。XML は `make_quad_ra_xml.py`。

    QUAD=tripo OUT=data/axis6/quad/quad_tripo_parts.json \\
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/build_quad_ra.py > ログ 2>&1
"""
import csv
import itertools
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJ = '/choreonoid_ws/install/share/choreonoid-2.3/robot_assembler/irsl'
QUAD = os.environ.get('QUAD', 'tripo')
OUT = os.environ.get('OUT', f'/userdir/StackelbergPPO/data/axis6/quad/quad_{QUAD}_parts.json')
TABLE = os.environ.get('TABLE', '/userdir/StackelbergPPO/data/axis6/irsl_link_all_v2.csv')
THIGH_MM = float(os.environ.get('THIGH_MM', '108.8'))
FOOT_MM = float(os.environ.get('FOOT_MM', '102.4'))
TIP_HINGE = os.environ.get('TIP_HINGE', 'Hinge_Short_Plug')
PLATE = os.environ.get('PLATE', 'plate_16_16')
RAIL = os.environ.get('RAIL', '')          # 股のサーボを胴の側面の外へ出す板（QUAD=walk。空なら胴の板の下に直接吊る）
HINGE_IN = 19.0                            # Hinge の side_connect_point から Hinge の中心（脚の中心線）まで、ホーンの軸に沿って（mm）
TPL = os.environ.get('TPL', f'/userdir/StackelbergPPO/assets/mujoco_envs/quad_{QUAD}_ra.xml')   # 近似の XML（0° の姿勢の向き）
# 近似（quad_tripo_ra.xml）の股の位置（胴の中心から、mm）。脚 1〜4 の順
HIPS = {'tripo': [(-60.96, 67.65), (60.93, -67.83), (61.17, 67.86), (-60.87, -67.62)],
        'x42': [(1, 1), (1, -1), (-1, 1), (-1, -1)]}   # x42 は股が胴の中心に重なる近似なので、象限だけを持つ（中心にいちばん近い付け方を探す）
CF = ['fixed', 'invert', 'rotate90', 'rotate-90']
TOL = 0.5
SERVO_PTS = ['right-side', 'left-side', 'bottom-side1', 'bottom-side2', 'bottom-side3', 'left-side2']


def rotmat(r):
    if r is None:
        return np.eye(3)
    ax = np.array(r[:3], float)
    if np.linalg.norm(ax) == 0:
        return np.eye(3)
    ax /= np.linalg.norm(ax)
    ang = np.radians(r[3])
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def log(*a):
    print(*a, flush=True)


def main():
    import cnoid.RobotAssembler as cra
    from joint_range_a1v import Shape, rot_about
    cfg = yaml.safe_load(open(PROJ + '/irsl_assembler_config.yaml'))
    P = {p['type']: p for p in cfg['PartsSettings']}
    ru = cra.RoboasmUtil(PROJ + '/irsl_assembler_config.yaml')
    CPT = {t: {c['name']: (np.array(c.get('translation', [0, 0, 0]), float), set(c.get('types', [])))
               for c in P[t]['connecting-points']} for t in P}

    def pose(p):
        wc = p.worldcoords
        wc = wc() if callable(wc) else wc
        cp = wc.cnoidPosition
        M = np.array(cp() if callable(cp) else cp, float)
        M[:3, 3] *= 1000.0
        return M

    cache = {}

    def rel(bt, bp, nt, np_, c):
        k = (bt, bp, nt, np_, c)
        if k not in cache:
            Rt = ru.makeRobotFromKey(bt)
            b = Rt.rootParts()
            N = ru.makeParts(nt)
            ok = Rt.attach(N, f'{N.name}/{np_}', f'{b.name}/{bp}', c)
            cache[k] = np.linalg.inv(pose(b)) @ pose(N) if ok else None
        return cache[k]

    def pts(t, kind):
        return [n for n, (_, ty) in CPT[t].items() if kind in ty]

    def wp(M, t, n):
        return M[:3, :3] @ CPT[t][n][0] + M[:3, 3]

    def corners(t, kind):
        ns = pts(t, kind)
        xy = np.array([CPT[t][n][0][:2] for n in ns])
        out = []
        for sx, sy in itertools.product((min, max), repeat=2):
            x, y = sx(xy[:, 0]), sy(xy[:, 1])
            out += [n for n, q in zip(ns, xy) if abs(q[0] - x) < 1e-6 and abs(q[1] - y) < 1e-6]
        return out

    def long_axis(M, t, kind):
        ns = pts(t, kind)
        q = np.array([wp(M, t, n) for n in ns])
        u, s, vt = np.linalg.svd(q - q.mean(0))
        return vt[0] if s[0] > s[1] + 1e-6 else None

    # ---- build_arm3.py と同じ LEGO の段の積み方（必須 ②③）----
    def make_layer(bases, out_kind, ltype, prev_axis):
        in_kind = 'SC1' if out_kind == 'PC1' else 'PC1'
        grid = [(bi, n, wp(M, t, n)) for bi, (M, t) in enumerate(bases) for n in pts(t, out_kind)]
        need = len(grid)
        npieces = 1 if ltype == 'plate_4_4' else 2

        def fits(Mn, covered):
            hit = set()
            for n in pts(ltype, in_kind):
                w = wp(Mn, ltype, n)
                g = [i for i, (_, _, q) in enumerate(grid) if np.linalg.norm(q - w) < TOL]
                if not g or g[0] in covered or g[0] in hit:
                    return None
                hit.add(g[0])
            return hit

        def search(covered, placed):
            if len(placed) == npieces:
                return placed if len(covered) == need else None
            for gi, (bi, bn, _) in enumerate(grid):
                if gi in covered:
                    continue
                Mb, tb = bases[bi]
                for npn in corners(ltype, in_kind):
                    for c in CF:
                        r = rel(tb, bn, ltype, npn, c)
                        if r is None:
                            continue
                        Mn = Mb @ r
                        h = fits(Mn, covered)
                        if h is None:
                            continue
                        if prev_axis is not None and ltype != 'plate_4_4':
                            ax = long_axis(Mn, ltype, in_kind)
                            if ax is not None and abs(ax @ prev_axis) > 0.1:
                                continue
                        res = search(covered | h, placed + [(bi, bn, npn, c, Mn)])
                        if res:
                            return res
            return None
        return search(set(), [])

    def stack_plan(Mh, ht, comp):
        out_kind = 'PC1' if pts(ht, 'PC1') else 'SC1'
        bases, layers, prev_axis = [(Mh, ht)], [], None
        for s in comp:
            lt = 'block_2_4' if s == 'B' else 'plate_4_4'
            lay = make_layer(bases, out_kind, lt, prev_axis)
            if lay is None:
                return None
            layers.append((lt, lay))
            bases = [(x[4], lt) for x in lay]
            if lt == 'block_2_4':
                prev_axis = long_axis(lay[0][4], lt, 'PC1' if out_kind == 'PC1' else 'SC1')
        return layers, bases, out_kind

    R = ru.makeRobotFromKey(PLATE)
    PL = R.rootParts()
    parts = [dict(name='胴のプレート', type=PLATE, p=PL, leg=0, seg='torso')]
    Mpl = pose(PL)
    q = np.array([CPT[PLATE][n][0] for n in pts(PLATE, 'PC1')])
    center = np.array([q[:, 0].mean(), q[:, 1].mean(), -1.6])          # プレートの厚みの中央
    log(f'胴 {PLATE}: 中心（プレートの座標）{center.tolist()} mm、質量 {P[PLATE]["mass-param"]["mass"]:.1f} g')
    holes = pts(PLATE, 'SC1')
    hole_w = np.array([wp(Mpl, PLATE, n) for n in holes])
    joints = []

    def build_layers(base_parts, layers, tag, leg, seg):
        cur = base_parts
        for li, (lt, lay) in enumerate(layers, start=1):
            new = []
            for k, (bi, bn, npn, c, _) in enumerate(lay):
                N = ru.makeParts(lt)
                assert R.attach(N, f'{N.name}/{npn}', f'{cur[bi].name}/{bn}', c)
                parts.append(dict(name=f'{tag}の LEGO {li} 段目 {k}', type=lt, p=N, leg=leg, seg=seg))
                new.append(N)
            cur = new
        return cur

    # ---- 腿の組み方（表の行）----
    # 9-325 の系統（connect_point ↔ right-side）を先に、膝のサーボを裏返して付ける 3 系統を後に試す（同じ Hinge・Frame・長さ）
    FAMS = [('connect_point', 'right-side'), ('connect_point', 'left-side'), ('connect_point2', 'left-side2'), ('connect_point2', 'bottom-side2')]
    allrows = [r for r in csv.DictReader(open(TABLE)) if r['Hinge'] == 'Hinge_Short_Plug' and r['Frame'] == 'Frame_Long_Socket']
    fam_rows = []
    for fs_, ss_ in FAMS:
        rs = [r for r in allrows if r['Frame→サーボ'] == fs_ and r['サーボの点'] == ss_]
        fam_rows.append(min(rs, key=lambda r: (abs(float(r['軸間距離_mm']) - THIGH_MM), int(r['部品の数']))))
    row = fam_rows[0]
    thigh_comp = '' if row['段'] == 'なし' else row['段']
    log(f'腿: 目標 {THIGH_MM} mm → 表の {row["軸間距離_mm"]} mm（段 {row["段"]}・Frame {row["Frame"]}・部品 {row["部品の数"]}）')

    # ---- 脛の段数: 膝のホーンから先端まで（ホーンの軸から外れた距離の最大）が FOOT_MM に近いもの ----
    def tip_reach(ht, comp):
        Mh = rel('XL-430', 'horn', ht, 'side_connect_point', 'fixed')
        sp = stack_plan(Mh, ht, comp)
        if sp is None:
            return None
        Ms = [(Mh, ht)] + [(x[4], lt) for lt, lay in sp[0] for x in lay]
        w = np.vstack([Shape(t, P).pts @ M[:3, :3].T + M[:3, 3] for M, t in Ms])
        return float(np.max(np.linalg.norm(w[:, :2], axis=1)))     # ホーンの軸は z（サーボの座標）
    cands = []
    for n in range(0, 9):
        r_ = tip_reach(TIP_HINGE, 'B' * n)
        if r_ is not None:
            cands.append((abs(r_ - FOOT_MM), n, r_))
            log(f'  脛の候補 {TIP_HINGE} ＋ ブロック {n} 段: 先端まで {r_:.1f} mm')
    _, shin_n, shin_reach = min(cands)
    log(f'脛: 目標 {FOOT_MM} mm → {TIP_HINGE} ＋ ブロック {shin_n} 段（{shin_reach:.1f} mm）')

    hz = np.array([0, 0, 1.0])
    # 近似の腿（股 → 膝）と脛（膝 → 足先）の向き。0° の姿勢をこれにそろえる（make_quad_ra_xml.py と同じ）
    import xml.etree.ElementTree as ET
    tb = {b.get('name'): b for b in ET.parse(TPL).getroot().iter('body')}
    vec = lambda s_: np.array([float(x) for x in s_.split()]) * 1000

    def wpts(lst):
        return np.vstack([Shape(t, P).pts @ M[:3, :3].T + M[:3, 3] for M, t in lst])

    def overlap(moving, static):
        """moving の部品の点が static の部品の芯（表面から 1 mm 削った内側）に入る体積（mm³）。joint_range_a1v.py と同じ判定"""
        n = 0
        for Ms, ts in static:
            sh = Shape(ts, P)
            Ti = np.linalg.inv(Ms)
            for Mm, tm in moving:
                q = Shape(tm, P).pts @ (Ti[:3, :3] @ Mm[:3, :3]).T + (Ti[:3, :3] @ Mm[:3, 3] + Ti[:3, 3])
                g = np.round(q).astype(int) - sh.lo
                ok = np.all((g >= 0) & (g < np.array(sh.core.shape)), axis=1)
                n += int(sh.core[tuple(g[ok].T)].sum())
        return n

    def signed(u, w, a):
        u = u - (u @ a) * a
        w = w - (w @ a) * a
        return float(np.degrees(np.arctan2(np.cross(u, w) @ a, u @ w)))

    def leg_chain(MS, ch, cs, row):
        """股のサーボの姿勢 MS から、腿の Hinge を向き ch・脛の Hinge を向き cs で付けた脚の部品 [(姿勢, 型)] と膝のサーボの姿勢（付けずに計算だけ）"""
        r0 = rel('XL-430', 'horn', row['Hinge'], 'side_connect_point', ch)
        if r0 is None:
            return None
        MH = MS @ r0
        sp = stack_plan(MH, row['Hinge'], thigh_comp)
        if sp is None:
            return None
        layers, _tops, _ = sp
        tops = [(MH, row['Hinge'])] if not layers else [(x[4], layers[-1][0]) for x in layers[-1][1]]
        Mt, tt = tops[int(row['最上段の部品'])]
        r1 = rel(tt, row['最上段の点'], row['Frame'], row['Frame の点'], row['つなぎ方'])
        if r1 is None:
            return None
        MF = Mt @ r1
        MK = MF @ rel(row['Frame'], row['Frame→サーボ'], 'XL-430', row['サーボの点'], 'fixed')
        thigh = [(MH, row['Hinge'])] + [(x[4], lt) for lt, lay in layers for x in lay] + [(MF, row['Frame']), (MK, 'XL-430')]
        r2 = rel('XL-430', 'horn', TIP_HINGE, 'side_connect_point', cs)
        if r2 is None:
            return None
        MT = MK @ r2
        spt = stack_plan(MT, TIP_HINGE, 'B' * shin_n)
        if spt is None:
            return None
        shin = [(MT, TIP_HINGE)] + [(x[4], lt) for lt, lay in spt[0] for x in lay]
        return thigh, shin, MK, layers, spt[0]

    def posed_score(MS, chain, leg, row, extra=()):
        """0° の姿勢にそろえ、股・膝 −30 / 0 / +30° の 9 姿勢で、脛が腿に・腿と脛が胴と股に入る体積の最大（mm³）"""
        thigh, shin, MK, _, _ = chain
        oh, ah = MS[:3, 3], MS[:3, :3] @ hz
        dv = MK[:3, 3] - oh
        Lg = float(np.linalg.norm(dv - (dv @ ah) * ah))
        if abs(Lg - float(row['軸間距離_mm'])) > 0.2 or abs((MK[:3, :3] @ hz) @ ah) < 0.999:
            return None
        th = signed(dv, vec(tb[f'11{leg}'].get('pos')), ah)
        M = rot_about(oh, ah, th)
        thigh = [(M @ m, t) for m, t in thigh]
        shin = [(M @ m, t) for m, t in shin]
        MK2 = M @ MK
        ok_, ak = MK2[:3, 3], MK2[:3, :3] @ hz
        w = wpts(shin) - ok_
        rad = np.linalg.norm(w - np.outer(w @ ak, ak), axis=1)
        tip = w[rad > rad.max() - 5.0].mean(0)        # 先端の面の中央（最も遠い 1 点はブロックの角で、左右の角のどちらかで ±9° 振れる）
        sh = signed(tip, vec(tb[f'11{leg}'].find('geom').get('fromto'))[3:], ak)
        M2 = rot_about(ok_, ak, sh)
        shin = [(M2 @ m, t) for m, t in shin]
        base = [(Mpl, PLATE), (MS, 'XL-430')] + list(extra)     # extra: 股の RAIL・フレーム（walk）
        worst = 0
        for hdeg in (-30, 0, 30):
            Mh_ = rot_about(oh, ah, hdeg)
            th_ = [(Mh_ @ m, t) for m, t in thigh]
            ko, ka = Mh_[:3, :3] @ ok_ + Mh_[:3, 3], Mh_[:3, :3] @ ak
            for kdeg in (-30, 0, 30):
                Mk_ = rot_about(ko, ka, kdeg)
                sh_ = [(Mk_ @ (Mh_ @ m), t) for m, t in shin]
                # 腿の Hinge はホーンと股のサーボに接して付くので、股のサーボを除いた胴の側（板・RAIL・フレーム）とだけ見る
                worst = max(worst, overlap(sh_, th_), overlap(th_[1:] + sh_, base), overlap(th_[:1], base[:1] + base[2:]))
        return worst, th, sh

    if QUAD not in HIPS and QUAD != 'x42':
        # 雛形の取り付けのカプセルの先（股）。y は足先の y ＋ HINGE_IN（組んだ脚の中心線が雛形の足先の面に来るホーンの位置）
        HIPS[QUAD] = []
        for k in range(1, 5):
            hp = vec(tb[str(k)].find('geom').get('fromto'))[3:]
            ft = hp + vec(tb[f'11{k}'].get('pos')) + vec(tb[f'11{k}'].find('geom').get('fromto'))[3:]
            HIPS[QUAD].append((hp[0], ft[1] + np.sign(ft[1]) * HINGE_IN))
        log(f'{QUAD}: 狙うホーンの xy（胴の中心から、mm）{[tuple(np.round(h, 1)) for h in HIPS[QUAD]]}')

    def servo_mounts(host_M, host_t, host_holes, sy, target, avoid):
        """host（胴の板か RAIL）の穴に Frame_Long_Plug を差し、ホーンが外向き（sy）のサーボを吊る付け方。avoid の部品に食い込むものは外す"""
        out = []
        hw = np.array([wp(host_M, host_t, n) for n in host_holes])
        for hn in host_holes:
            for fp in corners('Frame_Long_Plug', 'PC1'):
                for c in CF:
                    r = rel(host_t, hn, 'Frame_Long_Plug', fp, c)
                    if r is None:
                        continue
                    MF = host_M @ r
                    fw = np.array([wp(MF, 'Frame_Long_Plug', n) for n in pts('Frame_Long_Plug', 'PC1')])
                    if not all(np.min(np.linalg.norm(hw - f, axis=1)) < TOL for f in fw):
                        continue                       # フレームのポッチがすべて host の穴に入る
                    for fs in ('connect_point', 'connect_point2'):
                        for ss in SERVO_PTS:
                            r2 = rel('Frame_Long_Plug', fs, 'XL-430', ss, 'fixed')
                            if r2 is None:
                                continue
                            MS = MF @ r2
                            ax = MS[:3, :3] @ hz
                            if ax @ np.array([0, sy, 0]) < 0.999:
                                continue               # ホーンの軸が外向きの ±y
                            sv = Shape('XL-430', P).pts @ MS[:3, :3].T + MS[:3, 3]
                            if sv[:, 2].max() > center[2] - 1.6 + TOL:
                                continue               # サーボはプレートの下
                            d = float(np.linalg.norm(MS[:2, 3] - target))
                            if avoid and overlap([(MS, 'XL-430')], avoid):
                                continue
                            out.append((d, hn, fp, c, fs, ss, MS, MF))
        return out

    rails = {}
    if RAIL:
        # 胴の板の下に RAIL を付け、側面（±y）の外へはみ出させる。板の下に入るポッチはすべて穴に入り、入るのは 2 列（8 個）以上
        rp_all = pts(RAIL, 'PC1')
        seen_r = set()
        for hn in holes:
            for rp in corners(RAIL, 'PC1'):
                for c in CF:
                    r = rel(PLATE, hn, RAIL, rp, c)
                    if r is None:
                        continue
                    MR = Mpl @ r
                    rw = np.array([wp(MR, RAIL, n) for n in rp_all]) - center
                    inside = np.all(np.abs(rw[:, :2]) < 64.0, axis=1)
                    if inside.sum() < 8 or inside.all():
                        continue
                    if not all(np.min(np.linalg.norm(hole_w - (q + center), axis=1)) < TOL for q in rw[inside]):
                        continue
                    sy_ = np.sign(rw[~inside, 1].mean())
                    if not np.all(np.abs(rw[~inside, 1]) >= 64.0) or np.any(np.sign(rw[~inside, 1]) != sy_):
                        continue                       # はみ出すのは片方の側面（±y）の外だけ
                    key = (tuple(np.round(MR[:3, 3], 1)), tuple(np.round(MR[:3, :3].ravel(), 2)))
                    if key in seen_r:
                        continue
                    seen_r.add(key)
                    rails.setdefault(sy_, []).append((hn, rp, c, MR))
        log(f'{RAIL} を側面の外へはみ出させる付け方: +y {len(rails.get(1.0, []))}・−y {len(rails.get(-1.0, []))}')

    for leg, (tx, ty) in enumerate(HIPS[QUAD] if QUAD != 'x42' else [], start=1):
        sy = np.sign(ty)
        target = center[:2] + np.array([tx, ty])
        if RAIL:
            mounts = []
            for rhn, rp, rc, MR in rails[sy]:
                if abs(MR[:3, :3] @ np.mean([CPT[RAIL][n][0] for n in rp_all], 0) + MR[:3, 3] - np.r_[target, 0])[0] > 40:
                    continue                           # RAIL の中心が狙いの x から遠い
                for m in servo_mounts(MR, RAIL, pts(RAIL, 'SC1'), sy, target, [(Mpl, PLATE), (MR, RAIL)]):
                    mounts.append(m + ((rhn, rp, rc, MR),))
        else:
            mounts = [m + (None,) for m in servo_mounts(Mpl, PLATE, holes, sy, target, [])]
        assert mounts, f'脚 {leg}: 股の付け方が無い'
        dmin = min(m[0] for m in mounts)
        near = [m for m in mounts if m[0] <= dmin + 0.5]
        cand = []
        for fi, row in enumerate(fam_rows):
            for mi, (d, hn, fp, c, fs, ss, MS, MF, rl) in enumerate(near):
                extra = [(MF, 'Frame_Long_Plug')] + ([(rl[3], RAIL)] if rl else [])
                for ch in CF:
                    for cs in CF:
                        chain = leg_chain(MS, ch, cs, row)
                        if chain is None:
                            continue
                        sc = posed_score(MS, chain, leg, row, extra if rl else ())
                        if sc is None:
                            continue
                        cand.append((sc[0], abs(sc[2]), d, mi, ch, cs, sc[1], sc[2], fi))
            if cand and min(x[0] for x in cand) < 400:
                break                                  # 接している面のぶん（約 250 mm³）だけの組み方が見つかった
        assert cand, f'脚 {leg}: 表どおりの腿が組める付け方が無い'
        cand.sort(key=lambda x: (x[0], x[1]))
        log(f'脚 {leg}: 候補 {len(cand)}（股の付け方 {len(near)} × 腿の Hinge の向き × 脛の Hinge の向き）。'
            f'食い込みの最大（mm³）の少ない順: {[(x[0], FAMS[x[8]], x[4], x[5], round(x[7])) for x in cand[:4]]}')
        worst, _, d, mi, ch, cs, th, sh, fi = cand[0]
        _, hn, fp, c, fs, ss, _, _, rl = near[mi]
        row = fam_rows[fi]
        host = PL
        if rl:
            RL = ru.makeParts(RAIL)
            assert R.attach(RL, f'{RL.name}/{rl[1]}', f'{PL.name}/{rl[0]}', rl[2])
            parts.append(dict(name=f'脚{leg}の股の板（側面の外へ）', type=RAIL, p=RL, leg=leg, seg='mount'))
            host = RL
            log(f'  股の板 {RAIL}: 胴の板の穴 {rl[0]}・{rl[1]}・{rl[2]}、板の中心（胴の中心から）'
                f'{np.round(pose(RL)[:3, :3] @ np.mean([CPT[RAIL][n][0] for n in rp_all], 0) + pose(RL)[:3, 3] - center, 1).tolist()} mm')
        F = ru.makeParts('Frame_Long_Plug')
        assert R.attach(F, f'{F.name}/{fp}', f'{host.name}/{hn}', c)
        S = ru.makeParts('XL-430')
        assert R.attach(S, f'{S.name}/{ss}', f'{F.name}/{fs}', 'fixed')
        parts += [dict(name=f'脚{leg}の股のフレーム', type='Frame_Long_Plug', p=F, leg=leg, seg='mount'),
                  dict(name=f'脚{leg}の股のサーボ', type='XL-430', p=S, leg=leg, seg='mount')]
        hrel = pose(S)[:3, 3] - center
        log(f'  選んだ: 股のホーン（胴の中心から）{np.round(hrel, 1).tolist()} mm、目標の xy {[tx, ty]} とのずれ {d:.1f} mm'
            f'（穴 {hn}・{fp}・{c}・{fs}↔{ss}）、腿の Hinge {ch}・脛の Hinge {cs}、0° にそろえる回転 腿 {th:+.1f}°・脛 {sh:+.1f}°、'
            f'9 姿勢の食い込みの最大 {worst} mm³、腿の系統 {FAMS[fi]}（表の行の長さ {row["軸間距離_mm"]} mm）')
        # 腿
        H = ru.makeParts(row['Hinge'])
        assert R.attach(H, H.name + '/side_connect_point', S.name + '/horn', ch)
        parts.append(dict(name=f'脚{leg}の腿の Hinge', type=row['Hinge'], p=H, leg=leg, seg='thigh'))
        sp = stack_plan(pose(H), row['Hinge'], thigh_comp)
        tops = build_layers([H], sp[0], f'脚{leg}の腿', leg, 'thigh') if sp[0] else [H]
        Fk = ru.makeParts(row['Frame'])
        assert R.attach(Fk, f'{Fk.name}/{row["Frame の点"]}', f'{tops[int(row["最上段の部品"])].name}/{row["最上段の点"]}', row['つなぎ方'])
        K = ru.makeParts('XL-430')
        assert R.attach(K, f'{K.name}/{row["サーボの点"]}', f'{Fk.name}/{row["Frame→サーボ"]}', 'fixed')
        parts += [dict(name=f'脚{leg}の腿の Frame', type=row['Frame'], p=Fk, leg=leg, seg='thigh'),
                  dict(name=f'脚{leg}の膝のサーボ', type='XL-430', p=K, leg=leg, seg='thigh')]
        Mh, Mk = pose(S), pose(K)
        dv = Mk[:3, 3] - Mh[:3, 3]
        za = Mh[:3, :3] @ hz
        Lg = float(np.linalg.norm(dv - (dv @ za) * za))
        par = abs(float((Mk[:3, :3] @ hz) @ za))
        log(f'  腿: 本物で組んだ関節の間 {Lg:.1f} mm（表 {row["軸間距離_mm"]}）、膝の軸と股の軸の平行 {par:.4f}、軸方向のずれ {dv @ za:+.1f} mm')
        if abs(Lg - float(row['軸間距離_mm'])) > 0.2 or par < 0.999:
            raise RuntimeError(f'脚 {leg}: 腿が表どおりに組めていない')
        # 脛
        T = ru.makeParts(TIP_HINGE)
        assert R.attach(T, T.name + '/side_connect_point', K.name + '/horn', cs)
        parts.append(dict(name=f'脚{leg}の脛の Hinge', type=TIP_HINGE, p=T, leg=leg, seg='shin'))
        if shin_n:
            spt = stack_plan(pose(T), TIP_HINGE, 'B' * shin_n)
            build_layers([T], spt[0], f'脚{leg}の脛', leg, 'shin')
        joints.append(dict(leg=leg, hip=Mh.tolist(), knee=Mk.tolist(), overlap_mm3=worst, thigh_family=list(FAMS[fi])))


    # ======== QUAD=x42: 股は鉛直の軸（胴の下に Frame_Bottom_Plug で吊り、ホーンは下）、膝は股の軸と腿の向きの両方に垂直 ========
    def lever(d, ah, ak):
        """股の軸（ah）と膝の軸（ak）の 2 本の直線の間の、股の軸に垂直なてこの長さと向き。軸が平行なら d の ah に垂直な成分"""
        c = np.cross(ah, ak)
        if np.linalg.norm(c) < 0.01:          # 平行（約 0.6° 以内。数値の誤差で外積は 1e-6 を超える）
            v = d - (d @ ah) * ah
            return float(np.linalg.norm(v)), v / np.linalg.norm(v)
        n = c / np.linalg.norm(c)
        if d @ n < 0:
            n = -n
        return float(d @ n), n

    def search_link_orth(L):
        """股のサーボの座標（MA = 単位行列）で、ホーン → Hinge → LEGO → Frame → 膝のサーボ。膝の軸が股の軸と腿の向きの両方に垂直、
        軸方向（股の軸の向き）のずれが小さいものを選ぶ（build_arm3.py の search_link の「平行」を「直交」に替えた版）"""
        HINGES_ = ['Hinge_Short_Plug', 'Hinge_Short_Socket', 'Hinge_Long_Plug', 'Hinge_Long_Socket']
        FRAMES_ = ['Frame_Short_Plug', 'Frame_Short_Socket', 'Frame_Long_Plug', 'Frame_Long_Socket', 'Frame_Bottom_Plug', 'Frame_Bottom_Socket']
        best = None
        seen = {}            # 長さ（1 mm 刻み）ごとの最良（高さのずれ・部品の数）。ログに出す
        comps = [''] + [''.join(x) for n in range(1, 7) for x in [('B',) * n, ('B',) * n + ('P',)]]
        for ht in HINGES_:
            Mh = rel('XL-430', 'horn', ht, 'side_connect_point', 'fixed')
            for comp in comps:
                sp = stack_plan(Mh, ht, comp)
                if sp is None:
                    continue
                layers, tops, out_kind = sp
                in_kind = 'SC1' if out_kind == 'PC1' else 'PC1'
                for ft in FRAMES_:
                    fpts = pts(ft, in_kind)
                    if not fpts:
                        continue
                    for ti, (Mt, tt) in enumerate(tops):
                        for tn in pts(tt, out_kind):
                            for fn in corners(ft, in_kind):
                                for c in CF:
                                    r = rel(tt, tn, ft, fn, c)
                                    if r is None:
                                        continue
                                    MF = Mt @ r
                                    for fs in [n for n in CPT[ft] if not (CPT[ft][n][1] & {'PC1', 'SC1'})]:
                                        for ss in SERVO_PTS:
                                            r2 = rel(ft, fs, 'XL-430', ss, 'fixed')
                                            if r2 is None:
                                                continue
                                            MK = MF @ r2
                                            d = MK[:3, 3]
                                            ka = MK[:3, :3] @ hz
                                            if abs(ka[2]) > 0.001:
                                                continue        # 膝の軸が水平（股の軸 z に垂直）
                                            Lg, _n = lever(d, hz, ka)
                                            if Lg < 1.0:
                                                continue
                                            npart = 1 + sum(len(l[1]) for l in layers) + 1
                                            cand = dict(L=Lg, ax=float(d[2]), lat=float(d @ ka), n=npart, ht=ht, comp=comp, layers=layers,
                                                        top=(ti, tn), ft=ft, fn=fn, c=c, fs=fs, ss=ss)
                                            # 長さ 3 mm 以内を最優先、その中で膝の高さのずれ（5 mm 刻み）・部品の数・長さの差。3 mm 以内が無ければ長さの差
                                            key = lambda x: ((0, round(abs(x['ax']) / 5.0), x['n'], abs(x['L'] - L)) if abs(x['L'] - L) <= 3.0
                                                             else (1, abs(x['L'] - L), round(abs(x['ax']) / 5.0), x['n']))
                                            b_ = round(Lg)
                                            if b_ not in seen or (abs(cand['ax']), cand['n']) < (abs(seen[b_]['ax']), seen[b_]['n']):
                                                seen[b_] = cand
                                            if best is None or key(cand) < key(best):
                                                best = cand
            log(f'  {ht} まで: 最良 {best and round(best["L"], 1)} mm・軸方向のずれ {best and round(best["ax"], 1)} mm・部品 {best and best["n"]}')
        near = sorted(seen.values(), key=lambda x: abs(x['L'] - L))[:8]
        log('  長さの近い候補（てこ mm・高さのずれ mm・部品）: ' + str([(round(x['L'], 1), round(x['ax'], 1), x['n']) for x in near]))
        return best

    if QUAD == 'x42':
        tbx = {b.get('name'): b for b in ET.parse(TPL).getroot().iter('body')}
        v_th = {k: vec(tbx[str(k)].find('body').get('pos')) for k in range(1, 5)}         # 近似の股 → 膝（体 k の子の pos）
        v_sh = {k: vec(tbx[f'1{k}'].find('geom').get('fromto'))[3:] for k in range(1, 5)}  # 膝 → 脛のカプセルの終点
        L_t = float(np.linalg.norm(v_th[1]))
        log(f'x42: 近似の腿 {L_t:.1f} mm・脛のカプセルの終点まで {np.linalg.norm(v_sh[1]):.1f} mm')
        lk = search_link_orth(L_t)
        assert lk is not None, '腿: 直交の組み方が無い'
        log(f'腿: 目標 {L_t:.1f} → てこ {lk["L"]:.1f} mm（高さのずれ {lk["ax"]:+.1f} mm・膝のホーンの横のずれ {lk["lat"]:+.1f} mm）Hinge {lk["ht"]} 段 {lk["comp"] or "なし"} Frame {lk["ft"]}'
            f'（{lk["fn"]}・{lk["c"]}・{lk["fs"]}↔{lk["ss"]}）部品 {lk["n"]}')
        row = {'軸間距離_mm': f'{lk["L"]:.1f}', '段': lk['comp'] or 'なし'}
        # 股の付け方の候補（象限ごと）
        studs = pts('Frame_Bottom_Plug', 'PC1')
        mounts = {k: [] for k in range(1, 5)}
        for hn in holes:
            for fp in corners('Frame_Bottom_Plug', 'PC1'):
                for c in CF:
                    r = rel(PLATE, hn, 'Frame_Bottom_Plug', fp, c)
                    if r is None:
                        continue
                    MF = Mpl @ r
                    fw = np.array([wp(MF, 'Frame_Bottom_Plug', n) for n in studs])
                    if not all(np.min(np.linalg.norm(hole_w - f, axis=1)) < TOL for f in fw):
                        continue
                    for ss in ('bottom-side2', 'left-side2'):
                        r2 = rel('Frame_Bottom_Plug', 'connect_point', 'XL-430', ss, 'fixed')
                        if r2 is None:
                            continue
                        MS = MF @ r2
                        if (MS[:3, :3] @ hz)[2] > -0.999:
                            continue                   # ホーンが真下
                        h = MS[:2, 3] - center[:2]
                        sv = Shape('XL-430', P).pts @ MS[:3, :3].T + MS[:3, 3]
                        fr = Shape('Frame_Bottom_Plug', P).pts @ MF[:3, :3].T + MF[:3, 3]
                        box = np.vstack([sv, fr])
                        lo, hi = box.min(0)[:2] - center[:2], box.max(0)[:2] - center[:2]
                        for k, (qx, qy) in enumerate(HIPS['x42'], start=1):
                            # サーボとフレームがまるごと自分の象限に入る（中心の線から 0.5 mm 離す）→ 4 脚のどの組み合わせでも重ならない
                            inx = lo[0] >= 0.5 if qx > 0 else hi[0] <= -0.5
                            iny = lo[1] >= 0.5 if qy > 0 else hi[1] <= -0.5
                            if inx and iny:
                                mounts[k].append((float(np.linalg.norm(h)), hn, fp, c, ss, MF, MS, box.min(0), box.max(0)))
        for k in mounts:
            mounts[k].sort(key=lambda m: m[0])
            mounts[k] = mounts[k][:40]
            log(f'  脚 {k}: 股の付け方の候補 {len(mounts[k])}（中心からの距離 最小 {mounts[k][0][0]:.1f} mm）')

        def apart(a, b):
            return np.any(a[8] <= b[7] + 0.5) or np.any(b[8] <= a[7] + 0.5)   # 箱が離れている（0.5 mm の隙間）

        best4 = None
        for m1 in mounts[1]:
            for m2 in mounts[2]:
                if not apart(m1, m2) or (best4 and max(m1[0], m2[0]) >= best4[0]):
                    continue
                for m3 in mounts[3]:
                    if not (apart(m1, m3) and apart(m2, m3)) or (best4 and max(m1[0], m2[0], m3[0]) >= best4[0]):
                        continue
                    for m4 in mounts[4]:
                        if not (apart(m1, m4) and apart(m2, m4) and apart(m3, m4)):
                            continue
                        sc = max(m1[0], m2[0], m3[0], m4[0])
                        if best4 is None or sc < best4[0]:
                            best4 = (sc, [m1, m2, m3, m4])
        assert best4, '股のサーボ 4 個を重ねずに置けない'
        log(f'股のサーボ 4 個: 中心からの距離の最大 {best4[0]:.1f} mm（{[round(m[0], 1) for m in best4[1]]}）')

        # 腿を lk で組み、0° の姿勢にそろえて、9 姿勢の食い込みで腿・脛の Hinge の向きを選ぶ
        def chain_x42(MS, ch, cs):
            r0 = rel('XL-430', 'horn', lk['ht'], 'side_connect_point', ch)
            if r0 is None:
                return None
            MH = MS @ r0
            sp = stack_plan(MH, lk['ht'], lk['comp'])
            if sp is None:
                return None
            layers, tops, _ = sp
            Mt, tt = tops[lk['top'][0]]
            r1 = rel(tt, lk['top'][1], lk['ft'], lk['fn'], lk['c'])
            if r1 is None:
                return None
            MF = Mt @ r1
            MK = MF @ rel(lk['ft'], lk['fs'], 'XL-430', lk['ss'], 'fixed')
            thigh = [(MH, lk['ht'])] + [(x[4], lt) for lt, lay in layers for x in lay] + [(MF, lk['ft']), (MK, 'XL-430')]
            r2 = rel('XL-430', 'horn', TIP_HINGE, 'side_connect_point', cs)
            if r2 is None:
                return None
            MT = MK @ r2
            spt = stack_plan(MT, TIP_HINGE, 'B' * shin_n)
            if spt is None:
                return None
            return thigh, [(MT, TIP_HINGE)] + [(x[4], lt) for lt, lay in spt[0] for x in lay], MK

        allm = [(m[6], 'XL-430') for m in best4[1]] + [(m[5], 'Frame_Bottom_Plug') for m in best4[1]]
        for leg, m in enumerate(best4[1], start=1):
            _, hn, fp, c, ss, MFm, MS, _, _ = m
            oh, ah = MS[:3, 3], MS[:3, :3] @ hz
            cand = []
            for ch in CF:
                for cs in CF:
                    chn = chain_x42(MS, ch, cs)
                    if chn is None:
                        continue
                    thigh, shin, MK = chn
                    dv = MK[:3, 3] - oh
                    ka = MK[:3, :3] @ hz
                    Lg_, nl = lever(dv, ah, ka)
                    if abs(Lg_ - lk['L']) > 0.2 or abs(ka @ ah) > 0.001:
                        continue
                    th = signed(nl, v_th[leg], ah)      # てこの向きを近似の腿の向きへ
                    M = rot_about(oh, ah, th)
                    thigh = [(M @ mm, t) for mm, t in thigh]
                    shin = [(M @ mm, t) for mm, t in shin]
                    MK2 = M @ MK
                    ok_, ak = MK2[:3, 3], MK2[:3, :3] @ hz
                    w = wpts(shin) - ok_
                    rad = np.linalg.norm(w - np.outer(w @ ak, ak), axis=1)
                    sh = signed(w[rad > rad.max() - 5.0].mean(0), v_sh[leg], ak)
                    M2 = rot_about(ok_, ak, sh)
                    shin = [(M2 @ mm, t) for mm, t in shin]
                    base = [(Mpl, PLATE)] + allm           # 胴と 4 個の股のサーボ・フレーム（自分の股も）
                    worst = 0
                    for hdeg in (-30, 0, 30):
                        Mh_ = rot_about(oh, ah, hdeg)
                        th_ = [(Mh_ @ mm, t) for mm, t in thigh]
                        ko, kax = Mh_[:3, :3] @ ok_ + Mh_[:3, 3], Mh_[:3, :3] @ ak
                        for kdeg in (-30, 0, 30):
                            Mk_ = rot_about(ko, kax, kdeg)
                            sh_ = [(Mk_ @ (Mh_ @ mm), t) for mm, t in shin]
                            worst = max(worst, overlap(sh_, th_), overlap(th_[1:] + sh_, base))
                    cand.append((worst, abs(sh), ch, cs, th, sh))
            assert cand, f'脚 {leg}: 腿が組める向きが無い'
            cand.sort(key=lambda x: (x[0], x[1]))
            worst, _, ch, cs, th, sh = cand[0]
            F = ru.makeParts('Frame_Bottom_Plug')
            assert R.attach(F, f'{F.name}/{fp}', f'{PL.name}/{hn}', c)
            S = ru.makeParts('XL-430')
            assert R.attach(S, f'{S.name}/{ss}', f'{F.name}/connect_point', 'fixed')
            parts += [dict(name=f'脚{leg}の股のフレーム', type='Frame_Bottom_Plug', p=F, leg=leg, seg='mount'),
                      dict(name=f'脚{leg}の股のサーボ', type='XL-430', p=S, leg=leg, seg='mount')]
            H = ru.makeParts(lk['ht'])
            assert R.attach(H, H.name + '/side_connect_point', S.name + '/horn', ch)
            parts.append(dict(name=f'脚{leg}の腿の Hinge', type=lk['ht'], p=H, leg=leg, seg='thigh'))
            sp = stack_plan(pose(H), lk['ht'], lk['comp'])
            tops = build_layers([H], sp[0], f'脚{leg}の腿', leg, 'thigh') if sp[0] else [H]
            Fk = ru.makeParts(lk['ft'])
            assert R.attach(Fk, f'{Fk.name}/{lk["fn"]}', f'{tops[lk["top"][0]].name}/{lk["top"][1]}', lk['c'])
            K = ru.makeParts('XL-430')
            assert R.attach(K, f'{K.name}/{lk["ss"]}', f'{Fk.name}/{lk["fs"]}', 'fixed')
            parts += [dict(name=f'脚{leg}の腿の Frame', type=lk['ft'], p=Fk, leg=leg, seg='thigh'),
                      dict(name=f'脚{leg}の膝のサーボ', type='XL-430', p=K, leg=leg, seg='thigh')]
            T = ru.makeParts(TIP_HINGE)
            assert R.attach(T, T.name + '/side_connect_point', K.name + '/horn', cs)
            parts.append(dict(name=f'脚{leg}の脛の Hinge', type=TIP_HINGE, p=T, leg=leg, seg='shin'))
            if shin_n:
                spt = stack_plan(pose(T), TIP_HINGE, 'B' * shin_n)
                build_layers([T], spt[0], f'脚{leg}の脛', leg, 'shin')
            Mh, Mk = pose(S), pose(K)
            dv = Mk[:3, 3] - Mh[:3, 3]
            log(f'脚 {leg}: 股のホーン（胴の中心から）{np.round(Mh[:3, 3] - center, 1).tolist()} mm・腿のてこ '
                f'{lever(dv, ah, Mk[:3, :3] @ hz)[0]:.1f} mm・膝のホーンの横のずれ {dv @ (Mk[:3, :3] @ hz):+.1f} mm・高さ {dv @ ah:+.1f} mm・腿の Hinge {ch}・脛の Hinge {cs}・0° にそろえる回転 腿 {th:+.1f}°・脛 {sh:+.1f}°・'
                f'9 姿勢の食い込みの最大 {worst} mm³（候補 {len(cand)}）')
            joints.append(dict(leg=leg, hip=Mh.tolist(), knee=Mk.tolist(), overlap_mm3=worst))

    out = dict(quad=QUAD, plate=PLATE, center_mm=center.tolist(), thigh=dict(table_mm=float(row['軸間距離_mm']), 段=row['段']),
               shin=dict(hinge=TIP_HINGE, blocks=shin_n, reach_mm=shin_reach), joints=joints, parts=[])
    for d in parts:
        out['parts'].append(dict(name=d['name'], type=d['type'], T=pose(d['p']).tolist(), leg=d['leg'], seg=d['seg']))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1)
    m = sum(P[d['type']]['mass-param']['mass'] for d in parts)
    log(f'書き出した: {OUT}  部品 {len(parts)}  質量の合計 {m:.1f} g  試した attach {len(cache)}')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)
