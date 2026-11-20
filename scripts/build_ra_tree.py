"""関節の表（位置・軸の向き・どの鎖の何番目か）から、研究室の部品（RobotAssembler の IRSL 部品・XL430）で組む汎用版（軸6、2026-11-20、
ユーザー「robot assembler で 3D 形状とチャットで指定した軸の向きを元に robot を組むときの手順…こっちで使えるようにしたい」→ C）。

`build_quad_ra.py`（四脚専用。股の位置・腿と脛の 2 関節・付け方が QUAD ごとの分岐）の手順を、形に依らない入力で回す。
手順の図は docs/研究応用/設計/部品での自動組み立ての手順.drawio、説明は同じ名前の md。

入力 SPEC（JSON、長さは mm、胴の板の中心が原点・z が上）。手で書くか `ra_spec_from_xml.py` で XML から作る:
  {"name": "...", "torso": {"plate": "plate_16_16", "rail": "plate_4_8"（空なら板の下に直接吊る）},
   "range_deg": 30, "tip_hinge": "Hinge_Short_Plug", "link_names": ["thigh", "shin"]（省略で link1, link2, …）,
   "chains": [ {"name": "脚1",
                "joints": [{"pos": [x, y, z], "axis": "+y" または [ax, ay, az]}, …（根元 → 先端の順）],
                "tip": [x, y, z],                 先端の外形の点
                "mount_target_xy": [x, y],        省略可（省略なら最初の関節と先端の面から決める）
                "lengths_mm": [..], "tip_mm": ..  省略可（省略なら位置から出す）}, …]}

手順（図の ①〜⑦）:
  ① 軸を ±x / ±y / ±z に丸める（部品は直角にしかつながらない）。丸めた角度を出力する。
     鎖の中の隣り合う軸は平行でなければ止める（直交の組み方は四脚の x42 で組めなかった。系譜 9-342）
  ② 関節の間の長さ（軸に垂直な距離）・先端までの長さ・0° の向き（関節 → 次の関節、最後は → 先端）を出す
  ③ 部品どうしの相対位置を RobotAssembler の attach で 1 回ずつ試して表にする（rel。以降は行列の掛け算）
  ④ リンクの型: 組み方の表（irsl_link_all_v2.csv、9-325 の系統）から、系統ごとに長さがいちばん近い行。先端は Hinge ＋ ブロック 0〜8 段
  ⑤ 根元: 胴の板（か RAIL）の穴 × Frame_Long_Plug の角 × 付け方 4 通り × サーボの点を総当たりし、
     ポッチがすべて穴に入る・ホーンの軸が指定の向き・サーボが板の下、を満たすもののうち、狙う xy にいちばん近いもの（＋0.5 mm 以内）
  ⑥ 鎖ごとに 系統 × 根元の候補 × 各 Hinge の付け方（4^関節の数）を試す。0° の姿勢を ② の向きにそろえ、
     各関節を −range / 0 / +range に振った 3^関節の数 の姿勢で、部品が別の部品の芯（表面から 1 mm 内側）に入る体積の最大を数え、
     少ない順（同じなら 2 番目以降の関節の回転が小さい順）に選ぶ。接している面のぶん（400 mm³ 未満）の候補が出た系統で止める
  ⑦ 本物の attach で組み直し、関節の間の長さが表と 0.2 mm 以内・軸が平行を確かめる。JSON と .roboasm に書き、
     .roboasm を読み戻して書き直し、部品の名前ごとにつなぎ方と回転が一致するかを確かめる（CHECK_ONLY=<.roboasm> でこれだけ回す）

`build_quad_ra.py` との違い: 根元のフレーム（Frame_Long_Plug）は RAIL が無くても食い込みの相手に入れる（build_quad_ra.py は RAIL のときだけ）。
LEGACY_BASE=1 で build_quad_ra.py と同じにする（組み直して一致するかの確かめ用）。

    SPEC=data/axis6/tree/quad_walk_ra235_spec.json OUT=data/axis6/tree/quad_walk_ra235_parts.json \\
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/build_ra_tree.py > ログ 2>&1
出力: OUT（build_quad_ra.py と同じ形の JSON。関節が 2 つなら joints に hip・knee も書くので make_quad_ra_xml.py に通る）と
      ROBOASM（既定は OUT の _parts.json を .roboasm に替えた名前。RobotAssembler の GUI で開ける）
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
SPEC = os.environ.get('SPEC', '')
OUT = os.environ.get('OUT', '')
ROBOASM = os.environ.get('ROBOASM', '')
TABLE = os.environ.get('TABLE', '/userdir/StackelbergPPO/data/axis6/irsl_link_all_v2.csv')
LEGACY_BASE = os.environ.get('LEGACY_BASE', '0') == '1'
HINGE_IN = 19.0                 # Hinge の side_connect_point から Hinge の中心（リンクの中心線）まで、ホーンの軸に沿って（mm）
CF = ['fixed', 'invert', 'rotate90', 'rotate-90']
TOL = 0.5
OK_MM3 = 400                    # 接している面のぶん（約 250 mm³）だけとみなす上限
SERVO_PTS = ['right-side', 'left-side', 'bottom-side1', 'bottom-side2', 'bottom-side3', 'left-side2']
FAMS = [('connect_point', 'right-side'), ('connect_point', 'left-side'), ('connect_point2', 'left-side2'), ('connect_point2', 'bottom-side2')]
AXES = {'+x': (1, 0, 0), '-x': (-1, 0, 0), '+y': (0, 1, 0), '-y': (0, -1, 0), '+z': (0, 0, 1), '-z': (0, 0, -1)}


def log(*a):
    print(*a, flush=True)


def snap(axis):
    """軸を ±x / ±y / ±z に丸める。戻り値: 丸めた単位ベクトル・元の向きとの角度（度）"""
    if isinstance(axis, str):
        return np.array(AXES[axis], float), 0.0
    v = np.array(axis, float)
    v /= np.linalg.norm(v)
    i = int(np.argmax(np.abs(v)))
    s = np.zeros(3)
    s[i] = np.sign(v[i])
    return s, float(np.degrees(np.arccos(min(1.0, abs(v[i])))))


def derive(spec):
    """① ② 入力の表から、鎖ごとの丸めた軸・ホーンの向き・長さ・0° の向き・狙う xy を出す（Choreonoid は要らない）"""
    out = []
    for c in spec['chains']:
        J = [np.array(j['pos'], float) for j in c['joints']]
        A, ang = zip(*[snap(j['axis']) for j in c['joints']])
        for i in range(1, len(A)):
            if abs(A[i] @ A[0]) < 0.999:
                raise SystemExit(f'{c["name"]}: 関節 {i + 1} の軸 {A[i]} が関節 1 の軸 {A[0]} と平行でない。'
                                 '平行な軸の鎖だけを扱う（直交の組み方は四脚の x42 で組めなかった。系譜 9-342）')
        a = A[0]
        tip = np.array(c['tip'], float)
        h = a if J[0] @ a >= 0 else -a                       # ホーンは胴の外を向く側
        perp = lambda d: float(np.linalg.norm(d - (d @ a) * a))
        L = c.get('lengths_mm') or [perp(J[i + 1] - J[i]) for i in range(len(J) - 1)]
        tip_mm = c.get('tip_mm') or perp(tip - J[-1])
        dirs = [J[i + 1] - J[i] for i in range(len(J) - 1)] + [tip - J[-1]]
        if 'mount_target_xy' in c:
            tgt = np.array(c['mount_target_xy'], float)
        else:
            tgt = (J[0] + ((tip - J[0]) @ h + HINGE_IN) * h)[:2]   # ホーンを、先端の面から HINGE_IN だけ外に置く
        out.append(dict(name=c['name'], J=J, axis=a, horn=h, snap_deg=list(ang), L=list(L), tip_mm=float(tip_mm),
                        dirs=dirs, target=tgt))
    return out


def main():
    import cnoid.RobotAssembler as cra
    from joint_range_a1v import Shape, rot_about
    spec = json.load(open(SPEC))
    out_path = OUT or f'/userdir/StackelbergPPO/data/axis6/tree/{spec["name"]}_parts.json'
    ra_path = ROBOASM or out_path.removesuffix('_parts.json').removesuffix('.json') + '.roboasm'
    PLATE = spec['torso']['plate']
    RAIL = spec['torso'].get('rail', '')
    RANGE = float(spec.get('range_deg', 30))
    TIP_HINGE = spec.get('tip_hinge', 'Hinge_Short_Plug')
    chains = derive(spec)
    N = len(chains[0]['J'])
    names = spec.get('link_names') or [f'link{i + 1}' for i in range(N)]
    for ch in chains:
        log(f'① ② {ch["name"]}: 軸 {ch["axis"].tolist()}（丸めた角度 {np.round(ch["snap_deg"], 1).tolist()}°）・ホーン {ch["horn"].tolist()}・'
            f'関節の間 {np.round(ch["L"], 2).tolist()} mm・先端まで {ch["tip_mm"]:.2f} mm・狙う xy {np.round(ch["target"], 2).tolist()}')
        if len(ch['J']) != N:
            raise SystemExit('鎖ごとに関節の数が違う（今は同じ数だけ扱う）')

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

    # ---- ③ 部品どうしの相対位置（attach を 1 回ずつ試してキャッシュ）----
    cache = {}

    def rel(bt, bp, nt, np_, c):
        k = (bt, bp, nt, np_, c)
        if k not in cache:
            Rt = ru.makeRobotFromKey(bt)
            b = Rt.rootParts()
            Np = ru.makeParts(nt)
            ok = Rt.attach(Np, f'{Np.name}/{np_}', f'{b.name}/{bp}', c)
            cache[k] = np.linalg.inv(pose(b)) @ pose(Np) if ok else None
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

    # ---- LEGO の段の積み方（build_arm3.py・build_quad_ra.py と同じ。基準 ② 全部のポッチを覆う ③ 継ぎ目を直交）----
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
                Mb, tb_ = bases[bi]
                for npn in corners(ltype, in_kind):
                    for c in CF:
                        r = rel(tb_, bn, ltype, npn, c)
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

    # ---- 胴 ----
    R = ru.makeRobotFromKey(PLATE)
    PL = R.rootParts()
    parts = [dict(name='胴の板', type=PLATE, p=PL, leg=0, seg='torso')]
    Mpl = pose(PL)
    q = np.array([CPT[PLATE][n][0] for n in pts(PLATE, 'PC1')])
    center = np.array([q[:, 0].mean(), q[:, 1].mean(), -1.6])          # 板の厚みの中央
    half = (q[:, :2].max(0) - q[:, :2].min(0)) / 2 + 4.0                # ポッチの端 ＋ 半ピッチ ＝ 板の縁（plate_16_16 で 64 mm）
    holes = pts(PLATE, 'SC1')
    hole_w = np.array([wp(Mpl, PLATE, n) for n in holes])
    log(f'胴 {PLATE}: 中心（板の座標）{center.tolist()} mm・縁まで {half.tolist()} mm・質量 {P[PLATE]["mass-param"]["mass"]:.1f} g')

    def build_layers(base_parts, layers, tag, leg, seg):
        cur = base_parts
        for li, (lt, lay) in enumerate(layers, start=1):
            new = []
            for k, (bi, bn, npn, c, _) in enumerate(lay):
                Np = ru.makeParts(lt)
                assert R.attach(Np, f'{Np.name}/{npn}', f'{cur[bi].name}/{bn}', c)
                parts.append(dict(name=f'{tag}の LEGO {li} 段目 {k}', type=lt, p=Np, leg=leg, seg=seg))
                new.append(Np)
            cur = new
        return cur

    # ---- ④ リンクの型（組み方の表）と先端の段数 ----
    allrows = [r for r in csv.DictReader(open(TABLE)) if r['Hinge'] == 'Hinge_Short_Plug' and r['Frame'] == 'Frame_Long_Socket']

    def fam_row(fam, L):
        rs = [r for r in allrows if r['Frame→サーボ'] == fam[0] and r['サーボの点'] == fam[1]]
        return min(rs, key=lambda r: (abs(float(r['軸間距離_mm']) - L), int(r['部品の数'])))

    def comp_of(row):
        return '' if row['段'] == 'なし' else row['段']

    tip_cache = {}

    def tip_reach(n):
        if n not in tip_cache:
            Mh = rel('XL-430', 'horn', TIP_HINGE, 'side_connect_point', 'fixed')
            sp = stack_plan(Mh, TIP_HINGE, 'B' * n)
            if sp is None:
                tip_cache[n] = None
            else:
                Ms = [(Mh, TIP_HINGE)] + [(x[4], lt) for lt, lay in sp[0] for x in lay]
                w = np.vstack([Shape(t, P).pts @ M[:3, :3].T + M[:3, 3] for M, t in Ms])
                tip_cache[n] = float(np.max(np.linalg.norm(w[:, :2], axis=1)))     # ホーンの軸は z（サーボの座標）
        return tip_cache[n]

    hz = np.array([0, 0, 1.0])

    def wpts(lst):
        return np.vstack([Shape(t, P).pts @ M[:3, :3].T + M[:3, 3] for M, t in lst])

    def overlap(moving, static):
        """moving の部品の点が static の部品の芯（表面から 1 mm 削った内側）に入る体積（mm³）。joint_range_a1v.py と同じ判定"""
        n = 0
        for Ms, ts in static:
            sh = Shape(ts, P)
            Ti = np.linalg.inv(Ms)
            for Mm, tm in moving:
                qq = Shape(tm, P).pts @ (Ti[:3, :3] @ Mm[:3, :3]).T + (Ti[:3, :3] @ Mm[:3, 3] + Ti[:3, 3])
                g = np.round(qq).astype(int) - sh.lo
                ok = np.all((g >= 0) & (g < np.array(sh.core.shape)), axis=1)
                n += int(sh.core[tuple(g[ok].T)].sum())
        return n

    def signed(u, w, a):
        u = u - (u @ a) * a
        w = w - (w @ a) * a
        return float(np.degrees(np.arctan2(np.cross(u, w) @ a, u @ w)))

    def chain_parts(MS, orients, rows, tip_n):
        """根元のサーボの姿勢 MS から、各 Hinge を向き orients で付けた鎖の部品（部分ごと）と各関節のサーボの姿勢（付けずに計算だけ）"""
        segs, servos, M = [], [MS], MS
        for i, row in enumerate(rows):
            r0 = rel('XL-430', 'horn', row['Hinge'], 'side_connect_point', orients[i])
            if r0 is None:
                return None
            MH = M @ r0
            sp = stack_plan(MH, row['Hinge'], comp_of(row))
            if sp is None:
                return None
            layers = sp[0]
            tops = [(MH, row['Hinge'])] if not layers else [(x[4], layers[-1][0]) for x in layers[-1][1]]
            Mt, tt = tops[int(row['最上段の部品'])]
            r1 = rel(tt, row['最上段の点'], row['Frame'], row['Frame の点'], row['つなぎ方'])
            if r1 is None:
                return None
            MF = Mt @ r1
            MK = MF @ rel(row['Frame'], row['Frame→サーボ'], 'XL-430', row['サーボの点'], 'fixed')
            segs.append([(MH, row['Hinge'])] + [(x[4], lt) for lt, lay in layers for x in lay] + [(MF, row['Frame']), (MK, 'XL-430')])
            servos.append(MK)
            M = MK
        r2 = rel('XL-430', 'horn', TIP_HINGE, 'side_connect_point', orients[-1])
        if r2 is None:
            return None
        MT = M @ r2
        spt = stack_plan(MT, TIP_HINGE, 'B' * tip_n)
        if spt is None:
            return None
        segs.append([(MT, TIP_HINGE)] + [(x[4], lt) for lt, lay in spt[0] for x in lay])
        return segs, servos

    def tip_point(seg, o, a):
        w = wpts(seg) - o
        rad = np.linalg.norm(w - np.outer(w @ a, a), axis=1)
        return w[rad > rad.max() - 5.0].mean(0)        # 先端の面の中央（最も遠い 1 点はブロックの角で振れる）

    def posed_score(segs, servos, rows, dirs, base):
        """⑥ 0° の姿勢を dirs にそろえ、各関節を −RANGE / 0 / +RANGE に振った全姿勢で、食い込みの最大（mm³）"""
        for i, row in enumerate(rows):
            o, a = servos[i][:3, 3], servos[i][:3, :3] @ hz
            dv = servos[i + 1][:3, 3] - o
            if abs(float(np.linalg.norm(dv - (dv @ a) * a)) - float(row['軸間距離_mm'])) > 0.2 or abs((servos[i + 1][:3, :3] @ hz) @ a) < 0.999:
                return None
        segs = [list(s) for s in segs]
        sv = [m.copy() for m in servos]
        rots = []
        for i in range(len(segs)):
            o, a = sv[i][:3, 3], sv[i][:3, :3] @ hz
            cur = sv[i + 1][:3, 3] - o if i < len(rows) else tip_point(segs[i], o, a)
            ang = signed(cur, dirs[i], a)
            rots.append(ang)
            M = rot_about(o, a, ang)
            for j in range(i, len(segs)):
                segs[j] = [(M @ m, t) for m, t in segs[j]]
            for j in range(i + 1, len(sv)):
                sv[j] = M @ sv[j]
        worst = 0
        for degs in itertools.product((-RANGE, 0.0, RANGE), repeat=len(segs)):
            S = [list(s) for s in segs]
            V = [m.copy() for m in sv]
            for i, d in enumerate(degs):
                M = rot_about(V[i][:3, 3], V[i][:3, :3] @ hz, d)
                for j in range(i, len(S)):
                    S[j] = [(M @ m, t) for m, t in S[j]]
                for j in range(i + 1, len(V)):
                    V[j] = M @ V[j]
            w = 0
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    w = max(w, overlap(S[j], S[i]))
            moving = S[0][1:] + [x for s in S[1:] for x in s]
            # 最初の Hinge は根元のサーボのホーンに接して付くので、根元のサーボを除いた胴の側とだけ見る
            w = max(w, overlap(moving, base), overlap(S[0][:1], base[:1] + base[2:]))
            worst = max(worst, w)
        return worst, rots

    # ---- ⑤ 根元: 胴の板（か RAIL）に Frame_Long_Plug でサーボを吊る ----
    def servo_mounts(host_M, host_t, host_holes, h, target, avoid):
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
                        continue                       # フレームのポッチがすべて穴に入る
                    for fs in ('connect_point', 'connect_point2'):
                        for ss in SERVO_PTS:
                            r2 = rel('Frame_Long_Plug', fs, 'XL-430', ss, 'fixed')
                            if r2 is None:
                                continue
                            MS = MF @ r2
                            if (MS[:3, :3] @ hz) @ h < 0.999:
                                continue               # ホーンの軸が指定の向き
                            sv = Shape('XL-430', P).pts @ MS[:3, :3].T + MS[:3, 3]
                            if sv[:, 2].max() > center[2] - 1.6 + TOL:
                                continue               # サーボは板の下
                            d = float(np.linalg.norm(MS[:2, 3] - target))
                            if avoid and overlap([(MS, 'XL-430')], avoid):
                                continue
                            out.append((d, hn, fp, c, fs, ss, MS, MF))
        return out

    rail_cache = {}

    def rails_for(h):
        """胴の板の下に RAIL を付け、ホーンの向き h の側面の外へはみ出させる付け方。板の下に入るポッチはすべて穴に入り、8 個以上"""
        key_h = tuple(h)
        if key_h in rail_cache:
            return rail_cache[key_h]
        if abs(h[2]) > 0.5:
            raise SystemExit('ホーンが上下を向く関節に RAIL は使えない（側面の外へ出す板）')
        rp_all = pts(RAIL, 'PC1')
        need = float(np.abs(h[:2]) @ half)
        out, seen = [], set()
        for hn in holes:
            for rp in corners(RAIL, 'PC1'):
                for c in CF:
                    r = rel(PLATE, hn, RAIL, rp, c)
                    if r is None:
                        continue
                    MR = Mpl @ r
                    rw = np.array([wp(MR, RAIL, n) for n in rp_all]) - center
                    inside = np.all(np.abs(rw[:, :2]) < half, axis=1)
                    if inside.sum() < 8 or inside.all():
                        continue
                    if not all(np.min(np.linalg.norm(hole_w - (qq + center), axis=1)) < TOL for qq in rw[inside]):
                        continue
                    if not np.all(rw[~inside, :2] @ h[:2] >= need):
                        continue                       # はみ出すのはホーンの向きの側面の外だけ
                    key = (tuple(np.round(MR[:3, 3], 1)), tuple(np.round(MR[:3, :3].ravel(), 2)))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append((hn, rp, c, MR))
        log(f'{RAIL} をホーンの向き {h.tolist()} の側面の外へはみ出させる付け方: {len(out)}')
        rail_cache[key_h] = out
        return out

    joints = []
    for leg, ch in enumerate(chains, start=1):
        h = ch['horn']
        target = center[:2] + ch['target']
        rows_f = [[fam_row(f, L) for L in ch['L']] for f in FAMS]
        cands_tip = [(abs(tip_reach(n) - ch['tip_mm']), n) for n in range(0, 9) if tip_reach(n) is not None]
        _, tip_n = min(cands_tip)
        log(f'④ {ch["name"]}: リンク {[(r["軸間距離_mm"], r["段"]) for r in rows_f[0]]}（系統 {FAMS[0]}）・'
            f'先端 {TIP_HINGE} ＋ ブロック {tip_n} 段（{tip_reach(tip_n):.1f} mm、目標 {ch["tip_mm"]:.1f}）')
        if RAIL:
            rp_all = pts(RAIL, 'PC1')
            p_ = np.cross([0, 0, 1.0], h)
            mounts = []
            for rhn, rp, rc, MR in rails_for(h):
                rc_w = MR[:3, :3] @ np.mean([CPT[RAIL][n][0] for n in rp_all], 0) + MR[:3, 3]
                if abs((rc_w - np.r_[target, 0]) @ p_) > 40:
                    continue                           # RAIL の中心が狙いから（側面に沿って）遠い
                for m in servo_mounts(MR, RAIL, pts(RAIL, 'SC1'), h, target, [(Mpl, PLATE), (MR, RAIL)]):
                    mounts.append(m + ((rhn, rp, rc, MR),))
        else:
            mounts = [m + (None,) for m in servo_mounts(Mpl, PLATE, holes, h, target, [])]
        if not mounts:
            raise SystemExit(f'{ch["name"]}: 根元の付け方が無い（ホーンの向き {h.tolist()}）')
        dmin = min(m[0] for m in mounts)
        near = [m for m in mounts if m[0] <= dmin + 0.5]
        cand = []
        for fi, rows in enumerate(rows_f):
            for mi, (d, hn, fp, c, fs, ss, MS, MF, rl) in enumerate(near):
                base = [(Mpl, PLATE), (MS, 'XL-430')]
                if rl or not LEGACY_BASE:
                    base += [(MF, 'Frame_Long_Plug')]
                if rl:
                    base += [(rl[3], RAIL)]
                for orients in itertools.product(CF, repeat=N):
                    cp = chain_parts(MS, orients, rows, tip_n)
                    if cp is None:
                        continue
                    sc = posed_score(cp[0], cp[1], rows, ch['dirs'], base)
                    if sc is None:
                        continue
                    cand.append((sc[0], sum(abs(r) for r in sc[1][1:]), d, mi, orients, sc[1], fi))
            if cand and min(x[0] for x in cand) < OK_MM3:
                break
        if not cand:
            raise SystemExit(f'{ch["name"]}: 表どおりのリンクが組める付け方が無い')
        cand.sort(key=lambda x: (x[0], x[1]))
        log(f'⑥ {ch["name"]}: 候補 {len(cand)}（根元 {len(near)} × Hinge の向き {4 ** N}）。食い込みの最大（mm³）の少ない順: '
            f'{[(x[0], FAMS[x[6]], x[4], np.round(x[5], 0).tolist()) for x in cand[:4]]}')
        worst, _, d, mi, orients, rots, fi = cand[0]
        _, hn, fp, c, fs, ss, _, _, rl = near[mi]
        rows = rows_f[fi]
        host = PL
        if rl:
            RLp = ru.makeParts(RAIL)
            assert R.attach(RLp, f'{RLp.name}/{rl[1]}', f'{PL.name}/{rl[0]}', rl[2])
            parts.append(dict(name=f'{ch["name"]}の根元の板（側面の外へ）', type=RAIL, p=RLp, leg=leg, seg='mount'))
            host = RLp
        F = ru.makeParts('Frame_Long_Plug')
        assert R.attach(F, f'{F.name}/{fp}', f'{host.name}/{hn}', c)
        S = ru.makeParts('XL-430')
        assert R.attach(S, f'{S.name}/{ss}', f'{F.name}/{fs}', 'fixed')
        parts += [dict(name=f'{ch["name"]}の根元のフレーム', type='Frame_Long_Plug', p=F, leg=leg, seg='mount'),
                  dict(name=f'{ch["name"]}の関節 1 のサーボ', type='XL-430', p=S, leg=leg, seg='mount')]
        log(f'⑤ {ch["name"]}: 根元のホーン（胴の中心から）{np.round(pose(S)[:3, 3] - center, 1).tolist()} mm、狙いとのずれ {d:.1f} mm'
            f'（穴 {hn}・{fp}・{c}・{fs}↔{ss}）、Hinge の向き {orients}、0° にそろえる回転 {np.round(rots, 1).tolist()}°、'
            f'食い込みの最大 {worst} mm³、系統 {FAMS[fi]}')
        # ⑦ 本物の attach で組む
        servo, srv_poses = S, [pose(S)]
        for i, row in enumerate(rows):
            H = ru.makeParts(row['Hinge'])
            assert R.attach(H, H.name + '/side_connect_point', servo.name + '/horn', orients[i])
            parts.append(dict(name=f'{ch["name"]}の{names[i]}の Hinge', type=row['Hinge'], p=H, leg=leg, seg=names[i]))
            sp = stack_plan(pose(H), row['Hinge'], comp_of(row))
            tops = build_layers([H], sp[0], f'{ch["name"]}の{names[i]}', leg, names[i]) if sp[0] else [H]
            Fk = ru.makeParts(row['Frame'])
            assert R.attach(Fk, f'{Fk.name}/{row["Frame の点"]}', f'{tops[int(row["最上段の部品"])].name}/{row["最上段の点"]}', row['つなぎ方'])
            K = ru.makeParts('XL-430')
            assert R.attach(K, f'{K.name}/{row["サーボの点"]}', f'{Fk.name}/{row["Frame→サーボ"]}', 'fixed')
            parts += [dict(name=f'{ch["name"]}の{names[i]}の Frame', type=row['Frame'], p=Fk, leg=leg, seg=names[i]),
                      dict(name=f'{ch["name"]}の関節 {i + 2} のサーボ', type='XL-430', p=K, leg=leg, seg=names[i])]
            Ma, Mb = pose(servo), pose(K)
            dv = Mb[:3, 3] - Ma[:3, 3]
            za = Ma[:3, :3] @ hz
            Lg = float(np.linalg.norm(dv - (dv @ za) * za))
            par = abs(float((Mb[:3, :3] @ hz) @ za))
            log(f'⑦ {ch["name"]}の{names[i]}: 組んだ関節の間 {Lg:.1f} mm（表 {row["軸間距離_mm"]}・目標 {ch["L"][i]:.1f}）、軸の平行 {par:.4f}、軸方向のずれ {dv @ za:+.1f} mm')
            if abs(Lg - float(row['軸間距離_mm'])) > 0.2 or par < 0.999:
                raise RuntimeError(f'{ch["name"]}: {names[i]} が表どおりに組めていない')
            servo = K
            srv_poses.append(Mb)
        T = ru.makeParts(TIP_HINGE)
        assert R.attach(T, T.name + '/side_connect_point', servo.name + '/horn', orients[-1])
        parts.append(dict(name=f'{ch["name"]}の{names[-1]}の Hinge', type=TIP_HINGE, p=T, leg=leg, seg=names[-1]))
        if tip_n:
            spt = stack_plan(pose(T), TIP_HINGE, 'B' * tip_n)
            build_layers([T], spt[0], f'{ch["name"]}の{names[-1]}', leg, names[-1])
        j = dict(leg=leg, name=ch['name'], servos=[m.tolist() for m in srv_poses], overlap_mm3=worst, family=list(FAMS[fi]),
                 hinge_orients=list(orients), rot_to_zero_deg=rots, snap_deg=ch['snap_deg'],
                 table_mm=[float(r['軸間距離_mm']) for r in rows], tip=dict(hinge=TIP_HINGE, blocks=tip_n, reach_mm=tip_reach(tip_n)))
        if N == 2:
            j.update(hip=srv_poses[0].tolist(), knee=srv_poses[1].tolist(), thigh_family=list(FAMS[fi]))
        joints.append(j)

    out = dict(quad=spec['name'], spec=SPEC, plate=PLATE, rail=RAIL, center_mm=center.tolist(), link_names=names,
               joints=joints, parts=[])
    for d_ in parts:
        out['parts'].append(dict(name=d_['name'], type=d_['type'], T=pose(d_['p']).tolist(), leg=d_['leg'], seg=d_['seg']))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, 'w'), ensure_ascii=False, indent=1)
    m = sum(P[d_['type']]['mass-param']['mass'] for d_ in parts)
    log(f'書き出した: {out_path}  部品 {len(parts)}  質量の合計 {m:.1f} g  試した attach {len(cache)}')

    # ---- .roboasm（RobotAssembler の GUI で開ける組み立ての履歴）と読み戻しの確かめ ----
    f = cra.RoboasmFile()
    assert R.createRoboasm(f), 'createRoboasm が失敗した'
    assert f.dumpRoboasm(ra_path), 'dumpRoboasm が失敗した'
    check_roboasm(cra, ru, ra_path, len(parts), [np.array(T) for j in joints for T in j['servos']],
                  sum(P[d_['type']]['mass-param']['mass'] for d_ in parts))


def _rot(r):
    if r is None:
        return np.eye(3)
    ax = np.array(r[:3], float)
    if np.linalg.norm(ax) == 0:
        return np.eye(3)
    ax /= np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(r[3]) * K + (1 - np.cos(r[3])) * K @ K


def _hist(path):
    """部品の名前 → （型・自分の点・親・親の点・回転行列）。読み戻すと並び順（木の辿り方）と回転の書き方（+π と −π）が変わるので、名前と行列で比べる"""
    out = {}
    for x in yaml.safe_load(open(path))['history']:
        off = (x.get('connecting-offset') or {}).get('rotation')
        out[x['parts-name']] = (x['parts-type'], x.get('parts-point'), x.get('parent'), x.get('parent-point'), np.round(_rot(off), 6) + 0.0)
    return out


def check_roboasm(cra, ru, ra_path, n_parts, servos=None, mass_g=None):
    """.roboasm を RobotAssembler で読み戻し、書き直した履歴が元と同じ（名前ごとに、つなぎ方と回転が一致）かを確かめる"""
    R2 = ru.makeRobotFromFile(ra_path)
    f2 = cra.RoboasmFile()
    R2.createRoboasm(f2)
    tmp = ra_path + '.reread'
    f2.dumpRoboasm(tmp)
    a, b = _hist(ra_path), _hist(tmp)
    os.remove(tmp)
    bad = [n for n in a if n not in b or a[n][:4] != b[n][:4] or not np.allclose(a[n][4], b[n][4], atol=1e-6)]
    log(f'書き出した: {ra_path}  履歴 {len(a)} 行（部品 {n_parts}）。読み戻して書き直した履歴 {len(b)} 行・'
        f'名前ごとのつなぎ方と回転が{"すべて一致" if not bad and len(a) == len(b) else f" ⛔ {len(bad)} 行で不一致"}')
    if len(a) != n_parts or len(a) != len(b) or bad:
        raise RuntimeError(f'.roboasm の読み戻しが合わない: {bad[:5]}')
    # .roboasm から RobotAssembler に Body を作らせ、関節の数・位置・軸と質量を JSON と比べる
    body = cra.RoboasmBodyCreator(PROJ).createBodyRaw(R2)
    body.calcForwardKinematics()
    js = [body.joint(i) for i in range(body.numJoints)]
    bp = np.array([np.array(j.p) * 1000 for j in js])
    ba = np.array([np.array(j.R) @ np.array(j.jointAxis) for j in js])
    msg = f'.roboasm から作った Body: 関節 {body.numJoints}・リンク {body.numLinks}・質量 {body.mass * 1000:.1f} g'
    if servos is not None:
        dist = [float(np.min(np.linalg.norm(bp - M[:3, 3], axis=1))) for M in servos]
        par = [float(abs(ba[int(np.argmin(np.linalg.norm(bp - M[:3, 3], axis=1)))] @ M[:3, 2])) for M in servos]
        msg += (f'（JSON: 関節 {len(servos)}・質量 {mass_g:.1f} g）、関節の位置の差の最大 {max(dist):.3f} mm・軸の平行の最小 {min(par):.6f}')
        if body.numJoints != len(servos) or max(dist) > 0.01 or min(par) < 0.9999 or abs(body.mass * 1000 - mass_g) > 0.5:
            raise RuntimeError('⛔ .roboasm から作った Body が JSON と合わない: ' + msg)
    log(msg)


def check_only(path):
    import cnoid.RobotAssembler as cra
    ru = cra.RoboasmUtil(PROJ + '/irsl_assembler_config.yaml')
    check_roboasm(cra, ru, path, len(yaml.safe_load(open(path))['history']))


if __name__ == '__main__':
    try:
        if os.environ.get('CHECK_ONLY'):
            check_only(os.environ['CHECK_ONLY'])      # 既にある .roboasm の読み戻しだけを確かめる
        else:
            main()
    except SystemExit as e:
        print(f'止めた: {e}', flush=True)
        os._exit(2)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)
