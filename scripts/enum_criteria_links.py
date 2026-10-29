"""研究室の部品で、XL-430 から次の XL-430 までの 1 本のリンクに作れる長さを、軸6 §8「自動組み立ての基準」で組んだリンクだけで数え直す（系譜 9-323）。
長さ（0.1 mm 刻み）ごとに基準でいちばん良い組み方を 1 つ残し、その組み方の質量・重心を出す。9-277（1×1 の柱）と 9-316（その質量）の置き換え。

基準（`build_arm3.py` と同じ）:
  必須 ① 関節の軸どうしが平行で、軸方向のずれが 0
  必須 ② LEGO の各段は下の面のポッチをすべて覆う（2×4 ブロック 2 個、または 4×4 プレート 1 枚）
  必須 ③ ブロックの段は、すぐ下の段と継ぎ目の向きを直交させる（レンガ積み）
  同じ長さの中で ⑤ Frame と最上段の噛み合うポッチが多い → ⑥ 部品が少ない
段の組み合わせ: ブロック 0〜NB_MAX 段のあとに 4×4 プレート 0〜NP_MAX 枚。
部品の相対位置は RobotAssembler 自身の attach を小さいロボットで 1 回ずつ試して表にし、探索は行列で行う。
最後に、残した組み方のうち CHECK 本を本物の attach で組み直し、長さが一致するかを確かめる（合わなければ止める）。

⚠️ 部品どうしのめり込みは調べていない（RobotAssembler は調べない。9-316 の方法で足せる）。強度は見ていない。
⚠️ 質量は部品の設定ファイルの mass-param の和（ねじ・配線・ホーンの金具は入らない）。

    OUT=/userdir/StackelbergPPO/data/axis6/irsl_link_lengths_v2.csv \\
      PYTHONPATH=/choreonoid_ws/install/lib/choreonoid-2.3/python python3 scripts/enum_criteria_links.py > /tmp/enum2.log 2>&1
"""
import csv
import itertools
import os
import random
import sys
import time

import numpy as np
import yaml

PROJ = '/choreonoid_ws/install/share/choreonoid-2.3/robot_assembler/irsl'
OUT = os.environ.get('OUT', '/userdir/StackelbergPPO/data/axis6/irsl_link_lengths_v2.csv')
NB_MAX = int(os.environ.get('NB_MAX', '12'))
NP_MAX = int(os.environ.get('NP_MAX', '3'))
CHECK = int(os.environ.get('CHECK', '12'))
CF = ['fixed', 'invert', 'rotate90', 'rotate-90']
TOL = 0.5
HINGES = ['Hinge_Short_Plug', 'Hinge_Short_Socket', 'Hinge_Long_Plug', 'Hinge_Long_Socket']
FRAMES = ['Frame_Short_Plug', 'Frame_Short_Socket', 'Frame_Long_Plug', 'Frame_Long_Socket', 'Frame_Bottom_Plug', 'Frame_Bottom_Socket']
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
    print('##', *a, flush=True)


def main():
    t0 = time.time()
    import cnoid.RobotAssembler as cra
    cfg = yaml.safe_load(open(PROJ + '/irsl_assembler_config.yaml'))
    P = {p['type']: p for p in cfg['PartsSettings']}
    ru = cra.RoboasmUtil(PROJ + '/irsl_assembler_config.yaml')
    CPT = {t: {c['name']: (np.array(c.get('translation', [0, 0, 0]), float), set(c.get('types', [])))
               for c in P[t]['connecting-points']} for t in P}
    hz = rotmat(P['XL-430']['actuators'][0].get('rotation'))[:, 2]

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

    def loc(t, names):
        return np.array([CPT[t][n][0] for n in names])

    def wp(M, t, n):
        return M[:3, :3] @ CPT[t][n][0] + M[:3, 3]

    def corners(t, kind):
        ns = pts(t, kind)
        xy = loc(t, ns)[:, :2]
        out = []
        for sx, sy in itertools.product((min, max), repeat=2):
            x, y = sx(xy[:, 0]), sy(xy[:, 1])
            out += [n for n, q in zip(ns, xy) if abs(q[0] - x) < 1e-6 and abs(q[1] - y) < 1e-6]
        return out

    def long_axis(M, t, kind):
        q = (M[:3, :3] @ loc(t, pts(t, kind)).T).T
        u, s, vt = np.linalg.svd(q - q.mean(0))
        return vt[0] if s[0] > s[1] + 1e-6 else None

    def make_layer(bases, out_kind, ltype, prev_axis):
        in_kind = 'SC1' if out_kind == 'PC1' else 'PC1'
        grid = [(bi, n, wp(M, t, n)) for bi, (M, t) in enumerate(bases) for n in pts(t, out_kind)]
        G = np.array([g[2] for g in grid])
        need = len(grid)
        npieces = 1 if ltype == 'plate_4_4' else 2
        inl = loc(ltype, pts(ltype, in_kind))

        def fits(Mn, covered):
            w = (Mn[:3, :3] @ inl.T).T + Mn[:3, 3]
            D = np.linalg.norm(w[:, None, :] - G[None, :, :], axis=2)
            j = D.argmin(1)
            if (D[np.arange(len(w)), j] >= TOL).any() or len(set(j)) < len(j) or set(j) & covered:
                return None
            return set(j.tolist())

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
                prev_axis = long_axis(lay[0][4], lt, out_kind)
        return layers, bases, out_kind

    def mass_of(items):
        """items: [(型, 姿勢)]。質量 g と、世界座標の重心 mm。"""
        m_tot, c_sum = 0.0, np.zeros(3)
        for t, M in items:
            mp = P[t].get('mass-param')
            if not mp:
                continue
            m = float(mp['mass'])
            c = M[:3, :3] @ np.array(mp['center-of-mass'], float) + M[:3, 3]
            m_tot += m
            c_sum += m * c
        return m_tot, (c_sum / m_tot if m_tot else c_sum)

    servo_rel = {ft: [(fs, ss, rel(ft, fs, 'XL-430', ss, 'fixed'))
                      for fs in [n for n in CPT[ft] if not (CPT[ft][n][1] & {'PC1', 'SC1'})]
                      for ss in SERVO_PTS] for ft in FRAMES}
    servo_rel = {ft: [x for x in v if x[2] is not None] for ft, v in servo_rel.items()}
    log('Frame → サーボのつなぎ方:', {ft: len(v) for ft, v in servo_rel.items()})

    best = {}      # 長さ（0.1 mm）→ 候補
    n_valid = 0
    comps = [('B' * nb) + ('P' * npl) for nb in range(NB_MAX + 1) for npl in range(NP_MAX + 1)]
    MA = np.eye(4)
    za = hz
    for ht in HINGES:
        Mh = MA @ rel('XL-430', 'horn', ht, 'side_connect_point', 'fixed')
        for comp in comps:
            sp = stack_plan(Mh, ht, comp)
            if sp is None:
                log(f'  {ht} {comp or "なし"}: 基準 ②③ で積めない')
                continue
            layers, tops, out_kind = sp
            in_kind = 'SC1' if out_kind == 'PC1' else 'PC1'
            TG = np.array([wp(M, t, n) for M, t in tops for n in pts(t, out_kind)])
            for ft in FRAMES:
                fpts = pts(ft, in_kind)
                if not fpts or not servo_rel[ft]:
                    continue
                FL = loc(ft, fpts)
                for ti, (Mt, tt) in enumerate(tops):
                    for tn in pts(tt, out_kind):
                        for fn in fpts:
                            for c in CF:
                                r = rel(tt, tn, ft, fn, c)
                                if r is None:
                                    continue
                                MF = Mt @ r
                                w = (MF[:3, :3] @ FL.T).T + MF[:3, 3]
                                eng = int((np.linalg.norm(w[:, None, :] - TG[None, :, :], axis=2).min(1) < TOL).sum())
                                for fs, ss, r2 in servo_rel[ft]:
                                    MB = MF @ r2
                                    d = MB[:3, 3]
                                    ax = float(d @ za)
                                    if abs(ax) > 0.3 or abs(float((MB[:3, :3] @ hz) @ za)) < 0.999:
                                        continue
                                    L = float(np.linalg.norm(d - ax * za))
                                    n_valid += 1
                                    npart = 2 + sum(len(l[1]) for l in layers)
                                    key = round(L, 1)
                                    cand = (-eng, npart)
                                    if key not in best or cand < best[key]['rank']:
                                        best[key] = {'rank': cand, 'L': L, 'eng': eng, 'n': npart, 'ht': ht, 'comp': comp,
                                                     'layers': layers, 'tops': tops, 'top': (ti, tn), 'ft': ft, 'fn': fn, 'c': c,
                                                     'fs': fs, 'ss': ss, 'MF': MF, 'MB': MB, 'Mh': Mh}
            log(f'  {ht} {comp or "なし"}: 長さの種類 {len(best)}、有効な組み方 {n_valid}、attach の試し {len(cache)}、{time.time() - t0:.0f} 秒')

    # 質量と重心（リンクの座標: 原点 = A のホーン、x = A から B へ（軸に垂直）、z = A の軸）
    rows = []
    for key in sorted(best):
        b = best[key]
        items = [(b['ht'], b['Mh'])] + [(lt, x[4]) for lt, lay in b['layers'] for x in lay] + [(b['ft'], b['MF'])]
        m_s, c_s = mass_of(items)
        m_b, c_b = mass_of(items + [('XL-430', b['MB'])])
        d = b['MB'][:3, 3]
        ex = d - (d @ za) * za
        ex /= np.linalg.norm(ex)
        E = np.stack([ex, np.cross(za, ex), za])
        cs, cb = E @ c_s, E @ c_b
        rows.append({'軸間距離_mm': f'{b["L"]:.1f}', '噛み合い': b['eng'], '部品の数': b['n'], 'Hinge': b['ht'],
                     '段（B=2×4 ブロック 2 個・P=4×4 プレート）': b['comp'] or 'なし', 'Frame': b['ft'], 'Frame の点': b['fn'],
                     'つなぎ方': b['c'], 'Frame→サーボ': b['fs'], 'サーボの点': b['ss'],
                     '最上段の部品': b['top'][0], '最上段の点': b['top'][1],
                     '質量_構造_g': f'{m_s:.1f}', '重心_構造_x_mm': f'{cs[0]:.1f}', '重心_構造_y_mm': f'{cs[1]:.1f}', '重心_構造_z_mm': f'{cs[2]:.1f}',
                     '質量_サーボ込み_g': f'{m_b:.1f}', '重心_サーボ込み_x_mm': f'{cb[0]:.1f}'})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    Ls = np.array([float(r['軸間距離_mm']) for r in rows])
    g = np.diff(Ls)
    log(f'長さ {len(Ls)} 種類 {Ls.min():.1f}〜{Ls.max():.1f} mm、隙間 中央 {np.median(g):.2f} 最大 {g.max():.1f} mm、有効な組み方 {n_valid}')
    ms = np.array([float(r['質量_構造_g']) for r in rows])
    A = np.vstack([Ls, np.ones_like(Ls)]).T
    k, b0 = np.linalg.lstsq(A, ms, rcond=None)[0]
    res = ms - (k * Ls + b0)
    log(f'質量（構造）≈ {k:.3f}·L + {b0:.1f} g、残差 標準偏差 {res.std():.2f} 最大 {np.abs(res).max():.2f} g、噛み合い 中央 {np.median([r["噛み合い"] for r in rows]):.0f}')

    # 確かめ: 残した組み方の一部を本物の attach で組み直す
    random.seed(0)
    for key in random.sample(sorted(best), min(CHECK, len(best))):
        b = best[key]
        R = ru.makeRobotFromKey('XL-430')
        A_ = R.rootParts()
        H = ru.makeParts(b['ht'])
        assert R.attach(H, H.name + '/side_connect_point', A_.name + '/horn', 'fixed')
        cur = [H]
        for lt, lay in b['layers']:
            new = []
            for bi, bn, npn, c, _ in lay:
                N = ru.makeParts(lt)
                assert R.attach(N, f'{N.name}/{npn}', f'{cur[bi].name}/{bn}', c)
                new.append(N)
            cur = new
        F = ru.makeParts(b['ft'])
        assert R.attach(F, f'{F.name}/{b["fn"]}', f'{cur[b["top"][0]].name}/{b["top"][1]}', b['c'])
        S = ru.makeParts('XL-430')
        assert R.attach(S, f'{S.name}/{b["ss"]}', f'{F.name}/{b["fs"]}', 'fixed')
        d = pose(S)[:3, 3] - pose(A_)[:3, 3]
        Lr = float(np.linalg.norm(d - (d @ za) * za))
        if abs(Lr - b['L']) > 0.05:
            log(f'⛔ {key} mm の組み方を組み直したら {Lr:.2f} mm。止める')
            sys.stdout.flush()
            os._exit(1)
    log(f'✅ {min(CHECK, len(best))} 本を本物の attach で組み直し、長さが一致した。書き出した: {OUT}（{time.time() - t0:.0f} 秒）')
    os._exit(0)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
