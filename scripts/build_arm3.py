"""お辞儀型（A1v の 1/5 × SCALE）を研究室の部品で組む。リンクの組み方を「組み立ての基準」で選ぶ版。

基準（優先順。⭐ 軸6_部品ライブラリ md の「自動組み立ての基準」と同じ）:
  必須 ① 関節の軸どうしが平行で、軸方向のずれが 0（腕が 1 枚の平面で曲がる）
  必須 ② LEGO の各段は、下の面のポッチをすべて覆う（2×4 を 2 個、または 4×4 のプレート 1 枚）
  必須 ③ ブロックの段は、すぐ下の段と継ぎ目の向きを直交させる（レンガ積み。研究室の腕 simplehand.body と同じ）
  ④ 長さが目標に TOL_L 以内
  ⑤ Frame と最上段の噛み合うポッチの数が最大
  ⑥ 部品の数が少ない
  ⑦ 長さの誤差が小さい
部品の相対位置は RobotAssembler 自身の attach を小さいロボットで 1 回ずつ試して表にし（キャッシュ）、探索は行列の掛け算で行う。
選んだ組み方は最後に本物の腕のロボットで attach し直し、長さが探索と一致するかを確かめる。
"""
import itertools
import json
import os
import sys

import numpy as np
import yaml

PROJ = '/choreonoid_ws/install/share/choreonoid-2.3/robot_assembler/irsl'
SCALE = float(os.environ.get('SCALE', '1.5'))
TIP_LAYERS = int(os.environ.get('TIP_LAYERS', '3'))
TIP_HINGE = os.environ.get('TIP_HINGE', 'Hinge_Long_Plug')   # 1/5 の手先 55.3 mm には Hinge_Long_Socket（56.1 mm、9-319）
TOL_L = float(os.environ.get('TOL_L', '1.0'))      # mm
OUT = os.environ.get('OUT', '/userdir/StackelbergPPO/data/axis6/bow_arm_image/bow15v3_parts.json')
CF = ['fixed', 'invert', 'rotate90', 'rotate-90']
TOL = 0.5
HINGES = os.environ.get('FAM_HINGE', 'Hinge_Short_Plug,Hinge_Short_Socket,Hinge_Long_Plug,Hinge_Long_Socket').split(',')   # FAM_* で組み方の系統を固定できる（系譜 9-325）
FRAMES = os.environ.get('FAM_FRAME', 'Frame_Short_Plug,Frame_Short_Socket,Frame_Long_Plug,Frame_Long_Socket,Frame_Bottom_Plug,Frame_Bottom_Socket').split(',')
FAM_MOUNT = os.environ.get('FAM_MOUNT', '')   # 'Frame→サーボの点:サーボの点'（例 connect_point:right-side）
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
        """bt の部品の点 bp に、nt の部品の点 np_ をつなぎ方 c で付けたときの、nt の bt に対する姿勢（無理なら None）。"""
        k = (bt, bp, nt, np_, c)
        if k not in cache:
            Rt = ru.makeRobotFromKey(bt)
            b = Rt.rootParts()
            N = ru.makeParts(nt)
            ok = Rt.attach(N, f'{N.name}/{np_}', f'{b.name}/{bp}', c)
            cache[k] = np.linalg.inv(pose(b)) @ pose(N) if ok else None
        return cache[k]

    def pts(t, kind):
        """kind = 'PC1'（ポッチ）/ 'SC1'（穴）の点の名前。"""
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

    def make_layer(bases, out_kind, ltype, prev_axis):
        """bases: [(姿勢, 型)]。下の面（out_kind のポッチ）を ltype で覆う置き方を探す。
        返す: [(下の部品の番号, 下の点, 新しい部品の点, つなぎ方, 新しい姿勢)]。継ぎ目は prev_axis と直交させる（必須 ③）。"""
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
        """Hinge の上に comp（'B' ブロックの段 / 'P' 4×4 プレートの段）を積む。各段の置き方と最上段の [(姿勢, 型)]。"""
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

    def measure(MA, MB):
        d = MB[:3, 3] - MA[:3, 3]
        za = MA[:3, :3] @ hz
        return float(np.linalg.norm(d - (d @ za) * za)), abs(float((MB[:3, :3] @ hz) @ za)), float(d @ za)

    def search_link(L):
        MA = np.eye(4)
        best = None
        comps = [''] + [''.join(x) for n in range(1, 9) for x in [('B',) * n, ('B',) * n + ('P',)]]
        for ht in HINGES:
            Mh = MA @ rel('XL-430', 'horn', ht, 'side_connect_point', 'fixed')
            for comp in comps:
                sp = stack_plan(Mh, ht, comp)
                if sp is None:
                    continue
                layers, tops, out_kind = sp
                in_kind = 'SC1' if out_kind == 'PC1' else 'PC1'
                top_grid = [wp(M, t, n) for M, t in tops for n in pts(t, out_kind)]
                for ft in FRAMES:
                    fpts = pts(ft, in_kind)
                    if not fpts:
                        continue
                    for ti, (Mt, tt) in enumerate(tops):
                        for tn in pts(tt, out_kind):
                            for fn in fpts:
                                for c in CF:
                                    r = rel(tt, tn, ft, fn, c)
                                    if r is None:
                                        continue
                                    MF = Mt @ r
                                    eng = sum(any(np.linalg.norm(wp(MF, ft, n) - q) < TOL for q in top_grid) for n in fpts)
                                    if best and eng < best['eng'] and abs(best['L'] - L) <= TOL_L:
                                        continue
                                    for fs in [n for n in CPT[ft] if n not in fpts and not (CPT[ft][n][1] & {'PC1', 'SC1'})]:
                                        for ss in SERVO_PTS:
                                            if FAM_MOUNT and f'{fs}:{ss}' != FAM_MOUNT:
                                                continue
                                            r2 = rel(ft, fs, 'XL-430', ss, 'fixed')
                                            if r2 is None:
                                                continue
                                            Lg, par, ax = measure(MA, MF @ r2)
                                            if par < 0.999 or abs(ax) > 0.3:
                                                continue
                                            npart = 1 + sum(len(l[1]) for l in layers) + 1
                                            cand = {'L': Lg, 'eng': eng, 'n': npart, 'ht': ht, 'comp': comp, 'layers': layers,
                                                    'top': (ti, tn), 'ft': ft, 'fn': fn, 'c': c, 'fs': fs, 'ss': ss}
                                            key = lambda x: (abs(x['L'] - L) > TOL_L, -x['eng'], x['n'], abs(x['L'] - L))
                                            if best is None or key(cand) < key(best):
                                                best = cand
            log(f'  {ht} まで: 最良 {best and round(best["L"], 1)} mm 噛み合い {best and best["eng"]}')
        return best

    # ---- 本物の腕を組む ----
    R = ru.makeRobotFromKey('XL-430')
    J1 = R.rootParts()
    parts = [('J1', 'XL-430', J1)]
    servos = [J1]
    FS = ru.makeParts('Frame_Short')
    assert R.attach(FS, FS.name + '/horn', J1.name + '/horn', 'invert')
    J2 = ru.makeParts('XL-430')
    assert R.attach(J2, J2.name + '/bottom-side1', FS.name + '/side_connect_point2', 'fixed')
    parts += [('肩フレーム', 'Frame_Short', FS), ('J2', 'XL-430', J2)]
    servos.append(J2)

    def build_layers(base_parts, layers, tag):
        cur = base_parts
        for li, (lt, lay) in enumerate(layers, start=1):
            new = []
            for k, (bi, bn, npn, c, _) in enumerate(lay):
                N = ru.makeParts(lt)
                assert R.attach(N, f'{N.name}/{npn}', f'{cur[bi].name}/{bn}', c)
                parts.append((f'{tag}の LEGO {li} 段目 {k}', lt, N))
                new.append(N)
            cur = new
        return cur

    used = []
    for k, L in enumerate((403.1 * 0.2 * SCALE, 330.8 * 0.2 * SCALE)):
        tag = f'リンク{k+1}'
        b = search_link(L)
        if b is None:
            raise RuntimeError(f'{tag}: 基準を満たす組み方が無い')
        A = servos[-1]
        H = ru.makeParts(b['ht'])
        assert R.attach(H, H.name + '/side_connect_point', A.name + '/horn', 'fixed')
        parts.append((f'{tag}の Hinge', b['ht'], H))
        tops = build_layers([H], b['layers'], tag) if b['layers'] else [H]
        ti, tn = b['top']
        F = ru.makeParts(b['ft'])
        assert R.attach(F, f'{F.name}/{b["fn"]}', f'{tops[ti].name}/{tn}', b['c'])
        S = ru.makeParts('XL-430')
        assert R.attach(S, f'{S.name}/{b["ss"]}', f'{F.name}/{b["fs"]}', 'fixed')
        parts += [(f'{tag}の Frame', b['ft'], F), (f'J{k+3}', 'XL-430', S)]
        servos.append(S)
        Lg, par, ax = measure(pose(A), pose(S))
        log(f'{tag}: 目標 {L:.1f} → 探索 {b["L"]:.1f} / 本物 {Lg:.1f} mm（平行 {par:.3f}、軸方向 {ax:.1f}）'
            f' Hinge {b["ht"]} 段 {b["comp"] or "なし"} Frame {b["ft"]} 噛み合い {b["eng"]} 部品 {b["n"]}')
        if abs(Lg - b['L']) > 0.2:
            raise RuntimeError(f'{tag}: 本物で組んだ長さが探索と違う')
        used.append({'目標_mm': round(L, 1), '組めた_mm': round(Lg, 1), 'Hinge': b['ht'], '段': b['comp'], 'Frame': b['ft'],
                     '噛み合い': b['eng'], '部品': b['n']})
    # 手先: 長い Hinge ＋ ブロックの段（継ぎ目を直交）
    T = ru.makeParts(TIP_HINGE)
    assert R.attach(T, T.name + '/side_connect_point', servos[-1].name + '/horn', 'fixed')
    parts.append(('手先の Hinge', TIP_HINGE, T))
    if TIP_LAYERS:
        sp = stack_plan(pose(T), TIP_HINGE, 'B' * TIP_LAYERS)
        build_layers([T], sp[0], '手先')

    out = {'links': used, 'parts': []}
    for name, typ, p in parts:
        vis = []
        for g in P[typ].get('visual', []):
            Tl = np.eye(4)
            Tl[:3, :3] = rotmat(g.get('rotation'))
            Tl[:3, 3] = g.get('translation', [0, 0, 0])
            if 'box' in g:
                col = [0.95, 0.75, 0.15] if typ.startswith('block') else [0.25, 0.55, 0.9]
                vis.append({'box': g['box'], 'T': Tl.tolist(), 'color': col})
            elif g.get('type') == 'mesh':
                vis.append({'url': os.path.join(PROJ, g['url']), 'scale_mm': g.get('scale', 1.0) * 1000.0,
                            'T': Tl.tolist(), 'color': g.get('color')})
        out['parts'].append({'name': name, 'type': typ, 'T': pose(p).tolist(), 'visual': vis, 'servo': typ == 'XL-430'})
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1)
    log('書き出した:', OUT, '部品', len(parts), '試した attach', len(cache))
    os._exit(0)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
