#!/usr/bin/env python3
"""**四脚の立体（Tripo の GLB）を、マゼンタの関節球と「くびれ」（断面積の極小）で胴・腿・脛に分ける**（系譜 9-342 の追記、2026-11-06）。

なぜ: ユーザー「マゼンタとクビレに着目して先行研究と同じようなアルゴリズムで分割してみてよ」（興味本位、再現性は問わない）。
Text2Robot（Ringel et al., 2025）は、体積をそろえたメッシュを重心から横へ切り、**断面積が極小になる所で胴と脚を分けて肩の関節**にする
（輪講の素材 段 2）。この研究は関節を入力のマゼンタの球で指定する。同じ立体に両方を当て、切る位置がどれだけ違うかを見る。

やり方（学習しない・Choreonoid を使わない）:
  1. メッシュを PITCH の立方体で埋める（`voxelized().fill()`）。各ボクセルの色は最も近い頂点の色
  2. マゼンタの頂点を 1 cm のつながりで塊にし、中心が MERGE 以内の塊（MIN_BLOB 頂点を超えるもの）を 1 つにまとめ、大きい 8 つを関節とする。
     高い 4 つが股、低い 4 つが膝。x と z の符号で脚ごとに組にする
     ⚠️ まとめるのは、リンクが球を貫くと球が左右 2 つの塊に割れるため（quad_walk_0 の膝。8 つの最大が膝の半分 8 個になり、股を落とした）
  3. 足先 = その脚の側（x・z の符号が同じ）で、膝より低い点のうち膝から最も遠い点
  4. 脚ごとに軸「胴の中心 → 股 → 膝 → 足先」を引き、ボクセルを最も近い軸の区間に割り当て（軸から RADIUS 以内）、
     軸に沿った位置 s ごとのボクセル数＝断面積 A(s) を出す
  5. くびれ法: 「胴の中心〜膝」の A(s) の極小で胴と腿を、「股〜足先」の極小で腿と脛を分ける（Text2Robot の型）
     マゼンタ法: 股・膝の球の中心を通り軸に垂直な面で分ける（この研究の型）
  6. 2 つの分け方の切る位置の差（軸に沿って）と、ボクセルの割り当てが一致する割合を出し、図にする

    python3 scripts/segment_quad_mesh.py [--glb data/api/quad/tripo/quad_a1_0_seed42.glb] [--out data/api/quad/seg]

⚠️ 単位は Tripo の正規化した単位（最長の辺が 1）。高さは y。
"""
import argparse
import json
from pathlib import Path

import numpy as np

PITCH = 0.008
RADIUS = 0.12
DS = 0.01
MERGE = 0.12      # 同じ球の割れた塊の中心どうしの距離の上限（quad_walk_0 の膝で 0.08。股と膝は 0.29 以上離れている）
MIN_BLOB = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glb', default='data/api/quad/tripo/quad_a1_0_seed42.glb')
    ap.add_argument('--out', default='data/api/quad/seg')
    a = ap.parse_args()
    import trimesh
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    m = trimesh.load(a.glb).to_geometry()
    V = m.vertices
    col = m.visual.to_color().vertex_colors[:, :3].astype(int)
    r, g, b = col.T
    mag = (r > 150) & (b > 150) & (g < 120)

    # 1. ボクセル
    vg = m.voxelized(PITCH).fill()
    P = vg.points
    print(f'ボクセル {len(P)} 個（{PITCH} 刻み、中を埋めた）・頂点 {len(V)}')

    # 2. 関節の球
    Q = V[mag]
    pairs = cKDTree(Q).query_pairs(0.01, output_type='ndarray')
    A = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(Q), len(Q)))
    _n, lab = connected_components(A, directed=False)
    cnt = np.bincount(lab)
    blobs = [k for k in np.argsort(-cnt) if cnt[k] > MIN_BLOB]
    groups = []   # 中心が MERGE 以内の塊を 1 つの球にまとめる
    for k in blobs:
        c = Q[lab == k].mean(0)
        g = next((g for g in groups if np.linalg.norm(Q[np.isin(lab, g)].mean(0) - c) < MERGE), None)
        (g.append(k) if g is not None else groups.append([k]))
    groups.sort(key=lambda g: -cnt[g].sum())
    print(f'マゼンタの塊 {len(blobs)} 個（{MIN_BLOB} 頂点超）→ 球 {len(groups)} 個にまとめた（2 つ以上の塊をまとめた球 {sum(len(g) > 1 for g in groups)} 個）')
    if len(groups) < 8:
        raise SystemExit(f'⛔ 関節の球が 8 個に満たない（{len(groups)} 個）')
    sel = [np.isin(lab, g) for g in groups[:8]]
    J = np.array([Q[m_].mean(0) for m_ in sel])
    Jr = np.array([np.linalg.norm(Q[m_] - Q[m_].mean(0), axis=1).mean() for m_ in sel])
    hip_i = np.argsort(-J[:, 1])[:4]
    knee_i = np.argsort(-J[:, 1])[4:]
    center = P.mean(0)
    legs = []
    for hi in hip_i:
        h = J[hi]
        kn = min(knee_i, key=lambda k: np.linalg.norm(np.sign(J[k][[0, 2]]) - np.sign(h[[0, 2]])) * 10 + np.linalg.norm(J[k] - h))
        k = J[kn]
        side = (np.sign(V[:, 0]) == np.sign(h[0])) & (np.sign(V[:, 2]) == np.sign(h[2])) & (V[:, 1] < k[1])
        foot = V[side][np.argmax(np.linalg.norm(V[side] - k, axis=1))]
        legs.append(dict(name=f'{"前" if h[2] > 0 else "後"}{"左" if h[0] > 0 else "右"}', hip=h, knee=k, foot=foot,
                         hip_r=float(Jr[hi]), knee_r=float(Jr[kn])))
    print(f'胴の中心（ボクセルの平均） {np.round(center, 3)}')

    # 4. 軸に沿った断面積
    def project(pts, poly):
        """各点を折れ線 poly に最も近い点へ射影し、(折れ線に沿った位置 s, 軸からの距離, 垂線の足が区間の内側か) を返す。
        ⚠️ 折れ目の外側の点は区間の端へ寄せられ、断面積が関節の位置で鋭く立つ（最初の版でそうなった）。
        断面積には「垂線の足が区間の内側」の点だけを使う（折れ目の外側の点は数えない）"""
        best_d = np.full(len(pts), np.inf); best_s = np.zeros(len(pts)); inner = np.zeros(len(pts), bool); s0 = 0.0
        for p0, p1 in zip(poly[:-1], poly[1:]):
            v = p1 - p0; L = np.linalg.norm(v)
            t_raw = ((pts - p0) @ v) / L ** 2
            t = np.clip(t_raw, 0, 1)
            d = np.linalg.norm(pts - (p0 + t[:, None] * v), axis=1)
            upd = d < best_d
            best_d[upd] = d[upd]; best_s[upd] = s0 + t[upd] * L
            inner[upd] = (t_raw[upd] >= 0) & (t_raw[upd] < 1)
            s0 += L
        return best_s, best_d, inner

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # どの脚の軸に近いか（胴の中心から先の区間だけで比べる）
    dist_all = []
    for L in legs:
        L['poly'] = np.array([center, L['hip'], L['knee'], L['foot']])
        L['cum'] = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(L['poly'], axis=0), axis=1))])
        s, d, inner = project(P, L['poly'])
        L['s'], L['d'], L['inner'] = s, d, inner
        dist_all.append(d)
    owner = np.argmin(np.stack(dist_all), axis=0)

    label_mag = np.zeros(len(P), int)   # 0 胴・1 腿・2 脛
    label_neck = np.zeros(len(P), int)
    from scipy.signal import find_peaks

    def line_profile(mask, p0, p1, ext):
        """p0→p1 の直線（両端を ext だけ延ばす）に沿った断面積。直線からの距離が RADIUS 以内の点を、直線上の位置 t で数える。
        ⚠️ 折れ線に沿って測ると、折れ目で点の寄せ方が変わり、関節の位置に鋭い山（最初の版）か谷（2 版目）が出た。直線の区間ごとに測る"""
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        t = (P - p0) @ u
        perp = np.linalg.norm(P - p0 - t[:, None] * u, axis=1)
        sel = mask & (perp < RADIUS) & (t > -ext) & (t < np.linalg.norm(p1 - p0) + ext)
        edges = np.arange(-ext, np.linalg.norm(p1 - p0) + ext + DS, DS)
        area, _ = np.histogram(t[sel], edges)
        return (edges[:-1] + edges[1:]) / 2, np.convolve(area, np.ones(3) / 3, mode='same')

    def minima(tc, area, lo, hi):
        """谷（prominence 3 以上）を、際立つ順に (位置, prominence) で返す。lo〜hi の中だけ"""
        pk, pr = find_peaks(-area, prominence=3)
        ok = (tc[pk] > lo) & (tc[pk] < hi)
        o = np.argsort(-pr['prominences'][ok])
        return [(float(tc[pk[ok]][i]), float(pr['prominences'][ok][i])) for i in o]

    rows = []
    for li, L in enumerate(legs):
        mine = owner == li
        own = mine & (L['d'] < RADIUS)
        s = L['s'][own]
        C, H, K, F = L['poly']
        lCH, lHK = np.linalg.norm(H - C), np.linalg.norm(K - H)
        L['prof'] = {'胴の中心→股': line_profile(mine, C, H, 0.12), '股→膝': line_profile(mine, H, K, 0.12), '膝→足先': line_profile(mine, K, F, 0.03)}
        # くびれ法（Text2Robot の型）: 境目ごとに最も際立つ谷を 1 つ
        mA = minima(*L['prof']['胴の中心→股'], 0.05, lCH + 0.12)
        mB = minima(*L['prof']['股→膝'], L['hip_r'], lHK + 0.12)
        tA = mA[0][0] if mA else float('nan')
        tB = mB[0][0] if mB else float('nan')
        neck1 = tA if tA <= lCH else lCH + (tA - lCH)      # 股を越えた分は次の区間の上の位置として扱う（近似）
        neck2 = lCH + tB
        L.update(neck1=float(neck1), neck2=float(neck2), mA=mA, mB=mB)
        idx = np.where(own)[0]
        s_hip, s_knee = L['cum'][1], L['cum'][2]
        label_mag[idx] = np.where(s < s_hip, 0, np.where(s < s_knee, 1, 2))
        label_neck[idx] = np.where(s < neck1, 0, np.where(s < neck2, 1, 2))
        rows.append(dict(脚=L['name'], 胴と腿のくびれ_股から=round(tA - lCH, 3), 腿と脛のくびれ_膝から=round(tB - lHK, 3),
                         股の球の半径=round(L['hip_r'], 3), 膝の球の半径=round(L['knee_r'], 3),
                         胴から股の谷=[(round(t - lCH, 3), round(p)) for t, p in mA[:3]],
                         股から膝の谷=[(round(t - lHK, 3), round(p)) for t, p in mB[:3]]))
    agree = float((label_mag == label_neck).mean())
    print('\n脚ごとのくびれ（関節の球の中心から測った位置。負＝胴の側。谷は (位置, prominence) を際立つ順に 3 つまで）')
    for rw in rows:
        print('  ' + '  '.join(f'{k} {v}' for k, v in rw.items()))
    for nm, lb in (('マゼンタ法', label_mag), ('くびれ法', label_neck)):
        print(f'  {nm}: 胴 {np.mean(lb == 0):.3f}・腿 {np.mean(lb == 1):.3f}・脛 {np.mean(lb == 2):.3f}（ボクセルの割合）')
    print(f'  2 つの分け方でボクセルの割り当てが一致する割合 {agree:.3f}')
    # quad_mesh_to_xml.py に渡す骨格: 関節の位置・足先・リンクの太さ（断面積の平らな所の中央値から、同じ面積の円の半径）・胴の外形
    def plateau_r(tc, area, lo, hi):
        w = (tc > lo) & (tc < hi)
        return float(np.sqrt(np.median(area[w]) * PITCH ** 2 / np.pi))
    torso = P[label_mag == 0]
    skel = dict(unit='Tripo の正規化した単位・高さは y', center=center.tolist(),
                torso_min=np.percentile(torso, 1, axis=0).tolist(), torso_max=np.percentile(torso, 99, axis=0).tolist(),
                legs=[dict(name=L['name'], hip=L['hip'].tolist(), knee=L['knee'].tolist(), foot=L['foot'].tolist(),
                           hip_r=L['hip_r'], knee_r=L['knee_r'],
                           thigh_r=plateau_r(*L['prof']['股→膝'], L['hip_r'] + 0.03, np.linalg.norm(L['knee'] - L['hip']) - L['knee_r'] - 0.03),
                           shin_r=plateau_r(*L['prof']['膝→足先'], L['knee_r'] + 0.03, np.linalg.norm(L['foot'] - L['knee']) - 0.05))
                      for L in legs])
    json.dump(dict(rows=rows, agree=agree, pitch=PITCH, radius=RADIUS, skeleton=skel), open(out / 'seg_summary.json', 'w'), ensure_ascii=False, indent=1)

    # 6. 図
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')   # make_slide9_m1_variance_figure.py と同じ
    plt.rcParams['font.family'] = fm.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc').get_name()
    fig = plt.figure(figsize=(15, 9))
    cols = np.array([[0.35, 0.35, 0.35], [0.9, 0.2, 0.2], [0.2, 0.3, 0.95]])
    sub = np.random.default_rng(0).choice(len(P), min(len(P), 40000), replace=False)
    for k, (nm, lb) in enumerate((('マゼンタ法（球の中心で切る）', label_mag), ('くびれ法（断面積の極小で切る）', label_neck))):
        ax = fig.add_subplot(2, 3, 1 + k, projection='3d')
        ax.scatter(P[sub, 0], P[sub, 2], P[sub, 1], c=cols[lb[sub]], s=1)
        for L in legs:
            ax.plot(*L['poly'][:, [0, 2, 1]].T, 'k-', lw=1)
        ax.set_title(nm); ax.view_init(18, -60); ax.set_box_aspect((0.6, 1.0, 0.67)); ax.set_axis_off()
    ax = fig.add_subplot(2, 3, 3, projection='3d')
    ax.scatter(P[sub, 0], P[sub, 2], P[sub, 1], c=np.where((label_mag != label_neck)[sub], 'orange', '#dddddd'), s=1)
    ax.set_title(f'割り当てが違うボクセル（橙、一致 {agree:.1%}）'); ax.view_init(18, -60); ax.set_box_aspect((0.6, 1.0, 0.67)); ax.set_axis_off()
    L = legs[0]
    C, H, K, F = L['poly']
    marks = {'胴の中心→股': [(np.linalg.norm(H - C), 'm', '股の中心')],
             '股→膝': [(0.0, 'm', '股の中心'), (np.linalg.norm(K - H), 'm', '膝の中心')],
             '膝→足先': [(0.0, 'm', '膝の中心'), (np.linalg.norm(F - K), 'k', '足先')]}
    for k, (nm, (tc, area)) in enumerate(L['prof'].items()):
        ax = fig.add_subplot(2, 3, 4 + k)
        ax.plot(tc, area, 'k-')
        ymax = area.max() * 1.05
        for x, c, t in marks[nm]:
            ax.axvline(x, color=c, lw=1); ax.text(x, ymax, t, color=c, fontsize=8, rotation=90, va='top')
        cut = {'胴の中心→股': L['mA'], '股→膝': L['mB']}.get(nm, [])
        for j, (x, _p) in enumerate(cut[:1]):
            ax.axvline(x, color='g', ls='--', lw=1.5); ax.text(x, ymax * 0.6, 'くびれ（切る所）', color='g', fontsize=8, rotation=90, va='top')
        ax.set_title(f'{L["name"]}の脚: {nm} の断面積'); ax.set_xlabel('直線に沿った位置（正規化した単位）'); ax.set_ylabel('ボクセル数（3 区間の移動平均）')
    fig.tight_layout(); fig.savefig(out / 'seg.png', dpi=110)
    print(f'→ {out / "seg.png"}・{out / "seg_summary.json"}')


if __name__ == '__main__':
    main()
