#!/usr/bin/env python3
"""⭐ 腕のリンクどうしの**すり抜け（自己干渉）**が、既存の run でどれだけ起きているかを測る（2026-10-05）。

⛔ 学習は自己干渉を見ていない（Choreonoid の既定）。Target-Pusher のデモで上腕が伸びた台座の中に
  折りたたまれていた（ユーザー発見）。⭐ **学習し直す前に、どの run でどれだけ起きているかを測る。**

⭐ 軌跡（`single_run/<run>/trace/arm_trace.npz`）だけを使う。学習も Choreonoid も要らない。
- 各リンク = 関節位置 `xpos` から `xmat @ bone_offset` へ伸びる線分 + 半径 `geom_size`（カプセル）
- ⭐ 離れたリンクの組はそのまま、**隣り合う組は関節のまわり（2 本の半径の和）を両方から削ってから**見る
  （関節での自然な重なりは数えず、**子が親の上へ折り返す**重なりだけを拾う）
- 根元の取り付け用の球（先頭の body）は、骨の向きが運動学に効かない（9-154）ので**半径 r の球**として扱う
- ⭐ 2026-10-20（Bug 72）: **離れた組は本当のカプセル**（軌跡の `cap_fromto`、関節の後ろへのはみ出し `ext_start` を含む）で測る。
  ⛔ 旧版はボーンだけで測り、dist の `11` が台座の球に 13 cm 食い込んでいたのを 0 mm と出した。
  ⚠️ `cap_fromto` の無い古い軌跡は従来どおりボーンで測る（はみ出しは見えない。出力に「始点なし」と出す）。
  隣り合う組はボーンのまま（はみ出しは親の中へ入るのが普通で、Choreonoid も隣り合う組は当たり判定から外す）
- ⚠️ 軌跡は採用した話のうち中央値の 1 話（`xpos`）。全話ではない

    python3 scripts/check_self_penetration.py               # 全 run
    python3 scripts/check_self_penetration.py e2e_a1v_reach  # 指定 run
    python3 scripts/check_self_penetration.py --self-check  # 線分間距離の自己検査
"""
import glob
import os
import sys

import numpy as np

TOL = 0.005          # めり込みがこれ [m] を超えた step を「すり抜け」と数える
# ⭐ 2026-10-22（ユーザー「実機での gap の 1 つに衝突はあるから、これに対して言えないのはまずい」、系譜 9-287）:
#   隣り合う組を「関節のまわり（2 本の半径の和 d）を両方から削って」判定すると、削った端どうしの距離は 2d·sin(φ/2)
#   （φ は 2 本のなす角）で、d を切ると当たる → φ < 60°＝**まっすぐから 120° を超えて曲げると当たる**（半径に依らない）。
#   同じ平面に並べた実機では取れない姿勢の目安。⚠️ 120° は「削る量＝半径の和」という仮定から出る値で、実機の関節の組み方の裏付けは無い
FOLD_DEG = 120.0


def seg_seg_dist(p1, q1, p2, q2):
    """線分 p1-q1 と p2-q2 の最短距離（先頭の軸で並べて一度に解く）。Ericson の方法。"""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.einsum('...i,...i', d1, d1); e = np.einsum('...i,...i', d2, d2)
    f = np.einsum('...i,...i', d2, r);  c = np.einsum('...i,...i', d1, r)
    b = np.einsum('...i,...i', d1, d2)
    eps = 1e-12
    den = a * e - b * b
    s = np.where(den > eps, np.clip((b * f - c * e) / np.where(den > eps, den, 1), 0, 1), 0.0)
    s = np.where(a <= eps, 0.0, s)
    t = np.where(e > eps, (b * s + f) / np.where(e > eps, e, 1), 0.0)
    # t を [0,1] に収め、s を取り直す
    t0 = t < 0; t1 = t > 1
    s = np.where(t0 & (a > eps), np.clip(-c / np.where(a > eps, a, 1), 0, 1), s)
    s = np.where(t1 & (a > eps), np.clip((b - c) / np.where(a > eps, a, 1), 0, 1), s)
    t = np.clip(t, 0, 1)
    c1 = p1 + d1 * s[..., None]; c2 = p2 + d2 * t[..., None]
    return np.linalg.norm(c1 - c2, axis=-1)


def self_check():
    P = lambda *v: np.array(v, float)
    cases = [((P(0, 0, 0), P(1, 0, 0), P(0, 1, 0), P(1, 1, 0)), 1.0),      # 平行
             ((P(0, 0, 0), P(1, 0, 0), P(0.5, -1, 1), P(0.5, 1, 1)), 1.0),  # ねじれ・交差の上
             ((P(0, 0, 0), P(1, 0, 0), P(2, 0, 0), P(3, 0, 0)), 1.0),       # 同一直線・離れ
             ((P(0, 0, 0), P(0, 0, 0), P(0, 3, 4), P(0, 3, 4)), 5.0),       # 点と点
             ((P(0, 0, 0), P(2, 0, 0), P(1, -1, 0), P(1, 1, 0)), 0.0)]      # 交差
    for (a, b, c, d), want in cases:
        got = float(seg_seg_dist(a, b, c, d))
        assert abs(got - want) < 1e-9, (got, want)
    print('✅ 線分間距離の自己検査 5 件一致')
    # ⭐ Bug 56: 環境側の「自分のリンクどうしの初期接触の門」（pusher.py `_self_overlaps`）が、
    #   固着した実例を捕まえ、正常な形を捕まえないこと（軌跡があるときだけ）
    sys.path.insert(0, os.getcwd())
    from design_opt.envs.pusher import PusherEnv as E
    for r, want in [('tripo_pjdp_dist_sc', True), ('tripo_pjdp_dist_sc_s2', False), ('tripo_pjp_long_sc', False)]:
        f = f'single_run/{r}/trace/arm_trace.npz'
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True)
        x, R, b, g = d['xpos'][0], d['xmat'][0], d['bone_offset'], d['geom_size']
        segs = [(x[l], x[l] + R[l] @ b[l]) for l in range(len(b))]
        hit = E._self_overlaps(segs, list(g), [-1] + list(range(len(b) - 1)))
        assert bool(hit) == want, (r, hit)
    print('✅ 自分のリンクどうしの初期接触の門（Bug 56）: 固着した形を捕まえ、正常な形は通す')
    # ⭐ Bug 72: 関節の後ろへはみ出したカプセルの食い込みを拾えるか（既知の失敗の再現）。
    #   `arm_trace_cap.npz` は 2026-10-20 に新しい `record_arm_trace.py` で取り直した軌跡（元の arm_trace.npz は残してある）
    f = 'single_run/tripo_pjd_dist_phys/trace/arm_trace_cap.npz'
    if os.path.exists(f):
        far = measure(f)[4]
        assert far[2] and far[0] > 0.05 and far[3] == ('0', '11'), far
        print(f'✅ 関節の後ろへのはみ出し（Bug 72）: dist の 0×11 の食い込み {far[0]*1000:.0f} mm を拾う')
    # ⭐ 系譜 9-287: 到達の長腕 seed1 が手首（11×111）を 148° 折り返す姿勢を拾えるか、折り返さない中間は拾わないか
    for r, want in [('tripo_pj_long_phys_s2', True), ('tripo_pj_mid_phys', False)]:
        if os.path.exists(f'single_run/{r}/trace/arm_trace.npz'):
            fo = fold(r)
            assert (fo[1] > 0.5) == want, (r, fo)
            print(f'✅ 折り返し（9-287）: {r} 最大 {fo[0]:.1f}°・{FOLD_DEG:.0f}° 超 {fo[1]*100:.1f} %（{"拾う" if want else "拾わない"}）')


def measure(run):
    # run 名か、軌跡（.npz）のパス（run の軌跡を上書きせずに確かめるとき）
    p = run if run.endswith('.npz') else f'single_run/{run}/trace/arm_trace.npz'
    d = np.load(p, allow_pickle=True)
    x, R, b = d['xpos'], d['xmat'], d['bone_offset']
    rad = np.asarray(d['geom_size'], float) if 'geom_size' in d.files else np.full(len(b), 0.03)
    rad = np.where(np.isfinite(rad), rad, 0.03)
    T, L = x.shape[:2]
    p0 = x.copy()
    p1 = x + np.einsum('tlij,lj->tli', R, b)
    p1[:, 0] = p0[:, 0]                       # 根元は球として扱う（9-154）
    # ⭐ Bug 72: 離れた組は本当のカプセル（始点が関節の後ろにありうる）で測る
    cap = np.asarray(d['cap_fromto'], float) if 'cap_fromto' in d.files and len(d['cap_fromto']) == L else None
    if cap is not None and np.isfinite(cap).all():
        q0 = x + np.einsum('tlij,lj->tli', R, cap[:, :3])
        q1 = x + np.einsum('tlij,lj->tli', R, cap[:, 3:])
    else:
        cap, q0, q1 = None, p0, p1
    worst, frac, pair_w = 0.0, 0.0, None
    far = [0.0, 0.0]                          # 離れた組だけの最大めり込み・step の割合
    pair_far = None
    pairs = [(i, j, False) for i in range(L) for j in range(i + 2, L)]
    # ⭐ 隣り合う組（根元の球は除く）も見る。⛔ ただし関節では必ず重なるので、
    #   **関節のまわり（2 本の半径の和）を両方から削ってから**重なりを見る → 子が親の上へ**折り返す**ときだけ拾う。
    #   ⛔ 初稿は隣り合う組を丸ごと外しており、Target-Pusher の「上腕が台座の中に折りたたまれる」を数えなかった
    pairs += [(i, i + 1, True) for i in range(1, L - 1)]
    for i, j, adj in pairs:
        a0, a1, c0, c1 = (p0[:, i], p1[:, i], p0[:, j], p1[:, j]) if adj else (q0[:, i], q1[:, i], q0[:, j], q1[:, j])
        if adj:
            m = rad[i] + rad[j]
            da, dc = a1 - a0, c1 - c0
            la = np.linalg.norm(da, axis=-1, keepdims=True); lc = np.linalg.norm(dc, axis=-1, keepdims=True)
            if float(la.min()) <= m or float(lc.min()) <= m:
                continue                       # 短すぎて削ると残らない
            a1 = a1 - da / la * m              # 親は関節側（先端）を削る
            c0 = c0 + dc / lc * m              # 子は関節側（根元）を削る
        dist = seg_seg_dist(a0, a1, c0, c1)
        depth = rad[i] + rad[j] - dist
        f = float((depth > TOL).mean())
        if depth.max() > worst:
            worst, pair_w = float(depth.max()), (str(d['body_names'][i]), str(d['body_names'][j]))
        frac = max(frac, f)
        if not adj:
            if float(depth.max()) > far[0]:
                pair_far = (str(d['body_names'][i]), str(d['body_names'][j]))
            far = [max(far[0], float(depth.max())), max(far[1], f)]
    far.append(cap is not None)               # ⭐ Bug 72: はみ出しまで見たか
    far.append(pair_far)
    return worst, frac, pair_w, T, far


def fold(run):
    """隣り合うリンクの曲げ角（まっすぐ＝0°）。全話（`ep_xmat`）で、最大角・FOLD_DEG を超えた step の割合・その組を返す。"""
    p = run if run.endswith('.npz') else f'single_run/{run}/trace/arm_trace.npz'
    d = np.load(p, allow_pickle=True)
    if 'ep_xmat' in d.files:      # ⚠️ 話ごとに長さが違うと object 配列になるので、話ごとに扱う
        eps = [(np.asarray(x, float), np.asarray(b, float)) for x, b in zip(d['ep_xmat'], d['ep_bone'])]
    else:
        eps = [(np.asarray(d['xmat'], float), np.asarray(d['bone_offset'], float))]
    us = []
    for X, B in eps:                                        # (step, リンク, 3, 3)・(リンク, 3)
        u = np.einsum('tlij,lj->tli', X, B)
        us.append(u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-12))
    u = np.concatenate(us)
    worst, frac, pair = 0.0, 0.0, None
    for i in range(1, u.shape[1] - 1):                     # 根元の球（0）は骨の向きが運動学に効かない（9-154）
        ang = np.degrees(np.arccos(np.clip((u[:, i] * u[:, i + 1]).sum(-1), -1, 1)))
        f = float((ang > FOLD_DEG).mean())
        if f > frac or (f == frac and ang.max() > worst):
            pair = (str(d['body_names'][i]), str(d['body_names'][i + 1]))
        worst, frac = max(worst, float(ang.max())), max(frac, f)
    return worst, frac, pair, len(eps)


def main():
    if '--self-check' in sys.argv:
        return self_check()
    runs = [a for a in sys.argv[1:] if not a.startswith('-')]
    if not runs:
        runs = sorted(os.path.basename(os.path.dirname(os.path.dirname(f)))
                      for f in glob.glob('single_run/*/trace/arm_trace.npz') if '_aborted' not in f)
    thesis = ''.join(open(f, encoding='utf-8').read()
                     for f in glob.glob('docs/研究応用/修論ドラフト/第[1-5]章*.md')
                     + glob.glob('docs/研究応用/修論ドラフト/付録*.md'))
    rows = []
    for r in runs:
        try:
            w, fr, pr, T, far = measure(r)
            fo = fold(r)
        except Exception as e:
            print(f'  ⚠️ {r}: 読めない（{e!r}）'); continue
        rows.append((r, w, fr, pr, T, (f'`{r}`' in thesis) or (r in thesis), far, fo))
    rows.sort(key=lambda z: -z[1])
    hit = [z for z in rows if z[2] > 0]
    print(f'\n⭐ 対象 {len(rows)} run  /  すり抜けあり（めり込み > {TOL*1000:.0f} mm の step が 1 つ以上）: {len(hit)} run'
          f'  /  そのうち修論本文に名前が出る run: {sum(z[5] for z in hit)}')
    # ⭐ 離れた組を別に出す（2026-10-09、系譜 9-227）。⛔ 最悪の 1 組だけだと、隣り合う組の重なりの陰に
    #   離れた組の 39〜45 mm が隠れ、9-223 で「離れた組 0」と読み違えた
    nf = sum(z[7][1] > 0 for z in rows)
    print(f'⭐ 隣り合う組を {FOLD_DEG:.0f}° を超えて折り返す（同じ平面の実機では当たる）step がある: {nf} run（全話で数える）')
    print(f'{"run":44s} {"最大めり込み":>10s} {"step の割合":>10s}  {"組":14s} {"うち離れた組":>16s}  {"折り返し（最大・"+str(int(FOLD_DEG))+"°超・組）":24s} 修論')
    for r, w, fr, pr, T, th, far, fo in rows:
        if fr == 0 and fo[1] == 0 and len(runs) > 3:
            continue
        print(f'{r:44s} {w*1000:8.0f} mm {fr*100:9.1f} %  {str(pr):14s} {far[0]*1000:5.0f} mm {far[1]*100:5.1f} %  '
              f'{fo[0]:6.1f}° {fo[1]*100:5.1f} % {str(fo[2]) if fo[1] else "":16s} {"★" if th else ""}')


if __name__ == '__main__':
    main()
