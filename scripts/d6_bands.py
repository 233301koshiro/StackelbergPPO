#!/usr/bin/env python3
"""**D-6 の差し替えに使う値を、一次ログと軌跡から組ごとに並べる**（2026-10-23、系譜 9-291）。

⭐ なぜ要るか: 修論の第4章・第5章の結果待ちの値は本文で `???` にしてある。seed 1 の取り直し（`_ws`、Bug 76）が
そろったら、帯を作り直して埋める。帯は 2 通り作る — 系譜 9-285 の決定「seed 0 と新しい seed 1（`_ws`）で作り直し、
元の seed 1 の帯と並べる。帯の向きが変われば、その節の主張を取り下げる」と、9-281 の R1 の決定
「事前登録どおりの帯と種をそろえた帯の両方で読み、食い違ったら主張しない」のため。手で写すと取り違えるので機械で出す。

    python3 scripts/d6_bands.py              # 全部の組
    python3 scripts/d6_bands.py 4.4.3.3      # 節の番号で絞る
    python3 scripts/d6_bands.py --self-check # 既知の値（系譜 9-278・9-279・9-261・9-268・9-318）を再現するか

出すもの（run ごと）: 完走したか・best（exec_R_eps の最大）・ワーカーの種・軌跡からの量
（到達: 10 mm に入る step と最後の誤差の 5 話平均。押し: 箱の移動の 5 話平均。形: 各関節のギア比・先端の太さ・リンク長の合計）。
出すもの（組ごと）: 古い帯（seed 0 ＋ 元の seed 1）と新しい帯（seed 0 ＋ `_ws`）の上下と重なり。2 つで向きか重なりが
違えば ⛔ を出す。⚠️ 判定（事前登録の読み）はしない。読みはキューの頭と系譜にある。
⭐ 2026-11-01（系譜 9-318 の ㉘・㉙、D-6）: seed 2（`_s3`）を持つ形は 3 seed の帯（seed 0 ＋ 新しい seed 1 ＋ seed 2）も出す。
新しい帯と 3 seed の帯で関係が変われば ⚠️（9-318 の「重なればさらに狭める」の対象。⛔ の数には入れない）。
R2（種類 design）は best に加えて、関節ごとのギア比とリンク長の合計の帯も出す（㉙ の表）。
"""
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from check_before_conclusion import _worker_seeds  # noqa: E402

# 組: (節, 名前, 種類, [(形の名前, seed0, 元の seed1, 新しい seed1[, seed2])])。新しい seed1 が元と同じなら同じ名前を書く
#   seed2 は 9-318 で足した形だけ（㉘ の到達・押しの中間と長腕、㉙ の A1v）
#   ⭐ A1 の押しと R1 の凍結は、Bug 76 を直した後に起動したので元の seed 1 がすでに新しい種
GROUPS = [
    ('4.4.1', 'R1 形の凍結（押し、中間）', 'push', [
        ('凍結', 'tripo_pjp_mid_frozen_phys_fx', 'tripo_pjp_mid_frozen_phys_fx_s2', 'tripo_pjp_mid_frozen_phys_fx_s2', 'tripo_pjp_mid_frozen_phys_fx_s3'),
        ('最適化', 'tripo_pjp_mid_phys', 'tripo_pjp_mid_phys_s2', 'tripo_pjp_mid_phys_s2_ws', 'tripo_pjp_mid_phys_s3')]),
    ('4.4.2', 'R2 縦型 A1v（到達 対 押し）', 'design', [
        ('到達', 'e2e_a1v_reach_phys', 'e2e_a1v_reach_phys_s2', 'e2e_a1v_reach_phys_s2_ws', 'e2e_a1v_reach_phys_s3'),
        ('押し', 'e2e_a1v_pusher_phys', 'e2e_a1v_pusher_phys_s2', 'e2e_a1v_pusher_phys_s2_ws', 'e2e_a1v_pusher_phys_s3')]),
    ('4.4.3.1・4.4.3.2', 'R3 マトリクス 到達', 'reach', [
        ('短腕', 'tripo_pj_short_phys_cg', None, None),
        ('中間', 'tripo_pj_mid_phys', 'tripo_pj_mid_phys_s2', 'tripo_pj_mid_phys_s2_ws', 'tripo_pj_mid_phys_s3'),
        ('長腕', 'tripo_pj_long_phys', 'tripo_pj_long_phys_s2', 'tripo_pj_long_phys_s2_ws', 'tripo_pj_long_phys_s3')]),
    ('4.4.3.2', 'R3 マトリクス 押し', 'push', [
        ('短腕', 'tripo_pjp_short_phys_cg', None, None),
        ('中間', 'tripo_pjp_mid_phys', 'tripo_pjp_mid_phys_s2', 'tripo_pjp_mid_phys_s2_ws', 'tripo_pjp_mid_phys_s3'),
        ('長腕', 'tripo_pjp_long_phys', 'tripo_pjp_long_phys_s2', 'tripo_pjp_long_phys_s2_ws', 'tripo_pjp_long_phys_s3')]),
    ('4.4.3.3', 'R3 配分 到達', 'reach', [
        ('根元重', 'tripo_pjd_prox_phys', 'tripo_pjd_prox_phys_s2', 'tripo_pjd_prox_phys_s2_ws', 'tripo_pjd_prox_phys_s3'),
        ('先端重', 'tripo_pjd_dist_phys_cg', 'tripo_pjd_dist_phys_cg_s2', 'tripo_pjd_dist_phys_cg_s2_ws', 'tripo_pjd_dist_phys_cg_s3')]),
    ('4.4.3.3', 'R3 配分 押し', 'push', [
        ('根元重', 'tripo_pjdp_prox_phys', 'tripo_pjdp_prox_phys_s2', 'tripo_pjdp_prox_phys_s2_ws', 'tripo_pjdp_prox_phys_s3'),
        # seed 1 は門を直す前の run（9-279 の帯 [5.38, 7.04] が使った組）。取り直しは `_s2_ws`
        ('先端重', 'tripo_pjdp_dist_phys_cg', 'tripo_pjdp_dist_phys_s2', 'tripo_pjdp_dist_phys_s2_ws', 'tripo_pjdp_dist_phys_cg_s3')]),
    ('4.5.4', '閉ループ 伸ばす 到達（1.99 倍）', 'reach', [
        ('短腕', 'tripo_pj_short_phys_cg', None, None),
        ('1.99 倍', 'tripo_pjr199_phys', 'tripo_pjr199_phys_s2', 'tripo_pjr199_phys_s2_ws')]),
    ('4.5.4', '閉ループ 伸ばす 押し（2.11 倍）', 'push', [
        ('短腕', 'tripo_pjp_short_phys_cg', None, None),
        ('2.11 倍', 'tripo_pjr211p_phys', 'tripo_pjr211p_phys_s2', 'tripo_pjr211p_phys_s2_ws')]),
    # seed 2 は 9-334 ㊶（queue_1104_frozen_alloc_455.sh）
    ('4.5.5', '閉ループ 関節固定 到達（A1）', 'reach', [
        ('可動', 'e2e_a1_reach_phys', 'e2e_a1_reach_phys_s2', 'e2e_a1_reach_phys_s2_ws', 'e2e_a1_reach_phys_s3'),
        ('固定', 'e2e_a1_fix3_reach_phys', 'e2e_a1_fix3_reach_phys_s2', 'e2e_a1_fix3_reach_phys_s2_ws', 'e2e_a1_fix3_reach_phys_s3')]),
    ('4.5.5', '閉ループ 関節固定 押し（A1）', 'push', [
        ('可動', 'e2e_a1_pusher_phys', 'e2e_a1_pusher_phys_s2', 'e2e_a1_pusher_phys_s2', 'e2e_a1_pusher_phys_s3'),
        ('固定', 'e2e_a1_fix3_pusher_phys', 'e2e_a1_fix3_pusher_phys_s2', 'e2e_a1_fix3_pusher_phys_s2', 'e2e_a1_fix3_pusher_phys_s3')]),
    ('4.5.5', '第3層の提案 関節 1 固定 到達（中間）', 'reach', [
        # 関節 1 固定の seed 1 は新しい種でしか回していない（queue_1029_fix1_seed2.sh が `_s2_ws` と比べた）ので、可動も元・新とも `_s2_ws`
        ('可動', 'tripo_pj_mid_phys', 'tripo_pj_mid_phys_s2_ws', 'tripo_pj_mid_phys_s2_ws', 'tripo_pj_mid_phys_s3'),
        ('固定', 'tripo_pj_mid_phys_fix1', 'tripo_pj_mid_phys_fix1_s2', 'tripo_pj_mid_phys_fix1_s2', 'tripo_pj_mid_phys_fix1_s3')]),
    # 9-334 ㊵: 形を凍結した押しの配分（想定問答 Q58）。読みは凍結どうしの帯（seed 2 は 9-335 で足した）。最適化の帯は 4.4.3.3 の組
    ('9-334', '凍結の配分 押し', 'push', [
        ('根元重 凍結', 'tripo_pjdp_prox_frozen_phys_fx', 'tripo_pjdp_prox_frozen_phys_fx_s2', 'tripo_pjdp_prox_frozen_phys_fx_s2', 'tripo_pjdp_prox_frozen_phys_fx_s3'),
        ('先端重 凍結', 'tripo_pjdp_dist_frozen_phys_cg_fx', 'tripo_pjdp_dist_frozen_phys_cg_fx_s2', 'tripo_pjdp_dist_frozen_phys_cg_fx_s2', 'tripo_pjdp_dist_frozen_phys_cg_fx_s3')]),
]
EPOCHS = 200


def run_info(r):
    """完走したか・best・種・軌跡の量。run が無ければ None。"""
    if r is None:
        return None
    log = ROOT / 'single_run' / r / 'log' / 'log_train.txt'
    if not log.exists():
        return {'name': r, 'done': False, 'best': None, 'ep': 0}
    vals = [float(v) for v in re.findall(r'exec_R_eps\s+(-?[\d.]+)', log.read_text(errors='replace'))]
    info = {'name': r, 'ep': len(vals), 'done': len(vals) >= EPOCHS, 'best': max(vals) if vals else None}
    try:
        info['seeds'] = _worker_seeds(r)[0]
    except Exception:
        info['seeds'] = None
    tr = ROOT / 'single_run' / r / 'trace' / 'arm_trace.npz'
    if info['done'] and tr.exists():
        info.update(trace_metrics(tr))
    return info


def trace_metrics(p):
    """軌跡の 5 話から: 到達の 10 mm に入る step・最後の誤差、押しの箱の移動、形（ギア比・先端の太さ・リンク長の合計）。
    ⭐ 到達の 2 つの定義は系譜 9-278 の表（prox seed0 の 19〜21 step・0.28 mm など）を再現する（--self-check）"""
    z = np.load(p, allow_pickle=True)
    n = int(z['ep_count'])
    tg = np.asarray(z['target'], float) if 'target' in z.files else None
    steps, final = [], []
    for e in range(n):
        x = np.asarray(z['ep_xpos'][e], float)
        R = np.asarray(z['ep_xmat'][e], float)
        b = np.asarray(z['ep_bone'][e], float)
        tip = x[:, -1] + np.einsum('tij,j->ti', R[:, -1], b[-1])
        if tg is not None:
            d = np.linalg.norm(tip - tg, axis=1) * 1000
            steps.append(int(np.argmax(d < 10)) if (d < 10).any() else None)
            final.append(float(d[-1]))
    gear = np.asarray(z['ep_gear'][0], float)
    geom = np.asarray(z['ep_geom'][0], float)
    bone = np.asarray(z['ep_bone'][0], float)
    return {'steps10': steps, 'final_mm': float(np.mean(final)) if final else None,
            'cube_m': float(np.mean(np.asarray(z['ep_cube_dx'], float))) if 'ep_cube_dx' in z.files else None,
            'gears': [round(float(g), 1) for g in gear if np.isfinite(g)],
            'tip_r': round(float(geom[-1]), 3),
            'link_sum': round(float(np.linalg.norm(bone[1:], axis=1).sum()), 3)}


def band(*runs, key=lambda i: i['best']):
    """run（2 本か 3 本）の key の帯。1 本でも無い・未完走・値が無ければ None"""
    if any(r is None or not r['done'] for r in runs):
        return None
    try:
        v = [key(r) for r in runs]
    except (KeyError, IndexError, TypeError):
        return None
    if any(x is None for x in v):
        return None
    return (min(v), max(v))


def relation(b1, b2):
    """帯 b1 と b2: '上'（b1 が上で重ならない）・'下'・'重なる'"""
    if b1 is None or b2 is None:
        return None
    if b1[0] > b2[1]:
        return '上'
    if b1[1] < b2[0]:
        return '下'
    return '重なる'


def fmt_run(i, kind):
    if i is None:
        return '—'
    if not i['done']:
        return f"{i['name']}: 未完走（{i['ep']}/{EPOCHS}）"
    s = f"{i['name']}: best {i['best']:.2f}  種 {i['seeds']}"
    if kind == 'reach' and i.get('final_mm') is not None:
        s += f"  10 mm まで {i['steps10']} step・最後の誤差 {i['final_mm']:.2f} mm"
    if kind in ('push',) and i.get('cube_m') is not None:
        s += f"  箱の移動 {i['cube_m']:.3f} m"
    if i.get('gears') is not None:
        s += f"  ギア比 {i['gears']}  先端の太さ {i['tip_r']}  リンク長の合計 {i['link_sum']}"
    return s


def show(group):
    sec, name, kind, forms = group
    print(f'\n=== {sec} {name}')
    rows = []
    for label, s0, s1o, s1n, *rest in forms:
        s2 = rest[0] if rest else None
        a, bo = run_info(s0), run_info(s1o)
        bn = bo if s1n == s1o else run_info(s1n)
        c = run_info(s2)
        rows.append((label, a, bo, bn, c))
        print(f'  [{label}]')
        print(f'    seed0   {fmt_run(a, kind)}')
        if s1o:
            print(f'    seed1 元 {fmt_run(bo, kind)}')
        if s1n and s1n != s1o:
            print(f'    seed1 新 {fmt_run(bn, kind)}')
        if s2:
            print(f'    seed2   {fmt_run(c, kind)}')
        if s1o is None:
            print('    （1 シード）')
    has3 = any(r[4] is not None for r in rows)
    f = lambda b, d=2: '—' if b is None else f'[{b[0]:.{d}f}, {b[1]:.{d}f}]'  # noqa: E731
    bad = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            li, ai, boi, bni, ci = rows[i]
            lj, aj, boj, bnj, cj = rows[j]
            one = lambda r: (r[1]['best'], r[1]['best']) if r[1] and r[1]['done'] and r[2] is None else None  # noqa: E731
            old_i, old_j = band(ai, boi) or one(rows[i]), band(aj, boj) or one(rows[j])
            new_i, new_j = band(ai, bni) or one(rows[i]), band(aj, bnj) or one(rows[j])
            ro, rn = relation(old_i, old_j), relation(new_i, new_j)
            mark = ''
            if ro and rn and ro != rn:
                mark = '  ⛔ 古い帯と新しい帯で食い違う（9-285: 向きが変われば主張を取り下げる・9-281: R1 は主張しない）'
                bad += 1
            elif rn is None:
                mark = '  ⏳ 新しい帯がまだそろわない'
            print(f'  {li} 対 {lj}: 古い帯 {f(old_i)} 対 {f(old_j)} → {ro or "—"}  ／  新しい帯 {f(new_i)} 対 {f(new_j)} → {rn or "—"}{mark}')
            if has3:
                print('    best: ' + three(rows[i], rows[j], lambda r: r['best'], f))
            if kind == 'design':
                n = min(len(ai.get('gears') or []), len(aj.get('gears') or [])) if ai and aj else 0
                for k in range(n):
                    print(f'    ギア比 関節 {k + 1}: ' + three(rows[i], rows[j], lambda r, k=k: r['gears'][k], lambda b: f(b, 1)))
                print('    リンク長の合計: ' + three(rows[i], rows[j], lambda r: r['link_sum'], lambda b: f(b, 3)))
    return bad


def three(ri, rj, key, f):
    """新しい帯（seed0 ＋ 新しい seed1）と 3 seed の帯（＋ seed2）を並べる。関係が変われば ⚠️"""
    def bands(r):  # 1 シードの形（元の seed1 が無い）はその 1 本の値を 2 seed・3 seed の帯に使う
        _, a, bo, bn, c = r
        if bo is None:
            one = band(a, key=key)
            return one, one
        return band(a, bn, key=key), (band(a, bn, c, key=key) if c else None)
    (n_i, t_i), (n_j, t_j) = bands(ri), bands(rj)
    ci, cj = ri[4], rj[4]
    rn, rt = relation(n_i, n_j), relation(t_i, t_j)
    mark = ''
    if rn and rt and rn != rt:
        mark = '  ⚠️ 3 seed で関係が変わる（9-318: 重なればさらに狭める）'
    elif (ci or cj) and rt is None:
        mark = '  ⏳ 3 seed の帯がまだそろわない'
    return f'2 seed {f(n_i)} 対 {f(n_j)} → {rn or "—"}  ／  3 seed {f(t_i)} 対 {f(t_j)} → {rt or "—"}{mark}'


def self_check():
    """既知の値を再現できるか（再現できない道具の出力は D-6 に使わない）"""
    ok = True
    def chk(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f'  {"✅" if cond else "⛔"} {name}')
    p = run_info('tripo_pjd_prox_phys'); d = run_info('tripo_pjd_dist_phys_cg_s2')
    chk('配分 到達 prox seed0: 10 mm まで 19・21・19・19・20 step、最後の誤差 0.28 mm（9-278）',
        p['steps10'] == [19, 21, 19, 19, 20] and round(p['final_mm'], 2) == 0.28)
    chk('配分 到達 dist seed1: 最後の誤差 0.23 mm（9-278）', round(d['final_mm'], 2) == 0.23)
    chk('マトリクス 押し 長腕の古い帯 [22.68, 24.92]（9-259）',
        band(run_info('tripo_pjp_long_phys'), run_info('tripo_pjp_long_phys_s2')) == (22.68, 24.92))
    chk('配分 押し dist の古い帯 [5.38, 7.04]（9-279）',
        band(run_info('tripo_pjdp_dist_phys_cg'), run_info('tripo_pjdp_dist_phys_s2')) == (5.38, 7.04))
    chk('マトリクス 到達 中間 対 長腕は古い帯で「上」（9-268）',
        relation(band(run_info('tripo_pj_mid_phys'), run_info('tripo_pj_mid_phys_s2')),
                 band(run_info('tripo_pj_long_phys'), run_info('tripo_pj_long_phys_s2'))) == '上')
    a = run_info('e2e_a1v_reach_phys')
    chk('R2 到達 seed0 の best −25.42（4.4.2 の旧本文）', round(a['best'], 2) == -25.42)
    m = {k: run_info(k) for k in ('tripo_pj_mid_phys', 'tripo_pj_mid_phys_s2_ws', 'tripo_pj_mid_phys_s3',
                                  'tripo_pj_long_phys', 'tripo_pj_long_phys_s2_ws', 'tripo_pj_long_phys_s3',
                                  'tripo_pjp_mid_phys', 'tripo_pjp_mid_phys_s2_ws', 'tripo_pjp_mid_phys_s3',
                                  'tripo_pjp_long_phys', 'tripo_pjp_long_phys_s2_ws', 'tripo_pjp_long_phys_s3')}
    b3 = lambda f: band(m[f], m[f + '_s2_ws'], m[f + '_s3'])  # noqa: E731
    chk('㉘ 3 seed の帯: 到達 中間 [−1.85, −1.64]・長腕 [−9.68, −7.71]、押し 長腕 [20.71, 24.92]・中間 [9.37, 10.44]（9-318）',
        b3('tripo_pj_mid_phys') == (-1.85, -1.64) and b3('tripo_pj_long_phys') == (-9.68, -7.71)
        and b3('tripo_pjp_long_phys') == (20.71, 24.92) and b3('tripo_pjp_mid_phys') == (9.37, 10.44))
    R = [run_info(k) for k in ('e2e_a1v_reach_phys', 'e2e_a1v_reach_phys_s2_ws', 'e2e_a1v_reach_phys_s3')]
    P = [run_info(k) for k in ('e2e_a1v_pusher_phys', 'e2e_a1v_pusher_phys_s2_ws', 'e2e_a1v_pusher_phys_s3')]
    rd = lambda b, d: tuple(round(x, d) for x in b)  # noqa: E731
    g = lambda runs, k: rd(band(*runs, key=lambda r: r['gears'][k]), 1)  # noqa: E731
    ls = lambda runs: rd(band(*runs, key=lambda r: r['link_sum']), 3)  # noqa: E731
    chk('㉙ 関節 3: 到達 [78.2, 109.0] 対 押し [103.8, 149.9] で重なる（9-318）',
        g(R, 2) == (78.2, 109.0) and g(P, 2) == (103.8, 149.9) and relation(g(R, 2), g(P, 2)) == '重なる')
    chk('㉙ 関節 4: 到達 [93.5, 142.1] 対 押し [148.7, 150.0] で分かれる（9-318）',
        g(R, 3) == (93.5, 142.1) and g(P, 3) == (148.7, 150.0) and relation(g(R, 3), g(P, 3)) == '下')
    chk('㉙ リンク長の合計: 到達 [1.103, 1.78] 対 押し [1.957, 3.211]（9-318）',
        ls(R) == (1.103, 1.78) and ls(P) == (1.957, 3.211))
    chk('重なる帯は「重なる」', relation((1.0, 3.0), (2.0, 4.0)) == '重なる')
    chk('種は seed0 が [1, 2, 3, 4]、Bug 76 の後の seed1 は [1001, 1002, 1003, 1004]',
        run_info('tripo_pjp_mid_frozen_phys_fx')['seeds'] == [1, 2, 3, 4]
        and run_info('tripo_pjp_mid_frozen_phys_fx_s2')['seeds'] == [1001, 1002, 1003, 1004])
    return 0 if ok else 1


def main():
    args = sys.argv[1:]
    if args == ['--self-check']:
        return self_check()
    bad = sum(show(g) for g in GROUPS if not args or any(g[0].startswith(a) or a in g[0] for a in args))
    print(f'\n古い帯と新しい帯で食い違う組: {bad}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
