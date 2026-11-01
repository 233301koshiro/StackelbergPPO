#!/usr/bin/env python3
"""**押しで箱をどこで打ったか**（系譜 9-331 の追記、2026-11-02）。

箱が 2 mm 動き出す step とその 1 step 前のうち、箱に近い方の姿勢で、箱へ最も近いカプセル（リンク）と
その線分上の位置 s（0 = カプセルの始点、1 = 先端）を全話で出す。
分類: tip（最後のリンクで s > 0.97）・side-last（最後のリンクの側面）・other（ほかのリンク）・far（隙間が 20 mm を超え、判定しない）。
`cap_fromto` の無い古い軌跡は骨で代える（関節の後ろへのはみ出しを見ない。列 cap/bone）。

    python3 scripts/hit_part.py                 # 軌跡のある全 run
    python3 scripts/hit_part.py 'e2e_a1v_pusher_phys*'
"""
import sys, glob, zipfile, yaml, collections
import numpy as np
sys.path.insert(0, 'scripts')
from plot_run import cube_half
S = np.linspace(0, 1, 201)
def seg_box(a, b, lo, hi):
    pts = a + S[:, None] * (b - a)
    c, h = (lo + hi) / 2, (hi - lo) / 2
    q = np.abs(pts - c) - h
    d = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)
    i = int(np.argmin(d)); return float(d[i]), float(S[i])
thesis = ''.join(open(f, encoding='utf-8').read() for f in glob.glob('docs/研究応用/修論ドラフト/第[1-5]章*.md') + glob.glob('docs/研究応用/修論ドラフト/付録*.md'))
pat = sys.argv[1] if len(sys.argv) > 1 else '*'
rows = []
for p in sorted(glob.glob(f'single_run/{pat}/trace/arm_trace.npz')):
    r = p.split('/')[1]
    try:
        cfg = yaml.safe_load(open(f'single_run/{r}/.hydra/config.yaml'))
    except Exception:
        continue
    xml = cfg.get('xml_name'); ch = cube_half(xml) if xml else None
    if ch is None: continue
    d = np.load(p, allow_pickle=True)
    names = [str(n) for n in d['body_names']]
    if 'ep_xpos' in d.files and 'ep_bone' in d.files:
        E = range(len(d['ep_xpos']))
        get = lambda e: (np.asarray(d['ep_xpos'][e], float), np.asarray(d['ep_xmat'][e], float), np.asarray(d['ep_bone'][e], float),
                         np.asarray(d['ep_geom'][e] if d['ep_geom'][e] is not None else d['geom_size'], float).ravel(),
                         (np.asarray(d['ep_cap'][e], float) if 'ep_cap' in d.files and d['ep_cap'][e] is not None else (np.asarray(d['cap_fromto'], float) if 'cap_fromto' in d.files else None)),
                         np.asarray(d['ep_cubes'][e], float))
    else:
        E = [0]
        get = lambda e: (d['xpos'], d['xmat'], d['bone_offset'], np.asarray(d['geom_size'], float).ravel(),
                         np.asarray(d['cap_fromto'], float) if 'cap_fromto' in d.files else None, np.asarray(d['cube'], float))
    for e in E:
        xp, xm, bo, rad, cf, cube = get(e)
        if len(cube) == 0 or np.abs(cube[:, 0]).max() > 50: break   # 到達（箱を退避）
        mv = np.where(np.linalg.norm(cube - cube[0], axis=1) > 0.002)[0]
        if not len(mv) or mv[0] == 0: continue
        cand = []
        for k in (int(mv[0]) - 1, int(mv[0])):
            ends = xp[k] + np.einsum('lij,lj->li', xm[k], bo)
            if cf is not None and len(cf) == len(bo):
                a = xp[k] + np.einsum('lij,lj->li', xm[k], cf[:, :3]); b = xp[k] + np.einsum('lij,lj->li', xm[k], cf[:, 3:])
            else:
                a, b = xp[k].copy(), ends
            lo, hi = cube[k] - ch, cube[k] + ch
            best = None
            for i in range(1, len(bo)):
                dist, s_ = seg_box(a[i], b[i], lo, hi)
                g = dist - rad[i]
                if best is None or g < best[0]: best = (g, i, s_)
            cand.append((abs(best[0]), k, best))
        _, k, best = min(cand)
        g, i, s = best
        last = (i == len(bo) - 1)
        kind = ('far' if abs(g) > 0.02 else 'tip' if (last and s > 0.97) else ('side-last' if last else f'other:{names[i]}'))
        rows.append((r, e, k, kind, names[i], s, g, ('`' + r + '`' in thesis) or (r in thesis), cf is not None))
by = collections.defaultdict(list)
for row in rows: by[row[0]].append(row)
print(f'{"run":44s} 話 tip side-last other  内訳（s の中央値）  修論 cap')
tot = collections.Counter()
for r, rs in by.items():
    c = collections.Counter(x[3].split(':')[0] for x in rs); tot.update(c)
    oth = collections.Counter(x[4] for x in rs if x[3].startswith('other'))
    ss = np.median([x[5] for x in rs])
    print(f'{r:44s} {len(rs):2d} {c["tip"]:3d} {c["side-last"]:5d} {c["other"]:5d} far {c["far"]:2d}  s中央 {ss:.2f} gap中央 {np.median([x[6] for x in rs])*1000:5.1f}mm {dict(oth) if oth else ""} {"★" if rs[0][7] else ""} {"cap" if rs[0][8] else "bone"}')
print('合計', dict(tot), '話', sum(tot.values()), 'run', len(by))
