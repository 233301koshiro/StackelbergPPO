#!/usr/bin/env python3
"""**動く物体すべての極値を出し、境界を越えていないか**を全 run に掛ける（9-96 の一般化）。

9-96 の規律「タスクに動くものが 2 つあれば両方の極値を機械で出し、壁・可動域・目標の
境界を越えているものが無いかを機序を考える前に確かめる」を、1 run 手作業ではなく
**全軌跡へ一度に掛ける**形にした。

既に検査があるもの（ここでは再掲しない）:
  床下              plot_run.py が図の題に z 最小を焼く
  柱への侵入        check_obstacle_clearance.py
  cube への食い込み  check_cube_penetration.py（⚠️ 標本が粗いので深さは出さない。Bug 48）

⭐ **ここで初めて見るのは次の 2 つ。どちらも実際に踏んだ型である。**

| 見るもの | 踏んだ事故 |
|---|---|
| **対象物が slide の可動限界に達していないか** | **Bug 39**: cube の x 可動上限が 1.25 m 対 200 m と **160 倍違う**のに `fwd_cube` の生値を比べ、大小を逆に読んだ。⚠️ **上限に張り付いていれば、その値はタスクではなく上限を測っている** |
| **腕が壁の外へ出ていないか** | **9-80**: 再生も図もしたが**パックしか見ず**、同じファイルの腕の `xpos` を読まなかった。腕は壁の外面 0.90 に対し **1.31** まで出ていた。**7 時間の run を 1 本無駄にした** |

⭐ 2026-10-22（系譜 9-283 ③ の残り）: 以前は代表の 1 話（`xpos`・`cube`）しか見なかった。話ごとの記録（`ep_*`）があれば
全話を見る（無い古い軌跡は代表の話だけ）。可動限界と壁は 1 話でも越えれば ⛔、「対象が動かない」は全話が動かないときだけ ⛔
（一部の話で動かないのは方策の出来で、接触が効いていない症状ではない）。

    python3 scripts/check_extremes.py          # 全軌跡
    python3 scripts/check_extremes.py <run名>  # 1 本
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
TOL = 0.99          # 可動域のこの割合を超えたら「張り付き」とみなす


def limits(xml_name):
    """その XML の境界: cube の slide 可動域と、worldbody 直下の壁 geom。

    ⚠️ **worldbody 直下の geom は Choreonoid の物理には存在しない**（9-98）。
    それでも出すのは、**存在しない壁を越えていること自体が症状**だからである。
    """
    t = ET.parse(ROOT / 'assets' / 'mujoco_envs' / f'{xml_name}.xml')
    slide = {}
    for j in t.iter('joint'):
        if j.get('type') == 'slide' and j.get('range'):
            lo, hi = (float(x) for x in j.get('range').split())
            slide[j.get('name')] = (lo, hi)
    # 関節の値 0 の位置（cube の body の pos）。⛔ 2026-10-22: 以前は話の最初の位置を 0 とみなし、
    # 初期位置にくじがある課題（ホッケー y ±0.25）で変位を誤った
    cb = [b for b in t.iter('body') if b.get('name') == 'cube']
    slide['_origin'] = np.array([float(x) for x in cb[0].get('pos', '0 0 0').split()]) if cb else None
    walls = []
    for g in t.findall('worldbody/geom'):
        if g.get('type') == 'box' and g.get('size') and g.get('pos'):
            p = np.array([float(x) for x in g.get('pos').split()])
            s = np.array([float(x) for x in g.get('size').split()])
            walls.append((g.get('name') or '(no-name)', p, s))
    return slide, walls


def episodes(z):
    """話ごとの (xpos, xmat, bone, cube)。`ep_*` が無い古い軌跡は代表の 1 話だけ。
    `ep_bone` が無い軌跡（hockey_bank7・8）は代表の骨を使う（再生は方策の平均の行動なので、形は全話で同じ）。"""
    if 'ep_xpos' in z.files and len(z['ep_xpos']):
        out = []
        for e in range(len(z['ep_xpos'])):
            x = np.asarray(z['ep_xpos'][e], float)
            if x.ndim != 3 or not len(x):
                continue
            b = z['ep_bone'][e] if 'ep_bone' in z.files else z['bone_offset']
            out.append((x, np.asarray(z['ep_xmat'][e], float), np.asarray(b, float),
                        np.asarray(z['ep_cubes'][e], float)))
        if out:
            return out
    return [(z['xpos'], z['xmat'], z['bone_offset'], np.asarray(z['cube'], dtype=float))]


def check(run, quiet=False):
    d = ROOT / 'single_run' / run
    tz = d / 'trace' / 'arm_trace.npz'
    if not tz.exists():
        return None
    xml = yaml.safe_load(open(d / '.hydra' / 'config.yaml')).get('xml_name')
    try:
        slide, walls = limits(xml)
    except (FileNotFoundError, ET.ParseError):
        return None
    z = np.load(tz, allow_pickle=True)
    eps = episodes(z)
    # 全リンクの両端（全話を step の向きにつなぐ）
    pts = np.concatenate([np.concatenate([x, x + np.einsum('tlij,lj->tli', m, b)], axis=1)
                          for x, m, b, _ in eps], axis=0)
    cubes = [c for *_, c in eps if len(c)]
    cube = np.concatenate(cubes, axis=0) if cubes else np.zeros((0, 3))

    flags = []
    # 1. 対象物が slide の可動限界に達していないか（Bug 39）
    for ax, key in ((0, 'cube_slide'), (1, 'cube_slide2')):
        if key not in slide or not cubes:
            continue
        lo, hi = slide[key]
        org = slide['_origin']
        for e, c in enumerate(cubes):
            if np.abs(c[:, 0]).max() > 50:              # 到達で退避した箱（cube_x_offset=100、9-234）
                continue
            mv = c[:, ax] - (org[ax] if org is not None else c[0, ax])     # 関節の値
            # 上限は上から、下限は下から近づいたときだけ（⛔ 以前は abs で比べ、退避した x 100 を下限 −2 と取り違えうる）
            for v, bound, side, hit in ((mv.max(), hi, 'upper', mv.max() >= hi - (1 - TOL) * abs(hi)),
                                        (mv.min(), lo, 'lower', mv.min() <= lo + (1 - TOL) * abs(lo))):
                if bound and hit:
                    n = int(((mv >= hi - (1 - TOL) * abs(hi)) if side == 'upper' else (mv <= lo + (1 - TOL) * abs(lo))).sum())
                    flags.append(f'cube が {key} の{side} {bound:g} に達した（{v:.2f}、話 {e}、{n} step）')
    # 2. 腕が壁の内部へ入っていないか（9-80）
    for name, p, s in walls:
        out = (np.abs(pts[:, :, :3] - p) > s).all(axis=2)   # 3 軸とも箱の外
        n_in = int((~out).sum())
        if n_in:
            flags.append(f'腕が壁 `{name}` の内部に延べ {n_in} 標本入った')
    # 3. 対象物が初期位置から動いていない（接触が効いていない疑い）
    # ⚠️ **到達タスクでは cube が動かないのが正常。**ここで場合分けしないと
    #   全 Reach run で毎回出て、読まれない検査になる（§5-2 ①）。
    ov = d / '.hydra' / 'overrides.yaml'
    is_reach = 'use_reach=true' in (ov.read_text(encoding='utf-8') if ov.exists() else '')
    if not is_reach and cubes and all(float(np.abs(c[:, :2] - c[0, :2]).max()) < 0.001 for c in cubes):
        flags.append(f'cube が 1 mm も動いていない（押しタスクなのに対象が静止、全 {len(cubes)} 話）')

    if not quiet:
        print(f'=== {run}  xml={xml}  （全 {len(eps)} 話）')
        print(f'    腕 x[{pts[:,:,0].min():.2f},{pts[:,:,0].max():.2f}] '
              f'y[{pts[:,:,1].min():.2f},{pts[:,:,1].max():.2f}] '
              f'z[{pts[:,:,2].min():.2f},{pts[:,:,2].max():.2f}]')
        if len(cube):
            print(f'    cube x[{cube[:,0].min():.2f},{cube[:,0].max():.2f}] '
                  f'y[{cube[:,1].min():.2f},{cube[:,1].max():.2f}]')
        for f in flags:
            print(f'    ⛔ {f}')
        if not flags:
            print('    ✅ 境界を越えたものは無い')
    return flags


def main():
    if len(sys.argv) > 1:
        check(sys.argv[1])
        return
    runs = sorted(p.parent.parent.name for p in (ROOT / 'single_run').glob('*/trace/arm_trace.npz'))
    bad = {}
    for r in runs:
        f = check(r, quiet=True)
        if f:
            bad[r] = f
    print(f'軌跡 {len(runs)} 本を検査した。')
    if not bad:
        print('✅ 境界を越えた run は無い。')
        return
    print(f'⛔ **{len(bad)} 本で境界を越えていた。**\n')
    for r, fs in bad.items():
        print(f'  {r}')
        for x in fs:
            print(f'      ⛔ {x}')


main()
