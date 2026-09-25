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
    walls = []
    for g in t.findall('worldbody/geom'):
        if g.get('type') == 'box' and g.get('size') and g.get('pos'):
            p = np.array([float(x) for x in g.get('pos').split()])
            s = np.array([float(x) for x in g.get('size').split()])
            walls.append((g.get('name') or '(no-name)', p, s))
    return slide, walls


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
    xpos, xmat, bone = z['xpos'], z['xmat'], z['bone_offset']
    ends = xpos + np.einsum('tlij,lj->tli', xmat, bone)
    pts = np.concatenate([xpos, ends], axis=1)          # 全リンクの両端
    cube = np.asarray(z['cube'], dtype=float)

    flags = []
    # 1. 対象物が slide の可動限界に達していないか（Bug 39）
    for ax, key in ((0, 'cube_slide'), (1, 'cube_slide2')):
        if key not in slide or not len(cube):
            continue
        lo, hi = slide[key]
        mv = cube[:, ax] - cube[0, ax]                  # 初期位置からの変位 = 関節の値
        for v, bound, side in ((mv.max(), hi, 'upper'), (mv.min(), lo, 'lower')):
            if bound and abs(v) >= abs(bound) * TOL:
                flags.append(f'cube が {key} の{side} {bound:g} に達した（{v:.2f}）')
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
    if not is_reach and len(cube) and float(np.abs(cube[:, :2] - cube[0, :2]).max()) < 0.001:
        flags.append('cube が 1 mm も動いていない（押しタスクなのに対象が静止）')

    if not quiet:
        print(f'=== {run}  xml={xml}')
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
