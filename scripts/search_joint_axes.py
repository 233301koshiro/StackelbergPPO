#!/usr/bin/env python3
"""関節軸の組み合わせを**第1層で枝刈り**する（実験系譜 9-125・9-127）。

⛔ **何が問題か**: 関節軸は `mesh_to_params.py:264` の**1 行**で決まっており、
**平面アーム（全 Z）か お辞儀アーム（根元 Z・以降 Y）の 2 択しかない**（9-124 の弱点①）。
⚠️ **スケッチから形態を起こすと謳いながら、関節軸だけは人が決めた 2 択である。**

⚠️ **助教の指摘17 は「X/Y/XY の総当りは 3^N で爆発する」と述べた。実際に回らない:**

| 関節数 | 通り | 学習時間（200 epoch × 約 7 時間） |
|---|---|---|
| 3 | 27 | 7.9 日 |
| **4（本研究）** | **81** | ⛔ **23.6 日** |

⭐ **本スクリプトの答え: 学習の前に第1層で枝刈りする。**
`diagnose_morphology.layer1` は**可動域込みの到達可能性を学習なしで判定でき**（9-78・9-91）、
**実測 0.3 秒**である。⭐ **81 通りでも 24 秒。**

⭐⭐ **第1層は「棄却の側だけが確実」という非対称の保証を持つ**（9-78）ので、
**枝刈りで取りこぼす心配がない。**「届かない」と言われた配置は本当に届かない。

⚠️ **限界**: 第1層は「届くか」しか見ない。**押しやすさや精度は分からない。**
生き残りが多ければ、そこは結局学習で比べることになる。

    python3 scripts/search_joint_axes.py --xml e2e_a1v --task reach
    python3 scripts/search_joint_axes.py --xml e2e_a1v --task reach --axes x,y,z
"""
import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

AXIS_VEC = {'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--xml', required=True, help='元の XML 名（拡張子なし）')
    ap.add_argument('--task', default='reach', choices=['reach', 'pusher'])
    ap.add_argument('--axes', default='x,y,z', help='試す軸（既定 x,y,z ＝ 助教案の X/Y/XY 相当）')
    ap.add_argument('--target', nargs=3, type=float, default=None)
    ap.add_argument('--length-free', action='store_true')
    a = ap.parse_args()

    import diagnose_morphology as diag

    xml = str(ROOT / 'assets' / 'mujoco_envs' / f'{a.xml}.xml')
    geo0 = diag.parse_arm_xml(xml)
    n = len(geo0['axes'])
    cand = [c.strip() for c in a.axes.split(',') if c.strip() in AXIS_VEC]
    combos = list(itertools.product(cand, repeat=n))
    target = np.array(a.target if a.target else [0.8, 0.0, 0.15])

    print(f"=== {a.xml}: {n} 関節 × 軸 {cand} = **{len(combos)} 通り**")
    print(f"    タスク {a.task}  目標 {target}")
    print(f"⚠️ 学習で総当りすると 約 {len(combos)*7/24:.1f} 日。⭐ 第1層なら下記の秒数。\n")

    t0 = time.time()
    ok, ng = [], []
    cur = tuple(''.join(k for k, v in AXIS_VEC.items() if v == tuple(x)) or '?'
                for x in geo0['axes'])
    for combo in combos:
        geo = dict(geo0)
        geo['axes'] = [AXIS_VEC[c] for c in combo]
        try:
            findings, fatal = diag.layer1(geo, a.task, target,
                                          length_frozen=not a.length_free)
        except Exception as e:
            ng.append((combo, f'判定不能: {type(e).__name__}')); continue
        if fatal:
            why = next((m.split('\n')[0][:44] for lv, m in findings if lv == 'fatal'), '')
            ng.append((combo, why))
        else:
            ok.append(combo)
    el = time.time() - t0

    print(f"⭐ **{el:.1f} 秒**で {len(combos)} 通りを判定した\n")
    print(f"  ✅ 第1層を通過: **{len(ok)}** 通り")
    print(f"  ⛔ 棄却:        **{len(ng)}** 通り（⭐ **学習しなくてよい**）")
    if combos:
        print(f"  ⭐ 削減率: **{len(ng)/len(combos)*100:.0f} %**"
              f"（学習 {len(combos)*7/24:.1f} 日 → {len(ok)*7/24:.1f} 日）")

    print(f"\n=== 現在の実装が作る配置")
    print(f"  {'-'.join(cur)}  ← `mesh_to_params.py:264` の 1 行が決めている")
    print(f"  第1層の判定: {'✅ 通過' if tuple(cur) in ok else '⛔ 棄却 or 対象外'}")

    if ok:
        print(f"\n=== ✅ 通過した配置（学習に回す候補）")
        for c in ok[:20]:
            mark = '  ← ⭐ 現在の実装' if tuple(c) == tuple(cur) else ''
            print(f"  {'-'.join(c)}{mark}")
        if len(ok) > 20:
            print(f"  … ほか {len(ok)-20} 通り")
    if ng:
        print(f"\n=== ⛔ 棄却された配置（例）")
        for c, why in ng[:5]:
            print(f"  {'-'.join(c)}: {why}")
    print("\n⚠️ **第1層は「届くか」しか見ない。**押しやすさ・精度は学習で比べること。")
    return 0


if __name__ == '__main__':
    sys.exit(main())
