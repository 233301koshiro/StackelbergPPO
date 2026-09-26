#!/usr/bin/env python3
"""**狙える向きに当てられるか**を幾何で測る（ホッケー Phase 3 / 9-178）。

⭐ 9-177 で**反射解が実在する**ようになった（0 → 7 通り）。
⭐⭐ **ホッケーが Reach / Pusher に無い問いを出せるようになった**:
  Reach・Pusher は「**届くか**」を問う。ホッケーは「**要求される速度の向きを作れるか**」を問う。

**打点の幾何**: パック位置 P から向き û へ飛ばすには、先端は**反対側**に触れる必要がある。

    打点 S = P − (r_puck + r_tip)·û

⭐ **したがって「S に届くか」という到達判定に落ちる。**第1層の機構をそのまま使える。

⚠️ **本スクリプトは判定器を変えない。**まず「現在の腕で差が出るか」を測る。
⛔ **全部の向きに届くなら判定器に基準を足す意味が無い**（9-72 の規律: 今日の件数で仕様の是非を決めないが、
   **そもそも差が出ない量なら discriminator にならない**）。

    python3 scripts/probe_shot_strikeability.py e2e_hockey_wall3
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_morphology import parse_arm_xml, planar_chain_reach      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    xml = sys.argv[1] if len(sys.argv) > 1 else 'e2e_hockey_wall3'
    geo = parse_arm_xml(ROOT / 'assets' / 'mujoco_envs' / f'{xml}.xml')
    lengths = geo['lengths'] if isinstance(geo, dict) else geo[0]
    ranges = geo.get('ranges_deg') if isinstance(geo, dict) else None
    radii = geo.get('radii') if isinstance(geo, dict) else None
    print(f'=== {xml}')
    print(f'  リンク長 {np.round(lengths, 4)}  総リーチ {sum(lengths):.4f} m')
    if ranges:
        print(f'  可動域 {ranges}')
    r_tip = float(radii[-1]) if radii else 0.06
    r_puck = 0.05
    d_strike = r_puck + r_tip
    print(f'  先端半径 {r_tip:.3f} ＋ パック半幅 {r_puck:.3f} → 打点は中心から {d_strike:.3f} m')

    # 9-177 で反射解が出た初期 y と、そこで入った撃ち出し角
    ys = [0.0, 0.10, 0.20, 0.30, 0.40]
    angles = np.deg2rad(np.arange(-60, 61, 6))      # probe_shot_choreonoid と同じ刻み
    puck_x, puck_z = 0.55, 0.2125
    print(f'\n{"パック y":>9} {"狙える向き":>12} {"うち打点に届く":>16}')
    tot_dir = tot_ok = 0
    for y in ys:
        ok = 0
        for a in angles:
            u = np.array([np.cos(a), np.sin(a)])
            S = np.array([puck_x, y]) - d_strike * u       # 打点（水平面）
            d = float(np.hypot(S[0], S[1]))                # 根元からの水平距離
            done, reach, dmin, slack = planar_chain_reach(
                lengths, ranges or [], d, puck_z, steps=41)
            if not done or reach:
                ok += 1
        tot_dir += len(angles); tot_ok += ok
        mark = '⛔' if ok == 0 else ('⚠️' if ok < len(angles) else '⭐')
        print(f'{y:>9.2f} {len(angles):>12} {mark} {ok:>13} / {len(angles)}')
    print(f'\n  合計 {tot_ok} / {tot_dir}')
    if tot_ok == tot_dir:
        print('  ⛔⛔ **全ての向きに届く。この形態では discriminator にならない。**')
        print('  ⭐ 判定器に基準を足しても、この形態は棄却されない ＝ 差を出すには形態を変える必要がある')
    else:
        print('  ⭐⭐ **届かない向きがある。判定器の基準として機能しうる。**')


main()
