#!/usr/bin/env python3
"""読み 59（系譜 9-354、ばねの箱の取り直し 6 本）の量を数える。学習も Choreonoid も要らない（再生済みの軌跡を読むだけ）。

前提（先に回しておくもの）:
  各 run の best を 50 話再生して trace/arm_trace_ep50.npz に置く（1 本 1〜2 分）:
    EVAL_RESTORE_DIR=single_run/<run> EVAL_CHECKPOINT=best TRACE_EPISODES=50 \\
    TRACE_OUT=single_run/<run>/trace/arm_trace_ep50.npz USE_CHOREONOID=1 OMP_NUM_THREADS=1 \\
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/record_arm_trace.py > single_run/<run>/trace/ep50.log 2>&1
  第2層の出力は別に diagnose_morphology.py を EVAL_RESTORE_DIR で回して読む（このスクリプトは数えない）。

数えるもの（読み 59 の定義のまま）:
  触れた   見張りの再生（trace_mid/ep*.npz、各 5 話）か best の 50 話のどこかで、箱が xy で 1 mm 以上動いた話がある
  d_end    最後の step の箱と目標 (1.0, 0.30) の水平の距離。50 話の中央値。届いた＝中央値 ≤ 0.05 m
  旋回     旋回関節（body `1`）のギア比（ep_gear[:, 1]）。上限 150 の 20 % 以内＝125 以上
  床下     plot_run.py と同じ定義（カプセルの面、床に据えた根元の縦の柱は除く、FLOOR_TOL より下）。
           箱がずれている（初期位置から 1 mm 超）step のうち、腕が床下にある step を数える（床への突っ張りで保っていないか）
  --gap    箱が動き出す直前 4 step の、腕のカプセルと箱の面の隙間（mm）。腕が当たって動いたかの確かめ（既定は出さない）

使い方:
  python3 scripts/judge_reading59.py          # 6 本の表
  python3 scripts/judge_reading59.py --gap e2e_a1v_spring900_cw1
"""
import argparse
import glob
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from check_invariants import FLOOR_TOL  # noqa: E402
from plot_run import cube_half  # noqa: E402

R = os.path.join(ROOT, 'single_run')
TGT = np.array([1.0, 0.30])
RUNS = ['e2e_a1v_spring350_cw1', 'e2e_a1v_spring350_s2_ws_cw1', 'e2e_a1v_spring350_s3_cw1',
        'e2e_a1v_spring900_cw1', 'e2e_a1v_spring900_s2_ws_cw1', 'e2e_a1v_spring900_s3_cw1']
KEYS = ('ep_cubes', 'ep_xpos', 'ep_xmat', 'ep_cap', 'ep_geom', 'ep_bone')


def episodes(z):
    for e in range(len(z['ep_cubes'])):
        yield (e,) + tuple(np.asarray(z[k][e], float) for k in KEYS)


def cap_ends(xp, xm, cap):
    return xp + np.einsum('tlij,lj->tli', xm, cap[:, :3]), xp + np.einsum('tlij,lj->tli', xm, cap[:, 3:])


def capsule_floor(xp, xm, cap, rad, bo):
    """各 step のカプセルの面の最も低い z（plot_run.py と同じ: 根元の球と、床に据えた縦の柱は除く）。"""
    c0, c1 = cap_ends(xp, xm, cap)
    keep = [i for i in range(1, len(bo)) if not (i == 1 and abs(bo[1][2]) > 0.9 * np.linalg.norm(bo[1]) + 1e-9)]
    return np.nanmin((np.minimum(c0[..., 2], c1[..., 2]) - rad)[:, keep], axis=1)


def moved(cube):
    return np.linalg.norm(cube[:, :2] - cube[0, :2], axis=1)


def summary(run):
    d = os.path.join(R, run)
    mid = []
    for p in sorted(glob.glob(os.path.join(d, 'trace_mid', 'ep*.npz'))):
        z = np.load(p, allow_pickle=True)
        n = sum(moved(np.asarray(c, float)).max() >= 0.001 for c in z['ep_cubes'])
        mid.append(f'{os.path.basename(p)[2:6]}:{n}/{len(z["ep_cubes"])}')
    p = os.path.join(d, 'trace', 'arm_trace_ep50.npz')
    if not os.path.exists(p):
        return f'{run}\n  見張り {" ".join(mid)}\n  ⛔ {p} が無い（上の前提の再生を先に回す）'
    z = np.load(p, allow_pickle=True)
    E = len(z['ep_cubes'])
    touched, dend, nfl, nmov, nov, nstep, czmin, ov_eps = 0, [], 0, 0, 0, 0, np.inf, 0
    for e, cube, xp, xm, cap, rad, bo in episodes(z):
        disp = moved(cube)
        touched += disp.max() >= 0.001
        dend.append(np.linalg.norm(cube[-1, :2] - TGT))
        cz = capsule_floor(xp, xm, cap, rad, bo)
        fl, mv = cz < -FLOOR_TOL, disp > 0.001
        nfl += fl.sum(); nmov += mv.sum(); nov += (fl & mv).sum(); nstep += len(cz)
        czmin = min(czmin, np.nanmin(cz)); ov_eps += (fl & mv).any()
    g = float(np.asarray(z['ep_gear'][0], float)[1])
    med = float(np.median(dend))
    return (f'{run}\n  見張り {" ".join(mid)}\n'
            f'  best 50 話: 触れた {touched}/{E}  d_end 中央 {med:.3f} m [{min(dend):.3f}, {max(dend):.3f}]'
            f'（{"✅ 届いた" if med <= 0.05 else "届かない"}）  旋回のギア比 {g:.1f}（{"上限の 20 % 以内" if g >= 125 else "上限の 20 % の外"}）\n'
            f'  床下: 全 step の {nfl}/{nstep}（{nfl / nstep * 100:.2f} %、最下 {czmin * 1000:.0f} mm）。'
            f'箱がずれている {nmov} step のうち床下 {nov}（{nov / max(nmov, 1) * 100:.2f} %、{ov_eps} 話）')


def gaps(run):
    """箱が動き出す直前 4 step の、腕のカプセル（`1`〜先端）と箱の面の隙間。線分を 50 点に刻んで箱の直方体までの距離 − 半径。"""
    xml = yaml.safe_load(open(os.path.join(R, run, '.hydra', 'config.yaml')))['xml_name']
    ch = cube_half(xml)
    z = np.load(os.path.join(R, run, 'trace', 'arm_trace_ep50.npz'), allow_pickle=True)
    for e, cube, xp, xm, cap, rad, bo in episodes(z):
        disp = moved(cube)
        if disp.max() < 0.001:
            continue
        k = int(np.argmax(disp > 0.001))
        c0, c1 = cap_ends(xp, xm, cap)
        out = []
        for kk in range(max(k - 3, 0), k + 1):
            best = np.inf
            for i in range(1, c0.shape[1]):
                for t in np.linspace(0, 1, 50):
                    q = c0[kk, i] + (c1[kk, i] - c0[kk, i]) * t
                    best = min(best, np.linalg.norm(np.maximum(np.abs(q - cube[kk]) - ch, 0)) - rad[i])
            out.append(round(best * 1000))
        print(f'  話 {e + 1}: 動き出す step {k}  直前の隙間 {out} mm  箱の初期 {cube[0].round(3)}  最大のずれ {disp.max() * 1000:.0f} mm')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gap', metavar='RUN', help='この run の、箱が動いた話ごとに動き出す直前の隙間を出す')
    a = ap.parse_args()
    if a.gap:
        gaps(a.gap)
    else:
        for r in RUNS:
            print(summary(r))
