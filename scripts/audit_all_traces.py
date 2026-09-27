#!/usr/bin/env python3
"""**全軌跡を過去の事故の型で機械的に洗う**（2026-09-27 新設）。

⚠️⚠️ **9-155 の教訓**: 「検査は**見るべきと分かっている量**しか見ない」。
⛔ だから `plot_run.py` の床下判定と step 数だけでは足りない。
⭐⭐ **過去に実際に起きた事故を、片っ端から検査項目にする**のが本スクリプト。

| # | 検査 | 出所（実際に起きた事故） |
|---|---|---|
| 1 | エピソードが 50 step 未満 | 9-168（`tripo_v3_reach` が step=1 だった） |
| 2 | 可動リンクが床下 | 9-155（36 本中 10 本。数値の検査 5 項目は全部通っていた） |
| 3 | ⭐ **対象が可動域の端に張り付く** | 9-39 / 9-63（パックが壁を突き抜けて可動限界まで飛んだ） |
| 4 | ⭐ **腕が対象を通り抜ける** | 9-93（腕が壁をすり抜けてパックを押し込んでいた） |
| 5 | ⭐ **先端速度が非現実的** | 9-80（幾何の上にしか無い解を信じた）／打撃の妥当性 |
| 6 | ⭐ **対象がまったく動かない** | 9-135（`fwd_cube`≈0 を「動いていない」と 2 回誤読した。**逆に本当に動かない場合を拾う**） |

    python3 scripts/audit_all_traces.py            # 全部
    python3 scripts/audit_all_traces.py --only-hit # 引っかかったものだけ
"""
import argparse
import glob
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT = 0.04
TIP_SPEED_MAX = 60.0     # m/s。これを超えるのは発散（打撃の実測は 7〜21 m/s）


def cube_range(xml_name):
    """cube の slide 可動域（world 座標）。無ければ None。"""
    try:
        r = ET.parse(os.path.join(ROOT, 'assets', 'mujoco_envs', f'{xml_name}.xml')).getroot()
    except Exception:
        return None
    for b in r.find('worldbody').findall('body'):
        if b.get('name') != 'cube':
            continue
        pos = [float(v) for v in b.get('pos', '0 0 0').split()]
        out = {}
        for j in b.findall('joint'):
            rng = j.get('range')
            ax = j.get('axis', '1 0 0').split()
            if not rng:
                continue
            lo, hi = (float(v) for v in rng.split())
            k = 0 if ax[0] == '1' else 1
            out[k] = (pos[k] + lo, pos[k] + hi)
        return out or None
    return None


def audit(run):
    f = os.path.join(ROOT, 'single_run', run, 'trace', 'arm_trace.npz')
    z = np.load(f, allow_pickle=True)
    names = [str(x) for x in z['body_names']]
    xp, xm, bo = z['xpos'], z['xmat'], z['bone_offset']
    ends = xp + np.einsum('tlij,lj->tli', xm, bo)
    T = len(xp)
    hits = []

    if T < 50:
        hits.append(f'① エピソードが {T} step で終わっている')

    mov = min((min(xp[:, i, 2].min(), ends[:, i, 2].min()) for i in range(1, len(names))),
              default=0.0)
    if mov < -0.001:
        hits.append(f'② 可動リンクが床下 {mov*1000:.0f} mm')

    tip = ends[:, -1, :]
    sp = np.linalg.norm(np.diff(tip, axis=0), axis=1) / DT if T > 1 else np.array([0.0])
    if sp.max() > TIP_SPEED_MAX:
        hits.append(f'⑤ 先端速度 {sp.max():.1f} m/s（上限 {TIP_SPEED_MAX} を超過＝発散の疑い）')

    c = np.asarray(z['cube'], dtype=float) if 'cube' in z else None
    if c is not None and len(c):
        try:
            cfg = yaml.safe_load(open(os.path.join(ROOT, 'single_run', run, '.hydra', 'config.yaml')))
            xml = cfg.get('xml_name')
            rs = cfg.get('reward_specs') or {}
        except Exception:
            xml, rs = None, {}
        rng = cube_range(xml) if xml else None
        if rng:
            for k, (lo, hi) in rng.items():
                fin = float(c[-1, k])
                if abs(fin - lo) < 0.02 or abs(fin - hi) < 0.02:
                    hits.append(f'③ 対象が {"xy"[k]} の可動域の端に張り付いている'
                                f'（最終 {fin:.3f}、範囲 [{lo:.2f}, {hi:.2f}]）')
        # ⭐ 押し系なのに対象が動かない
        is_reach = bool(rs.get('use_reach'))
        dx = float(np.linalg.norm(c[-1, :2] - c[0, :2]))
        if not is_reach and dx < 0.01:
            hits.append(f'⑥ 押し系なのに対象が {dx*1000:.1f} mm しか動いていない')
        # ⭐ 腕が対象の向こう側へ出る（通り抜け）
        half = 0.05
        P = 11
        thr = 0
        for i in range(len(names)):
            a, b = xp[:, i, :], ends[:, i, :]
            for u in np.linspace(0, 1, P):
                p = a + (b - a) * u
                thr += int(np.sum((p[:, 0] > c[:, 0] + half)
                                  & (np.abs(p[:, 1] - c[:, 1]) < half)
                                  & (np.abs(p[:, 2] - c[:, 2]) < half)))
        if thr:
            hits.append(f'④ 腕が対象の向こう側へ出ている（延べ {thr} 標本）')
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only-hit', action='store_true')
    a = ap.parse_args()
    runs = sorted(p.split('/')[-3] for p in glob.glob(
        os.path.join(ROOT, 'single_run', '*', 'trace', 'arm_trace.npz')))
    n_hit = 0
    for r in runs:
        try:
            h = audit(r)
        except Exception as e:
            h = [f'⛔ 検査できない: {type(e).__name__} {e}']
        if h:
            n_hit += 1
            print(f'⛔ {r}')
            for x in h:
                print(f'      {x}')
        elif not a.only_hit:
            print(f'✅ {r}')
    print(f'\n{"="*56}')
    print(f'軌跡 {len(runs)} 本 / ⛔ 引っかかった {n_hit} 本')
    print('⚠️ **引っかからなかった＝正常ではない。**'
          '検査は「見るべきと分かっている量」しか見ない（9-155）')


main()
