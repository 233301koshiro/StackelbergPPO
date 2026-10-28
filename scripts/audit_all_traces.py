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

⭐ 2026-10-22（系譜 9-283 ③ の残り）: 以前は代表の 1 話（`xpos`・`cube`）しか見なかった。話ごとの記録（`ep_*`）があれば
全話を見る（無い古い軌跡は代表の話だけ）。①②③④⑤ は 1 話でも当たれば出し、⑥「動かない」は全話が動かないときだけ出す
（一部の話で動かないのは方策の出来）。到達で退避した箱（x 100、9-234）は ③④ から外す。

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


def episodes(z):
    """話ごとの (xpos, xmat, bone, cube)。`ep_*` が無い古い軌跡は代表の 1 話だけ。
    `ep_bone` が無い軌跡（hockey_bank7・8）は代表の骨（再生は方策の平均の行動なので形は全話で同じ）。"""
    cube0 = np.asarray(z['cube'], dtype=float) if 'cube' in z.files else None
    if 'ep_xpos' in z.files and len(z['ep_xpos']):
        out = []
        for e in range(len(z['ep_xpos'])):
            x = np.asarray(z['ep_xpos'][e], float)
            if x.ndim != 3 or not len(x):
                continue
            b = z['ep_bone'][e] if 'ep_bone' in z.files else z['bone_offset']
            c = np.asarray(z['ep_cubes'][e], float) if 'ep_cubes' in z.files else cube0
            out.append((x, np.asarray(z['ep_xmat'][e], float), np.asarray(b, float), c))
        if out:
            return out
    return [(z['xpos'], z['xmat'], z['bone_offset'], cube0)]


def audit(run):
    f = os.path.join(ROOT, 'single_run', run, 'trace', 'arm_trace.npz')
    z = np.load(f, allow_pickle=True)
    names = [str(x) for x in z['body_names']]
    eps = episodes(z)
    hits = []
    ne = len(eps)
    tag = (lambda bad: f'（{len(bad)}/{ne} 話: {bad[:5]}{"…" if len(bad) > 5 else ""}）') if ne > 1 else (lambda bad: '')

    short = [e for e, (x, *_) in enumerate(eps) if len(x) < 50]
    if short:
        hits.append(f'① エピソードが {min(len(eps[e][0]) for e in short)} step で終わっている{tag(short)}')

    movs, sps = [], []
    for xp, xm, bo, _ in eps:
        ends = xp + np.einsum('tlij,lj->tli', xm, bo)
        movs.append(min((min(xp[:, i, 2].min(), ends[:, i, 2].min()) for i in range(1, len(names))),
                        default=0.0))
        tip = ends[:, -1, :]
        sps.append(float((np.linalg.norm(np.diff(tip, axis=0), axis=1) / DT).max()) if len(tip) > 1 else 0.0)
    under = [e for e, m in enumerate(movs) if m < -0.001]
    if under:
        hits.append(f'② 可動リンクが床下 {min(movs)*1000:.0f} mm{tag(under)}')
    fast = [e for e, v in enumerate(sps) if v > TIP_SPEED_MAX]
    if fast:
        hits.append(f'⑤ 先端速度 {max(sps):.1f} m/s（上限 {TIP_SPEED_MAX} を超過＝発散の疑い）{tag(fast)}')

    if eps[0][3] is not None and len(eps[0][3]):
        try:
            cfg = yaml.safe_load(open(os.path.join(ROOT, 'single_run', run, '.hydra', 'config.yaml')))
            xml = cfg.get('xml_name')
            rs = cfg.get('reward_specs') or {}
        except Exception:
            xml, rs = None, {}
        rng = cube_range(xml) if xml else None
        live = [(e, ep) for e, ep in enumerate(eps) if np.abs(ep[3][:, 0]).max() <= 50]   # 退避した箱を外す
        if rng:
            for k, (lo, hi) in rng.items():
                stuck = [(e, float(ep[3][-1, k])) for e, ep in live
                         if abs(ep[3][-1, k] - lo) < 0.02 or abs(ep[3][-1, k] - hi) < 0.02]
                if stuck:
                    hits.append(f'③ 対象が {"xy"[k]} の可動域の端に張り付いている'
                                f'（最終 {stuck[0][1]:.3f}、範囲 [{lo:.2f}, {hi:.2f}]）{tag([e for e, _ in stuck])}')
        # ⭐ 押し系なのに対象が動かない（全話が動かないときだけ）
        is_reach = bool(rs.get('use_reach'))
        dxs = [float(np.linalg.norm(ep[3][-1, :2] - ep[3][0, :2])) for ep in eps]
        if not is_reach and max(dxs) < 0.01:
            hits.append(f'⑥ 押し系なのに対象が {max(dxs)*1000:.1f} mm しか動いていない'
                        f'{f"（全 {ne} 話）" if ne > 1 else ""}')
        # ⭐ 腕が対象の向こう側へ出る（通り抜け）
        half = 0.05
        P = 11
        thr, thr_eps = 0, []
        for e, (xp, xm, bo, c) in live:
            ends = xp + np.einsum('tlij,lj->tli', xm, bo)
            n0 = thr
            for i in range(len(names)):
                a, b = xp[:, i, :], ends[:, i, :]
                for u in np.linspace(0, 1, P):
                    p = a + (b - a) * u
                    thr += int(np.sum((p[:, 0] > c[:, 0] + half)
                                      & (np.abs(p[:, 1] - c[:, 1]) < half)
                                      & (np.abs(p[:, 2] - c[:, 2]) < half)))
            if thr > n0:
                thr_eps.append(e)
        if thr:
            hits.append(f'④ 腕が対象の向こう側へ出ている（延べ {thr} 標本）{tag(thr_eps)}')
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
