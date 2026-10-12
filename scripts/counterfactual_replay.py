#!/usr/bin/env python3
"""**物理のつまみを 1 つずつ変えて再生し、結果がどれだけ動くか**を表にする（反実仮想の再生、2026-10-12）。学習はしない。

[異常を能動的に見つける方法.md](../docs/リポジトリ説明/異常を能動的に見つける方法.md) の方法 3。
「効かないはず」のつまみで結果が大きく動いたら、方策がその物理の穴（または意図しない機構）に頼っている。
過去の実例: 9-241（制御入力を切ると密度の押しの箱が 19.6 → 14.9 m）・9-227（閾値で long の箱が 4.4 → 2.6 m）・9-235（台座を立てると 0 m）。

    python3 scripts/counterfactual_replay.py <run>        # → single_run/<run>/trace/cf_*.npz と cf_summary.txt

基準は自動再生の軌跡（`trace/arm_trace.npz`、学習時の環境・同じ 5 本のくじ）。つまみは学習時の環境（`launch_env.json`）から
1 つだけ反転させる（入っていれば外す、制御入力の切り方は入っていなければ入れる）。⚠️ 学習時と違う物理で動かすので、
落ち方は「その物理で学習したら」より大きめに出る。**大きく動いた行は ⚠️ にして人が見る（⛔ にはしない）。**
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CHOREONOID = '/choreonoid_ws/install/bin/choreonoid'
FLAG = 0.20      # 課題の量が 20 % 以上動いたら ⚠️


def knobs(le):
    """学習時の環境から、反転させるつまみの列 (名前, 環境変数の上書き)。空文字＝外す。"""
    out = [('制御入力を切る↔切らない', {'CNOID_CTRL_CLIP': '' if le.get('CNOID_CTRL_CLIP') else '1'})]
    if le.get('CNOID_CULLING_DEPTH'):
        out.append(('深い接触を捨てる（閾値 0.05）', {'CNOID_CULLING_DEPTH': ''}))
    if le.get('CNOID_SELF_COLLISION'):
        out.append(('自己干渉なし', {'CNOID_SELF_COLLISION': ''}))
    if le.get('HOCKEY_ARM_BLOCK'):
        out.append(('腕の壁ブロックなし', {'HOCKEY_ARM_BLOCK': ''}))
    if le.get('HOCKEY_WALL_RESTITUTION'):
        out.append(('壁の反射なし', {'HOCKEY_WALL_RESTITUTION': ''}))
    return out


def metric(d):
    """課題の量: 到達は最後の 200 step の先端と目標の誤差 [mm]（小さいほど良い）、それ以外は対象の移動 [m]。"""
    c = [np.asarray(x, float) for x in d['ep_cubes']]
    if max(np.abs(x[:, 0]).max() for x in c) > 50:          # 到達（箱は退避、9-234）
        tgt = np.asarray(d['target'], float)
        err = []
        for e in range(len(d['ep_xpos'])):
            xp = np.asarray(d['ep_xpos'][e], float)
            xm = np.asarray(d['ep_xmat'][e], float)
            bo = np.asarray(d['ep_bone'][e], float)
            tip = (xp + np.einsum('tlij,lj->tli', xm, bo))[:, -1]
            err.append(np.linalg.norm(tip - tgt, axis=1)[-200:].mean() * 1000)
        return '到達の誤差 mm', np.array(err)
    return '対象の移動 m', np.asarray(d['ep_cube_dx'], float)


def main():
    run = sys.argv[1]
    d0 = ROOT / 'single_run' / run
    base = d0 / 'trace' / 'arm_trace.npz'
    if not base.exists():
        sys.exit(f'{run}: 基準の軌跡（自動再生）が無い')
    le = json.loads((d0 / 'launch_env.json').read_text()) if (d0 / 'launch_env.json').exists() else {}
    b = np.load(base, allow_pickle=True)
    name, v0 = metric(b)
    lines = [f'=== {run}  （{name}、5 話の平均。基準＝学習時の物理 {le}）',
             f'  基準                         {v0.mean():8.2f}  {np.round(v0, 2).tolist()}']
    for label, ovr in knobs(le):
        out = d0 / 'trace' / f"cf_{'_'.join(f'{k}{v or 0}' for k, v in ovr.items())}.npz"
        env = dict(os.environ, EVAL_RESTORE_DIR=str(d0), EVAL_CHECKPOINT='best', TRACE_OUT=str(out),
                   USE_CHOREONOID='1', OMP_NUM_THREADS='1', **ovr)
        with open(str(out) + '.log', 'w') as f:
            subprocess.run(['timeout', '-k', '30', '900', CHOREONOID, '--no-window', '--python',
                            'scripts/record_arm_trace.py'], cwd=ROOT, env=env, stdout=f, stderr=f)
        if not out.exists():
            lines.append(f'  ⚠️ {label:<26} 再生できなかった（{out.name}.log）'); continue
        _, v = metric(np.load(out, allow_pickle=True))
        rel = (v.mean() - v0.mean()) / (abs(v0.mean()) + 1e-9)
        big = abs(rel) > FLAG and (name.startswith('対象') or abs(v.mean() - v0.mean()) > 5)
        lines.append(f"  {'⚠️' if big else '  '} {label:<26} {v.mean():8.2f}  {np.round(v, 2).tolist()}  ({rel*100:+.0f} %)")
    txt = '\n'.join(lines)
    print(txt)
    (d0 / 'trace' / 'cf_summary.txt').write_text(txt + '\n')


if __name__ == '__main__':
    main()
