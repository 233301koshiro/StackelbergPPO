#!/usr/bin/env python3
"""**ホッケー用の材質テーブルを足しても、既存 run の物理が変わらないこと**を実測する。

⚠️⚠️ **これはホッケー Phase 1 の関門である。**
修論 6.4.2 (7) は「接触を変えると既存 110 run と物理が変わり比較できなくなる」から
`ContactMaterial` を実装しないと決めた（2026-09-20、ユーザー判断）。
⭐ **その前提が成り立つかを測る。**

⭐ **機序**（9-143 の続き。2026-09-26 に判明）:
  Choreonoid の既定材質テーブルは `[Default, Default]` の **restitution が 0.0**。
  ⛔ 変換器はリンクに**物理材質を書かない**（`material:` は見た目の diffuseColor だけ）ので
  **全リンクが Default** であり、実測の e ≈ 0.055 は設定値ではなく残留の数値反発である。
  ⭐ したがってホッケー用テーブルは **`[Default, Default]` に触れず材質を足すだけ**にできる。

**測り方**: 同じ run・同じ checkpoint で軌跡を 2 回録る。
  ① 既定テーブル（`CNOID_MATERIAL_TABLE` 無し）
  ② ホッケー用テーブル（`[Puck, Rink]` 等を足しただけ）
⭐ **全リンクの位置が完全に一致すれば、足しても既存は変わらない。**

⚠️ **1 エピソードの再現性そのものを先に確かめる。**固定 body があるタスクは
再生ごとに揺れる（9-138）ので、**同一条件で 2 回録って一致することを先に見る**。
一致しないなら、この検査では判定できない。

    python3 scripts/probe_material_table_inert.py <run名>

環境変数: `PROBE_DIR`（軌跡の一時ファイルを置く場所。既定 /tmp）
"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(ROOT, 'assets', 'choreonoid', 'materials_hockey.yaml')


def record(run, out, table=None):
    env = dict(os.environ, EVAL_RESTORE_DIR=f'single_run/{run}', EVAL_CHECKPOINT='best',
               USE_CHOREONOID='1', OMP_NUM_THREADS='1', TRACE_OUT=out)
    env.pop('CNOID_MATERIAL_TABLE', None)
    if table:
        env['CNOID_MATERIAL_TABLE'] = table
    log = out + '.log'
    with open(log, 'w') as f:
        # ⚠️ timeout には必ず -k を付ける。Choreonoid は SIGTERM を無視する（Bug 48）
        subprocess.run(['timeout', '-k', '30', '600',
                        '/choreonoid_ws/install/bin/choreonoid', '--no-window',
                        '--python', os.path.join(ROOT, 'scripts', 'record_arm_trace.py')],
                       env=env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return os.path.exists(out)


def cmp(a, b, label):
    za, zb = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    if za['xpos'].shape != zb['xpos'].shape:
        print(f'  {label}: ⛔ step 数が違う {za["xpos"].shape} 対 {zb["xpos"].shape}')
        return None
    d = float(np.abs(za['xpos'] - zb['xpos']).max())
    print(f'  {label}: 全リンク位置の最大差 {d*1000:.6f} mm')
    return d


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'e2e_a1v_real_pusher'
    tmp = os.environ.get('PROBE_DIR', '/tmp')
    f = {k: os.path.join(tmp, f'mt_{k}.npz') for k in ('base1', 'base2', 'hockey')}
    print(f'=== {run}')
    print('① 既定テーブルで 2 回録る（再現性の確認）')
    for k in ('base1', 'base2'):
        if not record(run, f[k]):
            print(f'  ⛔ {k} の録画に失敗。{f[k]}.log を見ること'); return
    d0 = cmp(f['base1'], f['base2'], '既定 vs 既定')
    if d0 is None:
        return
    if d0 > 1e-9:
        print('  ⚠️ **同一条件でも一致しない。**この検査では材質テーブルの影響を判定できない（9-138 の揺れ）')
        return
    print('  ⭐ 完全に決定的。判定できる。')
    print('② ホッケー用テーブルを足して録る')
    if not record(run, f['hockey'], TABLE):
        print(f'  ⛔ 録画に失敗。{f["hockey"]}.log を見ること'); return
    d1 = cmp(f['base1'], f['hockey'], '既定 vs ホッケー用')
    print()
    if d1 is not None and d1 <= 1e-9:
        print('⭐⭐ **合格。材質を足しても既存 run の物理は変わらない。**')
        print('⭐ 6.4.2 (7) の「既存 110 run と物理が変わる」は、'
              '**テーブルを足すだけなら成り立たない。**')
    else:
        print('⛔⛔ **不合格。足しただけで既存の物理が変わる。**')
        print('⛔ 6.4.2 (7) の決定は覆せない。ホッケーは案 B（直接シュート）へ落とす。')


main()
