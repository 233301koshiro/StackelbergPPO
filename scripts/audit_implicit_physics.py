#!/usr/bin/env python3
"""XML に**書かれていない**物理パラメータを一覧にする（実験系譜 9-86）。

**なぜ要るか**: ホッケーで 6 回環境をいじった末、**反発係数が既定のままだったせいで
バンクショットが物理的に存在しなかった**と判明した（9-80 の訂正）。
⚠️ **`solref` は XML に一度も書かれていない。** だから `grep` にも物理監査にも掛からず、
**6 回の修正すべてを素通りした。**

⚠️ **同じ型の事故は 2 回目である。** `armature=1` も誰も選んでいない値で、
**慣性の 99 % を占めていた**（9-47）。こちらは明示されていたが、由来を問われなかった。

**明示された値は疑える。書かれていない値は、存在に気づかない。**
本スクリプトは**実効値と XML の記述を突き合わせ、既定のまま使っているものを挙げる。**

    python3 scripts/audit_implicit_physics.py assets/mujoco_envs/e2e_a1v.xml
    python3 scripts/audit_implicit_physics.py            # assets/mujoco_envs/*.xml 全部
"""
import glob
import io
import re
import sys

import mujoco
import numpy as np

# (表示名, モデルの属性を取る関数, XML で探すキー, 既定のままだと何が起きるか)
CHECKS = [
    ('接触 solref（時定数・減衰比）', lambda m: m.geom_solref[0].tolist(), 'solref',
     '減衰比 1 は臨界減衰＝**跳ね返らない**。反射を要するタスクは成立しない'),
    ('接触 solimp（剛性）', lambda m: m.geom_solimp[0].tolist(), 'solimp',
     'めり込み量が決まる。軽い物体ほど深く入る'),
    ('関節 armature（反射慣性）', lambda m: sorted(set(np.round(m.dof_armature, 3).tolist())), 'armature',
     '**慣性行列を支配しうる**。形態の寄与が消える（9-47 で 99 %）'),
    ('関節 damping', lambda m: sorted(set(np.round(m.dof_damping, 3).tolist())), 'damping',
     '質量スケールを変えたら τ=m/b が変わる（9-45）'),
    ('関節 frictionloss', lambda m: sorted(set(np.round(m.dof_frictionloss, 3).tolist())), 'frictionloss',
     '静止摩擦。0 だと微小トルクでも動く'),
    ('重力', lambda m: m.opt.gravity.tolist(), 'gravity', '既定 −9.81 は通常そのままでよい'),
    ('timestep / 積分器', lambda m: [m.opt.timestep, int(m.opt.integrator)], 'timestep',
     '接触の時定数の下限を決める（2×timestep）'),
]


def audit(path: str) -> int:
    txt = io.open(path, encoding='utf-8').read()
    m = mujoco.MjModel.from_xml_path(path)
    print(f"\n===== {path}")
    n_implicit = 0
    for name, get, key, why in CHECKS:
        written = bool(re.search(rf'\b{key}\s*=', txt))
        val = str(get(m))[:40]
        if written:
            print(f"  ✅ {name:26s} {val:42s} 明示")
        else:
            n_implicit += 1
            print(f"  ⚠️  {name:26s} {val:42s} **既定（誰も選んでいない）**")
            print(f"      └ {why}")
    print(f"  → 既定のまま使っているもの: {n_implicit} 件")
    return n_implicit


def main() -> int:
    paths = sys.argv[1:] or sorted(glob.glob('assets/mujoco_envs/*.xml'))
    if not paths:
        print('XML が見つからない'); return 1
    tot = sum(audit(p) for p in paths)
    print(f"\n合計 {tot} 件。")
    print("⚠️ **既定であること自体は誤りではない。** 問うのは"
          "**そのタスクが要求する挙動に、その既定値で足りるか**である。")
    print("⚠️ **新しいタスクを入れるときは、タスク名に出てくる動詞を物理で直接測ること。**")
    print("   ホッケーは『反射』と名乗りながら、反発係数を一度も測らなかった（9-80 の訂正）。")
    return 0


if __name__ == '__main__':
    sys.exit(main())
