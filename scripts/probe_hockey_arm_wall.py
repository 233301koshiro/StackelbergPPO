#!/usr/bin/env python3
"""腕を壁に衝突させたら、腕はまだパックに届くか（実験系譜 9-94）。

**なぜ要るか**: ホッケー台の壁は `conaffinity=2`、腕は `conaffinity=0` で積が 0 ＝
**腕は壁をすり抜ける**（9-93）。生成器にはその意図が
「腕が壁に引っかかって学習が壊れるのを防ぐ割り切り」と明記されている。

⚠️ **だがその懸念は一度も測られていない。** 9-65 の「検算」は
**すり抜け設定が意図どおりであること**を確かめただけで、
**衝突させたら本当に壊れるのか**は調べていない。本スクリプトがそれを測る。

    python3 scripts/probe_hockey_arm_wall.py
    python3 scripts/probe_hockey_arm_wall.py --n 40000 --xml e2e_hockey_wall

⚠️⚠️ **このプローブは MuJoCo で回る。学習は Choreonoid で回る（9-98）。**
**Choreonoid の変換器は `worldbody/body` しか読まないので、`worldbody` 直下の
`<geom>`（＝壁・板などの静的な障害物）は実走に存在しない。**
したがって**本スクリプトが壁について出す答えは、実走とは無関係である。**
⚠️ **壁を測るなら `scripts/probe_wall_choreonoid.py`**（Choreonoid 経由）を使うこと。
⭐ 壁を実在させるには `make_hockey_court_xml.py --wall-as-body`（9-99）で生成した
`e2e_hockey_wall3.xml` を使う。腕・パックの運動学や幾何だけを見る用途なら本スクリプトでよい。
"""
import argparse
import sys

import mujoco
import numpy as np

ASSET = 'assets/mujoco_envs/{}.xml'


def load(xml, wall_conaff):
    """wall_conaff=2 ですり抜け（現状）、3 で腕とも衝突する。"""
    m = mujoco.MjModel.from_xml_path(ASSET.format(xml))
    walls = [i for i in range(m.ngeom)
             if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or '').startswith(('wall', 'goal_board'))]
    for i in walls:
        m.geom_conaffinity[i] = wall_conaff
    return m, walls


def arm_geoms(m, walls):
    """腕の geom（worldbody 直下でない・壁でも床でもパックでもないもの）。"""
    out = []
    for i in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ''
        if i in walls or n in ('floor', 'cube_geom'):
            continue
        out.append(i)
    return out


def sweep(m, walls, arms, n, seed=0):
    """関節空間を一様に振り、先端位置と「腕が壁に当たるか」を集める。"""
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    # 腕の関節（パックの slide は除く）
    jids = [j for j in range(m.njnt)
            if not (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or '').startswith('cube')]
    lo = np.array([m.jnt_range[j][0] for j in jids])
    hi = np.array([m.jnt_range[j][1] for j in jids])
    adr = [m.jnt_qposadr[j] for j in jids]
    tip_body = m.ngeom and max(arms)
    tips, hits = [], []
    aset, wset = set(arms), set(walls)
    for _ in range(n):
        q = rng.uniform(lo, hi)
        mujoco.mj_resetData(m, d)
        for a, v in zip(adr, q):
            d.qpos[a] = v
        mujoco.mj_forward(m, d)
        tips.append(d.geom_xpos[tip_body].copy())
        hit = any((c.geom1 in aset and c.geom2 in wset) or
                  (c.geom2 in aset and c.geom1 in wset)
                  for c in (d.contact[k] for k in range(d.ncon)))
        hits.append(hit)
    return np.array(tips), np.array(hits)


def coverage(tips, hits, cell=0.05):
    """台の上（x∈[0,1.6], |y|<=0.5）を格子に切り、届く升の数を数える。"""
    ok = (tips[:, 0] >= 0) & (tips[:, 0] <= 1.6) & (np.abs(tips[:, 1]) <= 0.5)
    gx = np.floor(tips[:, 0] / cell).astype(int)
    gy = np.floor((tips[:, 1] + 0.5) / cell).astype(int)
    key = gx * 1000 + gy
    return set(key[ok]), set(key[ok & ~hits])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--xml', default='e2e_hockey_wall')
    ap.add_argument('--n', type=int, default=20000)
    a = ap.parse_args()

    print(f"=== {a.xml}: 関節空間を {a.n} 点サンプルし、先端が台のどこへ届くかを数える\n")
    m, walls = load(a.xml, 3)                     # 腕も壁に当たる設定で 1 度だけ回す
    arms = arm_geoms(m, walls)
    print(f"腕の geom {len(arms)} 個 / 壁の geom {len(walls)} 個")
    tips, hits = sweep(m, walls, arms, a.n)
    allc, freec = coverage(tips, hits)
    print(f"\n先端が壁に当たる姿勢の割合: {hits.mean() * 100:.1f} %")
    print(f"台の上の到達升（すり抜け時）    : {len(allc)}")
    print(f"台の上の到達升（壁と衝突させた）: {len(freec)}")
    if allc:
        print(f"⭐ **残る到達範囲: {len(freec) / len(allc) * 100:.1f} %**")
    # パックが実際に居る範囲（初期 x=0.55 付近・y は cube_y_noise=0.4 で ±0.4）
    def near_puck(s):
        return {k for k in s if 0.40 <= (k // 1000) * 0.05 <= 0.80
                and abs(((k % 1000) * 0.05) - 0.5) <= 0.45}
    na, nf = near_puck(allc), near_puck(freec)
    print(f"\nパックの居る帯（x 0.40〜0.80・|y|<=0.45）")
    print(f"  すり抜け時 {len(na)} 升 → 衝突させると {len(nf)} 升"
          f"（{len(nf) / len(na) * 100:.0f} %）" if na else "  （範囲外）")
    print("\n⚠️ **これは運動学の screen であって、学習できるかの証明ではない**"
          "（§5-2 ⑤-3: 幾何のプローブが「できる」と言っても幾何の話）。")
    return 0


if __name__ == '__main__':
    sys.exit(main())
