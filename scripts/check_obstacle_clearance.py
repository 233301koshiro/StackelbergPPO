#!/usr/bin/env python3
"""障害物 Reach の軌跡が **柱を避けているか** を機械で判定する（実験系譜 9-132）。

⚠️⚠️ **9-118 で踏んだ穴を塞ぐためのもの。**
柱は `cylinder` だと Choreonoid の変換器に無視され、**形だけ存在して物理に無い**。
**先端が到達しても、腕の途中が柱を突き抜けていれば「回避した」とは言えない。**

⭐ **9-96 の規律「動く物体すべての極値を先に出す」を障害物版にした。**
先端だけでなく**全リンク**を見る。9-118 では先端が迂回していたのにリンク2・3 が貫通していた。

    python3 scripts/check_obstacle_clearance.py e2e_a1v_obs_tall2_reach
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def obstacle_box(xml_name):
    """柱の (中心, 半寸法)。box 以外なら None（＝物理に存在しない可能性。9-118）。"""
    t = ET.parse(ROOT / 'assets' / 'mujoco_envs' / f'{xml_name}.xml')
    for b in t.findall('worldbody/body'):
        if b.get('name') != 'obstacle':
            continue
        g = b.find('geom')
        pos = np.array([float(x) for x in b.get('pos', '0 0 0').split()])
        size = np.array([float(x) for x in g.get('size').split()])
        return g.get('type'), pos, size
    return None, None, None


def main(run):
    d = ROOT / 'single_run' / run
    cfg = yaml.safe_load(open(d / '.hydra' / 'config.yaml'))
    xml = cfg.get('xml_name')
    typ, c, s = obstacle_box(xml)
    print(f"=== {run}   xml={xml}")
    if typ is None:
        print("⚠️ obstacle が無い XML。判定しない"); return
    if typ != 'box':
        print(f"⛔⛔ **柱の型が `{typ}`。Choreonoid の変換器は capsule/sphere/box しか扱わない**（9-118）。")
        print("⛔ **この run の障害物は物理に存在しない。結果を障害物回避として読んではいけない。**")
        return
    if len(s) == 2:          # box は 3 要素。念のため
        s = np.array([s[0], s[0], s[1]])
    lo, hi = c - s, c + s
    print(f"    柱(box) x[{lo[0]:.3f},{hi[0]:.3f}] y[{lo[1]:.3f},{hi[1]:.3f}] z[{lo[2]:.3f},{hi[2]:.3f}]")

    z = np.load(d / 'trace' / 'arm_trace.npz', allow_pickle=True)
    names, xpos, xmat, bone = list(z['body_names']), z['xpos'], z['xmat'], z['bone_offset']
    tgt = np.asarray(z['target'] if 'target' in z else [0.8, 0.0, 0.15], dtype=float)

    # ⭐ リンク i の実体は「自分の原点 → 原点 + R_i·bone_offset_i」の線分（＝カプセル本体）。
    #   ⚠️ 「親の原点 → 自分の原点」で取ると **先端リンクが丸ごと抜ける**
    #   （9-132 で実際に誤判定した。`check_before_conclusion.py:60` と同じ定義に揃えること）
    P = 21
    ends = xpos + np.einsum('tlij,lj->tli', xmat, bone)
    print(f"\n{'リンク':>6} {'柱の中に居た step':>18} {'最接近[mm]':>12}")
    worst = 0
    for i, n in enumerate(names):
        a, b = xpos[:, i, :], ends[:, i, :]
        inside = np.zeros(len(xpos), dtype=bool)
        dmin = np.inf
        for t in np.linspace(0, 1, P):
            p = a + (b - a) * t
            inside |= np.all((p >= lo) & (p <= hi), axis=1)
            dd = np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0), axis=1)
            dmin = min(dmin, dd.min())
        worst = max(worst, int(inside.sum()))
        mark = '  ⛔ **貫通**' if inside.any() else ''
        print(f"{n:>6} {int(inside.sum()):>10} / {len(xpos):<5} {dmin*1000:>10.1f}{mark}")

    tip = ends[:, -1, :]
    dist = np.linalg.norm(tip - tgt, axis=1)
    print(f"\n⭐ 先端の目標最小距離: **{dist.min()*1000:.1f} mm**（最終 {dist[-1]*1000:.1f} mm）")
    print(f"⭐ 先端 y の範囲: {tip[:,1].min():+.3f} 〜 {tip[:,1].max():+.3f} m"
          f"（|y| > {s[1]:.2f} なら柱の脇を通っている）")
    print(f"⭐ 先端 z の範囲: {tip[:,2].min():+.3f} 〜 {tip[:,2].max():+.3f} m"
          f"（> {hi[2]:.2f} なら柱の上をまたいでいる）")
    if worst == 0:
        print("\n✅ **どのリンクも柱の中に入っていない ＝ 本当に回避している**")
    else:
        print(f"\n⛔ **最大 {worst} step 柱の中に居る。回避とは言えない**（9-118 と同型）")


if __name__ == '__main__':
    for r in sys.argv[1:]:
        main(r); print()
