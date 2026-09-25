#!/usr/bin/env python3
"""**主戦場の Pusher で、腕が対象物（cube）を突き抜けていないか**を軌跡から判定する。

⚠️⚠️ **Bug 45 の未確認部分**（知見総括表 F 節）。
障害物 Reach では **柱は腕に当たるのに gear 100 の駆動トルクに押し切られる**ことが
分かっている（9-133）。⛔ **同じことが Pusher の cube で起きていないかを見る。**

⛔⛔ **最重要の制約（2026-09-25 に踏んだ）: 軌跡は接触を分解できない。**
軌跡は `frame_skip=4` ごとに 1 点しか記録しておらず、
**先端は記録 1 step あたり最大 0.31 m 動く**。一方で問題にしている食い込みは 10〜30 mm である。
⭐⭐ **接触の瞬間は標本と標本の間に落ちるので、食い込み量は原理的に測れない。**
⛔ **「触れていない」も「N mm 食い込んだ」も、どちらも標本化の産物になる。**

⭐ **そこで本スクリプトは 2 段構えにした。**

| 判定 | 意味 |
|---|---|
| ⭐ **貫通（通り抜け）** | **腕が cube の向こう側へ出たか。**⭐ これは標本が粗くても言える（9-93 のホッケー壁がこの型） |
| ⚠️ **食い込み量** | ⛔ **標本の刻みが隙間より大きいときは数値を出さない。**代わりに「分解できない」と言う |

⭐⭐ **なお「cube が動いたこと自体」が接触の証拠である**（cube にアクチュエータは無い）。
**力が伝わっていない 9-98 型ではない**ことは、それだけで言える。

    python3 scripts/check_cube_penetration.py            # Pusher の軌跡すべて
    python3 scripts/check_cube_penetration.py <run名>    # 1 本だけ
"""
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
P = 41          # 線分のサンプル数


def cube_half(xml_name):
    """cube の半寸法と固定 z。XML から読む（cube は x,y しか動かない）。"""
    import xml.etree.ElementTree as ET
    t = ET.parse(ROOT / 'assets' / 'mujoco_envs' / f'{xml_name}.xml')
    for b in t.findall('worldbody/body'):
        if b.get('name') != 'cube':
            continue
        g = b.find('geom')
        s = np.array([float(x) for x in g.get('size').split()], dtype=float)
        if len(s) == 1:
            s = np.repeat(s, 3)
        pos = np.array([float(x) for x in b.get('pos', '0 0 0').split()], dtype=float)
        return g.get('type'), s, pos[2]
    return None, None, None


def check(run):
    d = ROOT / 'single_run' / run
    tz = d / 'trace' / 'arm_trace.npz'
    if not tz.exists():
        print(f"=== {run}\n    ⚠️ 軌跡が無い。record_arm_trace.py を先に回す")
        return None
    cfg = yaml.safe_load(open(d / '.hydra' / 'config.yaml'))
    xml = cfg.get('xml_name')
    typ, s, cz = cube_half(xml)
    print(f"=== {run}   xml={xml}")
    if typ is None:
        print("    ⚠️ cube が無い XML。判定しない")
        return None
    if typ not in ('box', 'capsule', 'sphere'):
        print(f"    ⛔⛔ cube の型が `{typ}`。変換器は capsule/sphere/box しか扱わない（9-118）。"
              "**物理に存在しない可能性がある**")
        return None

    z = np.load(tz, allow_pickle=True)
    names, xpos, xmat, bone = list(z['body_names']), z['xpos'], z['xmat'], z['bone_offset']
    cube = np.asarray(z['cube'], dtype=float)
    # ⭐ 半径があれば表面どうしで判定する。無ければ軸だけの下限判定に落ちる（古い軌跡）。
    rad = np.asarray(z['geom_size'], dtype=float) if 'geom_size' in z else None
    T = len(xpos)
    # ⚠️ cube の z は固定（slide は x,y のみ）。軌跡の z は使わず XML の値を使う。
    c = np.column_stack([cube[:, 0], cube[:, 1], np.full(T, cz)])
    ends = xpos + np.einsum('tlij,lj->tli', xmat, bone)

    print(f"    cube 半寸法 {np.round(s,3)}  z={cz:.3f}  "
          f"x {c[:,0].min():.3f}→{c[:,0].max():.3f}  step={T}")
    print(f"\n{'リンク':>10} {'軸が cube 内の step':>20} {'最大の食い込み[mm]':>20} {'⭐ 軸〜箱表面の最小[mm]':>24}")
    worst = 0.0
    total = 0
    surf_pen = 0.0
    clear_min = {}
    for i, n in enumerate(names):
        a, b = xpos[:, i, :], ends[:, i, :]
        inside = np.zeros(T, dtype=bool)
        deep = np.zeros(T)
        for t in np.linspace(0, 1, P):
            p = a + (b - a) * t
            # 箱の内側なら、各軸で表面までの距離の最小値が食い込み量
            dd = s[None, :] - np.abs(p - c)
            ins = np.all(dd > 0, axis=1)
            inside |= ins
            deep = np.maximum(deep, np.where(ins, dd.min(axis=1), 0.0))
        # ⭐ 軸から箱の表面までの最短距離（外側なら正）。半径がこれを超えれば必ず食い込む。
        gap = np.full(T, np.inf)
        for t in np.linspace(0, 1, P):
            p = a + (b - a) * t
            q = np.maximum(np.abs(p - c) - s[None, :], 0.0)
            gap = np.minimum(gap, np.linalg.norm(q, axis=1))
        k = int(inside.sum())
        total += k
        worst = max(worst, deep.max())
        clear_min[n] = float(gap.min())
        mark = '⛔' if k else '✅'
        extra = ''
        if rad is not None and np.isfinite(rad[i]):
            pen = rad[i] - gap.min()
            extra = (f"  半径 {rad[i]*1000:.0f} mm → "
                     + (f"⛔ **食い込み {pen*1000:.1f} mm**" if pen > 0 else "✅ 接触せず"))
            if pen > 0:
                surf_pen = max(surf_pen, pen)
        print(f"{n:>10} {mark} {k:>6} / {T:<6} {deep.max()*1000:>18.1f} {gap.min()*1000:>22.1f}{extra}")
    # ⭐⭐ 分解能ガード: 記録 1 step あたりの移動量が「測ろうとしている隙間」より
    #   大きければ、接触の瞬間は標本の間に落ちている。**数値を出してはいけない。**
    tip = ends[:, -1, :]
    step_move = float(np.linalg.norm(np.diff(tip, axis=0), axis=1).max()) if T > 1 else 0.0
    gap_min = min(clear_min.values()) if clear_min else np.inf
    resolvable = step_move < gap_min

    # ⭐ 貫通（通り抜け）判定は標本が粗くても言える: 腕が cube の向こう側へ出たか。
    #   cube は x 方向へ押されるので、「リンクの x が cube の x を超えた step」を見る。
    through = 0
    for i in range(xpos.shape[1]):
        a_, b_ = xpos[:, i, :], ends[:, i, :]
        for u in np.linspace(0, 1, P):
            p_ = a_ + (b_ - a_) * u
            through += int(np.sum((p_[:, 0] > c[:, 0] + s[0])
                                  & (np.abs(p_[:, 1] - c[:, 1]) < s[1])
                                  & (np.abs(p_[:, 2] - c[:, 2]) < s[2])))

    moved = float(abs(c[-1, 0] - c[0, 0]))
    print()
    if through:
        print(f"⛔⛔ **腕が cube の向こう側へ出ている**（延べ {through} 標本）。"
              f"**通り抜けを疑う**（9-93 のホッケー壁と同型）。")
    else:
        print(f"⭐ **通り抜けは無い。**腕が cube の向こう側へ出た標本は 0。")
    if moved > 0.01:
        print(f"⭐ **cube は {moved:.2f} m 動いている ＝ 力は伝わっている。**"
              f"⛔ **9-98（対象が物理に存在しない）型ではない。**")
    else:
        print(f"⚠️ **cube がほとんど動いていない**（{moved*1000:.0f} mm）。"
              f"接触が効いていない可能性を別途確かめること。")

    if resolvable and rad is not None and surf_pen > 0:
        print(f"⛔⛔ **リンクのカプセルが cube に {surf_pen*1000:.1f} mm 食い込んでいる**"
              f"（標本の刻み {step_move*1000:.0f} mm < 隙間 {gap_min*1000:.0f} mm なので分解できている）。")
    elif not resolvable:
        print(f"⚠️⚠️ **食い込み量は判定しない。標本が粗すぎる。**"
              f"\n    記録 1 step あたりの先端移動 **{step_move*1000:.0f} mm** に対し、"
              f"標本上の最小の隙間は **{gap_min*1000:.0f} mm**。"
              f"\n    ⛔ **接触の瞬間は標本の間に落ちている。**"
              f"「触れていない」も「N mm 食い込んだ」も標本化の産物になる"
              f"（`frame_skip=4`。2026-09-25 に実際に誤読した）。")
    elif rad is not None:
        print(f"✅ **食い込みは無い**（半径込みで判定。標本の刻み {step_move*1000:.0f} mm "
              f"< 隙間 {gap_min*1000:.0f} mm）。")
    return total


def main():
    if len(sys.argv) > 1:
        check(sys.argv[1])
        return
    runs = sorted(p.parent.parent.name for p in
                  (ROOT / 'single_run').glob('*/trace/arm_trace.npz'))
    bad = []
    for r in runs:
        t = check(r)
        if t:
            bad.append((r, t))
        print()
    print('=' * 60)
    if bad:
        print(f"⛔⛔ **{len(bad)} 本で腕の軸が cube の内部に入っていた**")
        for r, t in bad:
            print(f"    {r}  ({t} step)")
    else:
        print("✅ 軸が cube に入った run は無い")


main()
