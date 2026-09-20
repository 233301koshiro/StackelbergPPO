#!/usr/bin/env python3
"""**軌跡を図にする**（実験系譜 9-155）。⭐ **数値の検査は「見るべきと分かっている量」しか見ない。**

⚠️⚠️ **2026-09-21 に実際に踏んだ**（9-154）: `check_before_conclusion.py` の 5 項目も
`check_obstacle_clearance.py` も通したのに、**リンクが床より 270 mm 下へ出ていた**。
⛔ **どの検査にも引っかからなかった。図にして初めて分かった。**

    python3 scripts/plot_run.py <run> [<run> ...]        # → single_run/<run>/trace/view.png
    python3 scripts/plot_run.py --all                    # 軌跡のある run 全部
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import yaml

ROOT = Path(__file__).resolve().parent.parent


def statics(xml_name):
    """柱・壁など、関節を持たない worldbody 直下の body を (lo, hi, name) で返す。"""
    out = []
    try:
        t = ET.parse(ROOT / 'assets' / 'mujoco_envs' / f'{xml_name}.xml').getroot()
        wb = t.find('worldbody')
        for i, b in enumerate(wb.findall('body')):
            if i == 0 or b.get('name') == 'cube' or b.find('joint') is not None:
                continue
            g = b.find('geom')
            if g is None:
                continue
            p = np.array([float(x) for x in b.get('pos', '0 0 0').split()])
            sz = [float(x) for x in g.get('size', '0').split()]
            ty = g.get('type', 'box')
            if ty == 'box' and len(sz) >= 3:
                h = np.array(sz[:3])
            elif ty in ('cylinder', 'capsule') and len(sz) >= 2:
                h = np.array([sz[0], sz[0], sz[1]])
            elif ty == 'sphere' and sz:
                h = np.array([sz[0]] * 3)
            else:
                continue
            out.append((p - h, p + h, b.get('name', '?'), ty))
    except Exception:
        pass
    return out


def plot(run):
    d = ROOT / 'single_run' / run
    p = d / 'trace' / 'arm_trace.npz'
    if not p.exists():
        print(f'⛔ {run}: 軌跡なし'); return None
    z = np.load(p, allow_pickle=True)
    xp, xm, bo = z['xpos'], z['xmat'], z['bone_offset']
    tgt = np.asarray(z['target'], float) if 'target' in z else None
    cube = z['cube'] if 'cube' in z else None
    ends = xp + np.einsum('tlij,lj->tli', xm, bo)
    T = len(xp)
    try:
        xml = yaml.safe_load(open(d / '.hydra' / 'config.yaml')).get('xml_name')
    except Exception:
        xml = None
    st = statics(xml) if xml else []

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, (ia, ib, la, lb) in zip(axes, [(0, 1, 'x', 'y'), (0, 2, 'x', 'z')]):
        for k in range(0, T, max(T // 25, 1)):
            for i in range(len(bo)):
                ax.plot([xp[k, i, ia], ends[k, i, ia]], [xp[k, i, ib], ends[k, i, ib]],
                        color=plt.cm.viridis(k / T), lw=.8, alpha=.55)
        ax.plot(ends[:, -1, ia], ends[:, -1, ib], 'r-', lw=1.6, label='tip')
        if cube is not None and np.abs(cube).sum() > 0:
            ax.plot(cube[:, ia], cube[:, ib], 'b--', lw=1.2, label='cube')
        if tgt is not None:
            ax.plot(tgt[ia], tgt[ib], 'k*', ms=16, label='target')
        for lo, hi, nm, ty in st:
            a, b = (lo[ia], hi[ia]), (lo[ib], hi[ib])
            ax.add_patch(plt.Rectangle((a[0], b[0]), a[1] - a[0], b[1] - b[0],
                         fc='orange', alpha=.4, ec='darkorange', lw=1.6))
        if ib == 2:
            ax.axhline(0, color='saddlebrown', lw=2.5, alpha=.7)   # ⭐ 床
        ax.set_xlabel(la); ax.set_ylabel(lb); ax.grid(alpha=.3); ax.set_aspect('equal')
        ax.set_title(f'{la}-{lb}')
    # ⭐ 極値を図に焼き込む（9-96: 動く物体すべての極値を先に出す）
    allp = np.concatenate([xp, ends], axis=1)
    zmin = float(allp[..., 2].min())
    note = (f'{run}   step={T}   全リンクの z 最小 = {zmin*1000:.0f} mm'
            + ('  ⛔ 床下!' if zmin < -0.001 else ''))
    if tgt is not None:
        dd = np.linalg.norm(ends[:, -1, :] - tgt, axis=1)
        note += f'   目標最小 {dd.min()*1000:.1f} mm'
    fig.suptitle(note, fontsize=11)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    out = d / 'trace' / 'view.png'
    plt.savefig(out, dpi=100); plt.close()
    print(f'⭐ {run:<28} z最小 {zmin*1000:7.1f} mm' + ('  ⛔ **床下**' if zmin < -0.001 else '') + f'  → {out}')
    return out


def main():
    args = sys.argv[1:]
    if args == ['--all'] or not args:
        runs = sorted(p.parent.parent.name for p in (ROOT / 'single_run').glob('*/trace/arm_trace.npz'))
    else:
        runs = args
    for r in runs:
        plot(r)


if __name__ == '__main__':
    main()
