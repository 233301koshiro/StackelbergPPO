#!/usr/bin/env python3
"""**軌跡を図にする**（実験系譜 9-155）。⭐ **数値の検査は「見るべきと分かっている量」しか見ない。**

⚠️⚠️ **2026-09-21 に実際に踏んだ**（9-154）: `check_before_conclusion.py` の 5 項目も
`check_obstacle_clearance.py` も通したのに、**リンクが床より 270 mm 下へ出ていた**。
⛔ **どの検査にも引っかからなかった。図にして初めて分かった。**

    python3 scripts/plot_run.py <run> [<run> ...]        # → single_run/<run>/trace/view.png
    python3 scripts/plot_run.py --trace <npz> <run>     # 途中の epoch の軌跡 → <npz>.png
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


def cube_half(xml_name):
    """箱（body `cube` の geom）の半寸法。無ければ None。"""
    try:
        t = ET.parse(ROOT / 'assets' / 'mujoco_envs' / f'{xml_name}.xml').getroot()
        for b in t.find('worldbody').findall('body'):
            if b.get('name') == 'cube':
                g = b.find('geom')
                sz = [float(x) for x in g.get('size', '0').split()]
                if g.get('type', 'box') == 'box' and len(sz) >= 3:
                    return np.array(sz[:3])
    except Exception:
        pass
    return None


def stadium(a, b, r, n=24):
    """カプセル（a〜b の線分＋半径 r）を平面へ投影した輪郭（両端が半円の長円）。a・b は 2 次元。"""
    d = b - a
    L = float(np.hypot(*d))
    th = np.arctan2(d[1], d[0]) if L > 1e-9 else 0.0
    u = np.linspace(-np.pi / 2, np.pi / 2, n)
    pb = b + r * np.c_[np.cos(th + u), np.sin(th + u)]
    pa = a + r * np.c_[np.cos(th + np.pi + u), np.sin(th + np.pi + u)]
    return np.vstack([pb, pa, pb[:1]])


def snapshot_steps(ends, cube, tgt):
    """カプセルを描く姿勢: 最初・最後と、目標に最も近い step（到達）か、箱が 2 mm 動く 1 step 前（押し。render_ra_forms.py と同じ）。"""
    T = len(ends)
    ks = {0: 'start', T - 1: 'end'}
    if cube is not None and np.abs(cube).sum() > 0 and np.abs(cube[:, 0]).max() < 50:
        mv = np.where(np.linalg.norm(cube - cube[0], axis=1) > 0.002)[0]
        if len(mv):
            ks[max(int(mv[0]) - 1, 0)] = 'hit-1'
    if tgt is not None:
        ks[int(np.argmin(np.linalg.norm(ends[:, -1, :] - tgt, axis=1)))] = 'closest'
    return sorted(ks.items())


def plot(run, trace=None, out=None):
    """trace / out: 途中の epoch の軌跡（watch_midrun.sh の trace_mid/ep<N>.npz）を描くとき。既定は trace/arm_trace.npz → trace/view.png"""
    d = ROOT / 'single_run' / run
    p = Path(trace) if trace else d / 'trace' / 'arm_trace.npz'
    if not p.exists():
        print(f'⛔ {run}: 軌跡なし'); return None
    z = np.load(p, allow_pickle=True)
    xp, xm, bo = z['xpos'], z['xmat'], z['bone_offset']
    tgt = np.asarray(z['target'], float) if 'target' in z else None
    cube = z['cube'] if 'cube' in z else None
    ends = xp + np.einsum('tlij,lj->tli', xm, bo)
    # ⚠️ 根（`0`、球の台座）の骨は運動学に効かない設計変数で（9-154）、線にすると床下へ伸びて見える。
    #   描かず、z 最小にも入れない（`0` の位置そのものは入れる）。⛔ 10-11 に「BELOW FLOOR -297 mm」と誤報した
    ends[:, 0] = xp[:, 0]
    T = len(xp)
    # ⭐ 2026-11-02（系譜 9-331）: カプセルの輪郭を重ねる。骨の線だけでは、太さと関節の後ろへのはみ出し（`ext_start`）による
    #   重なりが見えない（D-7 の直す前の形は伸ばしたまま `1`×`111` が 49 mm 重なっていた。Bug 72 も同じ型）。
    #   `cap_fromto` の無い古い軌跡（10-20 より前の再生）は骨で代える（はみ出しは描けない。題に cap=bone と出す）
    rad = np.asarray(z['geom_size'], float).ravel() if 'geom_size' in z.files else np.full(len(bo), 0.03)
    rad = np.where(np.isfinite(rad), rad, 0.03)
    has_cap = 'cap_fromto' in z.files and np.isfinite(np.asarray(z['cap_fromto'], float)).all()
    if has_cap:
        cf = np.asarray(z['cap_fromto'], float)
        c0 = xp + np.einsum('tlij,lj->tli', xm, cf[:, :3])
        c1 = xp + np.einsum('tlij,lj->tli', xm, cf[:, 3:])
    else:
        c0, c1 = xp, ends
    try:
        xml = yaml.safe_load(open(d / '.hydra' / 'config.yaml')).get('xml_name')
    except Exception:
        xml = None
    st = statics(xml) if xml else []
    ch = cube_half(xml) if xml else None
    snaps = snapshot_steps(ends, cube, tgt)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, (ia, ib, la, lb) in zip(axes, [(0, 1, 'x', 'y'), (0, 2, 'x', 'z')]):
        for k in range(0, T, max(T // 25, 1)):
            for i in range(len(bo)):
                ax.plot([xp[k, i, ia], ends[k, i, ia]], [xp[k, i, ib], ends[k, i, ib]],
                        color=plt.cm.viridis(k / T), lw=.8, alpha=.55)
        for (k, nm), col in zip(snaps, ['tab:blue', 'tab:green', 'tab:purple', 'tab:cyan']):
            for i in range(len(bo)):
                pts = stadium(c0[k, i, [ia, ib]], c1[k, i, [ia, ib]], rad[i])
                ax.plot(pts[:, 0], pts[:, 1], ':', color=col, lw=1.3, label=f'capsules @ {nm} (step {k})' if i == 0 else None)
            ax.plot(xp[k, 1:, ia], xp[k, 1:, ib], 'o', color=col, ms=3.5)        # 関節
            if ch is not None and cube is not None and np.abs(cube[k, 0]) < 50 and np.abs(cube).sum() > 0:
                ax.add_patch(plt.Rectangle((cube[k, ia] - ch[ia], cube[k, ib] - ch[ib]), 2 * ch[ia], 2 * ch[ib],
                             fill=False, ec=col, lw=1.0, ls='--'))
        ax.plot(ends[:, -1, ia], ends[:, -1, ib], 'r-', lw=1.6, label='tip')
        if cube is not None and np.abs(cube).sum() > 0 and np.abs(cube[:, 0]).max() < 50:   # 到達で退避した箱（x≈101、9-234）は描かない
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
    # ⚠️ 題は ASCII だけ（日本語はフォントが無く化け、「目標最小」を「隣接」と読み違えた。系譜 9-239）
    note = (f'{run}   step={T}   link z min = {zmin*1000:.0f} mm'
            + ('  BELOW FLOOR!' if zmin < -0.001 else ''))
    # ⭐ 2026-11-02: カプセルの面の最も下（骨の z 最小は太さを見ない）。根元の球と縦の柱は床に据えてあるので除く
    #   （check_invariants.py の床と同じ規則）
    keep = [i for i in range(1, len(bo)) if not (i == 1 and abs(bo[1][2]) > 0.9 * np.linalg.norm(bo[1]) + 1e-9)]
    czmin = float(np.nanmin((np.minimum(c0[..., 2], c1[..., 2]) - rad)[:, keep]))
    note += f'   moving capsule surface z min = {czmin*1000:.0f} mm' + ('  CAPSULE BELOW FLOOR!' if czmin < -0.005 else '')
    if tgt is not None:
        dd = np.linalg.norm(ends[:, -1, :] - tgt, axis=1)
        note += f'   tip-target min {dd.min()*1000:.1f} mm'
    try:
        sys.path.insert(0, str(ROOT / 'scripts'))
        from check_self_penetration import measure
        far = measure(str(p))[4]
        note += f'\nfar-pair overlap max {far[0]*1000:.0f} mm ({far[1]*100:.1f} % of steps{", " + "x".join(far[3]) if far[3] else ""})'
    except Exception as e:
        note += f'\nfar-pair overlap: n/a ({type(e).__name__})'
    note += '   cap=' + ('capsule' if has_cap else 'bone (old trace: no ext_start)')
    fig.suptitle(note, fontsize=11)
    axes[0].legend(fontsize=7, loc='upper left', bbox_to_anchor=(0, -0.12), ncol=3)
    plt.tight_layout()
    out = Path(out) if out else d / 'trace' / 'view.png'
    plt.savefig(out, dpi=100); plt.close()
    print(f'⭐ {run:<28} z最小 {zmin*1000:7.1f} mm' + ('  ⛔ **床下**' if zmin < -0.001 else '')
          + f'  カプセルの面 {czmin*1000:7.1f} mm' + ('  ⛔ **床下**' if czmin < -0.005 else '') + f'  → {out}')
    return out


def main():
    args = sys.argv[1:]
    if args[:1] == ['--trace']:     # --trace <npz> <run> → <npz>.png
        plot(args[2], args[1], args[1].replace('.npz', '.png')); return
    if args == ['--all'] or not args:
        runs = sorted(p.parent.parent.name for p in (ROOT / 'single_run').glob('*/trace/arm_trace.npz'))
    else:
        runs = args
    for r in runs:
        plot(r)


if __name__ == '__main__':
    main()
