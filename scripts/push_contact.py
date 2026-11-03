#!/usr/bin/env python3
"""**押しで、腕が箱に触れていた step 数と、箱がどれだけ・どれだけ速く動いたか**を全話で出す（2026-11-02）。

修論 4.4.3.2「シード 0 は細い先端で箱を短く弾き、シード 1・2 は太い先端で箱を長く押した」の「短く・長く」が、
接触の時間か箱の移動距離かを軌跡で確かめるために作った（ユーザー「短く長くは軌跡で確かめて」、執筆/文体の指摘_20261102.md）。
系譜 9-301 は seed 0・seed 1 の 2 本で「箱に触れている step」を数えたが、seed 2（`tripo_pjp_long_phys_s3`）は数えていない。

接触: 腕のカプセル（台座の body 0 を除く全リンク、`ep_cap` があれば関節の後ろへのはみ出しも含む）の面と箱の面の
隙間が `--tol`（既定 5 mm）以下の step。⚠️ 記録の刻みは 1 step（0.04 s）なので、すり抜け気味の打撃は 1 step に収まる。
隙間は負（めり込み）も接触に数える。

    python3 scripts/push_contact.py                          # 既定: 押しの長腕 3 seed（tripo_pjp_long_phys*）
    python3 scripts/push_contact.py tripo_pjp_long_phys tripo_pjp_long_phys_s2_ws tripo_pjp_long_phys_s3   # 4.4.3.2 の 3 seed
    python3 scripts/push_contact.py 'tripo_pjp_mid_phys*'    # 中間の腕（glob も可）
    python3 scripts/push_contact.py --tol 0.010              # 接触の閾値を変える
    python3 scripts/push_contact.py --root <dir>             # single_run の代わりの場所（合成データの確かめ）

出力（話ごと）: 接触 step 数・最初に動いた step・箱の移動（始点から終点の距離）・箱の最大の速さ（m/step）。
run ごとに中央値と範囲。系譜 9-301 の値（seed 0 は 3〜4 step・0.102〜0.112 m/step、seed 1〔_s2_ws〕は 12〜14 step・
0.063〜0.069 m/step）とそろうかを最初に見る。そろわなければ閾値か測り方の差を疑ってから seed 2 を読む。
"""
import sys, glob, argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
S = np.linspace(0, 1, 201)


def cube_half(xml_name):
    """箱（body `cube` の geom）の半寸法。plot_run.py の同名の関数と同じ読み方（matplotlib を要さないよう写した）。"""
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


def seg_box_gap(a, b, r, lo, hi):
    """線分 ab（半径 r のカプセル）と軸平行な箱 [lo, hi] の隙間（負はめり込み）。"""
    pts = a + S[:, None] * (b - a)
    c, h = (lo + hi) / 2, (hi - lo) / 2
    q = np.abs(pts - c) - h
    d = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)
    return float(d.min()) - r


def episodes(d):
    if 'ep_xpos' not in d.files or 'ep_bone' not in d.files:
        return
    for e in range(len(d['ep_xpos'])):
        xp = np.asarray(d['ep_xpos'][e], float)
        xm = np.asarray(d['ep_xmat'][e], float)
        bo = np.asarray(d['ep_bone'][e], float)
        rad = np.asarray(d['ep_geom'][e] if d['ep_geom'][e] is not None else d['geom_size'], float).ravel()
        cf = None
        if 'ep_cap' in d.files and d['ep_cap'][e] is not None:
            cf = np.asarray(d['ep_cap'][e], float)
        elif 'cap_fromto' in d.files and len(d['cap_fromto']) == len(bo):
            cf = np.asarray(d['cap_fromto'], float)
        cube = np.asarray(d['ep_cubes'][e], float)
        yield e, xp, xm.reshape(len(xp), -1, 3, 3), bo, rad, cf, cube


def measure(xp, xm, bo, rad, cf, cube, half, tol):
    T = min(len(xp), len(cube))
    contact = 0
    for k in range(T):
        lo, hi = cube[k] - half, cube[k] + half
        if cf is not None and len(cf) == len(bo):
            a = xp[k] + np.einsum('lij,lj->li', xm[k], cf[:, :3])
            b = xp[k] + np.einsum('lij,lj->li', xm[k], cf[:, 3:])
        else:
            a = xp[k]
            b = xp[k] + np.einsum('lij,lj->li', xm[k], bo)
        g = min(seg_box_gap(a[i], b[i], rad[i], lo, hi) for i in range(1, len(bo)))
        if g <= tol:
            contact += 1
    mv = np.where(np.linalg.norm(cube[:T] - cube[0], axis=1) > 0.002)[0]
    v = np.linalg.norm(np.diff(cube[:T], axis=0), axis=1)
    return contact, (int(mv[0]) if len(mv) else None), float(np.linalg.norm(cube[T - 1] - cube[0])), float(v.max() if len(v) else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('patterns', nargs='*', default=['tripo_pjp_long_phys*'])
    ap.add_argument('--tol', type=float, default=0.005)
    ap.add_argument('--root', default='single_run')
    args = ap.parse_args()
    paths = sorted({q for pat in args.patterns for q in glob.glob(f'{args.root}/{pat}/trace/arm_trace.npz')})
    if not paths:
        print(f'⛔ 軌跡が無い: {args.root}/{args.patterns}/trace/arm_trace.npz（record_arm_trace.py で再生してから）')
        sys.exit(1)
    for p in paths:
        run = Path(p).parent.parent
        try:
            xml = yaml.safe_load(open(run / '.hydra' / 'config.yaml')).get('xml_name')
        except Exception:
            xml = None
        half = cube_half(xml) if xml else None
        if half is None:
            print(f'{run.name}: 箱の寸法が取れない（xml_name={xml}）。飛ばす')
            continue
        d = np.load(p, allow_pickle=True)
        rows = []
        for e, xp, xm, bo, rad, cf, cube in episodes(d):
            if len(cube) == 0 or np.abs(cube[:, 0]).max() > 50:
                continue   # 到達（箱を退避）
            c, first, disp, vmax = measure(xp, xm, bo, rad, cf, cube, half, args.tol)
            rows.append((e, c, first, disp, vmax, float(rad[-1]), cf is not None))
        if not rows:
            print(f'{run.name}: 押しの話が無い')
            continue
        print(f'\n{run.name}（{xml}、{len(rows)} 話、接触の閾値 {args.tol * 1000:.0f} mm、'
              f'{"カプセル" if rows[0][6] else "骨（はみ出し無し）"}、先端の半径 {rows[0][5]:.3f} m）')
        print('  話  接触step  動き出し  箱の移動 m  箱の最大の速さ m/step')
        for e, c, first, disp, vmax, _, _ in rows:
            print(f'  {e:2d}  {c:7d}  {str(first):>8s}  {disp:9.3f}  {vmax:10.3f}')
        C = [r[1] for r in rows]; D = [r[3] for r in rows]; V = [r[4] for r in rows]
        print(f'  ⇒ 接触 {min(C)}〜{max(C)} step（中央 {np.median(C):.0f}）・箱の移動 {min(D):.2f}〜{max(D):.2f} m・'
              f'最大の速さ {min(V):.3f}〜{max(V):.3f} m/step')


if __name__ == '__main__':
    main()
