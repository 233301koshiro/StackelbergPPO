#!/usr/bin/env python3
"""ホッケーの壁での跳ね返りの比を、20 話の軌跡から出す（系譜 9-263 ⑦ の測り方、9-309）。学習も再生もしない。

    python3 scripts/hockey_bounce_ratio.py <run>[:<npz>] ...     # 既定の npz は trace/arm_trace_20.npz
    python3 scripts/hockey_bounce_ratio.py --reproduce            # 既知の失敗を拾えるかを先に確かめる

定義（9-263 ⑦）: パックが壁・板の面に最も近づいた記録 step を k とし、法線の速度の比
  |v_n[k+1]| ／ |v_n[k-2]|（衝突の 1 step 後 ／ 2 step 前。v[k] = (c[k+1] - c[k]) / DT）を出す。
  腕のカプセルが k-2〜k+1 のあいだ壁の箱（xy）から 0.3 m 以上離れている跳ね返りだけを数える。
  壁の反射（e = 0.75）に摩擦が乗るので、正常なら約 0.71。Bug 70 型（当て直しが切れる）は約 1.66。
⚠️ k を「速度の向きが変わった区間」で取ると、衝突をまたぐ区間の速度が混じって 0.12・0.53・0.94 が出る
  （9-263 ⑩ の「自作の数え方で 2〜3」と同じ型。9-309 で原因を特定した）。k は面に最も近い位置で取る。
⚠️ 角（側壁とゴール側の壁）に同時に当たると、片方の軸の比が下がる（0.18 など）。加速ではないので Bug 70 型ではない。
⚠️ 検出の条件は ⑥・⑦ と同じではないので、跳ね返りの数は一致しない（比の値は一致する）。
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
DT = 0.04            # 記録 1 step（timestep 0.01 × frame_skip 4）
NEAR = 0.03          # パックの中心と、半辺だけ広げた箱の距離 [m]
FAR = 0.30           # 腕が壁から離れている [m]
SLOW = 0.20          # これより遅い跳ね返りは比が乱れるので集計から外す [m/s]
BAD = 0.90           # Bug 70 型（比が 1 を超える）とみなす下限
NS = 21              # カプセル 1 本あたりの標本点


def walls(xml):
    """壁と板（worldbody 直下の body、Choreonoid に存在する形。9-98）の xy の中心と半辺。"""
    t = ET.parse(ROOT / 'assets' / 'mujoco_envs' / f'{xml}.xml')
    out = []
    for b in t.findall('worldbody/body'):
        n = b.get('name')
        if n.startswith('wall_') or n == 'goal_board':
            p = [float(x) for x in b.get('pos').split()]
            s = [float(x) for x in b.find('geom').get('size').split()]
            out.append((n, np.array(p[:2]), np.array(s[:2])))
    return out


def puck_half(xml):
    t = ET.parse(ROOT / 'assets' / 'mujoco_envs' / f'{xml}.xml')
    for b in t.iter('body'):
        if b.get('name') == 'cube':
            return float(b.find('geom').get('size').split()[0])
    raise SystemExit(f'{xml}: パックが無い')


def box_dist(pts, c, s):
    return np.linalg.norm(np.maximum(np.abs(pts - c) - s, 0), axis=-1)


def arm_pts(xp, xm, bone):
    t = np.linspace(0, 1, NS)
    ends = xp + np.einsum('lij,lj->li', xm, bone)
    return xp[:, None, :] + t[None, :, None] * (ends - xp)[:, None, :]


def bounces(run, fname='arm_trace_20.npz'):
    """[(話, k, 壁, 法線の軸, 比, 前の速さ)]"""
    d = np.load(ROOT / 'single_run' / run / 'trace' / fname, allow_pickle=True)
    xml = yaml.safe_load(open(ROOT / 'single_run' / run / '.hydra' / 'config.yaml'))['xml_name']
    W, half = walls(xml), puck_half(xml)
    res = []
    for e in range(int(d['ep_count'])):
        c = np.asarray(d['ep_cubes'][e], float)[:, :2]
        xp, xm = np.asarray(d['ep_xpos'][e], float), np.asarray(d['ep_xmat'][e], float)
        bone = np.asarray(d['ep_bone'][e], float)
        rad = np.asarray(d['ep_geom'][e], float).ravel()
        v = np.diff(c, axis=0) / DT
        for k in range(2, len(v) - 1):
            for n, wc, ws in W:
                se = ws + half
                if box_dist(c[k], wc, se) > NEAR:
                    continue
                rel = c[k] - wc
                ax = int(np.argmax(np.abs(rel) - se))        # はみ出している軸＝当たっている面
                sgn = np.sign(rel[ax])                       # 面の外向き法線
                pn = (c[k - 1:k + 2, ax] - wc[ax]) * sgn
                if not (pn[1] < pn[0] and pn[1] <= pn[2]):   # k が面に最も近い step でない
                    continue
                if not (v[k - 2, ax] * sgn < -0.05 and v[k + 1, ax] * sgn > 0.05):
                    continue
                if res and res[-1][0] == e + 1 and res[-1][2] == n and k - res[-1][1] <= 3:
                    continue
                arm = min(float((box_dist(arm_pts(xp[j], xm[j], bone)[..., :2], wc, ws) - rad[:, None]).min())
                          for j in range(k - 2, k + 2))
                if arm < FAR:
                    continue
                pre = abs(v[k - 2, ax])
                res.append((e + 1, k, n, 'xy'[ax], abs(v[k + 1, ax]) / pre, pre))
    return res


def report(run, fname='arm_trace_20.npz', verbose=True):
    res = bounces(run, fname)
    fast = np.array([r[4] for r in res if r[5] >= SLOW])
    print(f'=== {run}: 腕から {FAR} m 以上離れた跳ね返り {len(res)} 回（前の速さ ≥ {SLOW} m/s は {len(fast)} 回）')
    if len(fast):
        print(f'  比: 中央 {np.median(fast):.2f}  範囲 [{fast.min():.2f}, {fast.max():.2f}]'
              f'  ⛔ ＞ {BAD}（Bug 70 型）は {int((fast > BAD).sum())} 回')
    if verbose:
        for r in res:
            flag = f'  ⛔ ＞ {BAD}' if r[4] > BAD and r[5] >= SLOW else ('  （遅い）' if r[5] < SLOW else '')
            print('  話 {:2d} step {:4d} {:12s} 法線 {}  比 {:.2f}  （前 {:.2f} m/s）'.format(*r) + flag)
    return res


def reproduce():
    """9-263 ⑦ の既知の値: Bug 70 の前の _g10 は話 3・19・20 が 1.66〜1.67、直した後の _wfix は全部 0.71。"""
    ok = True
    bad = sorted({r[0] for r in report('hockey_bank9_phys_g10', verbose=False) if r[4] > BAD and r[5] >= SLOW})
    print(f'  → Bug 70 型の話 {bad}（期待 [3, 19, 20]）')
    ok &= bad == [3, 19, 20]
    fixed = [r[4] for r in report('hockey_bank9_phys_g10_wfix', verbose=False) if r[5] >= SLOW]
    print(f'  → 直した後の比 [{min(fixed):.2f}, {max(fixed):.2f}]（期待 すべて 0.70〜0.72）')
    ok &= all(0.70 <= x <= 0.72 for x in fixed)
    print('✅ 既知の失敗を拾えた' if ok else '⛔ 既知の値を再現できない（測り方を疑う）')
    return ok


def main():
    args = sys.argv[1:]
    if not args or args == ['--reproduce']:
        sys.exit(0 if reproduce() else 1)
    for a in args:
        run, _, fname = a.partition(':')
        report(run, fname or 'arm_trace_20.npz')


if __name__ == '__main__':
    main()
