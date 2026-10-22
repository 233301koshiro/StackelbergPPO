#!/usr/bin/env python3
"""ホッケーの取り直しを、事前登録した読みで機械判定する（系譜 9-230・9-237・9-238）。学習も再生もしない。

    python3 scripts/judge_hockey_goal.py                        # hockey_bank9_phys / _s2 の trace/arm_trace_20.npz
    python3 scripts/judge_hockey_goal.py hockey_bank9:arm_trace.npz   # 既知の結果で確かめる（9-229: seed0 は 3 話中 2 話ゴール）

話ごとに出すもの:
  - 置いた瞬間の隙間: 全リンク（カプセル）とパック（箱）の最小距離。10 cm 未満の話は「打撃」と数えない（Bug 55）
  - ゴール: パックの中心が x ≥ ゴール線 かつ |y| ≤ ゴール口の半幅 に入った step があるか（線と口は XML から読む）
  - 移動: パックの最大変位
  - ⚠️ 不審な加速: 腕がパックから 0.15 m 以上離れているのに、パックの速さが 1 記録 step で 0.5 m/s 以上増えた step
    （閾値 1.0 で板・壁との深い接触が押し戻されると、腕が触れずに加速しうる。壁の反射は e=0.75 で減速しかしない）
読み（`queue_hockey_goal_rate.sh` 冒頭、結果より前に登録）:
  ① どちらかの seed で隙間 ≥ 10 cm の話がゴール ② ゴールは無いが隙間 ≥ 10 cm の話で 0.1 m 以上動く ③ どれも無い
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from check_cube_penetration import cube_half  # noqa: E402

GAP_MIN = 0.10      # 置いた瞬間の隙間の下限 [m]（9-230）
MOVE_MIN = 0.10     # 「動いた」の下限 [m]
FAR = 0.15          # 腕が離れている [m]
JUMP = 0.5          # 不審な加速 [m/s / 記録 step]
DT = 0.04           # 記録 1 step（timestep 0.01 × frame_skip 4）
NS = 21             # カプセル 1 本あたりの標本点


def goal_geom(xml_name):
    """ゴール線 x と口の半幅を XML の壁から読む（ゴール側の壁の内側の端）。"""
    t = ET.parse(ROOT / 'assets' / 'mujoco_envs' / f'{xml_name}.xml')
    for b in t.findall('worldbody/body'):
        if b.get('name') == 'wall_goal_p':
            p = [float(x) for x in b.get('pos').split()]
            s = [float(x) for x in b.find('geom').get('size').split()]
            return p[0] - s[0], p[1] - s[1]
    raise SystemExit(f'{xml_name}: ゴール側の壁が無い')


def back_stop(xml_name):
    """パックの x の関節の下限（world 座標）。⭐ 2026-10-22: 台の手前の端（側壁の端 x 0.15）に壁は無く、
    この下限 x 0.20（＝0.15＋半幅 0.05）が跳ね返らない見えない後ろ壁になっている。_wfix seed0 のゴール 8 話のうち
    4 話がここに触れてから入った（系譜 9-288 の追記）。"""
    t = ET.parse(ROOT / 'assets' / 'mujoco_envs' / f'{xml_name}.xml')
    for b in t.iter('body'):
        if b.get('name') == 'cube':
            j = [j for j in b.findall('joint') if j.get('name') == 'cube_slide' and j.get('range')]
            if j:
                return float(b.get('pos').split()[0]) + float(j[0].get('range').split()[0])
    return None


def gap(xpos, xmat, bone, rad, puck, half):
    """全カプセルとパック（軸平行の箱）の最小距離 [m]。xpos (L,3) xmat (L,3,3) bone (L,3) rad (L,)"""
    t = np.linspace(0, 1, NS)
    ends = xpos + np.einsum('lij,lj->li', xmat, bone)
    pts = xpos[:, None, :] + t[None, :, None] * (ends - xpos)[:, None, :]       # (L, NS, 3)
    d = np.linalg.norm(np.maximum(np.abs(pts - puck) - half, 0), axis=-1)      # 点と箱
    return float((d - rad[:, None]).min())


def judge(run, fname):
    d = np.load(ROOT / 'single_run' / run / 'trace' / fname, allow_pickle=True)
    xml = yaml.safe_load(open(ROOT / 'single_run' / run / '.hydra' / 'config.yaml'))['xml_name']
    _, half, _ = cube_half(xml)
    gx, gy = goal_geom(xml)
    bx = back_stop(xml)
    rows = []
    for e in range(int(d['ep_count'])):
        bone = np.asarray(d['ep_bone'][e], float)
        rad = np.asarray(d['ep_geom'][e], float).ravel()
        ix, im, ic = (np.asarray(a, float) for a in d['ep_init'][e])
        g0 = gap(ix, im, bone, rad, ic, half)
        c = np.asarray(d['ep_cubes'][e], float)
        xp, xm = np.asarray(d['ep_xpos'][e], float), np.asarray(d['ep_xmat'][e], float)
        hit = (c[:, 0] >= gx) & (np.abs(c[:, 1]) <= gy)
        goal = bool(hit.any())
        end = int(np.argmax(hit)) if goal else len(c)
        back = int((c[:end, 0] <= bx + 0.0035).sum()) if bx is not None else 0    # ゴールより前に手前の止めにいた step
        move = float(np.linalg.norm(c[:, :2] - c[0, :2], axis=1).max())
        sp = np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1) / DT
        jumps = 0
        # sp[k] が sp[k-1] と sp[k-2] のどちらより JUMP 以上速い。⭐ 2026-10-22: 以前は sp[k-1] だけと比べ、ふつうの跳ね返り
        # （衝突を含む記録 step で速さが一度落ちて戻る。_blk3d の話 2・5 は比 0.73）も数えていた（check_invariants.py と同じ直し）
        for k in np.nonzero(sp[2:] - np.maximum(sp[1:-1], sp[:-2]) > JUMP)[0] + 2:
            if min(gap(xp[j], xm[j], bone, rad, c[j], half) for j in (k, k + 1)) > FAR:
                jumps += 1
        rows.append((g0, goal, move, jumps, back))
    return rows, gx, gy, bx


def main():
    args = sys.argv[1:] or ['hockey_bank9_phys:arm_trace_20.npz', 'hockey_bank9_phys_s2:arm_trace_20.npz']
    clean_goal = clean_move = False
    judged = 0
    for a in args:
        run, _, fname = a.partition(':')
        try:
            rows, gx, gy, bx = judge(run, fname or 'arm_trace_20.npz')
        except FileNotFoundError:
            print(f'⏳ {run}: 軌跡がまだ無い'); continue
        n = len(rows)
        judged += 1
        ok = [r for r in rows if r[0] >= GAP_MIN]
        print(f'\n=== {run}（{n} 話、ゴール線 x ≥ {gx:.2f}・|y| ≤ {gy:.2f}）')
        for i, (g0, goal, move, jumps, back) in enumerate(rows, 1):
            print(f'  話 {i:2d}  隙間 {g0*1000:5.0f} mm{"" if g0 >= GAP_MIN else " ⛔<10cm"}  移動 {move:4.2f} m'
                  f'  {"⭐ ゴール" if goal else "        "}{f"  ⚠️ 腕が離れて加速 {jumps} 回" if jumps else ""}'
                  f'{f"  ⚠️ 手前の見えない止め（x {bx:.2f}）に {back} step" if back else ""}')
        k = sum(r[1] for r in ok)
        print(f'  → 隙間 ≥ 10 cm の {len(ok)} 話のうち ゴール {k} 話・0.1 m 以上動いた {sum(r[2] >= MOVE_MIN for r in ok)} 話'
              f'  ／ 不審な加速のある話 {sum(r[3] > 0 for r in rows)}'
              f'  ／ 手前の止めに触れてから入ったゴール {sum(r[1] and r[4] > 0 for r in ok)} 話')
        clean_goal |= k > 0
        clean_move |= any(r[2] >= MOVE_MIN for r in ok)
    if not judged:
        print('\n読み: ⏳ 判定できない（軌跡がまだ無い）'); return
    print('\n読み:', '① ゴールは成立' if clean_goal else ('② 打撃は成立、ゴールは主張しない' if clean_move else '③ 動かない → 原因を調べる'),
          '（⚠️ 不審な加速がある話・手前の見えない止めに触れてから入った話は図で確かめてから数える）')


if __name__ == '__main__':
    main()
