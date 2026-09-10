#!/usr/bin/env python3
"""ホッケー台がパックを閉じ込められるかを**学習せずに**試す（Bug 39 / 9-65）。

**なぜ要るか**: Bug 39 は 200 epoch 回してもログに一切出なかった。
`fwd_cube` も `exec_R_eps` も正常に見え、**再生して初めて**パックが壁を
突き抜けていると分かった。**同じ穴を二度踏まないための物理だけの検査。**

やること: パックを台の各所から角度・速度を振って撃ち、台の外へ出るかを数える。
腕も方策も使わない。数秒で終わる。

⚠️ **限界**: 学習は Choreonoid 上で走るが、本プローブは **MuJoCo で直接**
XML を読む。接触の実装が完全に同じとは限らないので、**絶対的な保証ではない**。
そのため下の `--reproduce` で **既知の脱出（Bug 39）を再現できること**を先に確かめ、
再現できた場合にのみ「この検査は効いている」と扱う。

使い方:
  python3 scripts/probe_hockey_containment.py --reproduce   # 既知の脱出を再現するか
  python3 scripts/probe_hockey_containment.py               # いまの XML を検査
  python3 scripts/probe_hockey_containment.py --xml <path>
"""
import argparse, itertools, sys
import numpy as np
import mujoco

# 台の呼び値（make_hockey_court_xml.py と揃える。ズレたら FAIL させる）
COURT_Y1 = 0.50
GOAL_HALF = 0.15
GOAL_X = 1.55
PUCK_HALF = 0.05


BOARD_X = 0.95          # 中央の板の位置
BOARD_HALF_Y = 0.105    # 板の半長（GOAL_HALF*2*0.70/2）


def escaped(x, y, prev):
    """台の外に出たか／板を抜けたか。ゴール口を通った場合だけは「出てよい」。

    ⚠️ **板の貫通も失敗**である。板は「反射しないと入らない」を成立させる装置なので、
    そこを突き抜けられると**問い自体が壊れる**（Bug 39 と同じ壊れ方）。
    """
    if abs(y) > COURT_Y1 + 1e-6:                  # 側方へ抜けた
        return True, 'side'
    if x > GOAL_X + 1e-6 and abs(y) > GOAL_HALF:   # ゴールラインを口以外で越えた
        return True, 'through-goal-wall'
    if prev is not None:
        px, py = prev
        if px < BOARD_X <= x and abs(y) < BOARD_HALF_Y and abs(py) < BOARD_HALF_Y:
            return True, 'through-board'          # 板を正面から通り抜けた
    return False, ''


def shoot(model, data, x0, y0, speed, angle_deg, nstep=400):
    """パックを (x0,y0) から角度 angle_deg・速さ speed で撃つ。戻り値: (脱出したか, 理由, 軌跡)"""
    mujoco.mj_resetData(model, data)
    jx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'cube_slide')
    jy = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'cube_slide2')
    ax, ay = model.jnt_qposadr[jx], model.jnt_qposadr[jy]
    vx, vy = model.jnt_dofadr[jx], model.jnt_dofadr[jy]
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
    bx, by = model.body_pos[bid][0], model.body_pos[bid][1]
    data.qpos[ax], data.qpos[ay] = x0 - bx, y0 - by
    th = np.deg2rad(angle_deg)
    data.qvel[vx], data.qvel[vy] = speed*np.cos(th), speed*np.sin(th)
    traj, prev = [], None
    for _ in range(nstep):
        mujoco.mj_step(model, data)
        x, y = bx + data.qpos[ax], by + data.qpos[ay]
        out, why = escaped(x, y, prev)
        traj.append((x, y)); prev = (x, y)
        if out:
            return True, why, traj
    return False, '', traj


def sweep(model, data, speeds, angles, starts, verbose=False):
    fails = []
    n = 0
    for (x0, y0), sp, an in itertools.product(starts, speeds, angles):
        n += 1
        out, why, traj = shoot(model, data, x0, y0, sp, an)
        if out:
            fails.append((x0, y0, sp, an, why, traj[-1]))
            if verbose:
                print(f"  ✗ 脱出 start=({x0:.2f},{y0:+.2f}) v={sp:.1f} θ={an:+.0f}°  "
                      f"理由={why} 最終=({traj[-1][0]:.3f},{traj[-1][1]:+.3f})")
    return n, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xml', default='assets/mujoco_envs/e2e_hockey_wall.xml')
    ap.add_argument('--reproduce', action='store_true',
                    help='Bug 39 の既知の脱出（y の限界を滑って壁を抜ける）を再現するかだけ見る')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(a.xml)
    data = mujoco.MjData(model)

    if a.reproduce:
        # Bug 39 の実測: 腕に当てられた直後 (0.71, 0.02) から 4.14 m/s で斜め後方へ
        print("Bug 39 の再現を試す（実測の当たり方に近い条件）")
        worst = None
        for an in range(-80, 81, 5):
            out, why, traj = shoot(model, data, 0.71, 0.02, 4.14, an)
            if out:
                print(f"  ✗ 脱出: θ={an:+d}° 理由={why} 最終=({traj[-1][0]:.3f},{traj[-1][1]:+.3f})")
                worst = (an, why)
        if worst:
            print("\n✅ **このプローブは Bug 39 を再現できる。** 検査として有効。")
            return 0
        print("\n⚠️ **再現しなかった。** MuJoCo と Choreonoid の接触処理が違う可能性がある。"
              "\n   このプローブの合格を根拠にしないこと。")
        return 2

    speeds = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    angles = list(range(-80, 81, 10))
    starts = [(0.55, 0.0), (0.55, +0.30), (0.55, -0.30), (0.71, 0.02), (0.90, +0.40)]
    n, fails = sweep(model, data, speeds, angles, starts, verbose=a.verbose or True)
    print(f"\n{a.xml}")
    print(f"  試行 {n} 通り（開始 {len(starts)} × 速度 {len(speeds)} × 角度 {len(angles)}）")
    print(f"  脱出 {len(fails)} 件")
    if fails:
        by = {}
        for f in fails: by[f[4]] = by.get(f[4], 0) + 1
        for k, v in sorted(by.items()): print(f"    {k}: {v} 件")
        print("\n❌ **FAIL。この台では投入しない。**")
        return 1
    print("\n✅ **PASS。パックは台の外へ出ない。**")
    return 0


if __name__ == '__main__':
    sys.exit(main())
