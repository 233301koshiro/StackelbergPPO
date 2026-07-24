#!/usr/bin/env python3
"""
analyze_reach_kinematics.py: 平面直列Nリンク腕の到達可能性を解析的（IK的）に計算し、
co-designが実際に収束させた形態と比較する。

平面直列チェーン（全関節がZ軸回りの回転関節）の到達可能領域は、根本を中心とした
円環（annulus）になる:
  R_max = sum(l_i)                          （全リンクを一直線に伸ばした最大到達距離）
  R_min = max(0, 2*max(l_i) - sum(l_i))     （最長リンクが他の合計を上回る場合の「死角」）
target までの距離 D が R_min <= D <= R_max を満たせば、関節角度の選び方によって
必ず到達できる（決定論的なIKの存在条件）。学習は一切不要。

使い方:
  python3 scripts/analyze_reach_kinematics.py
"""
import numpy as np


def reachability(link_lengths, target_dist):
    l = np.asarray(link_lengths, dtype=float)
    r_max = l.sum()
    r_min = max(0.0, 2 * l.max() - l.sum())
    reachable = r_min <= target_dist <= r_max
    slack = r_max - target_dist  # 余剰リーチ（IK的に不要な超過分）
    return {
        'link_lengths': l.tolist(),
        'total_length': float(l.sum()),
        'r_min': r_min,
        'r_max': r_max,
        'target_dist': target_dist,
        'reachable': reachable,
        'slack': slack,
        'slack_ratio': slack / target_dist if target_dist > 0 else float('nan'),
    }


def print_report(name, link_lengths, target_dist):
    r = reachability(link_lengths, target_dist)
    status = 'OK' if r['reachable'] else 'NG(到達不能)'
    print(f'--- {name} ---')
    print(f'  link_lengths = {[f"{x:.3f}" for x in r["link_lengths"]]}  (n={len(link_lengths)})')
    print(f'  total_length = {r["total_length"]:.3f} m')
    print(f'  target_dist  = {r["target_dist"]:.3f} m')
    print(f'  reachable range: [{r["r_min"]:.3f}, {r["r_max"]:.3f}]  -> {status}')
    print(f'  slack (総リーチ - 必要距離) = {r["slack"]:.3f} m  '
          f'({r["slack_ratio"]*100:.0f}% 過剰)')
    print()


if __name__ == '__main__':
    target = np.array([0.8, 0.0, 0.15])
    base = np.array([0.0, 0.0, 0.15])  # 台座位置（z は同一平面なので実質2D距離）
    target_dist = float(np.linalg.norm((target - base)[:2]))
    print(f'target = {target}, target_dist(2D) = {target_dist:.3f} m\n')

    # co-design が実際に収束させた形態（今回の実測値、compare_morphology.py より）
    print_report('rrbot_L1 (2関節)', [0.654, 0.251], target_dist)
    print_report('TA (tripo 3関節, 狭可動域)', [0.500, 0.723, 0.513], target_dist)
    print_report('tripo_v2c_reach (3関節, 広可動域)', [0.632, 0.561, 0.370], target_dist)

    # 幾何的に「必要十分」な最小構成の例（等分配・多少の余裕を見込む）
    print('=== 参考: 幾何的に必要十分な最小構成の例 ===')
    print_report('最小構成の例（2リンク等分・余裕なし）', [target_dist/2, target_dist/2], target_dist)
    print_report('最小構成の例（3リンク等分・余裕なし）', [target_dist/3]*3, target_dist)
