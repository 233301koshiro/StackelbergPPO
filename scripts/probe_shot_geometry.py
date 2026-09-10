#!/usr/bin/env python3
"""パックの初期位置ごとに「反射が必須か」を幾何で測る（9-55 追記の再現・拡張）。

9-55 追記は撃ち出し角を振ってゴール口を抜けるかを数えたが、**スクリプトが残っていない**
（その場計算だった）。初期 x を変えると判定線が動くので、**再現できる形にした**。

物理エンジンは使わない。パックを質点+半幅として直進させ、側壁と板で反射させるだけ。
⚠️ **摩擦・減衰・接触の柔らかさは無視している。** 「その角度で入りうるか」の
上限を見るためのもので、学習が実際に到達できるかとは別。

    python3 scripts/probe_shot_geometry.py                # 既定 x=0.70（記録の再現）
    python3 scripts/probe_shot_geometry.py --x 0.55
"""
import argparse
import pathlib
import numpy as np

# ⚠️ **定数は make_hockey_court_xml.py から取り込む。ここに写さない。**
#   2026-09-10 まで BOARD_HT=0.03 と写し取っており、Bug 39 で板を 0.10 に
#   厚くしたときに**このファイルだけ古いまま**になった（CLAUDE.md §4-2 の
#   「2 箇所に同じことを書く構造」そのもの）。
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    '_court', pathlib.Path(__file__).with_name('make_hockey_court_xml.py'))
_court = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_court)

PUCK_HALF = _court.PUCK_HALF
COURT_X = _court.COURT_X
WALL_IN_Y = _court.WALL_FACE_Y                     # 側壁の内面（Bug 39 で 0.45 → 0.50）
BOARD_X = _court.COURT_X[1] - 0.60
BOARD_HY = _court.GOAL_HALF * 2 * 0.70 / 2
BOARD_HT = _court.BOARD_T                          # Bug 39 で 0.03 → 0.10
GOAL_X, GOAL_HY = _court.COURT_X[1], _court.GOAL_HALF


def shoot(x0, y0, deg, step=0.002, max_len=12.0):
    """撃ち出して (入ったか, 反射したか) を返す。"""
    p = np.array([x0, y0], float)
    v = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
    reflected = False
    for _ in range(int(max_len / step)):
        p = p + v * step
        # 側壁（パック中心が内面 − 半幅 で反射）
        lim = WALL_IN_Y - PUCK_HALF
        if abs(p[1]) >= lim:
            p[1] = np.sign(p[1]) * lim
            v[1] = -v[1]
            reflected = True
        # 板（x 方向の面で反射。中央を塞ぐ）
        if abs(p[0] - BOARD_X) <= BOARD_HT + PUCK_HALF and abs(p[1]) <= BOARD_HY + PUCK_HALF:
            v[0] = -v[0]
            p[0] += v[0] * step
            reflected = True
        # ゴールライン通過
        if p[0] >= GOAL_X:
            return (abs(p[1]) <= GOAL_HY - PUCK_HALF), reflected
        if p[0] <= COURT_X[0] + PUCK_HALF:      # 手前の端で終了
            return False, reflected
    return False, reflected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--x', type=float, default=0.70, help='パックの初期 x')
    ap.add_argument('--ys', type=float, nargs='+',
                    default=[0.40, 0.30, 0.20, 0.10, 0.00])
    ap.add_argument('--step-deg', type=float, default=4.0)
    args = ap.parse_args()

    print(f'  パック初期 x = {args.x}   （板 x={BOARD_X} / ゴール x={GOAL_X}）')
    print(f'  {"初期 y":>8} {"直線で入る":>10} {"反射で入る":>10}   判定')
    for y in args.ys:
        direct = refl = 0
        for deg in np.arange(-60, 60.001, args.step_deg):
            ok, r = shoot(args.x, y, deg)
            if ok:
                refl += 1 if r else 0
                direct += 0 if r else 1
        v = '⭐ 反射が必須' if direct == 0 and refl > 0 else \
            ('直線で入る' if direct > 0 else '❌ どちらでも入らない')
        print(f'  {y:8.2f} {direct:10d} {refl:10d}   {v}')


if __name__ == '__main__':
    main()
