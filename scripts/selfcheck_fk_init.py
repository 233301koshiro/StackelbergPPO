#!/usr/bin/env python3
"""`_safe_init_angle` の FK を、**シミュレータ無しで**既知の形状に当てて検算する。

⛔⛔ なぜ要るか（系譜 9-196）: 旧実装は腕を「原点から伸びる長さ R の直線の棒」として扱い、
`tip_y = R·sin(θ)` で初期角を決めていた。**その形状は存在しなかった。**
⛔ `qpos[0]` はヨー（z 軸回り）なのに、式はピッチを記述していた。
⛔ 結果 `hockey_bank5` は先端が 1201 step 中 **0 step** しかリンク内に入らなかった。

⭐ この検査は **FK が実際の XML から正しい先端位置を出すか**を見る。
⭐ 実測との突き合わせ: `single_run/hockey_bank5/trace/arm_trace.npz` の t=0 と比べる。

    python3 scripts/selfcheck_fk_init.py
"""
import sys
import numpy as np

sys.path.insert(0, '.')


class _FK:
    """pusher.PusherEnv から FK の 3 メソッドだけを借りる薄い殻。

    ⭐ env を作らずに検算できるようにしてある（Choreonoid の起動 105 秒を避ける）。
    """
    from design_opt.envs.pusher import PusherEnv as _P
    _parse_arm_chain = _P._parse_arm_chain
    _fk_points = _P._fk_points
    # ⚠️ staticmethod はクラス経由で取ると素の関数になるので、包み直さないと
    #   self が第1引数に入ってしまう
    _rot = staticmethod(_P._rot)
    _seg_point_dist = staticmethod(_P._seg_point_dist)


def main() -> int:
    fk = _FK()
    xml = open('assets/mujoco_envs/e2e_hockey_easy.xml', encoding='utf-8').read()
    chain = fk._parse_arm_chain(xml)
    bad = 0

    print(f'⭐ 連鎖 {len(chain)} 本: {[c["name"] for c in chain]}')
    for c in chain:
        print(f'   {c["name"]:5s} offset={np.round(c["offset"],4)} '
              f'axis={np.round(c["axis"],1)} bone={np.round(c["bone"],4)}')

    # ── ① 全角 0 なら腕は真上を向く（全 bone が局所 +z、全 offset も +z）──
    segs = fk._fk_points(chain, np.zeros(len(chain)))
    tip = segs[-1][1]
    exp_z = 0.020 + 0.1925 + 0.2951 + 0.3310 + 0.2239     # 台座 + 全リンク
    print(f'\n① 全角 0 → 先端 {np.round(tip,4)}   期待 (0, 0, {exp_z:.4f})')
    if abs(tip[0]) > 1e-9 or abs(tip[1]) > 1e-9 or abs(tip[2] - exp_z) > 1e-9:
        print('   ⛔ 不一致'); bad += 1
    else:
        print('   ✅ 一致')

    # ── ② ヨーだけ回しても真上の腕は動かない（⛔ 旧実装が見落とした点）──
    segs = fk._fk_points(chain, [np.pi / 2, 0, 0, 0])
    tip2 = segs[-1][1]
    print(f'\n② ヨー 90° のみ → 先端 {np.round(tip2,4)}')
    print('   ⭐ 旧実装の式 tip_y = R·sin(90°) なら y = 1.0425 になるはずだった')
    if np.linalg.norm(tip2 - tip) > 1e-9:
        print('   ⛔ ヨーで先端が動いた（真上の腕では動かないはず）'); bad += 1
    else:
        print('   ✅ 動かない。**ヨーにピッチの式を当てていたのが旧実装の誤り**')

    # ── ③ 第2関節を 90° 倒すと腕は水平（+x）へ ──
    segs = fk._fk_points(chain, [0, np.pi / 2, 0, 0])
    tip3 = segs[-1][1]
    exp_x = 0.2951 + 0.3310 + 0.2239
    exp_z3 = 0.020 + 0.1925
    print(f'\n③ 第2関節 90° → 先端 {np.round(tip3,4)}   期待 ({exp_x:.4f}, 0, {exp_z3:.4f})')
    if abs(tip3[0] - exp_x) > 1e-9 or abs(tip3[2] - exp_z3) > 1e-9:
        print('   ⛔ 不一致'); bad += 1
    else:
        print('   ✅ 一致')

    # ── ④ ③ の姿勢をヨー 90° すると +y へ回る ──
    segs = fk._fk_points(chain, [np.pi / 2, np.pi / 2, 0, 0])
    tip4 = segs[-1][1]
    print(f'\n④ ヨー90°+第2関節90° → 先端 {np.round(tip4,4)}   期待 (0, {exp_x:.4f}, {exp_z3:.4f})')
    if abs(tip4[1] - exp_x) > 1e-9 or abs(tip4[0]) > 1e-9:
        print('   ⛔ 不一致'); bad += 1
    else:
        print('   ✅ 一致')

    # ── ⑤ 線分と点の距離 ──
    d = fk._seg_point_dist(np.zeros(3), np.array([1., 0, 0]), np.array([0.5, 0.3, 0.]))
    print(f'\n⑤ 線分[(0,0,0),(1,0,0)] と 点(0.5,0.3,0) の距離 = {d:.4f}  期待 0.3000')
    if abs(d - 0.3) > 1e-9:
        print('   ⛔ 不一致'); bad += 1
    else:
        print('   ✅ 一致')

    print('\n' + ('⛔ %d 件 失敗' % bad if bad else '✅ FK の 5 項目すべて一致'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
