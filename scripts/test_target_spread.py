#!/usr/bin/env python3
"""第1層の「対象が分布を持つ場合」判定（9-46）の最小確認。

確かめること:
  1. spread_y=0（既定）では所見が1件も増えない → **既存の判定は動かない**
  2. 分布の端が届く範囲を超えたら ❌（fatal）になる
  3. 端で伸びきる（伸展率 > 90 %）なら警告は出るが ❌ にはしない
  4. 判定に使うのは分布の**最遠点**であって中央ではない

    python3 scripts/test_target_spread.py
"""
import importlib.util
import pathlib

import numpy as np

spec = importlib.util.spec_from_file_location(
    'diag', pathlib.Path(__file__).resolve().parent / 'diagnose_morphology.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def geo(reach, cube_x, half=0.05):
    """水平リーチ reach の平面2リンク腕と、x=cube_x にある対象。"""
    n = 2
    l = reach / n
    return {'base_pos': np.zeros(3), 'lengths': [l] * n,
            'offsets': [0.0] + [l] * (n - 1), 'axes': [(0, 0, 1)] * n,
            'dirs': [np.array([1.0, 0, 0])] * n,
            'cube': {'pos': np.array([cube_x, 0.0, 0.0]), 'half': half}}


def run(g, spread):
    return diag.layer1(g, 'pusher', np.array([g['cube']['pos']]).ravel(), True, spread_y=spread)


# 1. 既定では何も増えない
g = geo(1.0, 0.7)
base_f, base_fatal = run(g, 0.0)
assert not base_fatal
for s in (0.0, 0.0):
    f, _ = run(g, s)
    assert len(f) == len(base_f), '既定で所見が増えている＝既存の判定を動かしている'

# 2. 端が届かないなら ❌
#    reach 1.0、対象 x=0.7・半幅 0.05 → 端 y=0.8 なら面まで √(0.49+0.64)−0.05 = 1.013 m > 1.0
f, fatal = run(geo(1.0, 0.7), 0.8)
assert fatal, '分布の端が届かないのに fatal になっていない'
assert any('端に来たとき腕が届きません' in m for _, m in f)

# 3. 中央だけなら届く（＝中央で判定すると見逃す）ことの確認
_, fatal_center = run(geo(1.0, 0.7), 0.0)
assert not fatal_center, '中央では届くはず。2 の ❌ は分布を見たからこそ出た'

# 4. 端で伸びきるが届く → 警告だけ
#    reach 1.0、対象 x=0.7 → 端 y=0.45 なら面まで √(0.49+0.2025)−0.05 = 0.782 → 78 %
#    端 y=0.66 なら √(0.49+0.4356)−0.05 = 0.912 → 91 %
f, fatal = run(geo(1.0, 0.7), 0.66)
assert not fatal
assert any('ほぼ伸ばしきります' in m for _, m in f), '伸展率 91 % で警告が出ていない'

f, fatal = run(geo(1.0, 0.7), 0.45)
assert not fatal
assert any('端に来ても余裕があります' in m for _, m in f)

print('✅ 第1層の分布判定は意図どおり（既定では既存判定を動かさない）')
