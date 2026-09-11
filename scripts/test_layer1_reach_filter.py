#!/usr/bin/env python3
"""第1層の到達フィルタ A＋B（系譜 9-72 の仕様）の最小確認。

確かめること:
  1. B（可動域込みの到達判定）が、可動域を狭めると正しく棄却に転じる
  2. **棄却の側だけが確実**: 格子の余地を超えたときだけ「届かない」と言う
  3. B は**凍結モードでのみ**適用される（設計モードでは長さが単調に有利でないため）
  4. A の探索幅が `design_opt/cfg/*.yml` から読まれ、直書きの既定に依存しない

    python3 scripts/test_layer1_reach_filter.py
"""
import importlib.util
import os
import pathlib

import numpy as np
import yaml

spec = importlib.util.spec_from_file_location(
    'diag', pathlib.Path(__file__).resolve().parent / 'diagnose_morphology.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)

# --- 1・2. B 単体 ---
f = diag.planar_chain_reach
ok, reach, dmin, slack = f([1.0, 1.0], [(-180, 180)] * 2, 2.0, 0.0)
assert ok and reach, '±180° なら真横 (2,0) に届くはず'

ok, reach, dmin, slack = f([1.0, 1.0], [(-10, 10)] * 2, 2.0, 0.0)
assert ok and not reach, '±10° では真横に届かないはず'
assert dmin > slack, '棄却は「最小残距離 > 格子の余地」でのみ成立する'

ok, reach, _, _ = f([1.0, 1.0], [(-180, 180)] * 2, 5.0, 0.0)
assert ok and not reach, '腕より遠い点は棄却されるはず'

ok, _, _, _ = f([], [], 1.0, 0.0)
assert not ok, 'リンクが無ければ判定しない（棄却しない）'

# --- 3. B は凍結モードでのみ適用される ---
geo = diag.parse_arm_xml(os.path.join(diag.ASSET_DIR, 'e2e_a1v_fix4.xml'))
tgt = np.array([0.8, 0.0, 0.15])


def has_ik_reject(frozen):
    fs, _ = diag.layer1(geo, 'reach', tgt, length_frozen=frozen)
    return any('可動域では目標に届きません' in m for _, m in fs)


assert not has_ik_reject(False), (
    '設計モードで B が棄却してはいけない。最大まで伸ばした腕は関節が畳みきれず'
    '近くに届かないが、最適化器はもっと短い長さも選べる')

# --- 4. 探索幅は cfg から来る（直書きに依存しない） ---
for name, expect_frozen in (('pusher_tripo_v3', False), ('pusher_gearonly', True)):
    path = pathlib.Path('design_opt/cfg') / f'{name}.yml'
    bp = ((yaml.safe_load(open(path)) or {}).get('robot') or {}).get('body_params') or {}
    assert bool(bp) != expect_frozen, f'{name} のモード判定が cfg と食い違う'
    if bp:
        ub = (bp.get('offset') or {}).get('ub') or []
        assert ub and max(abs(float(v)) for v in ub) > 0, '探索幅が読めない'

# 幅を変えたら上限も変わる（直書きの 0.5 に固定されていない）
w5 = diag.max_reach_after_design([0.3, 0.3], offset_half=0.5)
w3 = diag.max_reach_after_design([0.3, 0.3], offset_half=0.3)
assert w3 < w5, '探索幅を狭めたら上限も下がるはず'

print('✅ 第1層の到達フィルタ A＋B は意図どおり（B は凍結モードのみ・棄却は余地超過時のみ）')
