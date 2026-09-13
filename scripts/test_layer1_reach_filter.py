#!/usr/bin/env python3
"""第1層の到達フィルタ A＋B（系譜 9-72・9-91 の仕様）の最小確認。

確かめること:
  1. B（可動域込みの到達判定、凍結モード）が、可動域を狭めると正しく棄却に転じる
  2. **棄却の側だけが確実**: 格子の余地を超えたときだけ「届かない」と言う
  3. B の設計モード版 `zonotope_chain_reach`（9-91）が同じ非対称の保証を満たす。
     箱制約つき最小二乗（角ではなく辺・面上の最短点）で厳密に距離を求めており、
     角までの距離だけを使う不健全な近似（過大評価）にはなっていない
  4. 凍結モードの判定は 9-91 の変更で一切変わらない（回帰確認）
  5. A の探索幅が `design_opt/cfg/*.yml` から読まれ、直書きの既定に依存しない

    python3 scripts/test_layer1_reach_filter.py
"""
import copy
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

# --- 3a. zonotope_chain_reach（B の設計モード版、9-91）単体 ---
zf = diag.zonotope_chain_reach

# 元の長さのまま届く点は残距離ほぼ0（dz が箱の中心なので厳密に届く）
ok, reach, dmin, slack = zf([1.0], [(-180, 180)], 1.0, 0.0, offset_half=0.5)
assert ok and reach and dmin < 1e-6, 'dz=元の長さそのままで届く点は残距離0のはず'

# 遠すぎる点は棄却される（最大でも hypot(1.5,0.5)=1.58 までしか届かない）
ok, reach, dmin, slack = zf([1.0], [(-180, 180)], 3.0, 0.0, offset_half=0.5)
assert ok and not reach, '腕を最大まで伸ばしても届かない点は棄却されるはず'

# ⚠️ 健全性の核心: 角までの距離だけを使うと過大評価する（不健全）。
# n=1・関節角を0に固定すると、dz∈[l-h,l+h]・dx∈[-h,h] の矩形（回転なし）になる。
# target を辺の外側・角の外側に置くと、真の最短点は**辺の中点**であって角ではない。
l, h = 0.3, 0.5
ok, reach, dmin, slack = zf([l], [(0.0, 0.0)], 0.6, 0.0, offset_half=h)
true_edge_dist = 0.6 - h  # dx=h に張り付き、dz=0（箱の内側）で正確に一致
corners = [(dx, dz) for dx in (-h, h) for dz in (l - h, l + h)]
naive_corner_dist = min(np.hypot(dx - 0.6, dz - 0.0) for dx, dz in corners)
assert abs(dmin - true_edge_dist) < 1e-6, (
    f'真の最短点（辺の中点）は {true_edge_dist:.4f} のはずが {dmin:.4f}')
assert naive_corner_dist > dmin + 0.05, (
    '角までの距離だけを使う近似は過大評価するはず（不健全さの実証）。'
    f'角: {naive_corner_dist:.4f} 対 真の最短距離: {dmin:.4f}')

# n=0 / n>4 は判定しない（棄却しない）
ok, _, _, _ = zf([], [], 1.0, 0.0, offset_half=0.5)
assert not ok, 'リンクが無ければ判定しない'
ok, _, _, _ = zf([0.3] * 5, [(-90, 90)] * 5, 1.0, 0.0, offset_half=0.5)
assert not ok, '5関節（n>4）は想定外として判定しない'

# --- 3b. B（凍結・設計とも）は同じ関数群を通り、棄却は格子の余地を超えたときだけ ---
geo = diag.parse_arm_xml(os.path.join(diag.ASSET_DIR, 'e2e_a1v_fix4.xml'))
tgt = np.array([0.8, 0.0, 0.15])


def has_ik_reject(frozen, g=geo, offset_half=diag.OFFSET_HALF):
    fs, _ = diag.layer1(g, 'reach', tgt, length_frozen=frozen, offset_half=offset_half)
    return any('可動域では目標に届きません' in m for _, m in fs)


assert not has_ik_reject(False), (
    '設計モードで B が棄却してはいけない。長さを縮めて畳める自由もあるので、'
    'ゾノトープで正しく計算すれば 0.8 m 先の目標は届く（9-91 で実測）')

# 設計モードでも「本当に無理な形」は棄却できることを確認する（可動域を極端に絞る）。
# root（par、ヨー）を絞っても B の計算には効かない（perp のみが対象）ため、
# perp 側の関節を絞る必要がある。
geo_tight = copy.deepcopy(geo)
geo_tight['ranges'] = [(-3.0, 3.0)] * len(geo_tight['ranges'])
assert has_ik_reject(False, g=geo_tight, offset_half=0.02), (
    '可動域を±3°まで絞り、探索幅も0.02mまで絞れば、設計モードでも'
    '「長さをどう選んでも届かない」と棄却できるはず')
# 同じ絞り方は凍結モードでも当然棄却する（対照）
assert has_ik_reject(True, g=geo_tight, offset_half=0.02)

# --- 4. 凍結モードの判定は 9-91 で一切変わらない（回帰の錨） ---
# 9-91 で 84 件（assets/mujoco_envs 全形態 × reach/pusher）を旧実装と突き合わせて
# 差分 0 を確認済み（実験系譜.md 9-91）。ここでは代表 2 件を恒久的な錨として残す。
# `layer1` の `if length_frozen:` 分岐は 9-91 で一切触っていないので、
# 崩れるとすれば呼び出し側（design_opt/cfg 追加など）の変化であり、それを拾うのが目的。
_b1v = diag.parse_arm_xml(os.path.join(diag.ASSET_DIR, 'e2e_b1v.xml'))
_, _b1v_fatal = diag.layer1(_b1v, 'reach', np.array([0.8, 0.0, 0.15]), length_frozen=True)
assert _b1v_fatal is True, 'e2e_b1v の凍結モードは総合判定で棄却されるはず（9-73: 0.680m）'
_, _a1v_fatal = diag.layer1(geo, 'reach', tgt, length_frozen=True)
assert _a1v_fatal is False, 'e2e_a1v_fix4 の凍結モードは棄却されないはず（9-73: 1.006m ok）'

# --- 5. 探索幅は cfg から来る（直書きに依存しない） ---
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

print('✅ 第1層の到達フィルタ A＋B（凍結・設計とも）は意図どおり（棄却は余地超過時のみ・角ではなく箱制約LSQで厳密）')
