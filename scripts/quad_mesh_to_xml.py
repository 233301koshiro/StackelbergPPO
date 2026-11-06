#!/usr/bin/env python3
"""**四脚の立体を分けた骨格（`segment_quad_mesh.py` の `seg_summary.json`）から、MuJoCo の XML を作る**（系譜 9-342 の追記、2026-11-06）。

なぜ: ユーザー「これをxmlに出来るかやってみよう」（興味本位）。関節の位置（マゼンタの球の中心）・足先・リンクの太さ・胴の外形は
`segment_quad_mesh.py` が立体から取った。それを、この repo の env（`design_opt/envs/ant.py`）と Choreonoid の変換器が読める形にする。

形の作り方:
  - 座標: Tripo（高さ y、体の長さ z）→ MuJoCo（高さ z、前 x）。(x, y, z)_Tripo → (z, x, y)_MuJoCo（巡回の入れ替えなので右手系のまま）。
    足先の最も低い点が床から CLEAR の高さに来るよう上下にずらす。長さの単位は Tripo の正規化した単位をそのまま m とする
  - 胴（`0`）: 箱（胴のボクセルの 1〜99 % 点の外形）。根元は free joint。
    ⚠️ 胴を球にしない: xml_robot.py は根元の体の球・カプセルを骨の上に置き直す（9-342 の最初の版で胴のカプセルの端が NaN）。箱は置き直さない
  - 脚ごとに 3 つの体:
    ① 取り付け（`1`〜`4`、関節なし）: 胴の中心 → 股の細いカプセル。
       ⚠️ xml_robot.py は子の体を親の骨の終わり（子の始点の平均）へ置き直すので、胴から直接 4 本の脚を出すと付け根が全部胴の中心へ寄る（9-342）。
       関節の無い体を 1 つ挟み、その骨を「胴の中心 → 股」にして、股の位置を保つ
    ② 腿（`11`〜`14`）: 股の関節（ピッチ、軸は横）。股 → 膝のカプセル（半径は立体の断面積から）
    ③ 脛（`111`〜`114`）: 膝の関節（ピッチ、軸は横）。膝 → 足先のカプセル（足先の面が立体の足先に来るよう、半径の分だけ手前で止める）
  - 物理の値（density 5・gear 150・armature 1・damping 1・可動域 ±30°）は quad_4x2.xml（ant の既定）と同じ。実在の部品から導いていない

    python3 scripts/quad_mesh_to_xml.py [--seg data/api/quad/seg/seg_summary.json] [--out assets/mujoco_envs/quad_tripo.xml]
        [--image data/api/quad/quad_a1_0.png]   # XML の注記に書く参照画（11-06 夜の quad_walk は data/api/quad/quad_walk_0.png）
"""
import argparse
import json

import numpy as np

CLEAR = 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seg', default='data/api/quad/seg/seg_summary.json')
    ap.add_argument('--out', default='assets/mujoco_envs/quad_tripo.xml')
    ap.add_argument('--image', default='data/api/quad/quad_a1_0.png')
    a = ap.parse_args()
    sk = json.load(open(a.seg))['skeleton']
    T = lambda p: np.array([p[2], p[0], p[1]], float)          # Tripo → MuJoCo
    lo, hi = T(sk['torso_min']), T(sk['torso_max'])
    legs = [dict(L, hip=T(L['hip']), knee=T(L['knee']), foot=T(L['foot'])) for L in sk['legs']]
    zmin = min(L['foot'][2] for L in legs)
    dz = CLEAR - zmin
    c = (lo + hi) / 2 + [0, 0, dz]
    half = (hi - lo) / 2
    f = lambda v: ' '.join(f'{x:.4f}'.rstrip('0').rstrip('.') if abs(x) > 5e-5 else '0' for x in v)
    body, act = [], []
    for k, L in enumerate(legs, 1):
        h, kn, ft = (L[n] + [0, 0, dz] for n in ('hip', 'knee', 'foot'))
        u = (ft - kn) / np.linalg.norm(ft - kn)
        ft_end = ft - u * L['shin_r']
        m, t, s = f'{k}', f'1{k}', f'11{k}'
        body.append(f'''      <body name="{m}" pos="0 0 0">
        <geom fromto="0 0 0 {f(h - c)}" size="0.03" type="capsule" />
        <body name="{t}" pos="{f(h - c)}">
          <joint axis="0 1 0" name="{t}_joint" pos="0 0 0" range="-30 30" type="hinge" />
          <geom fromto="0 0 0 {f(kn - h)}" size="{L['thigh_r']:.4f}" type="capsule" />
          <body name="{s}" pos="{f(kn - h)}">
            <joint axis="0 1 0" name="{s}_joint" pos="0 0 0" range="-30 30" type="hinge" />
            <geom fromto="0 0 0 {f(ft_end - kn)}" size="{L['shin_r']:.4f}" type="capsule" />
          </body>
        </body>
      </body>  <!-- {L['name']}の脚 -->''')
        act += [f'    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="{j}_joint" gear="150" name="{j}_joint" />' for j in (t, s)]
    model = a.out.rsplit('/', 1)[-1].removesuffix('.xml')
    xml = f'''<mujoco model="{model}">
  <!-- scripts/quad_mesh_to_xml.py が {a.seg} から作った（系譜 9-342 の追記、2026-11-06）。手で直さない。
       Gemini の参照画（{a.image}）→ Tripo の立体 → マゼンタの球で関節・断面積で太さ → この XML。
       ⚠️ 物理の値は ant の既定（density 5・gear 150）のままで、実在の部品から導いていない。 -->
  <compiler angle="degree" />
  <option integrator="RK4" timestep="0.01" />
  <default>
    <joint armature="1" damping="1" limited="true" />
    <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba=".7 .5 .3 1" />
  </default>
  <worldbody>
    <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="1 1 1 1" size="200 200 .125" type="plane" />
    <body name="0" pos="{f(c)}">
      <geom size="{f(half)}" type="box" />
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free" />
{chr(10).join(body)}
    </body>
  </worldbody>
  <actuator>
{chr(10).join(act)}
  </actuator>
</mujoco>
'''
    open(a.out, 'w').write(xml)
    print(f'→ {a.out}  胴の中心 {np.round(c, 3)}・胴の半分の外形 {np.round(half, 3)}・床から足先 {CLEAR} m')
    for L in legs:
        h, kn, ft = (L[n] + [0, 0, dz] for n in ('hip', 'knee', 'foot'))
        print(f"  {L['name']}: 股 {np.round(h, 3)}・膝 {np.round(kn, 3)}・足先 {np.round(ft, 3)}・腿 {np.linalg.norm(kn - h):.3f}（半径 {L['thigh_r']:.3f}）"
              f"・脛 {np.linalg.norm(ft - kn):.3f}（半径 {L['shin_r']:.3f}）")


if __name__ == '__main__':
    main()
