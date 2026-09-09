#!/usr/bin/env python3
"""ホッケー系の XML を作る（実験系譜 9-38 の B+C+D）。

`e2e_hockeyv.xml`（Pusher 系と同じ条件）から派生させ、3 点だけ変える。

  B  パックを小さく軽くする   0.30 m 角 2.70 kg → 0.10 m 角 0.03 kg（実物のパック相当）
  C  コートで囲う             関節可動域で拘束。壁 geom は足さない（下記）
  D  パックを手前へ           x = 1.00 → 0.70（ロボットの水平到達 0.848 m の内側）

⚠️ **この XML は既存の Pusher・Target-Pusher・Reach と比較できない。**
物理が変わっているので、比較はホッケー系の内部（パック固定 ↔ 左右に振る）でのみ行う。

**なぜ壁 geom ではなく関節可動域か**: パックは slide(x)/slide(y) の 2 自由度しか持たず、
床から浮いた平面を滑る（9-38）。壁 geom を足すと接触ペアが増えるだけで、
**可動域で止める方が同じ効果を確実に得られる**。
⚠️ ただし可動域は**跳ね返りが弱い**（ソルバの拘束なので反発係数を指定できない）。
ラリーを狙うなら壁 geom + restitution が要る。**いまは吹っ飛びを止めるのが目的。**

⚠️ **body 名 `cube` と関節の並び（slide x → slide y）は変えない。**
`pusher.py` が `get_body_com("cube")` と `model.nq - 2` で参照しているため。

    python3 scripts/make_hockey_court_xml.py
"""
import pathlib
import xml.etree.ElementTree as ET

SRC = pathlib.Path('assets/mujoco_envs/e2e_hockeyv.xml')
DST = pathlib.Path('assets/mujoco_envs/e2e_hockey_court.xml')
DST_WALL = pathlib.Path('assets/mujoco_envs/e2e_hockey_wall.xml')  # --wall（9-55）
# 表示用の .body も**同じパラメータから生成する**（9-58）。手で書くとズレる:
# 実際に 9-37 で手書きした版は 側壁 ±0.50（実際は ±0.45）・ゴール口 0.4 m（実際は 0.3 m）で、
# しかも **goal_board が描かれていなかった**。⚠️ **これは XML の派生物であり、正は XML。**
DST_BODY = pathlib.Path('assets/choreonoid/bodies/hockey/hockey_table_view.body')
DST_PUCK = pathlib.Path('assets/choreonoid/bodies/hockey/hockey_puck_view.body')

# コート（表示用 body と揃える）。ロボットは原点、正面が +x。
COURT_X = (0.15, 1.55)      # 奥行き 1.4 m
COURT_Y = (-0.50, 0.50)     # 幅 0.9 m … の外枠
PUCK_HALF = 0.05            # 一辺 0.10 m
PUCK_DENSITY = 30.0         # 0.10^3 * 30 = 0.03 kg（実物のパック相当）
PUCK_POS = (0.70, 0.0, 0.2125)   # z は肩の高さ（9-42）。**XML を直接編集したら必ずここも直す**
PUCK_DAMPING = 0.02              # τ=m/b を旧条件(1.35s)に保つ値（9-45）。質量 90 分の1 に対応
GOAL_HALF = 0.15                 # ゴール口の半幅（表示用 hockey_table_view.body のマーカーと一致）


def _box(parent, name, pos, size, rgba):
    """静的な板（worldbody 直下の geom）。

    ⚠️ contype=1 / conaffinity=2 にしてある。腕の geom は contype=1 / conaffinity=0 なので
    (1&2)=0・(1&0)=0 で**腕とは当たらない**。パックは conaffinity=1 なので
    (wall.contype 1 & cube.conaffinity 1)=1 で**パックとだけ当たる**。
    腕が壁に引っかかって学習が壊れるのを防ぐためのモデル化上の割り切り。
    """
    ET.SubElement(parent, 'geom', {
        'name': name, 'type': 'box',
        'pos': ' '.join(f'{v:.4f}' for v in pos),
        'size': ' '.join(f'{v:.4f}' for v in size),
        'rgba': rgba, 'contype': '1', 'conaffinity': '2',
        'friction': '0.5 0.1 0.1'})


def _add_walls(root, js):
    """側壁・ゴール端（口を開ける）・ゴール前の板を足し、x 可動域を口の先まで延ばす。

    9-38 は「壁 geom ではなく関節可動域で囲う」を選んだが、その docstring 自身が
    **「ラリーを狙うなら壁 geom が要る」**と例外を書いていた。反射を見たいのでここでは壁を置く。
    """
    wb = root.find('worldbody')
    # 半厚 3 cm。⚠️ **薄くしないこと。** パックが軽く(0.03 kg)速いため接触が柔らかく、
    # 実測で内面から約 3 cm めり込む。timestep=0.01 では solref の時定数を 0.02 未満に
    # できない（MuJoCo の推奨下限が 2×dt）ので、**厚みですり抜けを防ぐ**。
    # timestep を下げると既存 110 run と物理が変わるので触らない。
    t, zc, zh = 0.03, PUCK_POS[2], 0.06
    y_in = COURT_Y[1] - PUCK_HALF                 # 側壁の内面 = パック中心の可動域と同じ 0.45
    x_goal = COURT_X[1]                           # ゴールライン
    xc, xh = (COURT_X[0] + x_goal) / 2, (x_goal - COURT_X[0]) / 2
    for sgn in (+1, -1):
        _box(wb, f'wall_side_{"p" if sgn > 0 else "m"}',
             (xc, sgn * (y_in + t), zc), (xh, t, zh), '0.35 0.35 0.40 1')
        # ゴール端: 中央 GOAL_HALF*2 を開けて左右だけ塞ぐ
        y0, y1 = GOAL_HALF, y_in + 2 * t
        _box(wb, f'wall_goal_{"p" if sgn > 0 else "m"}',
             (x_goal + t, sgn * (y0 + y1) / 2, zc), (t, (y1 - y0) / 2, zh), '0.35 0.35 0.40 1')
    # ゴール前方 60 cm の板（ユーザー指定、長さはゴール口の 70 %）
    _box(wb, 'goal_board', (x_goal - 0.60, 0.0, zc),
         (t, GOAL_HALF * 2 * 0.70 / 2, zh), '0.20 0.40 0.90 1')
    # ゴール口を抜けたパックが前へ進めるよう x の可動域を延ばす。
    # 外したパックはゴール端の壁で中心 x=1.49 で止まるので、**通した方が報酬が大きくなる**。
    # 報酬（v_x の積算＝総移動距離）を変えずに「入れる」を得にする — 5.2.1 の罠を避ける形。
    js[0].set('range', f'{js[0].get("range").split()[0]} '
                       f'{x_goal + 0.25 - PUCK_POS[0]:.4f}')


def _write_body(js):
    """Choreonoid で見るための .body を XML と同じ寸法から書き出す（**表示専用**）。

    ⚠️ **物理には一切入らない。** 学習が使うのは `e2e_hockey_wall.xml` だけである。
    ⚠️ **このファイルは生成物。手で編集しない**（次の再生成で消える）。
    """
    t, zc, zh = 0.03, PUCK_POS[2], 0.06
    y_in = COURT_Y[1] - PUCK_HALF
    x_goal = COURT_X[1]
    xc, xh = (COURT_X[0] + x_goal) / 2, (x_goal - COURT_X[0]) / 2
    board_h = GOAL_HALF * 2 * 0.70 / 2
    surf = zc - PUCK_HALF                                   # 盤面 = パック底面

    def shape(name, pos, size, rgb):
        return (f"      -\n"
                f"        type: Shape\n"
                f"        translation: [ {pos[0]-xc:.4f}, {pos[1]:.4f}, {pos[2]-surf:.4f} ]\n"
                f"        geometry: {{ type: Box, size: [ {size[0]*2:.4f}, {size[1]*2:.4f}, {size[2]*2:.4f} ] }}\n"
                f"        appearance: {{ material: {{ diffuse: [ {rgb} ] }} }}   # {name}\n")

    els = [shape('surface', (xc, 0.0, surf - 0.005), (xh, y_in + 2*t, 0.005), '0.85, 0.90, 0.95')]
    for sgn in (+1, -1):
        sfx = 'p' if sgn > 0 else 'm'
        els.append(shape(f'wall_side_{sfx}', (xc, sgn*(y_in+t), zc), (xh, t, zh), '0.35, 0.35, 0.40'))
        y0, y1 = GOAL_HALF, y_in + 2*t
        els.append(shape(f'wall_goal_{sfx}', (x_goal+t, sgn*(y0+y1)/2, zc), (t, (y1-y0)/2, zh),
                         '0.35, 0.35, 0.40'))
    els.append(shape('goal_board', (x_goal-0.60, 0.0, zc), (t, board_h, zh), '0.20, 0.40, 0.90'))
    els.append(shape('goal_mouth', (x_goal+t, 0.0, surf), (t, GOAL_HALF, 0.003), '0.90, 0.20, 0.20'))

    DST_BODY.parent.mkdir(parents=True, exist_ok=True)
    DST_BODY.write_text(
        "format: ChoreonoidBody\n"
        "format_version: 2.0\n"
        "angle_unit: degree\n"
        "name: hockey_table_view\n"
        "\n"
        "# ⚠️ **自動生成物。手で編集しないこと**（`make_hockey_court_xml.py --wall` が上書きする）。\n"
        "# ⚠️ **表示専用で、物理には一切入らない。** 学習が使うのは\n"
        "#    `assets/mujoco_envs/e2e_hockey_wall.xml` であり、**そちらが正**。\n"
        "# 寸法は XML と同じ定数から出しているのでズレない（9-58）。\n"
        "#\n"
        "#   choreonoid data/test/hockey/e2e_hockeyv.urdf \\\n"
        "#     assets/choreonoid/bodies/hockey/hockey_table_view.body\n"
        "\n"
        "root_link: table_base\n"
        "\n"
        "links:\n"
        "  -\n"
        "    name: table_base\n"
        "    joint_type: fixed\n"
        f"    translation: [ {xc:.4f}, 0, {surf:.4f} ]\n"
        "    mass: 50.0\n"
        "    center_of_mass: [ 0, 0, 0 ]\n"
        "    inertia: [ 10, 0, 0, 0, 10, 0, 0, 0, 10 ]\n"
        "    elements:\n" + ''.join(els), encoding='utf-8')
    print(f'{DST_BODY}  （表示専用・自動生成）')

    # パックも同じ定数から。⚠️ 手書き版は高さ 0.116（実際 0.2125）・直径 0.08（実際 0.10）
    # と二重にずれていた。**物理は「箱」なので、見た目も箱にする**（円盤にすると当たりが嘘になる）。
    DST_PUCK.write_text(
        "format: ChoreonoidBody\n"
        "format_version: 2.0\n"
        "angle_unit: degree\n"
        "name: hockey_puck_view\n"
        "\n"
        "# ⚠️ **自動生成物。手で編集しないこと**（`make_hockey_court_xml.py --wall` が上書きする）。\n"
        "# ⚠️ **表示専用。** 物理側の実体は `e2e_hockey_wall.xml` の body `cube`。\n"
        "# ⚠️ **形は箱である。** 実物のパックは円盤だが、物理が箱なので見た目も箱にする\n"
        "#    （円盤で描くと当たる瞬間が嘘になる。9-37 で一度その案を出して撤回した）。\n"
        "\n"
        "root_link: puck\n"
        "\n"
        "links:\n"
        "  -\n"
        "    name: puck\n"
        "    joint_type: free\n"
        f"    translation: [ {PUCK_POS[0]:.4f}, {PUCK_POS[1]:.4f}, {PUCK_POS[2]:.4f} ]\n"
        f"    mass: {(2*PUCK_HALF)**3 * PUCK_DENSITY:.4f}\n"
        "    center_of_mass: [ 0, 0, 0 ]\n"
        "    inertia: [ 5.0e-05, 0, 0, 0, 5.0e-05, 0, 0, 0, 5.0e-05 ]\n"
        "    elements:\n"
        "      -\n"
        "        type: Shape\n"
        f"        geometry: {{ type: Box, size: [ {2*PUCK_HALF:.4f}, {2*PUCK_HALF:.4f}, {2*PUCK_HALF:.4f} ] }}\n"
        "        appearance: { material: { diffuse: [ 0.85, 0.15, 0.15 ] } }\n", encoding='utf-8')
    print(f'{DST_PUCK}  （表示専用・自動生成）')


def main(wall_mode=False):
    root = ET.parse(SRC).getroot()
    cube = root.find(".//body[@name='cube']")
    if cube is None:
        raise SystemExit('cube body が見つからない')

    cube.set('pos', f'{PUCK_POS[0]:.6f} {PUCK_POS[1]:.6f} {PUCK_POS[2]:.6f}')

    # C: 可動域でコートに閉じ込める（パック中心が入れる範囲）
    xlo = COURT_X[0] + PUCK_HALF - PUCK_POS[0]
    xhi = COURT_X[1] - PUCK_HALF - PUCK_POS[0]
    ylo = COURT_Y[0] + PUCK_HALF
    yhi = COURT_Y[1] - PUCK_HALF
    js = cube.findall('joint')
    assert len(js) == 2 and js[0].get('axis').startswith('1'), '関節の並びが想定と違う'
    js[0].set('range', f'{xlo:.4f} {xhi:.4f}')
    js[1].set('range', f'{ylo:.4f} {yhi:.4f}')

    for j in js:                       # 9-45: 質量を 90 分の1 にしたので damping も 90 分の1
        j.set('damping', f'{PUCK_DAMPING}')

    # B: 小さく軽く
    g = cube.find('geom')
    g.set('size', f'{PUCK_HALF} {PUCK_HALF} {PUCK_HALF}')
    g.set('density', f'{PUCK_DENSITY}')
    g.set('rgba', '0.85 0.15 0.15 1.0')

    if wall_mode:
        _add_walls(root, js)

    (DST_WALL if wall_mode else DST).write_bytes(ET.tostring(root))
    if wall_mode:
        _write_body(js)          # 9-58: 表示用 .body も同じ定数から生成する
    m = (2 * PUCK_HALF) ** 3 * PUCK_DENSITY
    print(f'{DST_WALL if wall_mode else DST}')
    print(f'  パック  一辺 {2*PUCK_HALF:.2f} m / {m:.3f} kg  位置 {PUCK_POS}')
    print(f'  コート  x [{COURT_X[0]}, {COURT_X[1]}]  y [{COURT_Y[0]}, {COURT_Y[1]}]')
    print(f'  可動域  x [{js[0].get("range")}]  y [{js[1].get("range")}]（関節値）')
    if wall_mode:
        print(f'  壁      側壁 内面 y=±{COURT_Y[1]-PUCK_HALF:.2f} / '
              f'ゴール口 |y|<={GOAL_HALF} / 板 x={COURT_X[1]-0.60:.2f}・長さ {GOAL_HALF*2*0.70:.3f} m')
        print(f'  ⚠️ x 可動域をゴールの先まで延ばした。外したパックは端の壁で止まるので'
              f'**通した方が総移動距離が大きい**（報酬は変えていない）')


if __name__ == '__main__':
    import sys
    main(wall_mode='--wall' in sys.argv)
