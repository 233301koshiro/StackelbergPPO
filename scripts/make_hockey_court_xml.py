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

# コート（表示用 body と揃える）。ロボットは原点、正面が +x。
COURT_X = (0.15, 1.55)      # 奥行き 1.4 m
COURT_Y = (-0.50, 0.50)     # 幅 0.9 m … の外枠
PUCK_HALF = 0.05            # 一辺 0.10 m
PUCK_DENSITY = 30.0         # 0.10^3 * 30 = 0.03 kg（実物のパック相当）
PUCK_POS = (0.70, 0.0, 0.15)


def main():
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

    # B: 小さく軽く
    g = cube.find('geom')
    g.set('size', f'{PUCK_HALF} {PUCK_HALF} {PUCK_HALF}')
    g.set('density', f'{PUCK_DENSITY}')
    g.set('rgba', '0.85 0.15 0.15 1.0')

    DST.write_bytes(ET.tostring(root))
    m = (2 * PUCK_HALF) ** 3 * PUCK_DENSITY
    print(f'{DST}')
    print(f'  パック  一辺 {2*PUCK_HALF:.2f} m / {m:.3f} kg  位置 {PUCK_POS}')
    print(f'  コート  x [{COURT_X[0]}, {COURT_X[1]}]  y [{COURT_Y[0]}, {COURT_Y[1]}]')
    print(f'  可動域  x [{xlo:.3f}, {xhi:.3f}]  y [{ylo:.3f}, {yhi:.3f}]（関節値）')


if __name__ == '__main__':
    main()
