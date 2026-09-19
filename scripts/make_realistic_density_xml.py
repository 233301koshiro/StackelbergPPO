#!/usr/bin/env python3
"""リンク密度に**根拠**を与えた XML を作る（実験系譜 9-125・9-126）。

⛔ **何が問題だったか**: リンク密度 5.0 kg/m³ は原論文（**移動ロボット**）からの引き継ぎで、
**本研究の設定には根拠が無い**（9-124 の弱点②）。腕全体が 45〜108 g にしかならず、
**押下対象の cube 2.7 kg の 25 分の 1** という物理的に成立しない設定になっていた。

⚠️ **ただし密度だけ実材料に変えても解決しない。**
リンクは**中実の丸太**としてモデル化されており（半径 0.0865 m ＝ 直径 17 cm）、
実材料の密度を当てると腕が **18〜47 kg** になる（9-125）。

⭐ **実際のロボットアームのリンクは薄肉パイプである。** 外径はそのままに中身を抜いた
**実効密度**を使えば、「なぜその値か」を**材料と肉厚で説明できる**。

    ρ_eff = ρ_材料 × (1 - (1 - t/r)²)

| 材料 | 肉厚 | 実効密度 | 腕の質量 |
|---|---|---|---|
| ABS 樹脂 | 3 mm | 110 | 1.91 kg |
| ⭐ **アルミ A6061** | **2 mm** | ⭐ **190** | ⭐ **3.30 kg** |
| 炭素繊維強化樹脂 | 2 mm | 113 | 1.96 kg |

⭐ **アルミ 2 mm を既定とする。**ロボットアームの標準材料で既製パイプが豊富であり、
cube 2.7 kg との比が **1.22** になる（現在は 0.04）。
**トルク密度（9-97）と同じ型の根拠づけ**である。

    python3 scripts/make_realistic_density_xml.py --base e2e_a1v --name e2e_a1v_rho
    python3 scripts/make_realistic_density_xml.py --base e2e_a1v --material abs --wall 0.003
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENVS = ROOT / 'assets' / 'mujoco_envs'

# 実在材料の密度 [kg/m³]。⚠️ **出典を持つ値だけを置く。**
MATERIALS = {
    'aluminum': (2700.0, 'アルミ合金 A6061（ロボットアームの標準材料。既製パイプが豊富）'),
    'abs':      (1050.0, 'ABS 樹脂（3D プリント）'),
    'cfrp':     (1600.0, '炭素繊維強化樹脂'),
    'steel':    (7850.0, '鋼'),
}


def effective_density(rho, wall, radii):
    """薄肉パイプの実効密度。外径 r・肉厚 t の円筒の中身を抜いた平均密度。"""
    vals = []
    for r in radii:
        if wall >= r:           # 肉厚が半径以上なら中実
            vals.append(rho)
        else:
            vals.append(rho * (1.0 - (1.0 - wall / r) ** 2))
    return sum(vals) / len(vals)


def capsule_radii(xml_text):
    """カプセル geom の半径を集める（size の第1要素）。"""
    out = []
    for m in re.finditer(r'<geom[^>]*type="capsule"[^>]*>', xml_text):
        g = m.group(0)
        s = re.search(r'size="([\d.eE+-]+)', g)
        if s:
            out.append(float(s.group(1)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='e2e_a1v', help='元の XML 名（拡張子なし）')
    ap.add_argument('--name', help='出力 XML 名。既定は <base>_rho')
    ap.add_argument('--material', default='aluminum', choices=sorted(MATERIALS))
    ap.add_argument('--wall', type=float, default=0.002, help='肉厚 [m]。既定 2 mm')
    ap.add_argument('--solref', default=None,
                    help='接触の solref（例 "0.02 0.4"）。⭐ 既定のままだと反発係数 0.135 で'
                         '実物（0.2〜0.5）を下回る（9-128）')
    a = ap.parse_args()

    src = ENVS / f'{a.base}.xml'
    if not src.exists():
        raise SystemExit(f'⛔ {src} が無い')
    txt = src.read_text(encoding='utf-8')

    radii = capsule_radii(txt)
    if not radii:
        raise SystemExit('⛔ カプセル geom が見つからない')
    rho, note = MATERIALS[a.material]
    eff = effective_density(rho, a.wall, radii)

    old = re.search(r'density="([\d.eE+-]+)"', txt)
    if not old:
        raise SystemExit('⛔ default の density が見つからない')
    out_txt = txt.replace(f'density="{old.group(1)}"', f'density="{eff:.1f}"', 1)

    # ⭐ 9-128: 接触の反発。⚠️ 既定（dampratio=1）は臨界減衰で e≈0.135。
    #   実物の金属アーム ↔ 箱 は e≈0.2〜0.5。dampratio=0.4 で e=0.318〜0.397（速度依存が最小）
    if a.solref:
        if 'solref=' in out_txt:
            out_txt = re.sub(r'solref="[^"]*"', f'solref="{a.solref}"', out_txt)
        else:
            out_txt = out_txt.replace(f'density="{eff:.1f}"',
                                      f'density="{eff:.1f}" solref="{a.solref}"', 1)

    name = a.name or f'{a.base}_rho'
    dst = ENVS / f'{name}.xml'
    dst.write_text(out_txt, encoding='utf-8')

    print(f'{dst}')
    print(f'  材料      {note}')
    print(f'  素材密度  {rho:.0f} kg/m³   肉厚 {a.wall*1000:.1f} mm')
    print(f'  カプセル半径 {[round(r,4) for r in radii]}')
    print(f'  ⭐ 実効密度 {old.group(1)} → **{eff:.1f} kg/m³**')
    if a.solref:
        print(f'  ⭐ solref  既定 → **{a.solref}**（9-128）')
    print(f'  ⚠️ 元の {a.base}.xml は変更していない（条件を足す形。9-105 の版の混入を避ける）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
