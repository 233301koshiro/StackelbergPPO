#!/usr/bin/env python3
"""M1 出力画像のマゼンタ関節マーカーから、腕の鉛直からの傾きを測る（9-68 / 9-79）。

**なぜ要るか**: 9-68 は「4 回に 1 回、向きの指示が破られる（42.5°）」と測ったが、
**その 42.5° が後段を止めるのかは確かめていない**。M3 のリンク分割は Z 座標で切るので
傾くと通らなくなるが、**どこから通らないのかの境界が無い**。
通った 5 例の M1 画像を同じ物差しで測れば、42.5° が範囲の内か外かが言える。

⚠️ **9-68 と同じ方法**（最下と最上のマゼンタクラスタの重心を結ぶ角）で実装してある。
既知の 4 値（5.4 / 42.5 / 23.4 / 7.6）を再現できることを `--reproduce` で先に確かめること。
再現できないプローブの結果は根拠にならない（CLAUDE.md §5-2 ⑤-3）。

    python3 scripts/probe_m1_tilt.py --reproduce   # 先にこれ
    python3 scripts/probe_m1_tilt.py              # 全 M1 画像を測る
"""
import argparse
import glob
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

# 生成物のマゼンタは #FF00FF から暗色化する（修論 3.7）。広めに取る。
def magenta_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    return (r > 110) & (b > 110) & (g < r - 45) & (g < b - 45)


def tilt_deg(path: str, min_frac: float = 0.02):
    """最下と最上のマゼンタクラスタの重心を結ぶ線の、鉛直からの角度 [deg]。"""
    im = np.asarray(Image.open(path).convert('RGB'))
    m = magenta_mask(im)
    if m.sum() < 20:
        return None, 0
    lab, n = ndimage.label(m)
    if n == 0:
        return None, 0
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= max(20, sizes.max() * min_frac)]
    cen = np.array(ndimage.center_of_mass(m, lab, keep))   # (row, col) = (y 下向き, x)
    if len(cen) < 2:
        return None, len(cen)
    lo = cen[cen[:, 0].argmax()]    # row が大きい = 画像の下 = 腕の根元
    hi = cen[cen[:, 0].argmin()]    # row が小さい = 画像の上 = 腕の先端
    dy = lo[0] - hi[0]              # 上向きを正にする
    dx = hi[1] - lo[1]
    return float(np.degrees(np.arctan2(abs(dx), abs(dy)))), len(cen)


KNOWN = {   # 9-68 の実測値（`make_slide9_m1_variance_figure.py` と同じ）
    'data/test/A1_m1var/sketch/A1_m1var_m1.jpeg': 5.4,
    'data/test/A1_m1var/sketch/A1_m1var_r2.jpeg': 42.5,
    'data/test/A1_m1var/sketch/A1_m1var_r3.jpeg': 23.4,
    'data/test/A1_m1var/sketch/A1_m1var_r4.jpeg': 7.6,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--reproduce', action='store_true',
                    help='9-68 の既知 4 値を再現できるか先に確かめる')
    ap.add_argument('--tol', type=float, default=8.0, help='再現とみなす許容 [deg]')
    a = ap.parse_args()

    if a.reproduce:
        print('=== 再現確認: 9-68 の 4 値と突き合わせる（±%.0f° 以内なら OK）\n' % a.tol)
        ok = True
        for p, want in KNOWN.items():
            got, n = tilt_deg(p)
            g = f'{got:5.1f}°' if got is not None else ' 測定不能'
            hit = got is not None and abs(got - want) <= a.tol
            ok &= hit
            print(f"  {p.split('/')[-1]:22s} 既知 {want:5.1f}°  実測 {g}  "
                  f"クラスタ {n}  {'✅' if hit else '❌'}")
        print('\n' + ('✅ 再現した。この物差しを他の画像へ当てられる。'
                      if ok else '❌ 再現しない。**この結果は根拠にならない**（§5-2 ⑤-3）。'))
        return 0 if ok else 1

    print(f"{'画像':34s}{'傾き':>8s}{'クラスタ':>9s}  後段")
    for p in sorted(glob.glob('data/test/*/sketch/*_m1*.jpeg')):
        got, n = tilt_deg(p)
        name = p.split('/')[2] + '/' + p.split('/')[-1]
        g = f'{got:6.1f}°' if got is not None else '   n/a'
        print(f'{name:34s}{g:>8s}{n:9d}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
