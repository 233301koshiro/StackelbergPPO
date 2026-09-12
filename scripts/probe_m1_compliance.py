#!/usr/bin/env python3
"""M1 出力画像の**仕様遵守率**を測る（実験系譜 9-68 / 9-79、研究方針「M1・M2 を API 化する」）。

**なぜ要るか**: 修論は 3 箇所で「手描き 5 枚では成功率を出せる件数ではない」と謝っている。
API で n 回生成できれば率が出るが、**何を測るかを先に決めておかないと率にならない。**
本スクリプトが**測る側**である。API 側（鍵・課金）が整えばそのまま n 枚に当てられる。

⚠️ **「向き」は 2D の角度で測らない（9-79）。** 通った A1 は 47.5°、止まった A3 は 49.8° で、
**この量は合否を分けない**。姿勢は **M2 の後に `check_glb_pose.py`** で見る
（関節の Z 広がり。閾値 0.25 は実測較正済み）。ここでは画像から測れる 2 項目に絞る。

| 項目 | 測り方 | ここで測るか |
|---|---|---|
| マーカー数 | マゼンタのクラスタ数が指定数と一致するか | ✅ |
| 比の保存 | マゼンタ重心の間隔比が指示比と一致するか | ✅ |
| 姿勢 | M2 後の関節 Z 広がり | ❌ `check_glb_pose.py` の担当 |
| 禁止物（寸法線・文字） | 画像の検出器が要る | ❌ **n が小さいうちは目視で数える** |

    python3 scripts/probe_m1_compliance.py                      # data/test/*/sketch/*_m1*.jpeg
    python3 scripts/probe_m1_compliance.py --dir <ディレクトリ>  # API で貯めた n 枚に当てる
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np

_spec = importlib.util.spec_from_file_location(
    'tilt', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'probe_m1_tilt.py'))
tilt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tilt)   # ⚠️ マゼンタ検出の実装は probe_m1_tilt が唯一


def centroids(path: str, min_frac: float = 0.02):
    """マゼンタクラスタの重心を、画像の下（腕の根元）から順に返す。"""
    from PIL import Image
    from scipy import ndimage
    im = np.asarray(Image.open(path).convert('RGB'))
    m = tilt.magenta_mask(im)
    if m.sum() < 20:
        return []
    lab, n = ndimage.label(m)
    if n == 0:
        return []
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= max(20, sizes.max() * min_frac)]
    cen = np.array(ndimage.center_of_mass(m, lab, keep))
    return cen[np.argsort(-cen[:, 0])]      # row 降順 = 下から上


def link_ratios(cen) -> list:
    """隣り合う重心の距離を、最初の区間を 1 とした比で返す。

    ⚠️ **関節は N 個、リンクは N 個**（台座→関節1 が第1リンク）だが、画像からは
    関節どうしの間隔しか測れないので **N−1 区間**しか出ない。
    指示比の**隣接する比**と突き合わせる形にする。
    """
    if len(cen) < 3:
        return []
    d = [float(np.linalg.norm(cen[i + 1] - cen[i])) for i in range(len(cen) - 1)]
    return [x / d[0] for x in d]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=None, help='画像を置いたディレクトリ（API で貯めた n 枚）')
    ap.add_argument('--markers', type=int, default=3, help='期待するマーカー数')
    ap.add_argument('--ratio-tol', type=float, default=0.15, help='比の許容（相対）')
    a = ap.parse_args()

    if a.dir:
        files = sorted(glob.glob(os.path.join(a.dir, '*.jpeg')) +
                       glob.glob(os.path.join(a.dir, '*.png')) +
                       glob.glob(os.path.join(a.dir, '*.jpg')))
        expect = None
    else:
        files = sorted(glob.glob('data/test/*/sketch/*_m1*.jpeg'))
        expect = 'per-dir'

    if not files:
        print('画像が見つからない'); return 1

    print(f"{'画像':32s}{'マーカー':>9s}{'区間比（第1区間=1）':>24s}{'指示比':>18s}  判定")
    ok_m = ok_r = n_r = 0
    for f in files:
        cen = centroids(f)
        nm = len(cen)
        r = link_ratios(cen)
        rs = ' / '.join(f'{x:.2f}' for x in r) if r else '測定不能'
        want = ''
        verdict_r = ''
        if expect:
            mj = os.path.join(os.path.dirname(f), 'measured.json')
            if os.path.exists(mj):
                base = json.load(open(mj)).get('ratios_base1')
                if base and len(base) >= 2:
                    # 指示比は台座高を 1 とした各リンク長。隣接比へ直して突き合わせる
                    w = [base[i] / base[0] for i in range(len(base))]
                    want = ' / '.join(f'{x:.2f}' for x in w[:len(r)])
                    if r and len(r) == len(w[:len(r)]):
                        n_r += 1
                        if all(abs(x - y) <= a.ratio_tol * max(y, 1e-9)
                               for x, y in zip(r, w[:len(r)])):
                            ok_r += 1; verdict_r = '✅'
                        else:
                            verdict_r = '❌'
        hit_m = (nm == a.markers)
        ok_m += hit_m
        print(f"{os.path.basename(f):32s}{nm:9d}{rs:>24s}{want:>18s}  "
              f"{'✅' if hit_m else '❌'}{verdict_r}")

    print(f"\nマーカー数の遵守率: {ok_m}/{len(files)}")
    if n_r:
        print(f"比の保存の遵守率  : {ok_r}/{n_r}（許容 ±{a.ratio_tol*100:.0f} %）")
    print("\n⚠️ **姿勢は本スクリプトでは測らない。** M2 後に `check_glb_pose.py` で見る（9-79）。")
    print("⚠️ **禁止物（寸法線・文字）は検出器が無い。** n が小さいうちは目視で数える。")
    return 0


if __name__ == '__main__':
    sys.exit(main())
