#!/usr/bin/env python3
"""GLB が M3 を通せる姿勢かを、パイプラインを走らせる前に判定する。

**なぜ要るか（2026-09-03、実験系譜 9-23）**: A3 で Tripo3D が 3/4 の斜め線を
「上へ伸びる」ではなく**「奥へ伸びる」と解釈して腕が寝た**。関節がほぼ同じ高さに並び、
M3 の分割（Z 座標を境界にする）が原理的に通らなくなった。
`run_tripo_pipeline.sh` を最後まで走らせてから落ちるのは無駄なので、**先に落とす**。

⚠️ **Bug 30 で関節の「検出」は 3 次元化したが、「分割」は今も Z のままである**（意図的）。
検出が 3D でも、腕が寝ていれば分割で落ちる。**「腕は立っていること」は今も前提。**

判定するもの:
  1. マゼンタのクラスタ数（期待値と一致するか）
  2. 関節の Z 方向の広がり（小さいと分割が通らない）
  3. 関節の Z が単調に並ぶか（順序が崩れていないか）

実測値（2026-09-03）:

  | GLB | 関節の Z | 広がり | 結果 |
  |---|---|---|---|
  | A1 | 0.220 / 0.496 / 0.726 | 0.51 | ✅ |
  | B2 | 0.082 / 0.436 / 0.710 | 0.63 | ✅ |
  | B1 | 0.532 / 0.760 / 0.887 | 0.35 | ✅ |
  | A2 | 0.315 / 0.494 / 0.624 | **0.31** | ✅（**通った中では最小**） |
  | A3 | 0.184 / 0.193 / 0.201 | **0.017** | ❌ 分割できず |

  ⚠️ **閾値 0.25 は通った最小（A2 の 0.31）と落ちた最大（A3 の 0.017）の間**に置いた。
  A2 が閾値に近いので、**0.25〜0.31 の GLB が来たら判定を疑い、実際に通してみる**こと。
  境界のサンプルが増えたら閾値を見直す。

使い方:
    python3 scripts/check_glb_pose.py data/test/A3/3D/A3.glb --joints 3
終了コード: 通せそうなら 0、問題があれば 1
"""
import argparse
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree

# 関節間の Z 差がこれを下回ると、分割の境界が潰れて通らない。
# 実測で通った 3 枚は最小 0.36、落ちた A3 は 0.14。あいだを取る。
MIN_Z_SPREAD = 0.25
# 隣り合う関節の Z 差の下限（分割は 1 リンクぶんのフェースを必要とする）
MIN_Z_GAP = 0.05


def magenta_clusters(mesh, radius_frac=0.02, min_frac=0.02):
    try:
        mesh.visual = mesh.visual.to_color()
    except Exception:
        pass
    v = mesh.vertices.copy()
    v = np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])          # Y-up → Z-up
    c = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(float)
    # マゼンタ色相域（3.8 節の自動キャリブレーションと同じ考え方の粗い版）
    mx, mn = c.max(axis=1), c.min(axis=1)
    mask = (c[:, 0] > 120) & (c[:, 2] > 100) & (c[:, 1] < 150) & ((mx - mn) > 60)
    if mask.sum() < 30:
        return np.empty((0, 3)), 0.0
    P = v[mask]
    scale = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    tree = cKDTree(P)
    parent = np.arange(len(P))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, js in enumerate(tree.query_ball_point(P, radius_frac * scale)):
        ri = find(i)
        for j in js:
            rj = find(j)
            if ri != rj:
                parent[rj] = ri
                ri = find(ri)
    lab = np.array([find(i) for i in range(len(P))])
    keep = [u for u in np.unique(lab) if (lab == u).sum() >= min_frac * len(P)]
    centres = np.array(sorted((P[lab == u].mean(axis=0) for u in keep), key=lambda c: c[2]))
    return centres, mask.sum() / len(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('glb')
    ap.add_argument('--joints', type=int, default=3, help='期待する関節数')
    args = ap.parse_args()

    mesh = trimesh.load(args.glb, force='mesh')
    e = mesh.bounds[1] - mesh.bounds[0]
    # Y-up の Y が鉛直。Z-up 変換後の Z にあたる
    print(f"  extents (Y-up): [{e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f}]  鉛直方向の広がり {e[1]:.3f}")

    centres, frac = magenta_clusters(mesh)
    print(f"  マゼンタ頂点 {frac*100:.1f} %  クラスタ {len(centres)} 個")
    issues = []

    if len(centres) != args.joints:
        issues.append(f"クラスタが {len(centres)} 個（期待 {args.joints} 個）。"
                      f"多いなら台座や金具が拾われている、少ないなら球が融合または欠落")
    if len(centres) < 2:
        print("\n❌ 判定不能（クラスタが少なすぎる）")
        return 1

    z = centres[:, 2]
    spread = float(z.max() - z.min())
    gaps = np.diff(z)
    print(f"  関節の Z: {' / '.join(f'{v:.3f}' for v in z)}")
    print(f"  Z 方向の広がり {spread:.3f}   隣接差 {' / '.join(f'{g:.3f}' for g in gaps)}")

    if spread < MIN_Z_SPREAD:
        issues.append(f"Z 方向の広がりが {spread:.3f} しかない（実測で通った 3 枚は 0.36〜0.63、"
                      f"落ちた A3 は 0.14）。**腕が寝ている**。"
                      f"M3 の分割は Z を境界にするので通らない")
    if (gaps < MIN_Z_GAP).any():
        issues.append(f"隣り合う関節の Z 差が {gaps.min():.3f} と小さい（下限 {MIN_Z_GAP}）。"
                      f"分割の境界が潰れる")

    print()
    if not issues:
        print("✅ 姿勢は問題なさそう。run_tripo_pipeline.sh へ進んでよい。")
        return 0
    for m in issues:
        print(f"❌ {m}")
    print("\n**まず同じ M1 画像で Tripo3D をやり直す**（再構成にはばらつきがある）。"
          "2 回目も同じなら入力側（M1 画像の構図）を見直す。詳細は実験系譜 9-23。")
    return 1


if __name__ == '__main__':
    sys.exit(main())
