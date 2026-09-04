#!/usr/bin/env python3
"""記録した軌跡と**実物のメッシュ**から、腕が動く動画を書き出す。

**なぜ要るか**: 発表で「描いた絵がそのまま動く」ことを見せたい。
`visualize_morph_changes.py` は形態の変化を静止画で比べるだけで、
シミュレータの GUI はコンテナから開けない。**近似の円柱ではなく生成メッシュで**動かす口が要る。

**メッシュをどう合わせるか**（ここが本体）:
STL は Tripo3D が出したままの姿勢（描いた絵のとおり傾いている）で、
原点は根元側の関節にある。一方 XML のボーンは縦型なら +Z を向いている。
さらに **co-design はリンク長を変える**ので、設計図の長さで並べると学習結果と違う絵になる。
そこで各リンクを次の順で変換する。

  1. `joints.json` の `frame_origins` から、**メッシュ座標でのボーン方向**を出す
     （自分の原点 → 次のリンクの原点。先端リンクだけは最遠点）
  2. その方向を +Z へ回す
  3. **最適化後のボーン長 ÷ 元の長さ**で Z 方向にだけ伸縮する（太さは変えない）
  4. XML のボーン方向へ回す
  5. 各時刻のワールド変換（`xpos` / `xmat`）を掛ける

⚠️ **太さは元のまま**である。co-design は太さも変えるが、
メッシュを太らせると別物に見えるので長さだけ合わせている。**動画は形の説明用で、
形態パラメータの証拠ではない**（証拠は `形態比較.md`）。

⚠️ 間引きは頂点クラスタリングの自前実装。`fast_simplification` も `skimage` も
この環境に無いため（2026-09-04 確認）。描画用なので水密性は要らない。

⚠️ **body 名とメッシュ名は一致しない。** 学習側の body は `0 / 1 / 11 / 111 / 1111` で、
**先頭の `0` は取り付け用の球でメッシュを持たない**。`--link-names` で
根元から順にメッシュ名を与え、余った先頭の body は読み飛ばす。

使い方:
  python3 scripts/render_arm_video.py \\
    --trace single_run/e2e_a1v_reach/trace/arm_trace.npz \\
    --meshes data/test/A1/meshes --out demo_a1v_reach.mp4
"""
import argparse
import json
import pathlib

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v2 as imageio

# ⚠️ 既定のフォントは日本語を持たず、題名が豆腐（□□□）になる（2026-09-04 に発生）。
# ⚠️ **`Droid Sans Fallback` では直らない。** 漢字は出るが**英数字と矢印が豆腐のまま**で、
# matplotlib は 1 書体で描くのでグリフ単位の代替が効かない。
# `/usr/share/fonts` の Noto CJK（和文も英数も持つ）を明示登録して使う。
for _ttc in sorted(pathlib.Path('/usr/share/fonts').rglob('*CJK*')):
    try:
        matplotlib.font_manager.fontManager.addfont(str(_ttc))
    except Exception:
        pass
for _f in ('Noto Sans CJK JP', 'Noto Serif CJK JP', 'Droid Sans Fallback'):
    if any(_x.name == _f for _x in matplotlib.font_manager.fontManager.ttflist):
        matplotlib.rcParams['font.family'] = _f
        break

# スケッチの配色に合わせる（参照画生成プロンプトと同じ）
COLORS = {'base': '#4a4a4a', 'upper_arm': '#e03131',
          'forearm': '#3b5bdb', 'hand': '#2f9e44'}
LIGHT = np.array([0.4, -0.6, 0.7]); LIGHT = LIGHT / np.linalg.norm(LIGHT)


def cluster_decimate(mesh, pitch):
    """頂点をグリッドへ丸めて代表点（セル重心）に寄せる素朴な間引き。"""
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    key = np.floor(V / pitch).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    nV = np.zeros((len(uniq), 3)); cnt = np.zeros(len(uniq))
    np.add.at(nV, inv, V); np.add.at(cnt, inv, 1)
    nV /= cnt[:, None]
    nF = inv[F]
    ok = (nF[:, 0] != nF[:, 1]) & (nF[:, 1] != nF[:, 2]) & (nF[:, 0] != nF[:, 2])
    return nV, np.unique(np.sort(nF[ok], axis=1), axis=0)


def align(a, b):
    """単位ベクトル a を b へ回す回転行列（ロドリゲス）。"""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', required=True)
    ap.add_argument('--meshes', required=True, help='STL と joints.json のあるディレクトリ')
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=int, default=25)
    ap.add_argument('--stride', type=int, default=2, help='何ステップに1枚描くか')
    ap.add_argument('--pitch', type=float, default=0.014, help='間引きの粗さ[m]')
    ap.add_argument('--elev', type=float, default=18)
    ap.add_argument('--azim', type=float, default=-62)
    ap.add_argument('--title', default='')
    ap.add_argument('--link-names', nargs='+',
                    default=['base', 'upper_arm', 'forearm', 'hand'],
                    help='根元から順のメッシュ名。body 側が多い分は先頭を読み飛ばす')
    ap.add_argument('--cube', action='store_true', help='対象物を描く（Pusher のとき）')
    ap.add_argument('--zoom', type=float, default=1.5, help='寄り。大きいほど腕が大きく映る')
    ap.add_argument('--tmax', type=int, default=0, help='この step までを描く（0 なら全部）')
    args = ap.parse_args()

    d = np.load(args.trace, allow_pickle=True)
    names = [str(n) for n in d['body_names']]
    xpos, xmat, bone = d['xpos'], d['xmat'], d['bone_offset']
    cube, target = d['cube'], d['target']
    T, L = xpos.shape[0], xpos.shape[1]
    tmax = min(args.tmax, T) if args.tmax > 0 else T

    mdir = pathlib.Path(args.meshes)
    origins = json.loads((mdir / 'joints.json').read_text())['frame_origins']

    mesh_names = args.link_names
    skip = L - len(mesh_names)          # 先頭の取り付け球など、メッシュを持たない body
    if skip < 0:
        raise SystemExit(f'--link-names が body 数 {L} より多い')
    print(f'  body {names} のうち先頭 {skip} 個はメッシュ無しとして読み飛ばす')

    # メッシュを body ローカルへ持ち込む変換を先に済ませる（毎フレームやらない）
    parts = [None] * L
    for j, mn in enumerate(mesh_names):
        i = skip + j
        f = mdir / f'{mn}.stl'
        if not f.exists():
            print(f'  ⚠️ {f} が無いので {mn} は描かない'); continue
        V, F = cluster_decimate(trimesh.load(f), args.pitch)
        # ① メッシュ座標でのボーン方向
        if j + 1 < len(mesh_names) and mesh_names[j + 1] in origins:
            bl = np.array(origins[mesh_names[j + 1]]) - np.array(origins[mn])
        else:                                   # 先端は最遠点（Bug 31 と同じ考え方）
            bl = V[np.argmax(np.linalg.norm(V, axis=1))]
        Lb = float(np.linalg.norm(bl))
        # ②→④ 元のボーン → +Z → 伸縮 → XML のボーン
        R1 = align(bl, np.array([0., 0., 1.]))
        bo = bone[i]; Lo = float(np.linalg.norm(bo))
        S = np.diag([1., 1., (Lo / Lb) if Lb > 1e-9 else 1.])
        R2 = align(np.array([0., 0., 1.]), bo) if Lo > 1e-9 else np.eye(3)
        parts[i] = ((R2 @ S @ R1 @ V.T).T, F, COLORS.get(mn, '#888888'))
        print(f'  {mn:10} (body {names[i]}) faces {len(F):5d}  '
              f'ボーン長 {Lb:.4f} → {Lo:.4f}（×{Lo/Lb:.3f}）')

    # 描画範囲は**全時刻の実際の点**から決めて固定する（途中で視点が動くと比較しにくい）。
    # リンク原点だけだと先端メッシュがはみ出るので、最長ボーン 1 本ぶんだけ広げる。
    # ⚠️ **cube は画角に入れない。** Pusher の学習済み方策は cube を 15 m 吹き飛ばすので、
    # 入れると腕が豆粒になる（2026-09-04 に実際にそうなった）。cube は枠外へ出てよい。
    pts = np.vstack([xpos[:tmax].reshape(-1, 3), target[None, :]])
    pad = float(np.linalg.norm(bone, axis=1).max())
    lo, hi = pts.min(axis=0) - pad, pts.max(axis=0) + pad
    lo[2] = min(lo[2], 0.0)                      # 床は必ず入れる
    ctr = (lo + hi) / 2
    rad = float((hi - lo).max()) / 2 * 1.05

    frames = range(0, tmax, args.stride)
    fig = plt.figure(figsize=(7, 6), dpi=110)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    w = imageio.get_writer(args.out, fps=args.fps, codec='libx264',
                           quality=8, macro_block_size=1)
    for k, t in enumerate(frames):
        fig.clf()
        ax = fig.add_subplot(111, projection='3d')
        for i, p in enumerate(parts):
            if p is None: continue
            Vl, F, col = p
            Vw = xpos[t, i] + (xmat[t, i] @ Vl.T).T
            tri = Vw[F]
            # 面法線と光源の内積で陰影を作る（立体感が無いと関節の動きが読めない）
            nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            ln = np.linalg.norm(nrm, axis=1, keepdims=True)
            nrm = nrm / np.where(ln < 1e-12, 1.0, ln)
            lit = np.clip(np.abs(nrm @ LIGHT), 0.0, 1.0)
            base_rgb = np.array(matplotlib.colors.to_rgb(col))
            fc = np.clip(base_rgb[None, :] * (0.45 + 0.55 * lit)[:, None], 0, 1)
            ax.add_collection3d(Poly3DCollection(tri, facecolors=fc, edgecolor='none'))
        ax.scatter(*target, s=90, marker='*', color='#f59f00', depthshade=False)
        if args.cube and cube.ndim == 2 and np.any(cube[t]):
            ax.scatter(*cube[t], s=70, marker='s', color='#c2255c', depthshade=False)
        # 床
        gg, hh = np.meshgrid(np.linspace(ctr[0]-rad, ctr[0]+rad, 2),
                             np.linspace(ctr[1]-rad, ctr[1]+rad, 2))
        ax.plot_surface(gg, hh, np.zeros_like(gg), color='#dee2e6', alpha=0.5, zorder=0)
        ax.set_xlim(ctr[0]-rad, ctr[0]+rad); ax.set_ylim(ctr[1]-rad, ctr[1]+rad)
        ax.set_zlim(ctr[2]-rad, ctr[2]+rad)
        # zoom を上げないと matplotlib は軸の立方体を figure の中で小さく描く
        ax.set_box_aspect((1, 1, 1), zoom=args.zoom)
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_axis_off()
        ax.set_title(f'{args.title}   t={t}', fontsize=11, y=0.94)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        w.append_data(img)
        if k % 25 == 0:
            print(f'    frame {k}/{len(frames)}')
    w.close()
    print(f'✅ {args.out}  ({len(frames)} フレーム / {args.fps} fps)')


if __name__ == '__main__':
    main()
