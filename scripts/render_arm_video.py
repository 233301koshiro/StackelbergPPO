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


def cluster_decimate(mesh, pitch, vcol=None):
    """頂点をグリッドへ丸めて代表点（セル重心）に寄せる素朴な間引き。

    `vcol`（頂点ごとの RGB, 0〜1）を渡すと、**面ごとの色**も返す（まとめた頂点の色を平均し、面は 3 頂点の平均）。
    """
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    key = np.floor(V / pitch).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    nV = np.zeros((len(uniq), 3)); cnt = np.zeros(len(uniq))
    np.add.at(nV, inv, V); np.add.at(cnt, inv, 1)
    nV /= cnt[:, None]
    nF = inv[F]
    ok = (nF[:, 0] != nF[:, 1]) & (nF[:, 1] != nF[:, 2]) & (nF[:, 0] != nF[:, 2])
    if vcol is None:
        return nV, np.unique(np.sort(nF[ok], axis=1), axis=0)
    nC = np.zeros((len(uniq), 3)); np.add.at(nC, inv, vcol); nC /= cnt[:, None]
    nF = np.unique(np.sort(nF[ok], axis=1), axis=0)
    return nV, nF, nC[nF].mean(axis=1)


def box_tris(center, half):
    """中心と半幅から、直方体の 12 枚の三角形 (12, 3, 3) を返す。"""
    c, h = np.asarray(center, float), np.asarray(half, float)
    s = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    P = c + s * h
    quads = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
    return np.array([[P[a], P[b], P[cc]] for a, b, cc, d in quads for (a, b, cc) in
                     ((a, b, cc), (a, cc, d))])


def scene_from_xml(xml_path):
    """⭐ デモ用（2026-10-05）: XML の worldbody から**動かない箱**と**対象の箱の半幅・色**を読む。

    動かない箱 = 腕（先頭の body）と `cube` 以外で、box の geom を持つ body
    （障害物 Reach の柱、ホッケーの側壁・ゴールの壁・中央の板）。⚠️ 回転（euler/quat）付きは扱わない。
    """
    import xml.etree.ElementTree as ET
    wb = ET.parse(xml_path).getroot().find('worldbody')
    bodies = wb.findall('body')
    static, cube = [], None
    for k, b in enumerate(bodies):
        bp = np.array([float(v) for v in b.get('pos', '0 0 0').split()])
        for g in b.findall('geom'):
            if g.get('type') != 'box':
                continue
            if g.get('euler') or g.get('quat'):
                print(f'  ⚠️ {b.get("name")}: 回転付きの箱は描かない'); continue
            half = np.array([float(v) for v in g.get('size').split()])
            gp = np.array([float(v) for v in g.get('pos', '0 0 0').split()])
            rgba = [float(v) for v in (g.get('rgba') or '0.5 0.5 0.5 1').split()][:3]
            if b.get('name') == 'cube':
                cube = (half, np.array(rgba))
            elif k > 0:
                static.append((b.get('name'), bp + gp, half, np.array(rgba)))
    return static, cube


def glb_vertex_colors(glb_path, origins):
    """⭐ 元の GLB の色を、リンクごとの STL の頂点へ移すための準備（2026-10-05、デモ用）。

    `glb_to_links.py` は GLB を **Y-up → Z-up**（(x, y, z) → (x, −z, y)）に回し、
    リンクごとに `frame_origins` を引いて STL にしている。⭐ **同じ変換を GLB に掛ければ頂点は一致する**
    （A1・hockey で距離の中央値 1e-9 m を確認）。返すのはリンク名 → (KD 木, 色) の関数。
    """
    import warnings
    from scipy.spatial import cKDTree
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    sc = trimesh.load(glb_path)
    g = sc.to_geometry() if hasattr(sc, 'to_geometry') else sc
    col = np.asarray(g.visual.to_color().vertex_colors[:, :3], dtype=float) / 255.0
    V = np.asarray(g.vertices)
    Vz = np.c_[V[:, 0], -V[:, 2], V[:, 1]]

    def colors_for(name, verts):
        t = cKDTree(Vz - np.asarray(origins[name]))
        dist, idx = t.query(verts)
        return col[idx], float(np.median(dist))
    return colors_for


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
    ap.add_argument('--no-target', action='store_true', dest='no_target',
                    help='目標の星を描かない（Pusher など目標の無いタスク。trace は Reach 用の既定の目標を持っているので、描くと誤解を招く）')
    ap.add_argument('--xml', default='',
                    help='環境の XML。省略時は trace のある run の .hydra から xml_name を読む。柱・壁・板と箱の寸法に使う')
    ap.add_argument('--glb', default='',
                    help='元の GLB。渡すと**元のモデルの色**で描く（無ければリンク名ごとの決め打ちの色）')
    args = ap.parse_args()

    d = np.load(args.trace, allow_pickle=True)
    names = [str(n) for n in d['body_names']]
    xpos, xmat, bone = d['xpos'], d['xmat'], d['bone_offset']
    cube, target = d['cube'], d['target']
    T, L = xpos.shape[0], xpos.shape[1]
    tmax = min(args.tmax, T) if args.tmax > 0 else T

    mdir = pathlib.Path(args.meshes)
    origins = json.loads((mdir / 'joints.json').read_text())['frame_origins']

    colors_for = glb_vertex_colors(args.glb, origins) if args.glb else None
    # ⭐ 動かない箱（柱・壁・板）と対象の箱の寸法を XML から読む
    xml_path = args.xml
    if not xml_path:
        try:
            import yaml
            run_dir = pathlib.Path(args.trace).resolve().parent.parent
            xn = yaml.safe_load((run_dir / '.hydra' / 'config.yaml').read_text()).get('xml_name')
            xml_path = f'assets/mujoco_envs/{xn}.xml' if xn else ''
        except Exception:
            xml_path = ''
    static, cube_box = scene_from_xml(xml_path) if xml_path and pathlib.Path(xml_path).exists() else ([], None)
    if xml_path:
        print(f'  環境 {xml_path}: 動かない箱 {len(static)} 個 {[s[0] for s in static]}')
    static_tris = [(box_tris(c, h), col) for _n, c, h, col in static]
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
        m = trimesh.load(f)
        fcol = None
        if colors_for is not None and mn in origins:
            vcol, med = colors_for(mn, np.asarray(m.vertices))
            if med < 1e-6:
                V, F, fcol = cluster_decimate(m, args.pitch, vcol)
            else:
                # ⚠️ 対応づけがずれている（`--link-rot` で回したリンクなど）。黙って変な色にしない
                print(f'  ⚠️ {mn}: GLB との距離の中央値 {med:.2e} m。元の色を使わず決め打ちの色にする')
        if fcol is None:
            V, F = cluster_decimate(m, args.pitch)
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
        parts[i] = ((R2 @ S @ R1 @ V.T).T, F,
                    fcol if fcol is not None else COLORS.get(mn, '#888888'))
        print(f'  {mn:10} (body {names[i]}) faces {len(F):5d}  '
              f'ボーン長 {Lb:.4f} → {Lo:.4f}（×{Lo/Lb:.3f}）')

    # 描画範囲は**全時刻の実際の点**から決めて固定する（途中で視点が動くと比較しにくい）。
    # リンク原点だけだと先端メッシュがはみ出るので、最長ボーン 1 本ぶんだけ広げる。
    # ⚠️ **cube は画角に入れない。** Pusher の学習済み方策は cube を 15 m 吹き飛ばすので、
    # 入れると腕が豆粒になる（2026-09-04 に実際にそうなった）。cube は枠外へ出てよい。
    pts = np.vstack([xpos[:tmax].reshape(-1, 3)] + ([] if args.no_target else [target[None, :]])
                    + [tr.reshape(-1, 3) for tr, _c in static_tris])   # ⭐ 柱・壁も画角に入れる
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
        # ⭐ 床を先に・不透明で塗り、物は必ずその上に描く（半透明の床を最後に重ねると柱の下半分がかすむ。2026-10-05）
        ax.computed_zorder = False
        fl = np.array([[ctr[0]-rad, ctr[1]-rad, 0], [ctr[0]+rad, ctr[1]-rad, 0],
                       [ctr[0]+rad, ctr[1]+rad, 0], [ctr[0]-rad, ctr[1]+rad, 0]])
        ax.add_collection3d(Poly3DCollection([fl], facecolors='#e3e5e8', edgecolor='none', zorder=0))
        # ⭐ 腕・箱・壁を**1 つの集まり**にして奥から順に塗る（部品ごとに足すと前後関係が崩れる）
        all_tri, all_fc = [], []

        def add(tri, rgb):
            nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            ln = np.linalg.norm(nrm, axis=1, keepdims=True)
            nrm = nrm / np.where(ln < 1e-12, 1.0, ln)
            lit = np.clip(np.abs(nrm @ LIGHT), 0.0, 1.0)
            rgb = np.atleast_2d(rgb)
            all_tri.append(tri)
            all_fc.append(np.clip(rgb * (0.45 + 0.55 * lit)[:, None], 0, 1))
        for tr, col in static_tris:
            add(tr, col)
        if args.cube and cube_box is not None and cube.ndim == 2 and np.any(cube[t]):
            add(box_tris(cube[t], cube_box[0]), cube_box[1])
        for i, p in enumerate(parts):
            if p is None: continue
            Vl, F, col = p
            Vw = xpos[t, i] + (xmat[t, i] @ Vl.T).T
            # 面法線と光源の内積で陰影を作る（立体感が無いと関節の動きが読めない）
            add(Vw[F], col if isinstance(col, np.ndarray)       # 面ごとの元の色（--glb）
                else np.array(matplotlib.colors.to_rgb(col))[None, :])
        if all_tri:
            ax.add_collection3d(Poly3DCollection(np.concatenate(all_tri), facecolors=np.concatenate(all_fc),
                                                 edgecolor='none', zorder=1))
        if not args.no_target:
            ax.scatter(*target, s=90, marker='*', color='#f59f00', depthshade=False, zorder=2)
        if args.cube and cube_box is None and cube.ndim == 2 and np.any(cube[t]):
            ax.scatter(*cube[t], s=70, marker='s', color='#c2255c', depthshade=False)
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
