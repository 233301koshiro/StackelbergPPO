#!/usr/bin/env python3
"""
glb_to_links.py: Tripo3D GLB → per-link STL (link-local 座標) + Choreonoid URDF

入力: Tripo3D 生成の GLB（Y-up、関節位置に magenta #FF00FF 球マーカーあり）
出力: link_0.stl, link_1.stl, ... (link-local 座標) + tripo_arm.urdf

使い方:
  # 基本（magenta マーカー自動検出）
  python3 scripts/glb_to_links.py \\
    --glb data/tripo_arm_colorful/mechanical_joystick_3d_model.glb \\
    --out-dir data/my_arm/meshes \\
    --urdf data/my_arm/tripo_arm.urdf

  # 関節 Z 位置を手動指定（マーカーが検出できない場合）
  python3 scripts/glb_to_links.py \\
    --glb data/tripo_arm_colorful/mechanical_joystick_3d_model.glb \\
    --out-dir data/my_arm/meshes \\
    --joints -0.070 0.277

  # リンク名・関節色を変更
  python3 scripts/glb_to_links.py \\
    --glb arm.glb --out-dir meshes \\
    --names upper_arm forearm hand \\
    --joint-color 255 0 255 --joint-tol 40
"""

import argparse
from pathlib import Path

import json
import numpy as np
import trimesh
from scipy.spatial import cKDTree

# Y-up → Z-up: (x, y, z)_Yup → (x, -z, y)_Zup
_R_YUP_ZUP = np.array([[1, 0, 0],
                         [0, 0, -1],
                         [0, 1, 0]], dtype=float)


def _bake_vertex_colors(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """TextureVisuals (UV テクスチャ) を頂点色 (ColorVisuals) に変換する。
    Tripo3D の GLB はテクスチャ形式のため、mesh.visual.vertex_colors に
    依存する下流処理（関節マーカー検出）の前に必須。"""
    try:
        if m.visual.kind != 'vertex':
            m.visual = m.visual.to_color()
    except Exception:
        pass
    return m


def _load_concat(glb_path: str) -> trimesh.Trimesh:
    scene_or_mesh = trimesh.load(glb_path)
    if isinstance(scene_or_mesh, trimesh.Scene):
        parts = [_bake_vertex_colors(p) for p in scene_or_mesh.geometry.values()]
        if not parts:
            raise ValueError(f"{glb_path}: geometry が空です")
        mesh = trimesh.util.concatenate(parts)
    else:
        mesh = _bake_vertex_colors(scene_or_mesh)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{glb_path}: Trimesh に変換できません (type={type(mesh)})")
    return mesh


def _apply_yup_zup(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.vertices = (_R_YUP_ZUP @ mesh.vertices.T).T
    return mesh


def _auto_calibrate_joint_color(mesh: trimesh.Trimesh,
                                hue_lo=285.0, hue_hi=345.0, sat_min=0.30,
                                val_min=80.0, min_frac=0.005):
    """マーカー色の自動キャリブレーション。

    Tripo3D は入力画像の純マゼンタ #FF00FF を暗色化する（実測 ≈[211,75,169]）ため、
    固定の目標色では世代・生成ごとの色ズレに追従できない。ここではマゼンタの
    色相域（hue 285–345°）かつ十分な彩度を持つ頂点クラスタを探し、その中央値を
    目標色、色の広がりから許容幅を決定する。

    Returns: (color_rgb tuple, tol int) / 見つからなければ (None, None)
    """
    try:
        vc = mesh.visual.vertex_colors[:, :3].astype(float)
    except Exception:
        return None, None
    mx = vc.max(axis=1)
    mn = vc.min(axis=1)
    delta = mx - mn
    sat = np.where(mx > 0, delta / np.maximum(mx, 1e-9), 0.0)
    r, g, b = vc[:, 0], vc[:, 1], vc[:, 2]
    hue = np.zeros(len(vc))
    m = (delta > 0) & (mx == r)
    hue[m] = (60.0 * ((g[m] - b[m]) / delta[m])) % 360.0
    m = (delta > 0) & (mx == g)
    hue[m] = 60.0 * ((b[m] - r[m]) / delta[m]) + 120.0
    m = (delta > 0) & (mx == b)
    hue[m] = 60.0 * ((r[m] - g[m]) / delta[m]) + 240.0

    # val_min: 暗い頂点（影・黒地）は彩度式の上ではマゼンタ色相に入りうるため明度で除外
    mask = (hue >= hue_lo) & (hue <= hue_hi) & (sat >= sat_min) & (mx >= val_min)
    n = int(mask.sum())
    if n < max(50, int(min_frac * len(vc))):
        return None, None
    cluster = vc[mask]
    color = tuple(int(round(c)) for c in np.median(cluster, axis=0))
    # 許容幅: クラスタ内の各チャネル 99 パーセンタイル偏差 + マージン（30〜80 に制限）
    dev = np.percentile(np.abs(cluster - np.median(cluster, axis=0)), 99, axis=0).max()
    tol = int(np.clip(dev + 15, 30, 80))
    frac = 100.0 * n / len(vc)
    print(f"  [auto-color] マゼンタ色相域クラスタ: {n} 頂点 ({frac:.1f}%) "
          f"→ 目標色 {color}, tol={tol}")
    return color, tol


def _marker_mask(mesh, color_rgb, tol):
    """マーカー色に一致する頂点のマスク。失敗時は None。"""
    try:
        vc = mesh.visual.vertex_colors[:, :3].astype(int)
    except Exception:
        return None
    r, g, b = color_rgb
    mask = ((np.abs(vc[:, 0] - r) <= tol) &
            (np.abs(vc[:, 1] - g) <= tol) &
            (np.abs(vc[:, 2] - b) <= tol))
    return mask if mask.any() else None


def _detect_joints_3d(mesh: trimesh.Trimesh, color_rgb=(255, 0, 255), tol=40,
                      link_radius=0.02, min_frac=0.02):
    """マーカー頂点を **3 次元で**連結成分に分け、各成分の重心を関節位置として返す。

    Returns: Z 昇順に並べた 3D 座標の配列（Z-up frame）。

    **なぜ 3D なのか（2026-09-02、B1 で判明）**: 従来は Z 座標だけを見て、
    隣接値の差が閾値を超えた位置で区切っていた。しかしマーカー球は有限の大きさを持つため、
    **リンクが短く腕が傾いていると隣り合う球の Z 範囲が重なり、2 個が 1 個に融合する**。
    実測（B1）では球の Z 方向の広がり約 0.14 に対し関節間の Z 差が 0.127 しかなく、
    3 個のうち 2 個が融合して検出が 2 個になった。

    球どうしは **3 次元では必ずリンク長ぶん離れている**（B1 で 0.24〜0.26、球の半径 0.07）
    ため、3D の連結成分で分ければ確実に分離できる。実測では連結半径 0.01〜0.03 の
    いずれでも同じ 3 個を返し、閾値に鈍感であることを確認した。

    この改良は第3章 3.8 節が「関節間隔やメッシュの外接寸法に対する相対量として
    定義し直すことは残された改良点」と述べていた箇所に対応する。

    link_radius: 連結とみなす距離。メッシュの外接寸法に対する相対量で与える。
    min_frac  : 採用する最小頂点数の割合（色滲みの迷い頂点を捨てる）。
    """
    mask = _marker_mask(mesh, color_rgb, tol)
    if mask is None:
        return np.empty((0, 3))

    P = np.asarray(mesh.vertices[mask], dtype=float)
    scale = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    r = link_radius * scale

    tree = cKDTree(P)
    parent = np.arange(len(P))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, js in enumerate(tree.query_ball_point(P, r)):
        ri = find(i)
        for j in js:
            rj = find(j)
            if ri != rj:
                parent[rj] = ri
                ri = find(ri)

    lab = np.array([find(i) for i in range(len(P))])
    min_count = max(10, int(min_frac * len(P)))
    centres = [P[lab == u].mean(axis=0)
               for u in np.unique(lab) if (lab == u).sum() >= min_count]
    if not centres:
        return np.empty((0, 3))
    return np.array(sorted(centres, key=lambda c: c[2]))


def _detect_joint_z(mesh: trimesh.Trimesh, color_rgb=(255, 0, 255), tol=40, gap=0.05,
                    min_frac=0.02):
    """
    magenta マーカーの頂点から関節 Z 位置を検出する。
    Returns sorted list of float (Z-up frame).

    min_frac: クラスタとして採用する最小頂点数の割合（マーカー頂点総数比）。
    テクスチャの色滲みによる迷い頂点（実測: パドル先端に1頂点）が
    偽クラスタを作るのを防ぐ。
    """
    try:
        vc = mesh.visual.vertex_colors[:, :3].astype(int)
    except Exception:
        return []

    r, g, b = color_rgb
    mask = ((np.abs(vc[:, 0] - r) <= tol) &
            (np.abs(vc[:, 1] - g) <= tol) &
            (np.abs(vc[:, 2] - b) <= tol))
    if not mask.any():
        return []

    zv = np.sort(mesh.vertices[mask, 2])
    clusters, cur = [], [zv[0]]
    for z in zv[1:]:
        if z - cur[-1] > gap:
            clusters.append(cur)
            cur = []
        cur.append(z)
    clusters.append(cur)

    min_count = max(10, int(min_frac * mask.sum()))
    clusters = [c for c in clusters if len(c) >= min_count]

    return sorted(float(np.mean(c)) for c in clusters)


def _joint_xyz(mesh: trimesh.Trimesh, jz: float, color_rgb, tol: int):
    """関節 Z に対応するマーカーの XY 重心を返す。"""
    try:
        vc = mesh.visual.vertex_colors[:, :3].astype(int)
        r, g, b = color_rgb
        mask_c = ((np.abs(vc[:, 0] - r) <= tol) &
                  (np.abs(vc[:, 1] - g) <= tol) &
                  (np.abs(vc[:, 2] - b) <= tol))
        mask_z = np.abs(mesh.vertices[:, 2] - jz) < 0.10
        near = mesh.vertices[mask_c & mask_z]
        if len(near) > 0:
            return near.mean(axis=0)[:2]
    except Exception:
        pass
    return np.array([0.0, 0.0])


def _split_by_z(mesh: trimesh.Trimesh, boundaries):
    """
    フェース重心の Z 値で mesh を len(boundaries)+1 個に分割。
    Returns list of trimesh.Trimesh.
    """
    cz = mesh.vertices[mesh.faces].mean(axis=1)[:, 2]
    limits = [-np.inf] + list(boundaries) + [np.inf]
    segments = []
    for lo, hi in zip(limits[:-1], limits[1:]):
        mask = (cz >= lo) & (cz < hi)
        if not mask.any():
            raise ValueError(
                f"Z=[{lo:.3f}, {hi:.3f}) にフェースが存在しません。"
                "関節位置（--joints）を確認してください。"
            )
        faces = mesh.faces[mask]
        used, inv = np.unique(faces, return_inverse=True)
        seg = trimesh.Trimesh(vertices=mesh.vertices[used],
                              faces=inv.reshape(-1, 3),
                              process=False)
        try:
            seg.visual.vertex_colors = mesh.visual.vertex_colors[used]
        except Exception:
            pass
        segments.append(seg)
    return segments


def _write_urdf(link_names, frame_origins, joint_globals, urdf_path, mesh_rel):
    """Choreonoid 用 URDF を生成する。"""
    colors = [
        ("red",    "0.7 0.2 0.2 1.0"),
        ("blue",   "0.2 0.2 0.8 1.0"),
        ("green",  "0.2 0.7 0.2 1.0"),
        ("yellow", "0.8 0.8 0.2 1.0"),
    ]
    lines = ['<?xml version="1.0"?>', '<robot name="tripo_arm">', '', '  <link name="world"/>']

    # world → link_0: continuous Z 回転（ベース旋回）
    lines += [
        '',
        '  <joint name="world_to_base" type="continuous">',
        '    <parent link="world"/>',
        f'    <child link="{link_names[0]}"/>',
        '    <origin xyz="0.000000 0.000000 0.020000" rpy="0 0 0"/>',
        '    <axis xyz="0 0 1"/>',
        '    <limit effort="100" velocity="5"/>',
        '  </joint>',
    ]

    for i, name in enumerate(link_names):
        cname, crgba = colors[i % len(colors)]
        lines += [
            '',
            f'  <!-- ===== {name} ===== -->',
            f'  <link name="{name}">',
            '    <visual>',
            '      <origin xyz="0 0 0" rpy="0 0 0"/>',
            '      <geometry>',
            f'        <mesh filename="{mesh_rel}/{name}.stl"/>',
            '      </geometry>',
            f'      <material name="{cname}"><color rgba="{crgba}"/></material>',
            '    </visual>',
            '    <collision>',
            '      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <geometry><mesh filename="{mesh_rel}/{name}.stl"/></geometry>',
            '    </collision>',
            '  </link>',
        ]
        if i < len(link_names) - 1:
            # 関節原点 = joint_global - parent_frame_global
            jg = joint_globals[i]
            fo = frame_origins[i]
            ox, oy, oz = jg[0] - fo[0], jg[1] - fo[1], jg[2] - fo[2]
            lines += [
                '',
                f'  <joint name="joint_to_{link_names[i+1]}" type="revolute">',
                f'    <parent link="{name}"/>',
                f'    <child link="{link_names[i+1]}"/>',
                f'    <origin xyz="{ox:.6f} {oy:.6f} {oz:.6f}" rpy="0 0 0"/>',
                '    <axis xyz="0.0000 1.0000 0.0000"/>',
                '    <limit lower="-1.5708" upper="1.5708" effort="100" velocity="5"/>',
                '  </joint>',
            ]

    lines += ['', '</robot>', '']
    Path(urdf_path).parent.mkdir(parents=True, exist_ok=True)
    Path(urdf_path).write_text('\n'.join(lines), encoding='utf-8')
    print(f"[glb_to_links] URDF → {urdf_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--glb', required=True)
    p.add_argument('--out-dir', required=True, help='STL 出力ディレクトリ')
    p.add_argument('--urdf', default=None, help='URDF 出力パス（省略時はスキップ）')
    p.add_argument('--joint-color', nargs=3, type=int, default=[255, 0, 255],
                   metavar=('R', 'G', 'B'))
    p.add_argument('--joint-tol', type=int, default=40)
    p.add_argument('--cluster-mode', choices=['3d', 'z'], default='3d',
                   help='マーカーのクラスタリング方法。既定の 3d は 3 次元の連結成分で分ける。'
                        'z は旧方式（Z 座標のみ）で、リンクが短く腕が傾いていると'
                        '隣り合う球が融合する（2026-09-02、B1 で実測）')
    p.add_argument('--cluster-radius', type=float, default=0.02,
                   help='3d モードで連結とみなす距離。メッシュの外接寸法に対する相対量')
    p.add_argument('--auto-color', action='store_true',
                   help='マーカー色をメッシュから自動キャリブレーション'
                        '（マゼンタ色相域 285-345° の支配クラスタを検出。'
                        'Tripo3D の暗色化 #FF00FF→≈[211,75,169] に自動追従）')
    p.add_argument('--joints', nargs='+', type=float, default=None,
                   help='手動指定 関節 Z 位置 [m]（Z-up）。--joints を省略すると magenta 自動検出')
    p.add_argument('--names', nargs='+', default=None,
                   help='リンク名（デフォルト: link_0, link_1, ...）')
    p.add_argument('--link-rot', nargs=2, action='append', default=None,
                   metavar=('NAME', 'DEG'),
                   help='指定リンクをローカル Z 軸（ボーン軸）周りに回転 [deg]。'
                        '例: --link-rot hand 90 — エンドエフェクタの面の向き修正用'
                        '（生成メッシュのヘラ面がスイング平面と平行=チョップ向きの場合に使う）')
    args = p.parse_args()

    print(f"[glb_to_links] 読み込み: {args.glb}")
    mesh = _load_concat(args.glb)
    print(f"  頂点={len(mesh.vertices)}  面={len(mesh.faces)}")

    print("[glb_to_links] Y-up → Z-up 変換...")
    mesh = _apply_yup_zup(mesh)

    color_rgb = tuple(args.joint_color)
    joint_tol = args.joint_tol
    if args.auto_color and not args.joints:
        print("[glb_to_links] マーカー色の自動キャリブレーション...")
        auto_c, auto_t = _auto_calibrate_joint_color(mesh)
        if auto_c is not None:
            color_rgb, joint_tol = auto_c, auto_t
        else:
            print("  [auto-color] マゼンタ色相域クラスタなし → 指定色にフォールバック "
                  f"{color_rgb} (tol={joint_tol})")
    joint_pts = None
    if args.joints:
        joint_z_vals = sorted(args.joints)
        print(f"[glb_to_links] 関節 Z（手動）: {joint_z_vals}")
    else:
        print(f"[glb_to_links] マーカー検出 color={color_rgb} (tol={joint_tol}), "
              f"mode={args.cluster_mode}...")
        if args.cluster_mode == '3d':
            joint_pts = _detect_joints_3d(mesh, color_rgb, joint_tol,
                                          link_radius=args.cluster_radius)
            joint_z_vals = [float(p[2]) for p in joint_pts]
        else:
            joint_pts = None
            joint_z_vals = _detect_joint_z(mesh, color_rgb, joint_tol)
        if not joint_z_vals:
            p.error(
                "magenta マーカーが見つかりません。\n"
                "  → --joint-tol を大きくする\n"
                "  → --joints Z1 Z2 ... で手動指定\n"
                "  → --joint-color R G B でマーカー色を変更"
            )
        print(f"  検出 Z 位置: {[f'{z:.4f}m' for z in joint_z_vals]}")

    n_links = len(joint_z_vals) + 1
    names = args.names or [f'link_{i}' for i in range(n_links)]
    if len(names) != n_links:
        p.error(f'--names の数 ({len(names)}) が リンク数 ({n_links}) と一致しません')

    print(f"[glb_to_links] 空間分割 → {n_links} リンク...")
    segments = _split_by_z(mesh, joint_z_vals)

    z_min = float(mesh.vertices[:, 2].min())
    if joint_pts is not None and len(joint_pts) == len(joint_z_vals):
        # 3D クラスタリング済みなら重心をそのまま使う（XY を測り直さない）
        joint_xyz = [np.asarray(p, dtype=float) for p in joint_pts]
    else:
        joint_xyz = []
        for jz in joint_z_vals:
            jxy = _joint_xyz(mesh, jz, color_rgb, args.joint_tol)
            joint_xyz.append(np.array([jxy[0], jxy[1], jz]))

    # Bug 28（2026-09-02）: 根元リンクのローカル原点を XY=(0,0) 決め打ちにしていた。
    # GLB の原点が台座の真下にあるとは限らない。A1 では台座が x∈[-0.332,-0.157] にあり、
    # 原点が形状の 0.157 m **外**に出たため、ヨー回転で台座が自分の外側を公転した。
    # 第1関節の XY（= ヨー軸が通る位置）を使う。台座は第1関節の真下にあるはずで、
    # 実測でも両者の XY のずれは A1 で 3.6 mm、v2c で 0.5 mm しかない。
    if joint_xyz:
        base_xy = (float(joint_xyz[0][0]), float(joint_xyz[0][1]))
    else:                                   # 関節が無い場合は形状の XY 重心へ
        c = mesh.vertices[:, :2].mean(axis=0)
        base_xy = (float(c[0]), float(c[1]))
    frame_origins = [np.array([base_xy[0], base_xy[1], z_min])] + joint_xyz

    joint_globals = frame_origins[1:]  # = frame_origins of child links

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    link_rots = {n: float(d) for n, d in (args.link_rot or [])}
    for n in link_rots:
        if n not in names:
            p.error(f'--link-rot のリンク名 "{n}" が --names に存在しません: {names}')

    for name, seg, fo in zip(names, segments, frame_origins):
        verts = np.asarray(seg.vertices) - fo
        if name in link_rots:
            th = np.deg2rad(link_rots[name])
            c, s = np.cos(th), np.sin(th)
            rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            verts = verts @ rz.T
            print(f"  [{name}] ローカル Z 軸周りに {link_rots[name]:.1f}° 回転")
        local = trimesh.Trimesh(vertices=verts, faces=seg.faces, process=False)
        stl_path = out_dir / f'{name}.stl'
        local.export(str(stl_path))
        ext = local.bounding_box.extents
        print(f"  [{name}] extents={ext.round(3)} → {stl_path}")

    # Bug 27（2026-09-02）: 分割境界はマーカー球の中心にあるため、各リンクの分割片は
    # 両端の球の半分ずつを含む。腕が傾いているとこの半球がリンク軸方向へも張り出し、
    # 下流（mesh_to_params）の OBB 主軸長が 18〜20 % 過大になる。
    # 関節の 3D 座標はここで既に求まっているので、**関節間距離**を JSON で下流へ渡す。
    # 子関節を持たない先端リンクだけは距離が定義できないため OBB のままとする。
    joints_json = out_dir / 'joints.json'
    link_len = {}
    for i, name in enumerate(names[:-1]):
        link_len[name] = float(np.linalg.norm(frame_origins[i + 1] - frame_origins[i]))

    # 先端リンクは子関節が無いので関節間距離が定義できない。当初は OBB 主軸長のまま
    # 通していたが（A1 では誤差 +0.7 %）、**リンクが短く球が相対的に大きいと破綻する**。
    # B1 では OBB 0.4316 m に対し実距離 0.2828 m（+52.6 %）だった。
    # 分割面が球の中心を通るため上半球が全方向へ張り出し、腕が傾いていると
    # OBB の主軸がその張り出しを拾ってしまう（Bug 27 の先端リンク版、2026-09-02）。
    # 運動学的な寄与は「最後の関節から最も遠い点までの距離」なのでそれを使う。
    if len(names) >= 2 and len(segments) == len(names):
        tip_v = np.asarray(segments[-1].vertices, dtype=float)
        if len(tip_v):
            link_len[names[-1]] = float(
                np.linalg.norm(tip_v - frame_origins[-1], axis=1).max())
    joints_json.write_text(json.dumps({
        'frame_origins': {n: fo.tolist() for n, fo in zip(names, frame_origins)},
        'joint_positions': [j.tolist() for j in joint_globals],
        # 子関節を持つリンクの「関節間距離」。先端リンクは含まない（OBB を使う）
        'link_lengths': link_len,
    }, indent=2), encoding='utf-8')
    print(f"[glb_to_links] 関節間距離 → {joints_json}")
    for n, d in link_len.items():
        print(f"    {n}: {d:.4f} m")

    if args.urdf:
        try:
            mesh_rel = str(out_dir.relative_to(Path(args.urdf).parent))
        except ValueError:
            mesh_rel = str(out_dir)
        _write_urdf(names, frame_origins, joint_globals, args.urdf, mesh_rel)

    print(f"\n[glb_to_links] 完了 ({n_links} リンク) → {out_dir}/")
    parts_str = ' '.join(str(out_dir / f'{n}.stl') for n in names)
    print(f"  次: python3 scripts/mesh_to_params.py --parts {parts_str} --output ...")


if __name__ == '__main__':
    main()
