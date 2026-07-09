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

import numpy as np
import trimesh

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
    if args.joints:
        joint_z_vals = sorted(args.joints)
        print(f"[glb_to_links] 関節 Z（手動）: {joint_z_vals}")
    else:
        print(f"[glb_to_links] magenta マーカー検出 (tol={args.joint_tol})...")
        joint_z_vals = _detect_joint_z(mesh, color_rgb, args.joint_tol)
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
    frame_origins = [np.array([0.0, 0.0, z_min])]
    for jz in joint_z_vals:
        jxy = _joint_xyz(mesh, jz, color_rgb, args.joint_tol)
        frame_origins.append(np.array([jxy[0], jxy[1], jz]))

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
