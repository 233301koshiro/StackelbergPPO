#!/usr/bin/env python3
"""
mesh_to_params.py: 分割済み GLB メッシュ → topology.json パラメータ抽出

Blender 等で手動分割したリンク GLB から、各リンクの
  - 長さ (bone_offset)
  - 半径 (geom.size、カプセル近似)
を OBB (Oriented Bounding Box) で計算し topology.json として出力する。

使い方:
  # 2関節アーム（上腕・前腕の2パーツ）
  python3 scripts/mesh_to_params.py \\
    --parts upper_arm.glb forearm.glb \\
    --names upper_arm forearm \\
    --output data/my_robot/my_topology.json

  # スケール指定（mm → m など）
  python3 scripts/mesh_to_params.py \\
    --parts link1.glb link2.glb \\
    --scale 0.001 \\
    --output data/my_robot/my_topology.json

  # 関節可動域・アクチュエータゲインを個別指定
  python3 scripts/mesh_to_params.py \\
    --parts upper_arm.glb forearm.glb \\
    --ranges '-60 60' '-90 90' \\
    --gears 150 100 \\
    --output data/my_robot/my_topology.json

引数:
  --parts     GLBファイルパスのリスト（根元→先端の順）
  --names     各リンクの名前（デフォルト: link0, link1, ...）
  --output    出力 topology.json のパス
  --scale     長さの単位変換係数（デフォルト: 1.0、mm→m なら 0.001）
  --ranges    各関節の可動域 [deg]（デフォルト: -90 90）
  --gears     各関節のアクチュエータゲイン（デフォルト: 100）
  --validate  出力 JSON から topology_to_xml.py で XML を生成し検証

出力 topology.json のスキーマは topology_to_xml.py / dynamic_body_updater.py
と共通。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh


# ---------- OBB からリンクパラメータを抽出 ----------

def load_mesh(path: str) -> trimesh.Trimesh:
    scene_or_mesh = trimesh.load(path, force='mesh')
    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = list(scene_or_mesh.geometry.values())
        if not meshes:
            raise ValueError(f"{path}: geometry が空です")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene_or_mesh
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{path}: Trimesh に変換できません（type={type(mesh)}）")
    return mesh


def obb_params(mesh: trimesh.Trimesh, scale: float) -> dict:
    """
    OBB から以下を抽出して返す。
      length  : 最長辺の長さ [m]（bone_offset の大きさになる）
      radius  : 短辺2辺の平均半径 [m]（カプセル近似）
      obb_axes: OBB の軸ベクトル3本（列ベクトル、長い順）
      center  : OBB 中心 [m]
    """
    obb = mesh.bounding_box_oriented
    extents = np.array(obb.primitive.extents)     # [ex, ey, ez] 各辺の長さ
    transform = np.array(obb.primitive.transform)  # 4x4、回転部分が軸方向

    # スケール適用
    extents = extents * scale

    # 長い順にソート
    order = np.argsort(extents)[::-1]  # 長い→短い
    extents_sorted = extents[order]

    # 軸ベクトル（transform の回転部分の列 = OBB の辺方向）
    rot = transform[:3, :3]
    axes = [rot[:, order[i]] for i in range(3)]  # 長い順

    length = float(extents_sorted[0])
    # カプセル半径 = 短辺2本の平均 / 2
    radius = float((extents_sorted[1] + extents_sorted[2]) / 4.0)

    center = np.array(obb.primitive.transform[:3, 3]) * scale

    return dict(length=length, radius=radius,
                obb_axes=axes, center=center,
                extents=extents_sorted.tolist())


def bone_offset_from_obb(params: dict) -> list:
    """
    OBB の最長軸をタスク座標 X 軸にマッピングして bone_offset を返す。
    骨格の「長さ方向」を常に +x にそろえる。

    Note: メッシュが斜めに置かれていても OBB で吸収される。
    ただし最長辺が「長さ方向」になるよう、GLB 作成時にリンクを
    X/Y/Z いずれかの軸に沿わせておくことを推奨。
    """
    length = params['length']
    # 短辺と長辺の比が 1.5 未満の場合（ほぼ立方体）は警告
    extents = params['extents']
    if extents[0] < extents[1] * 1.5:
        import sys
        print(f"  ⚠️  警告: 最長辺 ({extents[0]:.3f}m) と次辺 ({extents[1]:.3f}m) の差が小さい。"
              f" リンクが正しく向いているか確認してください。", file=sys.stderr)
    return [round(length, 4), 0.0, 0.0]


# ---------- topology.json 組み立て ----------

def build_topology(parts: list, names: list, scale: float,
                   ranges: list, gears: list,
                   output_path: str) -> dict:
    """
    parts  : [(path, mesh), ...]  根元→先端順
    names  : [str, ...]
    ranges : [[lo, hi], ...]  各関節の可動域 [deg]
    gears  : [float, ...]
    """
    bodies = []
    for i, (path, mesh) in enumerate(parts):
        params = obb_params(mesh, scale)
        boff   = bone_offset_from_obb(params)
        radius = round(params['radius'], 4)
        parent = 'root' if i == 0 else names[i - 1]

        lo, hi = ranges[i]
        gear   = gears[i]

        print(f"  [{names[i]}]")
        print(f"    OBB extents (scaled): {[f'{e:.4f}' for e in params['extents']]} m")
        print(f"    bone_offset: {boff}")
        print(f"    capsule radius: {radius:.4f} m")
        print(f"    joint range: [{lo}, {hi}] deg  gear: {gear}")

        bodies.append({
            "name": names[i],
            "parent": parent,
            "joint": {
                "type": "hinge",
                "axis": [0.0, 0.0, 1.0],
                "range": [float(lo), float(hi)],
                "note": f"Z軸回転（水平 XY 面プッシャータスク用）。元メッシュ: {Path(path).name}"
            },
            "bone_offset": boff,
            "geom": {
                "type": "capsule",
                "size": radius,
                "ext_start": 0.0,
                "source_mesh": str(path),
            },
            "actuator": {"gear": float(gear)},
        })

    topo = {
        "description": f"{len(parts)}-joint serial arm (from mesh)",
        "bodies": bodies,
        "stackelberg_param_bounds": {
            "bone_offset_xy": {"lb": [-0.5, -0.5], "ub": [0.5, 0.5]},
            "geom_size":      {"lb": 0.03, "ub": 0.10},
            "geom_ext_start": {"lb": 0.0,  "ub": 0.2},
            "actuator_gear":  {"lb": 20,   "ub": 400}
        }
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(topo, f, indent=2, ensure_ascii=False)
    print(f"\n[mesh_to_params] → {out}")
    return topo


# ---------- CLI ----------

def parse_range(s: str) -> list:
    parts = s.strip().split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"range は '下限 上限' の形式で指定してください: {s!r}")
    return [float(parts[0]), float(parts[1])]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--parts',    nargs='+', required=True,
                        help='GLB ファイルパス（根元→先端の順）')
    parser.add_argument('--names',    nargs='+', default=None,
                        help='各リンクの名前（デフォルト: link0, link1, ...）')
    parser.add_argument('--output',   required=True,
                        help='出力 topology.json のパス')
    parser.add_argument('--scale',    type=float, default=1.0,
                        help='長さの単位変換係数（mm→m なら 0.001）')
    parser.add_argument('--ranges',   nargs='+', default=None, type=parse_range,
                        help="各関節の可動域 e.g. '-60 60' '-90 90'")
    parser.add_argument('--gears',    nargs='+', type=float, default=None,
                        help='各関節のアクチュエータゲイン（デフォルト: 100）')
    parser.add_argument('--validate', action='store_true',
                        help='生成した JSON から XML を生成して MuJoCo で検証')
    args = parser.parse_args()

    n = len(args.parts)
    names  = args.names  or [f'link{i}' for i in range(n)]
    ranges = args.ranges or [[-90.0, 90.0]] * n
    gears  = args.gears  or [100.0] * n

    if len(names) != n:
        parser.error(f'--names の数 ({len(names)}) が --parts の数 ({n}) と一致しません')
    if len(ranges) != n:
        parser.error(f'--ranges の数 ({len(ranges)}) が --parts の数 ({n}) と一致しません')
    if len(gears) != n:
        parser.error(f'--gears の数 ({len(gears)}) が --parts の数 ({n}) と一致しません')

    print(f"[mesh_to_params] {n} パーツを読み込み中...")
    parts = []
    for path in args.parts:
        print(f"  読み込み: {path}")
        mesh = load_mesh(path)
        print(f"    頂点数={len(mesh.vertices)}  面数={len(mesh.faces)}")
        parts.append((path, mesh))

    print(f"\n[mesh_to_params] OBB からパラメータを抽出中（scale={args.scale}）...")
    topo = build_topology(parts, names, args.scale, ranges, gears, args.output)

    if args.validate:
        print("\n[mesh_to_params] --validate: MuJoCo XML を生成して検証中...")
        import subprocess, tempfile, os
        with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as tf:
            xml_path = tf.name
        try:
            result = subprocess.run(
                [sys.executable, 'scripts/topology_to_xml.py',
                 '--topology', args.output,
                 '--output',   xml_path,
                 '--validate'],
                capture_output=True, text=True
            )
            print(result.stdout)
            if result.returncode != 0:
                print(result.stderr)
                print('[validate] NG')
            else:
                print('[validate] OK')
        finally:
            os.unlink(xml_path)


if __name__ == '__main__':
    main()
