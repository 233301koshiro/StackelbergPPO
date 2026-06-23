#!/usr/bin/env python3
"""
dynamic_body_updater.py: topology.json → Choreonoid .body 変換・動的更新モジュール

役割:
  - topology.json を読み込み、Choreonoid .body YAML を直接生成する
  - 学習ループ内で bone_offset が更新されるたびに .body を再生成する API を提供する
  - ChoreonoidSimWorld.load_model() が受け取る body_defs 形式 [(name, yaml_str)] を返す

topology_to_xml.py との分業:
  topology_to_xml.py   → topology.json → MuJoCo XML  (StackelbergPPO の学習入力)
  dynamic_body_updater → topology.json → .body YAML   (Choreonoid のロード入力)

使い方:
  # スタンドアロン CLI
  python3 scripts/dynamic_body_updater.py \\
    --topology data/rrbot_description/rrbot_topology.json \\
    --output-dir /tmp/body_out/

  # Python API（学習ループ内）
  from scripts.dynamic_body_updater import DynamicBodyUpdater
  updater = DynamicBodyUpdater("data/rrbot_description/rrbot_topology.json")
  body_defs = updater.generate_body_defs()   # → [(name, yaml_str), ...]
  updater.update({"upper_arm": [0.35, 0.0, 0.0]})  # bone_offset を更新
  body_defs = updater.generate_body_defs()   # 再生成

  # ChoreonoidSimWorld との統合
  sim_world.load_model_from_body_defs(body_defs, frame_skip=4)
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

# ---------- topology.json 読み込み（topology_to_xml.py と同じロジック） ----------

DEFAULT_ROOT = {
    "fixed_base": True,
    "pos": [0.0, 0.0, 0.15],
    "geom": {"type": "sphere", "size": 0.1},
}

DEFAULT_CUBE = {
    "pos": [1.0, 0.0, 0.20],
    "size": 0.15,
    "slide_range": [-2.0, 200.0],
    "slide2_range": [-100.0, 100.0],
    "damping": 2.0,
    "density": 100,
}


def _load_topology(path: str) -> dict:
    import json
    with open(path) as f:
        topo = json.load(f)

    root_cfg = dict(DEFAULT_ROOT)
    root_cfg.update(topo.get("root", {}))
    root_cfg["geom"] = dict(DEFAULT_ROOT["geom"])
    root_cfg["geom"].update(topo.get("root", {}).get("geom", {}))
    topo["root"] = root_cfg

    for b in topo.get("bodies", []):
        j = b.get("joint")
        if j:
            if "adapted_axis" in j and "axis" not in j:
                j["axis"] = j["adapted_axis"]
            if "range_deg" in j and "range" not in j:
                j["range"] = j["range_deg"]

    task = topo.get("task", {})
    cube_cfg = dict(DEFAULT_CUBE)
    cube_cfg.update(task.get("cube", {}))
    task["cube"] = cube_cfg
    topo["task"] = task

    return topo


# ---------- 物理量計算（mujoco_env_choreonoid.py と同等、独立実装） ----------

def _capsule_inertia(length: float, radius: float, density: float):
    """カプセル（Y軸方向）の慣性テンソルを返す (mass, Iperp, Iaxial)"""
    r, l = radius, length
    m_cyl = density * math.pi * r**2 * l
    m_cap = density * (4.0 / 3.0) * math.pi * r**3
    m = m_cyl + m_cap
    d_hemi = l / 2.0 - 3.0 * r / 8.0
    Iperp = m_cyl * (r**2 / 4.0 + l**2 / 12.0) + m_cap * (2.0 * r**2 / 5.0 + d_hemi**2)
    Iaxial = m * r**2 / 2.0
    return m, Iperp, Iaxial  # Y軸カプセル: Ixx=Izz=Iperp, Iyy=Iaxial


def _sphere_inertia(radius: float, density: float):
    m = density * (4.0 / 3.0) * math.pi * radius**3
    I = 0.4 * m * radius**2
    return m, I


def _rot_y_to_vec(d) -> tuple:
    """Y軸を方向 d に向ける回転の (axis, angle_deg) を返す"""
    d = np.asarray(d, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        return [1.0, 0.0, 0.0], 0.0
    d = d / norm
    y = np.array([0.0, 1.0, 0.0])
    dot = float(np.clip(np.dot(y, d), -1.0, 1.0))
    if abs(dot - 1.0) < 1e-9:
        return [1.0, 0.0, 0.0], 0.0
    if abs(dot + 1.0) < 1e-9:
        return [1.0, 0.0, 0.0], 180.0
    angle_deg = math.degrees(math.acos(dot))
    axis = np.cross(y, d)
    axis /= np.linalg.norm(axis)
    return axis.tolist(), angle_deg


def _inertia9(m: float, Iperp: float, Iaxial: float, shape_type: str) -> list:
    """3x3慣性テンソルを9要素リストで返す（Y軸カプセル想定）"""
    if shape_type == "capsule":
        # capsule along Y → Ixx=Izz=Iperp, Iyy=Iaxial
        return [Iperp, 0, 0, 0, Iaxial, 0, 0, 0, Iperp]
    else:
        return [Iperp, 0, 0, 0, Iperp, 0, 0, 0, Iperp]


# ---------- topology.json からlinkリストを構築 ----------

def _make_link(name, parent, jtype, jname, jaxis, jrange, translation,
               mass, com, inertia, shape, joint_id, damping=None):
    return dict(name=name, parent=parent, jtype=jtype, jname=jname,
                jaxis=jaxis, jrange=jrange, translation=translation,
                mass=mass, com=com, inertia=inertia, shape=shape,
                joint_id=joint_id, damping=damping)


def _body_local_pos(body: dict, body_map: dict) -> list:
    """親ローカル座標でのjoint位置（= 親のbone_offset）"""
    parent_name = body.get("parent", "root")
    if parent_name == "root":
        return [0.0, 0.0, 0.0]
    parent = body_map[parent_name]
    return list(parent.get("bone_offset", [0.0, 0.0, 0.0]))


def _capsule_shape(body: dict, density: float) -> dict:
    """bodyのgeomからcapsule形状dictを返す"""
    bone_offset = np.array(body.get("bone_offset", [0.3, 0.0, 0.0]), dtype=float)
    ext_start = body.get("geom", {}).get("ext_start", 0.0)
    radius = float(body.get("geom", {}).get("size", 0.05))

    norm = np.linalg.norm(bone_offset)
    bone_dir = bone_offset / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0])
    p0 = bone_dir * ext_start
    p1 = bone_offset
    center = (p0 + p1) / 2.0
    diff = p1 - p0
    length = float(np.linalg.norm(diff))

    m, Iperp, Iaxial = _capsule_inertia(length, radius, density)
    rot_axis, rot_angle = _rot_y_to_vec(diff)
    return dict(type="capsule", center=center.tolist(), length=length,
                radius=radius, rot_axis=rot_axis, rot_angle=rot_angle,
                mass=m, Iperp=Iperp, Iaxial=Iaxial)


def _sphere_shape(root_cfg: dict, density: float) -> dict:
    geom = root_cfg.get("geom", {})
    radius = float(geom.get("size", 0.1))
    m, I = _sphere_inertia(radius, density)
    return dict(type="sphere", center=[0.0, 0.0, 0.0], radius=radius, mass=m, I=I)


def _build_links(topo: dict, density: float) -> list:
    """
    topology.json から link dict のリストを構築する。
    mujoco_xml_to_body の links と同じ構造。
    """
    bodies = topo.get("bodies", [])
    body_map = {b["name"]: b for b in bodies}
    root_cfg = topo["root"]
    task = topo.get("task", {})
    cube_cfg = task.get("cube", {})

    links = []
    joint_id_ctr = [0]

    def next_jid():
        jid = joint_id_ctr[0]
        joint_id_ctr[0] += 1
        return jid

    # ---- root ----
    shape = _sphere_shape(root_cfg, density)
    m, I = shape["mass"], shape["I"]
    links.append(_make_link(
        name="0",
        parent=None,
        jtype="fixed" if root_cfg.get("fixed_base", True) else "free",
        jname="root_joint" if not root_cfg.get("fixed_base", True) else None,
        jaxis=None,
        jrange=None,
        translation=list(root_cfg.get("pos", [0.0, 0.0, 0.15])),
        mass=m,
        com=[0.0, 0.0, 0.0],
        inertia=[I, 0, 0, 0, I, 0, 0, 0, I],
        shape=shape,
        joint_id=-1,
    ))

    # ---- bodies (DFS order, parent="root" → parent="0") ----
    def get_children(parent_name):
        return [b for b in bodies if b.get("parent") == parent_name]

    def assign_id(body_name, parent_id, sibling_idx):
        pname = "" if parent_id == "0" else parent_id
        return str(sibling_idx + 1) + pname

    def process_body(body_name, parent_mujoco_id, sibling_idx):
        body = body_map[body_name]
        mid = assign_id(body_name, parent_mujoco_id, sibling_idx)

        local_pos = _body_local_pos(body, body_map)

        geom_type = body.get("geom", {}).get("type", "capsule")
        if geom_type == "capsule":
            shape = _capsule_shape(body, density)
            mass = shape["mass"]
            com = shape["center"]
            inertia = _inertia9(mass, shape["Iperp"], shape["Iaxial"], "capsule")
        else:
            shape = None
            mass = 0.001
            com = [0.0, 0.0, 0.0]
            inertia = [1e-6, 0, 0, 0, 1e-6, 0, 0, 0, 1e-6]

        j = body.get("joint", {})
        jtype = "revolute" if j.get("type", "hinge") == "hinge" else j.get("type", "revolute")
        jname = f"{mid}_joint"
        jaxis = j.get("axis", [0.0, 0.0, 1.0])
        jrange = j.get("range", [-60.0, 60.0])
        jid = next_jid()

        links.append(_make_link(
            name=mid,
            parent=parent_mujoco_id,
            jtype=jtype,
            jname=jname,
            jaxis=jaxis,
            jrange=jrange,
            translation=local_pos,
            mass=mass,
            com=com,
            inertia=inertia,
            shape=shape,
            joint_id=jid,
        ))

        children = get_children(body_name)
        for i, child in enumerate(children):
            process_body(child["name"], mid, i)

    root_children = get_children("root")
    for i, child in enumerate(root_children):
        process_body(child["name"], "0", i)

    return links


def _build_cube_links(cube_cfg: dict, density: float = 100.0) -> list:
    """
    cubeボディのlinkリスト生成。
    slide x2 なので: fixed_root → virt (slide x) → cube (slide y + box)
    """
    pos = list(cube_cfg.get("pos", [1.0, 0.0, 0.20]))
    size = float(cube_cfg.get("size", 0.15))
    slide_range = cube_cfg.get("slide_range", [-2.0, 200.0])
    slide2_range = cube_cfg.get("slide2_range", [-100.0, 100.0])
    d_val = float(cube_cfg.get("density", density))

    m = d_val * (2 * size) ** 3
    Ixx = m * (size**2 + size**2) / 3.0  # box half-size = size, full = 2*size
    box_shape = dict(type="box", center=[0.0, 0.0, 0.0],
                     size=[2*size, 2*size, 2*size],
                     mass=m, Ixx=Ixx, Iyy=Ixx, Izz=Ixx)

    links = []
    # fixed root （ダミー：Choreonoidはrootがfixedでもjoint_idカウントを正しく保つ）
    links.append(_make_link(
        name="cube_fixed_root",
        parent=None,
        jtype="fixed",
        jname=None,
        jaxis=None,
        jrange=None,
        translation=pos,
        mass=0.001,
        com=[0.0, 0.0, 0.0],
        inertia=[1e-6, 0, 0, 0, 1e-6, 0, 0, 0, 1e-6],
        shape=None,
        joint_id=-1,
    ))
    damping = float(cube_cfg.get("damping", 2.0))
    # cube_virt0: slide x
    links.append(_make_link(
        name="cube_virt0",
        parent="cube_fixed_root",
        jtype="prismatic",
        jname="cube_slide",
        jaxis=[1.0, 0.0, 0.0],
        jrange=list(slide_range),
        translation=[0.0, 0.0, 0.0],
        mass=0.001,
        com=[0.0, 0.0, 0.0],
        inertia=[1e-6, 0, 0, 0, 1e-6, 0, 0, 0, 1e-6],
        shape=None,
        joint_id=0,
        damping=damping,
    ))
    # cube: slide y + box shape
    links.append(_make_link(
        name="cube",
        parent="cube_virt0",
        jtype="prismatic",
        jname="cube_slide2",
        jaxis=[0.0, 1.0, 0.0],
        jrange=list(slide2_range),
        translation=[0.0, 0.0, 0.0],
        mass=m,
        com=[0.0, 0.0, 0.0],
        inertia=[Ixx, 0, 0, 0, Ixx, 0, 0, 0, Ixx],
        shape=box_shape,
        joint_id=1,
        damping=damping,
    ))
    return links


# ---------- .body YAML シリアライズ ----------

def _fv(v, prec=8) -> str:
    return "[ " + ", ".join(f"{x:.{prec}g}" for x in v) + " ]"


def _serialize_links_to_yaml(links: list, root_name: str, body_name: str) -> str:
    out = [
        "format: ChoreonoidBody",
        "format_version: 2.0",
        "angle_unit: degree",
        f"name: {body_name}",
        f'root_link: "{root_name}"',
        "links:",
    ]
    for lk in links:
        out.append("  -")
        out.append(f'    name: "{lk["name"]}"')
        if lk["parent"] is not None:
            out.append(f'    parent: "{lk["parent"]}"')
        if lk["jname"]:
            out.append(f'    joint_name: {lk["jname"]}')
        out.append(f'    joint_type: {lk["jtype"]}')
        if lk.get("joint_id", -1) >= 0:
            out.append(f'    joint_id: {lk["joint_id"]}')
        if lk["jaxis"] is not None:
            out.append(f'    joint_axis: {_fv(lk["jaxis"], 6)}')
        if lk["jrange"] is not None:
            out.append(f'    joint_range: {_fv(lk["jrange"], 6)}')
        if lk.get("damping") is not None:
            out.append(f'    joint_damping: {lk["damping"]:.6g}')
        out.append(f'    translation: {_fv(lk["translation"])}')
        out.append(f'    mass: {lk["mass"]:.6g}')
        out.append(f'    center_of_mass: {_fv(lk["com"])}')
        m = lk["inertia"]
        out.append("    inertia: [")
        out.append(f"      {m[0]:.6g}, {m[1]:.6g}, {m[2]:.6g},")
        out.append(f"      {m[3]:.6g}, {m[4]:.6g}, {m[5]:.6g},")
        out.append(f"      {m[6]:.6g}, {m[7]:.6g}, {m[8]:.6g} ]")
        shape = lk.get("shape")
        if shape:
            st = shape["type"]
            cx, cy, cz = shape["center"]
            out.append("    elements:")
            out.append("      -")
            out.append("        type: Shape")
            out.append(f"        translation: [ {cx:.6g}, {cy:.6g}, {cz:.6g} ]")
            if st == "capsule":
                ax, ay, az = shape["rot_axis"]
                ang = shape["rot_angle"]
                if abs(ang) > 0.01:
                    out.append(f"        rotation: [ {ax:.6g}, {ay:.6g}, {az:.6g}, {ang:.4g} ]")
                r, h = shape["radius"], shape["length"]
                out.append(f"        geometry: {{ type: Capsule, radius: {r:.6g}, height: {h:.6g} }}")
            elif st == "sphere":
                out.append(f"        geometry: {{ type: Sphere, radius: {shape['radius']:.6g} }}")
            elif st == "box":
                sx, sy, sz = shape["size"]
                out.append(f"        geometry: {{ type: Box, size: [ {sx:.6g}, {sy:.6g}, {sz:.6g} ] }}")
    return "\n".join(out) + "\n"


# ---------- メインクラス ----------

class DynamicBodyUpdater:
    """
    topology.json から Choreonoid .body YAML を生成・動的更新するクラス。

    返す body_defs は ChoreonoidSimWorld.load_model_from_body_defs() に渡せる
    [(item_name, yaml_str), ...] 形式。robot が先頭、cube が後続。
    """

    def __init__(self, topology_path: str, density: float = 5.0):
        self._topo = _load_topology(topology_path)
        self._density = density

    @property
    def topo(self) -> dict:
        return self._topo

    def update(self, bone_offsets: dict):
        """
        bone_offsets: {body_name: [x, y, z]} の辞書でtopology.jsonのbone_offsetを上書きする。
        generate_body_defs() を呼ぶ前に更新する。
        """
        body_map = {b["name"]: b for b in self._topo.get("bodies", [])}
        for name, offset in bone_offsets.items():
            if name in body_map:
                body_map[name]["bone_offset"] = list(offset)

    def generate_body_defs(self) -> list:
        """
        [(item_name, yaml_str), ...] を返す。
        ChoreonoidSimWorld.load_model_from_body_defs() に渡せる形式。
        robot_name が先頭、pusherタスクなら cube が後続。
        """
        robot_links = _build_links(self._topo, self._density)
        robot_name = self._topo.get("description", "robot")[:20].replace(" ", "_")
        robot_yaml = _serialize_links_to_yaml(robot_links, "0", robot_name)

        body_defs = [(robot_name, robot_yaml)]

        task = self._topo.get("task", {})
        if task.get("type", "pusher") == "pusher":
            cube_links = _build_cube_links(task["cube"])
            cube_yaml = _serialize_links_to_yaml(cube_links, "cube_fixed_root", "cube")
            body_defs.append(("cube", cube_yaml))

        return body_defs

    def write(self, output_dir: str) -> list:
        """
        .body ファイルをディレクトリに書き出し、パスのリストを返す。
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        body_defs = self.generate_body_defs()
        paths = []
        for name, yaml_str in body_defs:
            path = out_dir / f"{name}.body"
            path.write_text(yaml_str)
            paths.append(str(path))
        return paths


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="topology.json → Choreonoid .body 変換"
    )
    parser.add_argument("--topology", required=True,
                        help="入力 topology.json のパス")
    parser.add_argument("--output-dir", required=True,
                        help="出力 .body ファイルを格納するディレクトリ")
    parser.add_argument("--density", type=float, default=5.0,
                        help="ロボットボディの密度 [kg/m^3]（デフォルト 5.0）")
    args = parser.parse_args()

    updater = DynamicBodyUpdater(args.topology, density=args.density)
    paths = updater.write(args.output_dir)

    print(f"[dynamic_body_updater] {len(paths)} ファイルを出力:")
    for p in paths:
        print(f"  {p}")

    # 内容の概要を表示
    topo = updater.topo
    bodies = topo.get("bodies", [])
    print(f"  ロボットボディ: root + {len(bodies)} リンク")
    for b in bodies:
        print(f"    {b['name']}  parent={b.get('parent')}  "
              f"bone_offset={b.get('bone_offset')}")
    task = topo.get("task", {})
    if task.get("type", "pusher") == "pusher":
        cube = task["cube"]
        print(f"  cube: pos={cube['pos']}  size={cube['size']}")


if __name__ == "__main__":
    main()
