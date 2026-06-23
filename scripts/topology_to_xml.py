#!/usr/bin/env python3
"""
topology_to_xml.py: topology.json → MuJoCo XML 変換スクリプト

JSON スキーマ:
  root:   {fixed_base:bool, pos:[x,y,z], geom:{type,size}}   (省略可、デフォルトあり)
  bodies: [
    {name, parent("root" or body_name),
     joint:{type, axis:[x,y,z], range:[lo,hi]},
     bone_offset:[x,y,z],
     geom:{type, size, ext_start},
     actuator:{gear}}
  ]
  task:   {type:"pusher", cube:{pos:[x,y,z], size:float}}     (省略可)

joint フィールドの後方互換エイリアス:
  "adapted_axis" → "axis"
  "range_deg"    → "range"

使い方:
  python3 scripts/topology_to_xml.py \\
    --topology data/rrbot_description/rrbot_topology.json \\
    --output   assets/mujoco_envs/my_robot.xml \\
    [--validate]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from lxml import etree


# ---------- デフォルト値 ----------

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
    "friction": [0.5, 0.1, 0.1],
    "density": 100,
}


# ---------- JSON 読み込み・正規化 ----------

def load_topology(path: str) -> dict:
    with open(path) as f:
        topo = json.load(f)

    # root セクションのデフォルト補完
    root_cfg = dict(DEFAULT_ROOT)
    root_cfg.update(topo.get("root", {}))
    root_cfg["geom"] = dict(DEFAULT_ROOT["geom"])
    root_cfg["geom"].update(topo.get("root", {}).get("geom", {}))
    topo["root"] = root_cfg

    # bodies の後方互換フィールド正規化
    for b in topo.get("bodies", []):
        j = b.get("joint")
        if j:
            if "adapted_axis" in j and "axis" not in j:
                j["axis"] = j["adapted_axis"]
            if "range_deg" in j and "range" not in j:
                j["range"] = j["range_deg"]

    # cube デフォルト補完
    task = topo.get("task", {})
    cube_cfg = dict(DEFAULT_CUBE)
    cube_cfg.update(task.get("cube", {}))
    task["cube"] = cube_cfg
    topo["task"] = task

    return topo


# ---------- ID 割り当て（StackelbergPPO 命名規則） ----------

def assign_mujoco_ids(bodies: list) -> dict:
    """
    body name → mujoco_id ("0", "1", "11", "21", ...) のマップを返す。
    規則: ind = 兄弟内の 1-indexed 順位、parent_suffix = (parent=="0" ? "" : parent_id)
    → mujoco_id = str(ind) + parent_suffix
    """
    id_map = {}  # body_name → mujoco_id

    def children_of(parent_name):
        return [b for b in bodies if b.get("parent") == parent_name]

    def recurse(parent_name, parent_id):
        for idx, child in enumerate(children_of(parent_name)):
            pname = "" if parent_id == "0" else parent_id
            mid = str(idx + 1) + pname
            id_map[child["name"]] = mid
            recurse(child["name"], mid)

    recurse("root", "0")
    return id_map


# ---------- 各 body の MuJoCo ローカル pos を計算 ----------

def compute_local_pos(body: dict, body_map: dict) -> list:
    """
    親のローカル座標系における、このボディの関節位置を返す。
    = 親の bone_offset（親のカプセルエンドポイント）
    rootの直接の子は [0,0,0]（root中心に関節）。
    """
    parent_name = body.get("parent", "root")
    if parent_name == "root":
        return [0.0, 0.0, 0.0]
    parent = body_map[parent_name]
    return list(parent.get("bone_offset", [0.0, 0.0, 0.0]))


# ---------- capsule fromto 計算 ----------

def compute_fromto(body: dict) -> str:
    """
    geom fromto = "start_x start_y start_z  end_x end_y end_z" (ローカル座標)
    start: ext_start * normalize(bone_offset)
    end:   bone_offset
    """
    bone_offset = np.array(body.get("bone_offset", [0.3, 0.0, 0.0]), dtype=float)
    ext_start = body.get("geom", {}).get("ext_start", 0.0)

    norm = np.linalg.norm(bone_offset)
    if norm < 1e-8:
        bone_dir = np.array([1.0, 0.0, 0.0])
    else:
        bone_dir = bone_offset / norm

    start = bone_dir * ext_start
    end = bone_offset

    return (f"{start[0]:.6f} {start[1]:.6f} {start[2]:.6f}  "
            f"{end[0]:.6f} {end[1]:.6f} {end[2]:.6f}")


# ---------- XML 生成 ----------

def _pos_str(pos) -> str:
    return " ".join(f"{x:.6f}" for x in pos)


def _add_asset(mujoco_elem):
    asset = etree.SubElement(mujoco_elem, "asset")
    etree.SubElement(asset, "texture", name="skybox", type="skybox",
                     builtin="gradient", rgb1=".2 .3 .4", rgb2="0 0 0",
                     width="800", height="800", mark="random", markrgb="1 1 1")
    etree.SubElement(asset, "texture", name="grid", type="2d",
                     builtin="checker", rgb1=".2 .1 .2", rgb2=".3 .2 .3",
                     width="300", height="300", mark="edge", markrgb=".3 .2 .3")
    etree.SubElement(asset, "material", name="grid", texture="grid",
                     texrepeat="1 1", texuniform="true", reflectance=".2")
    etree.SubElement(asset, "material", name="self", rgba=".7 .5 .3 1")


def _add_visual(mujoco_elem):
    visual = etree.SubElement(mujoco_elem, "visual")
    etree.SubElement(visual, "headlight", ambient=".4 .4 .4",
                     diffuse=".8 .8 .8", specular="0 0 0")
    etree.SubElement(visual, "map", znear=".001")
    etree.SubElement(visual, "quality", shadowsize="16384")


def _add_defaults(mujoco_elem):
    defaults = etree.SubElement(mujoco_elem, "default")
    etree.SubElement(defaults, "joint", armature="1", damping="1", limited="true")
    etree.SubElement(defaults, "geom", conaffinity="0", condim="3", density="5.0",
                     friction="1.0 0.5 0.5", margin="0.01", rgba=".7 .5 .3 1")


def _add_root_body(worldbody, root_cfg):
    pos = root_cfg.get("pos", [0.0, 0.0, 0.15])
    root_body = etree.SubElement(worldbody, "body", name="0", pos=_pos_str(pos))
    etree.SubElement(root_body, "camera", name="track", mode="trackcom",
                     pos="0 -3 0.3", xyaxes="1 0 0 0 0 1")

    geom_cfg = root_cfg.get("geom", {})
    gtype = geom_cfg.get("type", "sphere")
    gsize = geom_cfg.get("size", 0.1)
    etree.SubElement(root_body, "geom", pos="0 0 0", size=str(gsize), type=gtype)

    # free joint（移動ロボット）vs 固定根本
    if not root_cfg.get("fixed_base", True):
        etree.SubElement(root_body, "joint", armature="0", damping="0",
                         limited="false", margin="0.01", name="root",
                         pos="0 0 0", type="free")

    return root_body


def _add_body_recursive(parent_xml, body_name, bodies, body_map, id_map, actuator_elem):
    children = [b for b in bodies if b.get("parent") == body_name]
    for child in children:
        mid = id_map[child["name"]]
        local_pos = compute_local_pos(child, body_map)

        body_xml = etree.SubElement(parent_xml, "body",
                                    name=mid,
                                    pos=_pos_str(local_pos))

        # joint
        j = child.get("joint", {})
        jtype = j.get("type", "hinge")
        axis = j.get("axis", [0.0, 0.0, 1.0])
        rng = j.get("range", [-60.0, 60.0])
        etree.SubElement(body_xml, "joint",
                         name=f"{mid}_joint",
                         type=jtype,
                         pos="0 0 0",
                         axis=_pos_str(axis),
                         range=f"{rng[0]} {rng[1]}")

        # geom
        geom_cfg = child.get("geom", {})
        gtype = geom_cfg.get("type", "capsule")
        gsize = geom_cfg.get("size", 0.05)
        if gtype == "capsule":
            fromto = compute_fromto(child)
            etree.SubElement(body_xml, "geom",
                             type="capsule",
                             fromto=fromto,
                             size=str(gsize))
        else:
            etree.SubElement(body_xml, "geom",
                             type=gtype,
                             size=str(gsize))

        # actuator
        act = child.get("actuator")
        if act:
            gear = str(act.get("gear", 150))
            etree.SubElement(actuator_elem, "motor",
                             ctrllimited="true",
                             ctrlrange="-1.0 1.0",
                             joint=f"{mid}_joint",
                             gear=gear,
                             name=f"{mid}_joint")

        _add_body_recursive(body_xml, child["name"], bodies, body_map, id_map, actuator_elem)


def _add_cube(worldbody, cube_cfg):
    pos = cube_cfg.get("pos", [1.0, 0.0, 0.20])
    size = cube_cfg.get("size", 0.15)
    damping = str(cube_cfg.get("damping", 2.0))
    slide_range = cube_cfg.get("slide_range", [-2.0, 200.0])
    slide2_range = cube_cfg.get("slide2_range", [-100.0, 100.0])
    friction = cube_cfg.get("friction", [0.5, 0.1, 0.1])
    density = str(cube_cfg.get("density", 100))

    cube_body = etree.SubElement(worldbody, "body", name="cube", pos=_pos_str(pos))
    etree.SubElement(cube_body, "joint",
                     name="cube_slide", type="slide", axis="1 0 0",
                     damping=damping,
                     range=f"{slide_range[0]} {slide_range[1]}")
    etree.SubElement(cube_body, "joint",
                     name="cube_slide2", type="slide", axis="0 1 0",
                     damping=damping,
                     range=f"{slide2_range[0]} {slide2_range[1]}")
    etree.SubElement(cube_body, "geom",
                     name="cube_geom", type="box",
                     size=f"{size} {size} {size}",
                     rgba="0.8 0.2 0.2 1.0",
                     friction=" ".join(str(f) for f in friction),
                     density=density,
                     contype="1", conaffinity="1")


def generate_xml(topo: dict) -> str:
    bodies = topo.get("bodies", [])
    body_map = {b["name"]: b for b in bodies}
    root_cfg = topo["root"]
    id_map = assign_mujoco_ids(bodies)

    model_name = topo.get("description", "robot")[:30].replace(" ", "_")
    mujoco_elem = etree.Element("mujoco", model=model_name)

    _add_asset(mujoco_elem)
    _add_visual(mujoco_elem)

    etree.SubElement(mujoco_elem, "compiler", angle="degree")
    etree.SubElement(mujoco_elem, "option", integrator="RK4", timestep="0.01")

    _add_defaults(mujoco_elem)

    worldbody = etree.SubElement(mujoco_elem, "worldbody")
    etree.SubElement(worldbody, "light",
                     cutoff="100", diffuse="1 1 1", dir="0.3 0 -1.3",
                     directional="true", exponent="1",
                     pos="-30 0 130", specular=".1 .1 .1")
    etree.SubElement(worldbody, "geom",
                     conaffinity="1", condim="3", name="floor",
                     pos="0 0 0", rgba="1 1 1 1", size="200 200 .125",
                     type="plane", material="grid")

    actuator_elem = etree.SubElement(mujoco_elem, "actuator")

    root_body_xml = _add_root_body(worldbody, root_cfg)
    _add_body_recursive(root_body_xml, "root", bodies, body_map, id_map, actuator_elem)

    # pusher タスク固有: cube
    task = topo.get("task", {})
    if task.get("type", "pusher") == "pusher":
        _add_cube(worldbody, task["cube"])

    return etree.tostring(mujoco_elem, pretty_print=True).decode()


# ---------- MuJoCo 検証 ----------

def validate_xml(xml_str: str, n_steps: int = 10) -> bool:
    try:
        import mujoco
    except ImportError:
        print("[SKIP] mujoco が見つからないため検証をスキップ", file=sys.stderr)
        return True

    try:
        model = mujoco.MjModel.from_xml_string(xml_str)
    except Exception as e:
        print(f"[FAIL] XML ロードエラー: {e}", file=sys.stderr)
        return False

    data = mujoco.MjData(model)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)

    if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
        print(f"[FAIL] {n_steps} ステップ後に数値発散", file=sys.stderr)
        return False

    max_vel = float(np.abs(data.qvel).max()) if data.qvel.size > 0 else 0.0
    if max_vel > 50.0:
        print(f"[WARN] 速度が大きすぎる: max_vel={max_vel:.1f} m/s", file=sys.stderr)
        return False

    print(f"[OK] {n_steps} ステップ正常完了 (max_vel={max_vel:.3f} m/s)")
    return True


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="topology.json → MuJoCo XML 変換"
    )
    parser.add_argument("--topology", required=True,
                        help="入力 topology.json のパス")
    parser.add_argument("--output", required=True,
                        help="出力 MuJoCo XML のパス")
    parser.add_argument("--validate", action="store_true",
                        help="MuJoCo で 10 ステップ検証を行う")
    parser.add_argument("--no-cube", action="store_true",
                        help="cube ボディを出力しない（pusher 以外のタスク）")
    args = parser.parse_args()

    topo = load_topology(args.topology)

    if args.no_cube:
        topo["task"]["type"] = "none"

    xml_str = generate_xml(topo)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml_str)
    print(f"[topology_to_xml] 出力: {out_path}")

    # 生成されたボディ構成を表示
    bodies = topo.get("bodies", [])
    id_map = assign_mujoco_ids(bodies)
    print(f"  ボディ数: {len(bodies)} + root")
    for b in bodies:
        mid = id_map.get(b["name"], "?")
        print(f"    [{mid}] {b['name']}  parent={b.get('parent')}  "
              f"bone_offset={b.get('bone_offset')}")

    if args.validate:
        ok = validate_xml(xml_str)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
