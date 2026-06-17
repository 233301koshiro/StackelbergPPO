#!/usr/bin/env python3
"""
Convert MuJoCo XML files from coordinate="global" (MuJoCo 2.x) to MuJoCo 3.x format.

In global mode:
  - body.pos = world-frame position
  - joint.pos / geom.pos / site.pos inside a body = world-frame position
  - geom.fromto endpoints = world-frame positions

In local mode (MuJoCo 3.x default):
  - body.pos = position relative to parent body frame
  - joint/geom/site.pos inside a body = position relative to that body frame
  - geom.fromto endpoints = relative to that body frame

Since none of these bodies have orientation (no euler/quat), world frame and body frame
differ only by translation, so: local_pos = global_pos - body_world_pos.

Also removes the deprecated inertiafromgeom="true" compiler attribute.
"""
import xml.etree.ElementTree as ET
import numpy as np
import sys
from pathlib import Path


def parse_pos(s):
    return np.array([float(v) for v in s.strip().split()])


def fmt_pos(arr):
    # Use %g to drop trailing zeros; maintain enough precision
    return " ".join(f"{v:.10g}" for v in arr)


def process_body(body_el, parent_world_pos):
    """Recursively convert body subtree from global to local coordinates."""
    # This body's position was stored in world frame; compute local
    body_world_pos = parse_pos(body_el.get("pos", "0 0 0"))
    local_body_pos = body_world_pos - parent_world_pos
    if np.allclose(local_body_pos, 0.0):
        body_el.set("pos", "0 0 0")
    else:
        body_el.set("pos", fmt_pos(local_body_pos))

    # Convert direct children
    for child in body_el:
        tag = child.tag
        if tag == "body":
            process_body(child, body_world_pos)
        elif tag in ("geom", "joint", "site", "camera"):
            # pos attribute: convert global → local (relative to body frame)
            if "pos" in child.attrib:
                g_pos = parse_pos(child.get("pos"))
                l_pos = g_pos - body_world_pos
                if np.allclose(l_pos, 0.0):
                    child.set("pos", "0 0 0")
                else:
                    child.set("pos", fmt_pos(l_pos))
            # fromto: both endpoints global → local
            if "fromto" in child.attrib:
                vals = [float(v) for v in child.get("fromto").split()]
                pt1 = np.array(vals[:3]) - body_world_pos
                pt2 = np.array(vals[3:]) - body_world_pos
                child.set("fromto", fmt_pos(pt1) + " " + fmt_pos(pt2))
            # axis, euler, quat, ref: NOT positions, leave as-is


def convert_file(src_path, dst_path=None):
    if dst_path is None:
        dst_path = src_path  # in-place

    # Parse without resolving entities
    ET.register_namespace("", "")
    tree = ET.parse(str(src_path))
    root = tree.getroot()

    # --- Fix compiler ---
    for compiler in root.iter("compiler"):
        compiler.attrib.pop("coordinate", None)
        compiler.attrib.pop("inertiafromgeom", None)

    # --- Convert worldbody ---
    worldbody = root.find("worldbody")
    if worldbody is not None:
        for child in worldbody:
            if child.tag == "body":
                # worldbody is at world origin (0,0,0)
                process_body(child, np.zeros(3))
            # Direct geoms/sites/lights in worldbody stay as-is
            # (worldbody IS the world origin, local == global for its direct children)

    # Write back, preserving the original declaration-free style
    # ElementTree doesn't preserve whitespace; re-indent for readability
    _indent(root)
    tree.write(str(dst_path), encoding="unicode", xml_declaration=False)
    print(f"  Converted: {Path(dst_path).name}")


def _indent(elem, level=0):
    """Add pretty-print indentation (Python <3.9 compatible)."""
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
    if not level and not elem.tail:
        elem.tail = "\n"


def main():
    xml_dir = Path(__file__).parent
    targets = sorted(xml_dir.glob("*.xml"))
    # Exclude this script itself and any backup files
    targets = [p for p in targets if not p.name.startswith("convert_")]

    if len(sys.argv) > 1:
        targets = [Path(p) for p in sys.argv[1:]]

    print(f"Converting {len(targets)} XML file(s) ...")
    for path in targets:
        # Make a .bak backup first
        bak = path.with_suffix(".xml.bak")
        if not bak.exists():
            bak.write_text(path.read_text())
            print(f"  Backup  : {bak.name}")
        convert_file(path)

    print("\nDone.")


if __name__ == "__main__":
    main()
