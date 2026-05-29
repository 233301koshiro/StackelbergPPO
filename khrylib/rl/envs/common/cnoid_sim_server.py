"""
Choreonoid simulation server for StackelbergPPO.

Run inside Choreonoid via:
    xvfb-run /choreonoid_ws/install/bin/choreonoid \
        --python path/to/cnoid_sim_server.py

Communicates with mujoco_env_choreonoid.py (Python 3.9) via ZeroMQ REQ/REP.

Design principle:
  - MuJoCo XML generation (xml_robot.py) is kept as-is.
  - This file replaces ONLY the simulation execution (MjSim).
  - XML received from client → converted to URDF → loaded into Choreonoid.
"""

import sys
import os
import math
import json
import tempfile
import numpy as np
import zmq

# Choreonoid Python 3.8 bindings
CNOID_PYTHON_PATH = '/choreonoid_ws/install/lib/choreonoid-2.0/python'
if CNOID_PYTHON_PATH not in sys.path:
    sys.path.insert(0, CNOID_PYTHON_PATH)

from cnoid.Base import RootItem
from cnoid.Body import Body, BodyLoader
from cnoid.BodyPlugin import (
    WorldItem, BodyItem, AISTSimulatorItem, SimulationBody
)
import cnoid.IRSLUtil as IU

# irsl utilities (provides addSimulator, findItemsByClass, etc.)
IRSL_PYTHON_PATH = '/choreonoid_ws/install/lib/python3/dist-packages'
if IRSL_PYTHON_PATH not in sys.path:
    sys.path.insert(0, IRSL_PYTHON_PATH)
import irsl_choreonoid.cnoid_base as ib

PORT = 5556


# ===========================================================================
# MuJoCo XML → URDF converter (minimal, covers StackelbergPPO morphologies)
# ===========================================================================

def mujoco_xml_to_urdf(xml_str: str) -> str:
    """
    Convert MuJoCo XML string to URDF string.
    Handles: free root joint, hinge joints, capsule/sphere geoms, motor actuators.
    Inertia is estimated from geometry (same as MuJoCo inertiafromgeom=true).
    """
    from lxml import etree

    tree = etree.fromstring(xml_str.encode())

    # Parse default geom settings
    default_density = 5.0
    default_el = tree.find('default/geom')
    if default_el is not None:
        default_density = float(default_el.get('density', default_density))

    # Parse option
    opt_el = tree.find('option')
    timestep = float(opt_el.get('timestep', '0.01')) if opt_el is not None else 0.01

    # Parse actuators: joint_name → ctrlrange
    actuators = {}
    for motor in tree.findall('actuator/motor'):
        jname = motor.get('joint')
        cr = motor.get('ctrlrange', '-1 1')
        lo, hi = [float(x) for x in cr.split()]
        gear = float(motor.get('gear', '1'))
        actuators[jname] = {'ctrlrange': [lo, hi], 'gear': gear, 'name': motor.get('name', jname)}

    # URDF root element
    robot_name = tree.get('model', 'robot')
    urdf_root = etree.Element('robot', name=robot_name)
    etree.SubElement(urdf_root, 'link', name='world')

    links_added = set()
    joints_added = []
    body_order = []  # (body_name, parent_name)

    def parse_vec(s):
        return [float(x) for x in s.split()]

    def capsule_inertia(length, radius, density):
        """
        Inertia of a capsule (cylinder of length l + 2 hemispheres of radius r).
        'length' here is the distance between hemisphere centers (= cylinder length),
        matching MuJoCo's fromto convention.

        I_transverse (perpendicular to capsule axis, about capsule COM):
          I_cyl  = m_cyl * (r²/4 + l²/12)
          I_caps = m_cap * (2r²/5 + (l/2 - 3r/8)²)
                   ^-- hemisphere COM is at 3r/8 inward from flat face
        """
        r, l = radius, length
        m_cyl = density * math.pi * r**2 * l
        m_cap = density * (4.0/3.0) * math.pi * r**3   # both hemispheres = 1 sphere
        m = m_cyl + m_cap
        d_hemi = l/2.0 - 3.0*r/8.0   # hemisphere COM distance from capsule center
        Ixx = m_cyl * (r**2/4.0 + l**2/12.0) + m_cap * (2.0*r**2/5.0 + d_hemi**2)
        Izz = m * r**2 / 2.0          # longitudinal (along axis)
        return m, Ixx, Izz

    def sphere_inertia(radius, density):
        m = density * (4/3) * math.pi * radius**3
        I = 0.4 * m * radius**2
        return m, I

    def add_link(name, geom_el, density):
        link_el = etree.SubElement(urdf_root, 'link', name=name)
        inertial_el = etree.SubElement(link_el, 'inertial')

        if geom_el is not None:
            gtype = geom_el.get('type', 'sphere')

            if gtype == 'capsule' and 'fromto' in geom_el.attrib:
                fromto = parse_vec(geom_el.get('fromto'))
                p0 = np.array(fromto[:3])
                p1 = np.array(fromto[3:])
                center = (p0 + p1) / 2.0
                diff = p1 - p0
                length = float(np.linalg.norm(diff))
                radius = float(geom_el.get('size', '0.08'))

                m, Ixx, Izz = capsule_inertia(length, radius, density)

                etree.SubElement(inertial_el, 'origin', xyz=f'{center[0]} {center[1]} {center[2]}', rpy='0 0 0')
                etree.SubElement(inertial_el, 'mass', value=str(m))
                etree.SubElement(inertial_el, 'inertia',
                                 ixx=str(Ixx), ixy='0', ixz='0',
                                 iyy=str(Ixx), iyz='0', izz=str(Izz))

                origin_xyz = f'{center[0]} {center[1]} {center[2]}'
                # compute orientation rpy to align cylinder z-axis with capsule direction
                import math as _math
                axis_dir = diff / (np.linalg.norm(diff) + 1e-12)
                # rotation from z-axis to capsule direction
                z = np.array([0, 0, 1.0])
                cp = np.cross(z, axis_dir)
                cp_norm = np.linalg.norm(cp)
                if cp_norm > 1e-6:
                    angle = _math.atan2(cp_norm, np.dot(z, axis_dir))
                    ax = cp / cp_norm
                    # angle-axis → RPY (approximate via rotation matrix)
                    c, s = _math.cos(angle), _math.sin(angle)
                    R = np.array([
                        [c+ax[0]**2*(1-c),      ax[0]*ax[1]*(1-c)-ax[2]*s, ax[0]*ax[2]*(1-c)+ax[1]*s],
                        [ax[1]*ax[0]*(1-c)+ax[2]*s, c+ax[1]**2*(1-c),      ax[1]*ax[2]*(1-c)-ax[0]*s],
                        [ax[2]*ax[0]*(1-c)-ax[1]*s, ax[2]*ax[1]*(1-c)+ax[0]*s, c+ax[2]**2*(1-c)],
                    ])
                    roll  = _math.atan2(R[2,1], R[2,2])
                    pitch = _math.atan2(-R[2,0], _math.sqrt(R[2,1]**2+R[2,2]**2))
                    yaw   = _math.atan2(R[1,0], R[0,0])
                    rpy_s = f'{roll} {pitch} {yaw}'
                else:
                    rpy_s = '0 0 0'

                for tag in ('visual', 'collision'):
                    el = etree.SubElement(link_el, tag)
                    etree.SubElement(el, 'origin', xyz=origin_xyz, rpy=rpy_s)
                    geom_el2 = etree.SubElement(el, 'geometry')
                    etree.SubElement(geom_el2, 'cylinder', radius=str(radius), length=str(length))

            elif gtype == 'sphere':
                radius = float(geom_el.get('size', '0.25'))
                pos_s = geom_el.get('pos', '0 0 0')
                m, I = sphere_inertia(radius, density)

                etree.SubElement(inertial_el, 'origin', xyz=pos_s, rpy='0 0 0')
                etree.SubElement(inertial_el, 'mass', value=str(m))
                etree.SubElement(inertial_el, 'inertia',
                                 ixx=str(I), ixy='0', ixz='0',
                                 iyy=str(I), iyz='0', izz=str(I))

                for tag in ('visual', 'collision'):
                    el = etree.SubElement(link_el, tag)
                    etree.SubElement(el, 'origin', xyz=pos_s, rpy='0 0 0')
                    etree.SubElement(etree.SubElement(el, 'geometry'), 'sphere', radius=str(radius))

            elif gtype == 'box':
                size_s = geom_el.get('size', '1 1 1')
                pos_s  = geom_el.get('pos', '0 0 0')
                sx, sy, sz = [float(x) for x in size_s.split()]
                m = density * 8 * sx * sy * sz
                Ixx = m * (sy**2 + sz**2) / 12
                Iyy = m * (sx**2 + sz**2) / 12
                Izz = m * (sx**2 + sy**2) / 12

                etree.SubElement(inertial_el, 'origin', xyz=pos_s, rpy='0 0 0')
                etree.SubElement(inertial_el, 'mass', value=str(m))
                etree.SubElement(inertial_el, 'inertia',
                                 ixx=str(Ixx), ixy='0', ixz='0',
                                 iyy=str(Iyy), iyz='0', izz=str(Izz))

                for tag in ('visual', 'collision'):
                    el = etree.SubElement(link_el, tag)
                    etree.SubElement(el, 'origin', xyz=pos_s, rpy='0 0 0')
                    etree.SubElement(etree.SubElement(el, 'geometry'), 'box', size=f'{2*sx} {2*sy} {2*sz}')
        else:
            # Dummy inertia for links without geoms
            etree.SubElement(inertial_el, 'mass', value='0.001')
            etree.SubElement(inertial_el, 'inertia',
                             ixx='1e-6', ixy='0', ixz='0',
                             iyy='1e-6', iyz='0', izz='1e-6')

        links_added.add(name)
        return link_el

    def add_joint_urdf(jname, jtype, parent, child, pos, axis, jrange, damping, armature):
        joint_el = etree.SubElement(urdf_root, 'joint', name=jname, type=jtype)
        etree.SubElement(joint_el, 'parent', link=parent)
        etree.SubElement(joint_el, 'child',  link=child)
        etree.SubElement(joint_el, 'origin', xyz=f'{pos[0]} {pos[1]} {pos[2]}', rpy='0 0 0')
        if axis is not None:
            etree.SubElement(joint_el, 'axis', xyz=f'{axis[0]} {axis[1]} {axis[2]}')
        if jrange is not None:
            etree.SubElement(joint_el, 'limit',
                             lower=str(jrange[0]), upper=str(jrange[1]),
                             effort='100', velocity='10')
        dyn_el = etree.SubElement(joint_el, 'dynamics',
                                  damping=str(damping), friction='0')
        joints_added.append(jname)

    # joint_name → armature (collected during parse, applied after Choreonoid load)
    joint_armatures = {}

    def _add_one_joint(joint_el, parent_lk, child_lk, bpos):
        """Convert one MuJoCo joint element to URDF and record armature."""
        jname    = joint_el.get('name', f'{child_lk}_joint')
        jtype_mj = joint_el.get('type', 'hinge')
        armature = float(joint_el.get('armature', default_armature))
        # Use explicitly set damping first, fall back to default
        damping  = float(joint_el.get('damping', default_damping))

        if jtype_mj == 'free':
            add_joint_urdf(jname=f'{child_lk}_to_world', jtype='floating',
                           parent=parent_lk, child=child_lk, pos=bpos,
                           axis=None, jrange=None, damping=0.0, armature=0.0)
        elif jtype_mj == 'hinge':
            axis    = parse_vec(joint_el.get('axis', '0 0 1'))
            rng_str = joint_el.get('range', '-180 180')
            rng     = [math.radians(float(x)) for x in rng_str.split()]
            add_joint_urdf(jname=jname, jtype='revolute',
                           parent=parent_lk, child=child_lk, pos=bpos,
                           axis=axis, jrange=rng, damping=damping, armature=armature)
            joint_armatures[jname] = armature
        elif jtype_mj in ('slide', 'prismatic'):
            axis    = parse_vec(joint_el.get('axis', '1 0 0'))
            rng_str = joint_el.get('range', '-10 10')
            rng     = [float(x) for x in rng_str.split()]
            add_joint_urdf(jname=jname, jtype='prismatic',
                           parent=parent_lk, child=child_lk, pos=bpos,
                           axis=axis, jrange=rng, damping=damping, armature=0.0)

    def process_body(body_el, parent_link_name):
        bname = body_el.get('name')
        body_order.append(bname)
        bpos  = parse_vec(body_el.get('pos', '0 0 0'))

        geom_el = body_el.find('geom')
        add_link(bname, geom_el, default_density)

        joint_els = body_el.findall('joint')
        if not joint_els:
            # Fixed
            add_joint_urdf(jname=f'{parent_link_name}_to_{bname}_fixed',
                           jtype='fixed', parent=parent_link_name, child=bname,
                           pos=bpos, axis=None, jrange=None, damping=0.0, armature=0.0)
        elif len(joint_els) == 1:
            _add_one_joint(joint_els[0], parent_link_name, bname, bpos)
        else:
            # Multiple joints: create intermediate virtual links
            # e.g. cube with slide_x + slide_y
            prev_lk = parent_link_name
            for idx, j_el in enumerate(joint_els):
                if idx < len(joint_els) - 1:
                    virt_name = f'{bname}_virt{idx}'
                    etree.SubElement(urdf_root, 'link', name=virt_name)
                    _add_one_joint(j_el, prev_lk, virt_name, bpos if idx == 0 else [0,0,0])
                    prev_lk = virt_name
                else:
                    _add_one_joint(j_el, prev_lk, bname, [0,0,0])

        for child_body in body_el.findall('body'):
            process_body(child_body, bname)

    # Parse default joint settings from MuJoCo XML
    default_joint_el = tree.find('default/joint')
    default_armature = float(default_joint_el.get('armature', '0')) if default_joint_el is not None else 0.0
    default_damping  = float(default_joint_el.get('damping',  '1')) if default_joint_el is not None else 1.0

    for body_el in tree.findall('worldbody/body'):
        process_body(body_el, 'world')

    urdf_str = etree.tostring(urdf_root, pretty_print=True).decode()
    return urdf_str, body_order, actuators, timestep, joint_armatures


# ===========================================================================
# Simulation state extraction helpers
# ===========================================================================

def get_model_info(body, actuators_map, body_order, timestep):
    """
    Build the model-info dict that mujoco_env_choreonoid.py needs,
    mimicking MuJoCo model attributes.
    """
    # Joint qpos addresses: for each link (= MuJoCo body), record which
    # joints belong to it and their qpos index.
    # Choreonoid body.joint(i) iterates over actuated joints in order.
    njoints = body.numJoints
    nlinks  = body.numLinks

    # Build body_names, body_jntadr, body_jntnum, jnt_qposadr
    # matching MuJoCo's model.body_jntadr / jnt_qposadr layout.
    body_names = [body.link(i).name for i in range(nlinks)]

    # qpos layout: [root_free(7 if floating), joint0(1), joint1(1), ...]
    # Determine if root is floating
    root = body.rootLink
    is_floating = (root.jointType == root.FreeJoint)

    qpos_offset = 7 if is_floating else 0
    nq = qpos_offset + njoints
    nv = (6 if is_floating else 0) + njoints

    # jnt_qposadr[i] = qpos index for joint i
    jnt_qposadr = list(range(qpos_offset, qpos_offset + njoints))

    # body_jntadr[i] = index of first joint of link i in joint list (-1 if none)
    # body_jntnum[i] = number of joints in link i
    # Build name→joint_index map
    joint_name_to_idx = {}
    for i in range(njoints):
        joint_name_to_idx[body.joint(i).jointName] = i

    body_jntadr = []
    body_jntnum = []
    for i in range(nlinks):
        lk = body.link(i)
        # Each Choreonoid link has at most one joint
        jname = lk.jointName if hasattr(lk, 'jointName') else lk.name
        if jname in joint_name_to_idx:
            idx = joint_name_to_idx[jname]
            body_jntadr.append(idx)
            body_jntnum.append(1)
        else:
            body_jntadr.append(-1)
            body_jntnum.append(0)

    # Actuator info (ordered as in MuJoCo XML)
    actuator_names = [v['name'] for v in actuators_map.values()]
    ctrlrange = [v['ctrlrange'] for v in actuators_map.values()]

    # Initial state: all joints at 0
    init_qpos = [0.0] * nq
    if is_floating:
        init_qpos[2] = 0.4  # lift root slightly above ground

    return {
        'nq': nq,
        'nv': nv,
        'timestep': timestep,
        'actuator_names': actuator_names,
        'ctrlrange': ctrlrange,
        'body_names': body_names,
        'body_jntadr': body_jntadr,
        'body_jntnum': body_jntnum,
        'jnt_qposadr': jnt_qposadr,
        'init_qpos': init_qpos,
        'init_qvel': [0.0] * nv,
    }


def _rot_to_quat_wxyz(R):
    """Rotation matrix (3x3 numpy) → quaternion [w, x, y, z]."""
    R = np.asarray(R)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [w, x, y, z]


def _quat_wxyz_to_rot(quat):
    """Quaternion [w, x, y, z] → rotation matrix (3x3 numpy)."""
    w, x, y, z = quat
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def get_state_dict(sim_body):
    """
    Read current simulation state from a Choreonoid SimulationBody.
    Returns qpos, qvel, and body_xpos/xmat for all links.
    """
    b = sim_body.body()
    root = b.rootLink

    is_floating = (root.jointType == root.FreeJoint)
    njoints = b.numJoints

    qpos = []
    qvel = []

    if is_floating:
        p = root.translation
        R = np.asarray(root.rotation)
        quat = _rot_to_quat_wxyz(R)   # [w, x, y, z]
        qpos += list(p) + quat         # 7 values
        # root.v = linear velocity, root.w = angular velocity
        # (root.dv = linear acceleration — do NOT use here)
        qvel += list(root.v) + list(root.w)   # 6 values

    for i in range(njoints):
        j = b.joint(i)
        qpos.append(j.q)
        qvel.append(j.dq)

    # Body poses
    body_xpos = {}
    body_xmat = {}
    for i in range(b.numLinks):
        lk = b.link(i)
        body_xpos[lk.name] = list(lk.translation)
        body_xmat[lk.name] = np.asarray(lk.rotation).flatten().tolist()

    return {
        'qpos': qpos,
        'qvel': qvel,
        'body_xpos': body_xpos,
        'body_xmat': body_xmat,
        'ctrl': [0.0] * njoints,
    }


def set_state(sim_body, qpos, qvel):
    """Set joint states on a Choreonoid SimulationBody."""
    b = sim_body.body()
    root = b.rootLink
    is_floating = (root.jointType == root.FreeJoint)
    offset = 0

    if is_floating:
        root.setTranslation(qpos[:3])
        # qpos[3:7] = quaternion [w, x, y, z] (MuJoCo convention)
        R = _quat_wxyz_to_rot(qpos[3:7])
        root.setRotation(R)
        root.v = np.array(qvel[:3])   # linear velocity
        root.w = np.array(qvel[3:6])  # angular velocity (NOT dv)
        offset = 7

    for i in range(b.numJoints):
        j = b.joint(i)
        j.q  = qpos[offset + i]
        j.dq = qvel[(6 if is_floating else 0) + i]

    b.calcForwardKinematics()


# ===========================================================================
# Choreonoid world management
# ===========================================================================

class ChoreonoidSimWorld:
    def __init__(self):
        self.world_item = None
        self.sim_item   = None
        self.body_items = {}  # name → BodyItem
        self.sim_bodies = {}  # name → SimulationBody
        self.actuators_map = {}
        self.frame_skip = 4
        self.is_running = False
        self._setup_world()

    def _setup_world(self):
        self.world_item = WorldItem()
        RootItem.instance.addChildItem(self.world_item)

        # Floor
        floor_path = '/choreonoid_ws/install/share/choreonoid-2.0/model/misc/floor.body'
        if os.path.exists(floor_path):
            floor_item = BodyItem()
            floor_item.load(floor_path)
            self.world_item.addChildItem(floor_item)

        # AIST simulator in manual (non-realtime) mode
        self.sim_item = AISTSimulatorItem()
        self.sim_item.setTimeStep(0.01)
        self.sim_item.setRealtimeSyncMode(3)  # manual mode: step only on tickRequest
        self.world_item.addChildItem(self.sim_item)

    def load_model(self, xml_str: str, frame_skip: int) -> dict:
        """Convert MuJoCo XML to URDF, load into world, return model info."""
        self.frame_skip = frame_skip

        # Remove existing robot body items (keep floor)
        for name, item in list(self.body_items.items()):
            item.detachFromParentItem()
        self.body_items.clear()
        self.sim_bodies.clear()

        # Convert XML → URDF
        urdf_str, body_order, actuators_map, timestep, joint_armatures = mujoco_xml_to_urdf(xml_str)
        self.actuators_map = actuators_map
        self.sim_item.setTimeStep(timestep)

        # Write URDF to temp file and load
        with tempfile.NamedTemporaryFile(suffix='.urdf', mode='w', delete=False) as f:
            f.write(urdf_str)
            urdf_path = f.name

        robot_item = BodyItem()
        loaded = robot_item.load(urdf_path)
        os.unlink(urdf_path)

        if not loaded:
            raise RuntimeError(f"Failed to load URDF")

        # Apply MuJoCo armature to each joint (equivalent rotational inertia)
        # MuJoCo armature=1 adds 1 kg·m² per joint — critical for correct dynamics
        b = robot_item.body
        for i in range(b.numJoints):
            j = b.joint(i)
            arm = joint_armatures.get(j.jointName, 0.0)
            if arm > 0:
                j.setEquivalentRotorInertia(arm)

        # Fix root link if MuJoCo model uses a floating root joint.
        # Without this Choreonoid URDFLoader makes every body free-floating,
        # but we handle the free joint explicitly in get_state / set_state.
        # → keep root as FreeJoint; just log its type for debugging.
        root = b.rootLink
        # root.jointType values: FreeJoint=0, FixedJoint=1, etc.

        robot_item.storeInitialState()
        self.world_item.addChildItem(robot_item)
        self.body_items['robot'] = robot_item

        # Start simulation
        if self.is_running:
            self.sim_item.stopSimulation()
        self.sim_item.startSimulation(doReset=True)
        self.is_running = True

        # Wait one tick for sim bodies to be initialized
        self.sim_item.tickRequest(True)
        IU.processEvent()

        sim_body = self.sim_item.findSimulationBody(robot_item.name)
        if sim_body is None:
            raise RuntimeError("SimulationBody not found after startSimulation")
        self.sim_bodies['robot'] = sim_body

        body = sim_body.body()
        info = get_model_info(body, actuators_map, body_order, timestep)
        return info

    def reset(self) -> dict:
        """Reset simulation to initial state."""
        self.sim_item.stopSimulation()
        for item in self.body_items.values():
            item.restoreInitialState(True)
        self.sim_item.startSimulation(doReset=True)
        self.sim_item.tickRequest(True)
        IU.processEvent()

        # Re-fetch sim bodies after restart
        for name, item in self.body_items.items():
            sb = self.sim_item.findSimulationBody(item.name)
            if sb is not None:
                self.sim_bodies[name] = sb

        sb = self.sim_bodies.get('robot')
        if sb is None:
            return {'qpos': [], 'qvel': [], 'body_xpos': {}, 'body_xmat': {}}
        return get_state_dict(sb)

    def step(self, ctrl: list, n_frames: int) -> dict:
        """Apply control, advance n_frames steps, return state."""
        sb = self.sim_bodies.get('robot')
        if sb is None:
            return {'qpos': [], 'qvel': [], 'body_xpos': {}, 'body_xmat': {}}

        b = sb.body()

        # Map control to joint torques (actuator gear scaling)
        actuator_list = list(self.actuators_map.values())
        for i, (jname, ainfo) in enumerate(self.actuators_map.items()):
            # Find joint by name
            j = b.joint(jname)
            if j is not None and i < len(ctrl):
                j.u = float(ctrl[i]) * ainfo['gear']

        # Advance simulation
        for _ in range(n_frames):
            self.sim_item.tickRequest(True)
            IU.processEvent()

        return get_state_dict(sb)

    def set_state_cmd(self, qpos: list, qvel: list) -> dict:
        sb = self.sim_bodies.get('robot')
        if sb is None:
            return {'qpos': qpos, 'qvel': qvel, 'body_xpos': {}, 'body_xmat': {}}
        set_state(sb, qpos, qvel)
        return get_state_dict(sb)


# ===========================================================================
# ZMQ server main loop
# ===========================================================================

def run_server():
    world = ChoreonoidSimWorld()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{PORT}")
    print(f"[cnoid_sim_server] Listening on port {PORT}")

    while True:
        msg = sock.recv_json()
        cmd = msg.get('cmd')

        try:
            if cmd in ('load_model', 'reload_model'):
                result = world.load_model(msg['xml_str'], msg.get('frame_skip', 4))

            elif cmd == 'reset':
                result = world.reset()

            elif cmd == 'step':
                result = world.step(msg['ctrl'], msg.get('n_frames', world.frame_skip))

            elif cmd == 'set_state':
                result = world.set_state_cmd(msg['qpos'], msg['qvel'])

            elif cmd == 'ping':
                result = {'status': 'ok'}

            else:
                result = {'error': f'unknown command: {cmd}'}

        except Exception as e:
            import traceback
            result = {'error': str(e), 'traceback': traceback.format_exc()}

        sock.send_json(result)


# Entry point when run via: choreonoid --python cnoid_sim_server.py
run_server()
