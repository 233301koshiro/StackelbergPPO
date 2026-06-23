"""
ChoreonoidEnv: direct drop-in for MujocoEnv using Choreonoid Python 3.12 bindings.

Must run inside Choreonoid (e.g. choreonoid --no-window --python ...),
because WorldItem / AISTSimulatorItem require the Qt application context.
"""

import os
import math
import tempfile
import numpy as np
from pathlib import Path
from gym import spaces
from gym.utils import seeding

from cnoid.Base import RootItem
from cnoid.Body import BodyLoader
from cnoid.BodyPlugin import WorldItem, BodyItem, AISTSimulatorItem
import cnoid.IRSLUtil as IU

DEFAULT_SIZE = 500


# ---------------------------------------------------------------------------
# MuJoCo XML → URDF (minimal converter for StackelbergPPO morphologies)
# ---------------------------------------------------------------------------

def mujoco_xml_to_urdf(xml_str: str):
    """
    Convert MuJoCo XML string to URDF string.
    Returns (urdf_str, body_order, actuators_map, timestep, joint_armatures).
    """
    from lxml import etree

    tree = etree.fromstring(xml_str.encode())

    # coordinate="global" means all body/geom positions are in world frame.
    # URDF requires parent-relative offsets, so we must subtract parent positions.
    compiler_el = tree.find('compiler')
    is_global_coord = (compiler_el is not None and
                       compiler_el.get('coordinate', 'local') == 'global')

    default_density = 5.0
    default_el = tree.find('default/geom')
    if default_el is not None:
        default_density = float(default_el.get('density', default_density))

    opt_el = tree.find('option')
    timestep = float(opt_el.get('timestep', '0.01')) if opt_el is not None else 0.01

    actuators = {}
    for motor in tree.findall('actuator/motor'):
        jname = motor.get('joint')
        cr = motor.get('ctrlrange', '-1 1')
        lo, hi = [float(x) for x in cr.split()]
        gear = float(motor.get('gear', '1'))
        actuators[jname] = {'ctrlrange': [lo, hi], 'gear': gear, 'name': motor.get('name', jname)}

    robot_name = tree.get('model', 'robot')
    urdf_root = etree.Element('robot', name=robot_name)
    etree.SubElement(urdf_root, 'link', name='world')

    links_added = set()
    joints_added = []
    body_order = []
    joint_armatures = {}

    def parse_vec(s):
        return [float(x) for x in s.split()]

    def capsule_inertia(length, radius, density):
        r, l = radius, length
        m_cyl = density * math.pi * r**2 * l
        m_cap = density * (4.0/3.0) * math.pi * r**3
        m = m_cyl + m_cap
        d_hemi = l/2.0 - 3.0*r/8.0
        Ixx = m_cyl * (r**2/4.0 + l**2/12.0) + m_cap * (2.0*r**2/5.0 + d_hemi**2)
        Izz = m * r**2 / 2.0
        return m, Ixx, Izz

    def sphere_inertia(radius, density):
        m = density * (4/3) * math.pi * radius**3
        I = 0.4 * m * radius**2
        return m, I

    def add_link(name, geom_el, density, body_global_pos=None):
        if body_global_pos is None:
            body_global_pos = np.zeros(3)
        link_el = etree.SubElement(urdf_root, 'link', name=name)
        inertial_el = etree.SubElement(link_el, 'inertial')

        if geom_el is not None:
            gtype = geom_el.get('type', 'sphere')

            if gtype == 'capsule' and 'fromto' in geom_el.attrib:
                fromto = parse_vec(geom_el.get('fromto'))
                # In coordinate="global" mode, fromto is in world frame; convert to body-local.
                p0 = np.array(fromto[:3]) - body_global_pos
                p1 = np.array(fromto[3:]) - body_global_pos
                center = (p0 + p1) / 2.0
                diff = p1 - p0
                length = float(np.linalg.norm(diff))
                radius = float(geom_el.get('size', '0.08'))

                m, Ixx, Izz = capsule_inertia(length, radius, density)
                etree.SubElement(inertial_el, 'origin',
                                 xyz=f'{center[0]} {center[1]} {center[2]}', rpy='0 0 0')
                etree.SubElement(inertial_el, 'mass', value=str(m))
                etree.SubElement(inertial_el, 'inertia',
                                 ixx=str(Ixx), ixy='0', ixz='0',
                                 iyy=str(Ixx), iyz='0', izz=str(Izz))

                origin_xyz = f'{center[0]} {center[1]} {center[2]}'
                axis_dir = diff / (np.linalg.norm(diff) + 1e-12)
                z = np.array([0, 0, 1.0])
                cp = np.cross(z, axis_dir)
                cp_norm = np.linalg.norm(cp)
                if cp_norm > 1e-6:
                    angle = math.atan2(cp_norm, np.dot(z, axis_dir))
                    ax = cp / cp_norm
                    c, s = math.cos(angle), math.sin(angle)
                    R = np.array([
                        [c+ax[0]**2*(1-c),           ax[0]*ax[1]*(1-c)-ax[2]*s, ax[0]*ax[2]*(1-c)+ax[1]*s],
                        [ax[1]*ax[0]*(1-c)+ax[2]*s,  c+ax[1]**2*(1-c),          ax[1]*ax[2]*(1-c)-ax[0]*s],
                        [ax[2]*ax[0]*(1-c)-ax[1]*s,  ax[2]*ax[1]*(1-c)+ax[0]*s, c+ax[2]**2*(1-c)],
                    ])
                    roll  = math.atan2(R[2,1], R[2,2])
                    pitch = math.atan2(-R[2,0], math.sqrt(R[2,1]**2+R[2,2]**2))
                    yaw   = math.atan2(R[1,0], R[0,0])
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
                pos_local = np.array(parse_vec(geom_el.get('pos', '0 0 0'))) - body_global_pos
                pos_s = f'{pos_local[0]} {pos_local[1]} {pos_local[2]}'
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
                pos_local = np.array(parse_vec(geom_el.get('pos', '0 0 0'))) - body_global_pos
                pos_s = f'{pos_local[0]} {pos_local[1]} {pos_local[2]}'
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
            etree.SubElement(inertial_el, 'mass', value='0.001')
            etree.SubElement(inertial_el, 'inertia',
                             ixx='1e-6', ixy='0', ixz='0',
                             iyy='1e-6', iyz='0', izz='1e-6')

        links_added.add(name)
        return link_el

    def add_joint_urdf(jname, jtype, parent, child, pos, axis, jrange, damping):
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
        etree.SubElement(joint_el, 'dynamics', damping=str(damping), friction='0')
        joints_added.append(jname)

    default_joint_el = tree.find('default/joint')
    default_armature = float(default_joint_el.get('armature', '0')) if default_joint_el is not None else 0.0
    default_damping  = float(default_joint_el.get('damping',  '1')) if default_joint_el is not None else 1.0

    def _add_one_joint(joint_el, parent_lk, child_lk, bpos):
        jname    = joint_el.get('name', f'{child_lk}_joint')
        jtype_mj = joint_el.get('type', 'hinge')
        armature = float(joint_el.get('armature', default_armature))
        damping  = float(joint_el.get('damping', default_damping))

        if jtype_mj == 'free':
            add_joint_urdf(jname=f'{child_lk}_to_world', jtype='floating',
                           parent=parent_lk, child=child_lk, pos=bpos,
                           axis=None, jrange=None, damping=0.0)
        elif jtype_mj == 'hinge':
            axis    = parse_vec(joint_el.get('axis', '0 0 1'))
            rng_str = joint_el.get('range', '-180 180')
            rng     = [math.radians(float(x)) for x in rng_str.split()]
            add_joint_urdf(jname=jname, jtype='revolute',
                           parent=parent_lk, child=child_lk, pos=bpos,
                           axis=axis, jrange=rng, damping=damping)
            joint_armatures[jname] = armature
        elif jtype_mj in ('slide', 'prismatic'):
            axis    = parse_vec(joint_el.get('axis', '1 0 0'))
            rng_str = joint_el.get('range', '-10 10')
            rng     = [float(x) for x in rng_str.split()]
            add_joint_urdf(jname=jname, jtype='prismatic',
                           parent=parent_lk, child=child_lk, pos=bpos,
                           axis=axis, jrange=rng, damping=0.0)

    def process_body(body_el, parent_link_name, parent_global_pos=None):
        if parent_global_pos is None:
            parent_global_pos = np.zeros(3)
        bname = body_el.get('name')
        body_order.append(bname)
        global_pos = np.array(parse_vec(body_el.get('pos', '0 0 0')))
        # In coordinate="global" mode, body pos is world-frame; URDF needs parent-relative.
        bpos = (global_pos - parent_global_pos).tolist() if is_global_coord else global_pos.tolist()
        geom_el = body_el.find('geom')
        add_link(bname, geom_el, default_density,
                 body_global_pos=global_pos if is_global_coord else np.zeros(3))

        joint_els = body_el.findall('joint')
        if not joint_els:
            add_joint_urdf(jname=f'{parent_link_name}_to_{bname}_fixed',
                           jtype='fixed', parent=parent_link_name, child=bname,
                           pos=bpos, axis=None, jrange=None, damping=0.0)
        elif len(joint_els) == 1:
            _add_one_joint(joint_els[0], parent_link_name, bname, bpos)
        else:
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
            process_body(child_body, bname, global_pos if is_global_coord else np.zeros(3))

    for body_el in tree.findall('worldbody/body'):
        process_body(body_el, 'world', np.zeros(3))

    urdf_str = etree.tostring(urdf_root, pretty_print=True).decode()
    return urdf_str, body_order, actuators, timestep, joint_armatures


# ---------------------------------------------------------------------------
# MuJoCo XML → Choreonoid .body (supports native Capsule geometry)
# ---------------------------------------------------------------------------

def mujoco_xml_to_body(xml_str: str):
    """
    Convert MuJoCo XML to Choreonoid .body format (format_version 2.0).
    Returns (body_defs, body_order, actuators_map, timestep, joint_armatures).

    body_defs is a list of (item_name, yaml_str) — one entry per top-level
    worldbody child.  Each child becomes the root of its own BodyItem so that
    Choreonoid never sees a non-root FreeJoint (which triggers an interactive
    dialog).  For most envs body_defs has one entry (robot).  For pusher it has
    two: robot ("0" as FreeJoint root) and cube ("cube_virt0" as prismatic root).
    """
    from lxml import etree

    tree = etree.fromstring(xml_str.encode())

    compiler_el = tree.find('compiler')
    is_global_coord = (compiler_el is not None and
                       compiler_el.get('coordinate', 'local') == 'global')
    angle_is_degree = (compiler_el is None or
                       compiler_el.get('angle', 'degree') == 'degree')

    default_density = 5.0
    default_el = tree.find('default/geom')
    if default_el is not None:
        default_density = float(default_el.get('density', default_density))

    opt_el = tree.find('option')
    timestep = float(opt_el.get('timestep', '0.01')) if opt_el is not None else 0.01

    actuators = {}
    for motor in tree.findall('actuator/motor'):
        jname = motor.get('joint')
        cr = motor.get('ctrlrange', '-1 1')
        lo, hi = [float(x) for x in cr.split()]
        gear = float(motor.get('gear', '1'))
        actuators[jname] = {'ctrlrange': [lo, hi], 'gear': gear,
                             'name': motor.get('name', jname)}

    robot_name = tree.get('model', 'robot')
    body_order = []
    joint_armatures = {}
    links = []   # list of dicts, one per .body link entry

    default_joint_el = tree.find('default/joint')
    default_armature = float(default_joint_el.get('armature', '0')) if default_joint_el is not None else 0.0
    default_damping  = float(default_joint_el.get('damping',  '1')) if default_joint_el is not None else 1.0

    def parse_vec(s):
        return [float(x) for x in s.split()]

    def capsule_inertia(length, radius, density):
        r, l = radius, length
        m_cyl = density * math.pi * r**2 * l
        m_cap = density * (4.0/3.0) * math.pi * r**3
        m = m_cyl + m_cap
        d_hemi = l/2.0 - 3.0*r/8.0
        Iperp  = m_cyl*(r**2/4.0 + l**2/12.0) + m_cap*(2.0*r**2/5.0 + d_hemi**2)
        Iaxial = m * r**2 / 2.0
        return m, Iperp, Iaxial   # capsule along Y: Ixx=Izz=Iperp, Iyy=Iaxial

    def sphere_inertia(radius, density):
        m = density * (4/3) * math.pi * radius**3
        I = 0.4 * m * radius**2
        return m, I

    def rot_y_to_vec(d):
        """Rotation (axis, angle_deg) that maps Y-axis onto direction d."""
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

    def make_shape(geom_el, body_global_pos):
        if geom_el is None:
            return None
        bpos = np.asarray(body_global_pos, dtype=float)
        gtype = geom_el.get('type', 'sphere')

        if gtype == 'capsule' and 'fromto' in geom_el.attrib:
            fv  = parse_vec(geom_el.get('fromto'))
            p0  = np.array(fv[:3])
            p1  = np.array(fv[3:])
            if is_global_coord:
                p0 -= bpos; p1 -= bpos
            center = (p0 + p1) / 2.0
            diff   = p1 - p0
            length = float(np.linalg.norm(diff))
            radius = float(geom_el.get('size', '0.08'))
            m, Iperp, Iaxial = capsule_inertia(length, radius, default_density)
            rot_axis, rot_angle = rot_y_to_vec(diff)
            return dict(type='capsule', center=center.tolist(), length=length,
                        radius=radius, rot_axis=rot_axis, rot_angle=rot_angle,
                        mass=m, Iperp=Iperp, Iaxial=Iaxial)

        elif gtype == 'sphere':
            radius  = float(geom_el.get('size', '0.25'))
            pos_raw = np.array(parse_vec(geom_el.get('pos', '0 0 0')))
            pos_loc = (pos_raw - bpos) if is_global_coord else pos_raw
            m, I    = sphere_inertia(radius, default_density)
            return dict(type='sphere', center=pos_loc.tolist(), radius=radius, mass=m, I=I)

        elif gtype == 'box':
            sz      = [float(x) for x in geom_el.get('size', '1 1 1').split()]
            pos_raw = np.array(parse_vec(geom_el.get('pos', '0 0 0')))
            pos_loc = (pos_raw - bpos) if is_global_coord else pos_raw
            sx, sy, sz_v = sz[0], sz[1], sz[2]
            m   = default_density * 8 * sx * sy * sz_v
            Ixx = m * (sy**2 + sz_v**2) / 12
            Iyy = m * (sx**2 + sz_v**2) / 12
            Izz = m * (sx**2 + sy**2)   / 12
            return dict(type='box', center=pos_loc.tolist(),
                        size=[2*sx, 2*sy, 2*sz_v], mass=m, Ixx=Ixx, Iyy=Iyy, Izz=Izz)

        return None

    def inertia9(shape):
        """Return 3x3 inertia as 9-element list (row-major)."""
        if shape is None:
            e = 1e-6
            return [e,0,0, 0,e,0, 0,0,e]
        t = shape['type']
        if t == 'capsule':
            Ip, Ia = shape['Iperp'], shape['Iaxial']
            return [Ip,0,0, 0,Ia,0, 0,0,Ip]   # Y-axis capsule
        if t == 'sphere':
            I = shape['I']
            return [I,0,0, 0,I,0, 0,0,I]
        if t == 'box':
            return [shape['Ixx'],0,0, 0,shape['Iyy'],0, 0,0,shape['Izz']]
        e = 1e-6
        return [e,0,0, 0,e,0, 0,0,e]

    # Per-.body file joint ID counter (reset for each worldbody child).
    # Choreonoid requires explicit joint_id >= 0 on each non-root, non-fixed
    # link for Body::numJoints() and Body::joint(i) to work correctly.
    _joint_id_ctr = [0]

    def add_link(name, parent, jtype, jname, jaxis, jrange_deg,
                 translation, mass, com, inertia, shape, damping=None):
        if jtype in ('revolute', 'prismatic'):
            jid = _joint_id_ctr[0]
            _joint_id_ctr[0] += 1
        else:
            jid = -1
        links.append(dict(name=name, parent=parent, jtype=jtype, jname=jname,
                          jaxis=jaxis, jrange=jrange_deg, translation=translation,
                          mass=mass, com=com, inertia=inertia, shape=shape,
                          joint_id=jid, damping=damping))

    def process_one_joint(j_el, parent_name, child_name, translation,
                          mass, com, inr, shape):
        jtype_mj = j_el.get('type', 'hinge')
        jname    = j_el.get('name', f'{child_name}_joint')
        armature = float(j_el.get('armature', default_armature))
        joint_armatures[jname] = armature

        if jtype_mj == 'free':
            add_link(child_name, parent_name, 'free', jname,
                     None, None, translation, mass, com, inr, shape)
        elif jtype_mj == 'hinge':
            axis    = parse_vec(j_el.get('axis', '0 0 1'))
            rng_raw = [float(x) for x in j_el.get('range', '-180 180').split()]
            rng_deg = rng_raw if angle_is_degree else [math.degrees(r) for r in rng_raw]
            add_link(child_name, parent_name, 'revolute', jname,
                     axis, rng_deg, translation, mass, com, inr, shape)
        elif jtype_mj in ('slide', 'prismatic'):
            axis    = parse_vec(j_el.get('axis', '1 0 0'))
            rng_raw = [float(x) for x in j_el.get('range', '-10 10').split()]
            damping = float(j_el.get('damping', default_damping))
            add_link(child_name, parent_name, 'prismatic', jname,
                     axis, rng_raw, translation, mass, com, inr, shape,
                     damping=damping)

    def process_body(body_el, parent_name, parent_global_pos=None):
        if parent_global_pos is None:
            parent_global_pos = np.zeros(3)

        bname      = body_el.get('name')
        body_order.append(bname)
        global_pos = np.array(parse_vec(body_el.get('pos', '0 0 0')))
        trans      = ((global_pos - parent_global_pos) if is_global_coord
                      else global_pos).tolist()

        shape = make_shape(body_el.find('geom'),
                           global_pos if is_global_coord else np.zeros(3))
        mass  = shape['mass'] if shape else 0.001
        com   = shape['center'] if shape else [0.0, 0.0, 0.0]
        inr   = inertia9(shape)

        joint_els = body_el.findall('joint')

        if not joint_els:
            add_link(bname, parent_name, 'fixed', None,
                     None, None, trans, mass, com, inr, shape)
            last_name = bname
        elif len(joint_els) == 1:
            process_one_joint(joint_els[0], parent_name, bname,
                              trans, mass, com, inr, shape)
            last_name = bname
        else:
            # Multi-joint body (e.g. cube with 2 slide joints).
            # When this body is the tree root (parent_name is None), insert a
            # fixed dummy root so all prismatic/revolute joints are counted in
            # body.numJoints regardless of Choreonoid's root-joint convention.
            if parent_name is None:
                fixed_root_name = f'{bname}_fixed_root'
                add_link(fixed_root_name, None, 'fixed', None,
                         None, None, trans, 0.001, [0.0,0.0,0.0],
                         inertia9(None), None)
                prev = fixed_root_name
                first_child_trans = [0.0, 0.0, 0.0]
            else:
                prev = parent_name
                first_child_trans = trans
            for idx, j_el in enumerate(joint_els):
                if idx < len(joint_els) - 1:
                    vname = f'{bname}_virt{idx}'
                    process_one_joint(j_el, prev, vname,
                                      first_child_trans if idx == 0 else [0.0,0.0,0.0],
                                      0.001, [0.0,0.0,0.0],
                                      inertia9(None), None)
                    prev = vname
                else:
                    process_one_joint(j_el, prev, bname,
                                      [0.0,0.0,0.0],
                                      mass, com, inr, shape)
            last_name = bname

        for child in body_el.findall('body'):
            process_body(child, last_name,
                         global_pos if is_global_coord else np.zeros(3))

    # ---- Serialize helper -----------------------------------------------
    def fv(v, prec=8):
        return '[ ' + ', '.join(f'{x:.{prec}g}' for x in v) + ' ]'

    def serialize_links_to_yaml(links_list, root_name, body_name):
        out = [
            'format: ChoreonoidBody',
            'format_version: 2.0',
            'angle_unit: degree',
            f'name: {body_name}',
            f'root_link: "{root_name}"',
            'links:',
        ]
        for lk in links_list:
            out.append('  -')
            out.append(f'    name: "{lk["name"]}"')
            if lk['parent'] is not None:
                out.append(f'    parent: "{lk["parent"]}"')
            if lk['jname']:
                out.append(f'    joint_name: {lk["jname"]}')
            out.append(f'    joint_type: {lk["jtype"]}')
            if lk.get('joint_id', -1) >= 0:
                out.append(f'    joint_id: {lk["joint_id"]}')
            if lk['jaxis'] is not None:
                out.append(f'    joint_axis: {fv(lk["jaxis"], 6)}')
            if lk['jrange'] is not None:
                out.append(f'    joint_range: {fv(lk["jrange"], 6)}')
            if lk.get('damping') is not None:
                out.append(f'    joint_damping: {lk["damping"]:.6g}')
            out.append(f'    translation: {fv(lk["translation"])}')
            out.append(f'    mass: {lk["mass"]:.6g}')
            out.append(f'    center_of_mass: {fv(lk["com"])}')
            m = lk['inertia']
            out.append( '    inertia: [')
            out.append(f'      {m[0]:.6g}, {m[1]:.6g}, {m[2]:.6g},')
            out.append(f'      {m[3]:.6g}, {m[4]:.6g}, {m[5]:.6g},')
            out.append(f'      {m[6]:.6g}, {m[7]:.6g}, {m[8]:.6g} ]')
            shape = lk['shape']
            if shape:
                st = shape['type']
                cx, cy, cz = shape['center']
                out.append('    elements:')
                out.append('      -')
                out.append('        type: Shape')
                out.append(f'        translation: [ {cx:.6g}, {cy:.6g}, {cz:.6g} ]')
                if st == 'capsule':
                    ax, ay, az = shape['rot_axis']
                    ang = shape['rot_angle']
                    if abs(ang) > 0.01:
                        out.append(f'        rotation: [ {ax:.6g}, {ay:.6g}, {az:.6g}, {ang:.4g} ]')
                    r, h = shape['radius'], shape['length']
                    out.append(f'        geometry: {{ type: Capsule, radius: {r:.6g}, height: {h:.6g} }}')
                elif st == 'sphere':
                    out.append(f'        geometry: {{ type: Sphere, radius: {shape["radius"]:.6g} }}')
                elif st == 'box':
                    out.append(f'        geometry: {{ type: Box, size: {fv(shape["size"], 6)} }}')
        return '\n'.join(out) + '\n'

    # Process each top-level worldbody child as its own independent .body file.
    # Each child becomes the root of its BodyItem — no shared "world" wrapper —
    # so Choreonoid never encounters a non-root FreeJoint (which triggers the
    # "all link position recording" dialog and breaks headless operation).
    all_body_defs = []      # list of (item_name, yaml_str)
    all_body_order = []
    all_armatures = {}

    for body_idx, child_el in enumerate(tree.findall('worldbody/body')):
        links.clear()
        body_order.clear()
        joint_armatures.clear()
        _joint_id_ctr[0] = 0

        process_body(child_el, None, np.zeros(3))

        root_name = links[0]['name']
        # Each BodyItem needs a unique Choreonoid body name for findSimulationBody().
        # Robot (first child) keeps robot_name; extras use their XML name attribute.
        if body_idx == 0:
            body_name = robot_name
        else:
            body_name = child_el.get('name', f'extra_{body_idx}')
        yaml_str = serialize_links_to_yaml(links, root_name, body_name)
        all_body_defs.append((body_name, yaml_str))
        all_body_order.extend(body_order)
        all_armatures.update(joint_armatures)

    return all_body_defs, all_body_order, actuators, timestep, all_armatures


# ---------------------------------------------------------------------------
# State extraction helpers
# ---------------------------------------------------------------------------

def _rot_to_quat_wxyz(R):
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
    w, x, y, z = quat
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def _get_model_info(sim_body_list, actuators_map, timestep):
    """Build combined model info from a list of SimulationBodies (robot first)."""
    body_names     = []
    jnt_qposadr    = []
    body_jntadr_all = []
    body_jntnum_all = []
    nq_total       = 0
    nv_total       = 0
    njoints_total  = 0

    for sim_body in sim_body_list:
        b       = sim_body.body()
        njoints = b.numJoints
        nlinks  = b.numLinks
        root    = b.rootLink
        is_floating  = (root.jointType == root.FreeJoint)
        qpos_offset  = 7 if is_floating else 0

        body_names += [b.link(i).name for i in range(nlinks)]

        # Absolute qpos indices for this body's non-free joints
        qpos_joint_start = nq_total + qpos_offset
        jnt_qposadr += list(range(qpos_joint_start, qpos_joint_start + njoints))

        joint_name_to_local = {b.joint(i).jointName: i for i in range(njoints)}
        for li in range(nlinks):
            lk    = b.link(li)
            jname = lk.jointName if hasattr(lk, 'jointName') else lk.name
            if jname in joint_name_to_local:
                body_jntadr_all.append(njoints_total + joint_name_to_local[jname])
                body_jntnum_all.append(1)
            else:
                body_jntadr_all.append(-1)
                body_jntnum_all.append(0)

        nq_total      += qpos_offset + njoints
        nv_total      += (6 if is_floating else 0) + njoints
        njoints_total += njoints

    actuator_names = [v['name'] for v in actuators_map.values()]
    ctrlrange      = [v['ctrlrange'] for v in actuators_map.values()]

    # Build init_qpos from actual body positions (not zeros).
    # Floating bodies (cube etc.) carry real world poses that must be preserved.
    init_qpos = []
    init_qvel = []
    for sim_body in sim_body_list:
        b    = sim_body.body()
        root = b.rootLink
        is_floating = (root.jointType == root.FreeJoint)
        if is_floating:
            p    = list(root.translation)
            quat = _rot_to_quat_wxyz(np.asarray(root.rotation))
            init_qpos += p + list(quat)
            init_qvel += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i in range(b.numJoints):
            init_qpos.append(b.joint(i).q)
            init_qvel.append(0.0)

    return {
        'nq': nq_total, 'nv': nv_total, 'timestep': timestep,
        'actuator_names': actuator_names, 'ctrlrange': ctrlrange,
        'body_names': body_names,
        'body_jntadr': body_jntadr_all, 'body_jntnum': body_jntnum_all,
        'jnt_qposadr': jnt_qposadr,
        'init_qpos': init_qpos, 'init_qvel': init_qvel,
    }


def _get_state_dict(sim_body_entries):
    """
    sim_body_entries: list of (sim_body, is_floating) — robot first, extras after.
    Returns combined qpos/qvel/body_xpos/body_xmat across all bodies.
    """
    qpos, qvel = [], []
    body_xpos, body_xmat = {}, {}
    n_nonfree_joints = 0

    for (sim_body, is_floating) in sim_body_entries:
        b    = sim_body.body()
        root = b.rootLink

        if is_floating:
            p    = root.translation
            quat = _rot_to_quat_wxyz(np.asarray(root.rotation))
            qpos += list(p) + quat
            qvel += list(root.v) + list(root.w)

        for i in range(b.numJoints):
            j = b.joint(i)
            qpos.append(j.q)
            qvel.append(j.dq)

        n_nonfree_joints += b.numJoints

        for i in range(b.numLinks):
            lk = b.link(i)
            body_xpos[lk.name] = list(lk.translation)
            body_xmat[lk.name] = np.asarray(lk.rotation).flatten().tolist()

    return {
        'qpos': qpos, 'qvel': qvel,
        'body_xpos': body_xpos, 'body_xmat': body_xmat,
        'ctrl': [0.0] * n_nonfree_joints,
    }


def _set_state(sim_body_entries, qpos, qvel):
    """
    sim_body_entries: list of (sim_body, is_floating) — robot first.
    qpos/qvel: combined arrays covering all bodies in order.
    """
    qpos_off = 0
    qvel_off = 0
    for (sim_body, is_floating) in sim_body_entries:
        b       = sim_body.body()
        root    = b.rootLink
        njoints = b.numJoints

        if is_floating:
            root.setTranslation(qpos[qpos_off:qpos_off+3])
            root.setRotation(_quat_wxyz_to_rot(qpos[qpos_off+3:qpos_off+7]))
            root.v = np.array(qvel[qvel_off:qvel_off+3])
            root.w = np.array(qvel[qvel_off+3:qvel_off+6])
            qpos_off += 7
            qvel_off += 6

        for i in range(njoints):
            j     = b.joint(i)
            j.q   = qpos[qpos_off + i]
            j.dq  = qvel[qvel_off + i]

        qpos_off += njoints
        qvel_off += njoints
        b.calcForwardKinematics()


# ---------------------------------------------------------------------------
# Choreonoid simulation world (one instance per ChoreonoidEnv)
# ---------------------------------------------------------------------------

class ChoreonoidSimWorld:
    def __init__(self):
        self.world_item        = None
        self.sim_item          = None
        self.body_items        = {}   # insertion-ordered: 'robot' first
        self.sim_bodies        = {}
        self._sim_body_entries = []   # [(sim_body, is_floating), ...] robot first
        self.actuators_map     = {}
        self.frame_skip        = 4
        self.is_running        = False
        self._setup_world()

    def _build_sim_body_entries(self):
        entries = []
        for sb in self.sim_bodies.values():
            b           = sb.body()
            is_floating = (b.rootLink.jointType == b.rootLink.FreeJoint)
            entries.append((sb, is_floating))
        self._sim_body_entries = entries

    def _setup_world(self):
        self.world_item = WorldItem()
        RootItem.instance.addChildItem(self.world_item)

        for candidate in (
            '/choreonoid_ws/install/share/choreonoid-2.3/model/misc/floor.body',
            '/choreonoid_ws/install/share/choreonoid-2.0/model/misc/floor.body',
        ):
            if os.path.exists(candidate):
                floor_item = BodyItem()
                floor_item.load(candidate)
                self.world_item.addChildItem(floor_item)
                break

        self.sim_item = AISTSimulatorItem()
        self.sim_item.setTimeStep(0.01)
        self.sim_item.setRealtimeSyncMode(3)  # manual / non-realtime
        self.world_item.addChildItem(self.sim_item)

    def _clear_bodies(self):
        for item in list(self.body_items.values()):
            item.detachFromParentItem()
        self.body_items.clear()
        self.sim_bodies.clear()
        self._sim_body_entries = []

    def _load_body_defs(self, body_defs: list, actuators_map: dict,
                        timestep: float, joint_armatures: dict) -> dict:
        """body_defs = [(name, yaml_str), ...] を Choreonoid にロードする共通処理。"""
        self.actuators_map = actuators_map
        self.sim_item.setTimeStep(timestep)

        for i, (item_name, body_yaml) in enumerate(body_defs):
            body_key = 'robot' if i == 0 else f'extra_{i-1}'

            with tempfile.NamedTemporaryFile(suffix='.body', mode='w', delete=False) as f:
                f.write(body_yaml)
                body_path = f.name

            body_item = BodyItem()
            loaded = body_item.load(body_path)
            os.unlink(body_path)
            if not loaded:
                raise RuntimeError(f"Failed to load .body for '{item_name}'")

            b = body_item.body
            for j in range(b.numJoints):
                jnt = b.joint(j)
                arm = joint_armatures.get(jnt.jointName, 0.0)
                if arm > 0:
                    jnt.setEquivalentRotorInertia(arm)

            body_item.storeInitialState()
            self.world_item.addChildItem(body_item)
            self.body_items[body_key] = body_item

        if self.is_running:
            self.sim_item.stopSimulation()
            IU.processEvent()  # flush stop signal before restart
        self.sim_item.startSimulation(doReset=True)
        self.is_running = True

        self.sim_item.tickRequest(True)
        IU.processEvent()

        for key, item in self.body_items.items():
            sb = self.sim_item.findSimulationBody(item.name)
            if sb is None:
                raise RuntimeError(f"SimulationBody not found for '{key}'")
            self.sim_bodies[key] = sb

        self._build_sim_body_entries()
        return _get_model_info(
            [e[0] for e in self._sim_body_entries], actuators_map, timestep
        )

    def load_model(self, xml_str: str, frame_skip: int) -> dict:
        self.frame_skip = frame_skip
        self._clear_bodies()

        body_defs, _all_order, actuators_map, timestep, joint_armatures = \
            mujoco_xml_to_body(xml_str)
        return self._load_body_defs(body_defs, actuators_map, timestep, joint_armatures)

    def load_model_from_body_defs(self, body_defs: list, frame_skip: int,
                                   actuators_map: dict = None,
                                   timestep: float = 0.01,
                                   joint_armatures: dict = None) -> dict:
        """
        DynamicBodyUpdater.generate_body_defs() の出力を直接受け取ってロードする。
        actuators_map: {joint_name: {ctrlrange, gear, name}} （省略可）
        """
        self.frame_skip = frame_skip
        self._clear_bodies()
        return self._load_body_defs(
            body_defs,
            actuators_map or {},
            timestep,
            joint_armatures or {},
        )

    def reset(self) -> dict:
        self.sim_item.stopSimulation()
        IU.processEvent()  # flush stop signal before restart
        for item in self.body_items.values():
            item.restoreInitialState(True)
        self.sim_item.startSimulation(doReset=True)
        self.sim_item.tickRequest(True)
        IU.processEvent()

        for key, item in self.body_items.items():
            sb = self.sim_item.findSimulationBody(item.name)
            if sb is not None:
                self.sim_bodies[key] = sb

        self._build_sim_body_entries()
        if not self._sim_body_entries:
            return {'qpos': [], 'qvel': [], 'body_xpos': {}, 'body_xmat': {}}
        return _get_state_dict(self._sim_body_entries)

    def step(self, ctrl: list, n_frames: int) -> dict:
        if not self._sim_body_entries:
            return {'qpos': [], 'qvel': [], 'body_xpos': {}, 'body_xmat': {}}

        # Apply ctrl to the robot body only (first entry)
        robot_sb, _ = self._sim_body_entries[0]
        b = robot_sb.body()
        for i, (jname, ainfo) in enumerate(self.actuators_map.items()):
            j = b.joint(jname)
            if j is not None and i < len(ctrl):
                j.u = float(ctrl[i]) * ainfo['gear']

        for _ in range(n_frames):
            self.sim_item.tickRequest(True)
            IU.processEvent()

        return _get_state_dict(self._sim_body_entries)

    def set_state_cmd(self, qpos: list, qvel: list) -> dict:
        if not self._sim_body_entries:
            return {'qpos': qpos, 'qvel': qvel, 'body_xpos': {}, 'body_xmat': {}}
        _set_state(self._sim_body_entries, qpos, qvel)
        return _get_state_dict(self._sim_body_entries)


# ---------------------------------------------------------------------------
# Public ChoreonoidEnv  (MujocoEnv-compatible public API)
# ---------------------------------------------------------------------------

class ChoreonoidEnv:
    def __init__(self, fullpath, frame_skip, mujoco_xml=None):
        self.frame_skip = frame_skip
        self._world = ChoreonoidSimWorld()

        if mujoco_xml is not None:
            xml_str = mujoco_xml
        else:
            if not os.path.exists(fullpath):
                fullpath = os.path.join(
                    Path(__file__).parent.parent.parent.parent,
                    'assets/mujoco_models', os.path.basename(fullpath)
                )
            with open(fullpath, 'r') as f:
                xml_str = f.read()

        self._last_xml = xml_str
        info = self._world.load_model(xml_str, frame_skip)
        self._apply_model_info(info)

        self.viewer    = None
        self._viewers  = {}
        self.np_random = None
        self.is_inited = False

        self._set_action_space()
        action = self.action_space.sample()
        observation, _reward, term, trunc, _info = self.step(action)
        assert not (term or trunc)
        self._set_observation_space(observation)
        self.seed()
        self.is_inited = True

    # ------------------------------------------------------------------
    # Internal helpers

    def _apply_model_info(self, info: dict):
        self.nq = info['nq']
        self.nv = info['nv']
        self._actuator_names    = info['actuator_names']
        self._actuator_ctrlrange = np.array(info['ctrlrange'], dtype=np.float64)
        self._timestep          = info['timestep']
        self.init_qpos          = np.array(info['init_qpos'], dtype=np.float64)
        self.init_qvel          = np.array(info['init_qvel'], dtype=np.float64)
        self._body_names        = info['body_names']
        self._body_jntadr       = info['body_jntadr']
        self._body_jntnum       = info['body_jntnum']
        self._jnt_qposadr       = info['jnt_qposadr']
        self._qpos     = np.zeros(self.nq)
        self._qvel     = np.zeros(self.nv)
        self._ctrl     = np.zeros(len(self._actuator_names))
        self._body_xpos = {n: np.zeros(3)  for n in self._body_names}
        self._body_xmat = {n: np.eye(3)    for n in self._body_names}

    def _cache_state(self, resp: dict):
        self._qpos      = np.array(resp['qpos'], dtype=np.float64)
        self._qvel      = np.array(resp['qvel'], dtype=np.float64)
        self._body_xpos = {k: np.array(v)              for k, v in resp.get('body_xpos', {}).items()}
        self._body_xmat = {k: np.array(v).reshape(3,3) for k, v in resp.get('body_xmat', {}).items()}
        self._ctrl      = np.array(resp.get('ctrl', [0.0]*len(self._actuator_names)))

    # ------------------------------------------------------------------
    # MujocoEnv-compatible properties

    @property
    def dt(self):
        return self._timestep * self.frame_skip

    @property
    def model(self):
        return _ModelProxy(self)

    @property
    def data(self):
        return _DataProxy(self)

    # ------------------------------------------------------------------
    # Public API

    def _set_action_space(self):
        low  = self._actuator_ctrlrange[:, 0].astype(np.float32)
        high = self._actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        from collections import OrderedDict
        if isinstance(observation, dict):
            self.observation_space = spaces.Dict(OrderedDict([
                (k, self._box_from_array(v)) for k, v in observation.items()
            ]))
        elif isinstance(observation, np.ndarray):
            low  = np.full(observation.shape, -np.inf, dtype=np.float32)
            high = np.full(observation.shape,  np.inf, dtype=np.float32)
            self.observation_space = spaces.Box(low, high, dtype=observation.dtype)
        elif isinstance(observation, list):
            self.observation_space = None
        return self.observation_space

    def _box_from_array(self, arr):
        arr = np.asarray(arr)
        low  = np.full(arr.shape, -np.inf, dtype=np.float32)
        high = np.full(arr.shape,  np.inf, dtype=np.float32)
        return spaces.Box(low, high, dtype=arr.dtype)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        resp = self._world.reset()
        self._cache_state(resp)
        return self.reset_model()

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.nq,) and qvel.shape == (self.nv,)
        resp = self._world.set_state_cmd(qpos.tolist(), qvel.tolist())
        self._cache_state(resp)

    def do_simulation(self, ctrl, n_frames):
        resp = self._world.step(ctrl.tolist(), n_frames)
        self._cache_state(resp)

    def reload_sim_model(self, xml_str: str):
        self._last_xml = xml_str
        info = self._world.load_model(xml_str, self.frame_skip)
        self._apply_model_info(info)
        self._set_action_space()

    def state_vector(self):
        return np.concatenate([self._qpos, self._qvel])

    def get_body_com(self, body_name: str) -> np.ndarray:
        return self._body_xpos[body_name]

    def vec_body2world(self, body_name: str, vec: np.ndarray) -> np.ndarray:
        return (self._body_xmat[body_name] @ vec[:, None]).ravel()

    def pos_body2world(self, body_name: str, pos: np.ndarray) -> np.ndarray:
        xpos = self._body_xpos[body_name]
        xmat = self._body_xmat[body_name]
        return (xmat @ pos[:, None]).ravel() + xpos

    def close(self):
        if self._world.is_running:
            self._world.sim_item.stopSimulation()
            self._world.is_running = False

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        pass

    # ------------------------------------------------------------------
    # Subclass hooks

    def step(self, action):
        raise NotImplementedError

    def reset_model(self):
        raise NotImplementedError

    def viewer_setup(self):
        pass


# ---------------------------------------------------------------------------
# Proxies so env code using self.model.nq / self.data.qpos keeps working
# ---------------------------------------------------------------------------

class _ModelProxy:
    def __init__(self, env: ChoreonoidEnv):
        self._env = env

    @property
    def nq(self):                   return self._env.nq
    @property
    def nv(self):                   return self._env.nv
    @property
    def nu(self):                   return len(self._env._actuator_names)
    @property
    def actuator_names(self):       return self._env._actuator_names
    @property
    def actuator_ctrlrange(self):   return self._env._actuator_ctrlrange
    @property
    def body_names(self):           return self._env._body_names
    @property
    def body_jntadr(self):          return self._env._body_jntadr
    @property
    def body_jntnum(self):          return self._env._body_jntnum
    @property
    def jnt_qposadr(self):          return self._env._jnt_qposadr

    class _Opt:
        def __init__(self, timestep): self.timestep = timestep
    @property
    def opt(self): return self._Opt(self._env._timestep)

    def _camera_name2id(self): return {}

    @property
    def _body_name2id(self):
        return {n: i for i, n in enumerate(self._env._body_names)}


class _DataProxy:
    def __init__(self, env: ChoreonoidEnv):
        self._env = env

    @property
    def qpos(self): return self._env._qpos
    @property
    def qvel(self): return self._env._qvel
    @property
    def ctrl(self): return self._env._ctrl

    @property
    def body_xpos(self):
        names = self._env._body_names
        arr = np.zeros((len(names), 3))
        for i, n in enumerate(names):
            if n in self._env._body_xpos:
                arr[i] = self._env._body_xpos[n]
        return arr

    def get_body_xpos(self, name: str) -> np.ndarray:
        return self._env._body_xpos.get(name, np.zeros(3))

    def get_body_xmat(self, name: str) -> np.ndarray:
        return self._env._body_xmat.get(name, np.eye(3))
