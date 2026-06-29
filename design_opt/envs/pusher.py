import numpy as np
import os
from gym import utils
if os.environ.get('USE_CHOREONOID', '0') == '1':
    from khrylib.rl.envs.common.mujoco_env_choreonoid import ChoreonoidEnv as MujocoEnv
else:
    from khrylib.rl.envs.common.mujoco_env_gym import MujocoEnv
from khrylib.robot.xml_robot import Robot
from khrylib.utils import get_single_body_qposaddr, get_graph_fc_edges
from khrylib.utils.transformation import quaternion_matrix
from copy import deepcopy
try:
    import mujoco_py
except Exception:
    mujoco_py = None
import time
import os
import shutil
import os.path as osp

class PusherEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, cfg, agent):  
        self.cur_t = 0
        self.cfg = cfg
        self.env_specs = cfg.env_specs
        self.agent = agent
        if self.cfg.xml_name == "default":
            self.model_xml_file = os.path.join(cfg.project_path, "assets", "mujoco_envs", "pusher.xml")
        else:
            self.model_xml_file = os.path.join(cfg.project_path, "assets", "mujoco_envs", f"{self.cfg.xml_name}.xml")
        # robot xml
        self.robot = Robot(cfg.robot_cfg, xml=self.model_xml_file)
        self.init_xml_str = self.robot.export_xml_string()
        self.cur_xml_str = self.init_xml_str.decode('utf-8')
        # design options
        self.clip_qvel = cfg.obs_specs.get('clip_qvel', False)
        self.use_projected_params = cfg.obs_specs.get('use_projected_params', True)
        self.abs_design = cfg.obs_specs.get('abs_design', False)
        self.use_body_ind = cfg.obs_specs.get('use_body_ind', False)
        self.use_body_depth_height = cfg.obs_specs.get('use_body_depth_height', False)
        self.use_shortest_distance = cfg.obs_specs.get('use_shortest_distance', False)
        self.use_position_encoding = cfg.obs_specs.get('use_position_encoding', False)
        self.design_ref_params = self.get_attr_design()
        self.design_cur_params = self.design_ref_params.copy()
        self.design_param_names = self.robot.get_params(get_name=True)
        self.attr_design_dim = self.design_ref_params.shape[-1]
        self.index_base = 5
        self.stage = 'skeleton_transform'    # transform or execute
        self.control_nsteps = 0
        self.sim_specs = set(cfg.obs_specs.get('sim', []))
        self.attr_specs = set(cfg.obs_specs.get('attr', []))
        MujocoEnv.__init__(self, self.model_xml_file, 4)
        utils.EzPickle.__init__(self)
        self.control_action_dim = 1
        self.skel_num_action = 3 if cfg.enable_remove else 2
        self.sim_obs_dim = self.get_sim_obs().shape[-1]
        self.attr_fixed_dim = self.get_attr_fixed().shape[-1]
        # Potential-based rewards: φ(s) = 1/(1+dist).
        # Both potentials store φ at episode start; step() uses Δφ so static
        # morphology/pose yields zero reward — preserving the Stackelberg coupling
        # (Follower must actively move; Leader is rewarded for enabling that motion).
        self.prev_contact_potential = None  # φ = 1/(1+dist(arm_tip, cube))
        self.prev_cube_potential = None     # φ = 1/(1+dist(cube, target))  [use_target only]

    def allow_add_body(self, body):
        add_body_condition = self.cfg.add_body_condition
        max_nchild = add_body_condition.get('max_nchild', 3)
        min_nchild = add_body_condition.get('min_nchild', 0)
        return body.depth >= self.cfg.min_body_depth and body.depth < self.cfg.max_body_depth - 1 and len(body.child) < max_nchild and len(body.child) >= min_nchild
    
    def allow_remove_body(self, body):
        if body.depth >= self.cfg.min_body_depth + 1 and len(body.child) == 0:
            if body.depth == 1:
                return body.parent.child.index(body) > 0
            else:
                return True
        return False

    def apply_skel_action(self, skel_action):
        bodies = list(self.robot.bodies)
        for body, a in zip(bodies, skel_action):
            if a == 1 and self.allow_add_body(body):
                self.robot.add_child_to_body(body)
            if a == 2 and self.allow_remove_body(body):
                self.robot.remove_body(body)

        xml_str = self.robot.export_xml_string()
        self.cur_xml_str = xml_str.decode('utf-8')
        try:
            self.reload_sim_model(xml_str.decode('utf-8'))
        except:
            print(self.cur_xml_str)
            return False      
        self.design_cur_params = self.get_attr_design()
        return True

    def set_design_params(self, in_design_params):
        design_params = in_design_params
        for params, body in zip(design_params, self.robot.bodies):
            body.set_params(params, pad_zeros=True, map_params=True)
            body.sync_node()
        xml_str = self.robot.export_xml_string()
        self.cur_xml_str = xml_str.decode('utf-8')
        try:
            self.reload_sim_model(xml_str.decode('utf-8'))
        except:
            print(self.cur_xml_str)
            return False
        if self.use_projected_params:
            self.design_cur_params = self.get_attr_design()
        else:
            self.design_cur_params = in_design_params.copy()
        return True

    def action_to_control(self, a):
        ctrl = np.zeros_like(self.data.ctrl)
        assert a.shape[0] == len(self.robot.bodies)
        for body, body_a in zip(self.robot.bodies[1:], a[1:]):
            aname = body.get_actuator_name()
            if aname in self.model.actuator_names:
                aind = self.model.actuator_names.index(aname)
                ctrl[aind] = body_a.item()
        return ctrl        

    def step(self, a):
        if not self.is_inited:
            return self._get_obs(), 0, False, False, {'use_transform_action': False, 'stage': 'execution', 'reward_ctrl': 0.0}

        self.cur_t += 1
        # skeleton transform stage
        if self.stage == 'skeleton_transform':
            if getattr(self.cfg, 'fix_skeleton', False):
                self.transit_attribute_transform()
                ob = self._get_obs()
                return ob, 0.0, False, False, {'use_transform_action': True, 'stage': 'skeleton_transform', 'reward_ctrl': 0.0}

            skel_a = a[:, -1]
            succ = self.apply_skel_action(skel_a)
            if not succ:
                return self._get_obs(), 0.0, True, False, {'use_transform_action': True, 'stage': 'skeleton_transform', 'reward_ctrl': 0.0}

            if self.cur_t == self.cfg.skel_transform_nsteps:
                self.transit_attribute_transform()

            ob = self._get_obs()
            reward = 0.0
            termination = truncation = False
            return ob, reward, termination, truncation, {'use_transform_action': True, 'stage': 'skeleton_transform', 'reward_ctrl': 0.0}
        # attribute transform stage
        elif self.stage == 'attribute_transform':
            design_a = a[:, self.control_action_dim:-1]
            # Clamp NaN/Inf in action before applying (prevents design_cur_params corruption)
            if not np.isfinite(design_a).all():
                design_a = np.nan_to_num(design_a, nan=0.0, posinf=0.0, neginf=0.0)
            if self.abs_design:
                design_params = design_a * self.cfg.robot_param_scale
            else:
                design_params = self.design_cur_params + design_a * self.cfg.robot_param_scale
            succ = self.set_design_params(design_params)
            if not succ:
                return self._get_obs(), 0.0, True, False, {'use_transform_action': True, 'stage': 'attribute_transform', 'reward_ctrl': 0.0}
            reward = 0.0
            if self.cur_t == self.cfg.skel_transform_nsteps + 1:
                succ = self.transit_execution()
                if not succ:
                    return self._get_obs(), 0.0, True, False, {'use_transform_action': True, 'stage': 'attribute_transform', 'reward_ctrl': 0.0}
                # R^L: one-shot leader-only bonus for a morphology whose reach
                # annulus geometrically covers the cube (see タスク設計と報酬関数.md セクション8).
                # Additive on top of the inherited follower return — does not
                # touch execution-phase reward, preserving Stackelberg coupling.
                reward = self.compute_reach_bonus()

            ob = self._get_obs()
            termination = truncation = False
            return ob, reward, termination, truncation, {'use_transform_action': True, 'stage': 'attribute_transform', 'reward_ctrl': 0.0}
        # execution stage
        else:
            self.control_nsteps += 1
            assert np.all(a[:, self.control_action_dim:] == 0)
            control_a = a[:, :self.control_action_dim]
            ctrl = self.action_to_control(control_a)
            ctrl_cost_coeff = self.cfg.reward_specs.get('ctrl_cost_coeff', 1e-4)
            xposbefore = self.get_body_com("cube")[0]
            yposbefore = self.get_body_com("cube")[1]
            try:
                self.do_simulation(ctrl, self.frame_skip)
            except:
                print(self.cur_xml_str)
                return self._get_obs(), 0, True, False, {'use_transform_action': False, 'stage': 'execution', 'reward_ctrl': 0.0}
            
            xposafter = self.get_body_com("cube")[0]
            yposafter = self.get_body_com("cube")[1]

            use_target = self.cfg.reward_specs.get('use_target_reward', False)
            if use_target:
                target_x = self.cfg.reward_specs.get('target_x', 1.5)
                target_y = self.cfg.reward_specs.get('target_y', 0.0)
                dist_to_target = np.linalg.norm(
                    np.array([xposafter, yposafter]) - np.array([target_x, target_y]))
                # PBRS: Δφ_cube so static cube yields zero reward, preserving
                # the Stackelberg coupling (Follower must push to get reward).
                curr_cube_potential = 1.0 / (1.0 + dist_to_target)
                if self.prev_cube_potential is None:
                    self.prev_cube_potential = curr_cube_potential
                reward_fwd_cube = curr_cube_potential - self.prev_cube_potential
                self.prev_cube_potential = curr_cube_potential
            else:
                reward_fwd_cube = (xposafter - xposbefore) / self.dt - 0.1 * np.abs(yposafter - yposbefore) / self.dt

            # Potential-Based Reward Shaping: reward = φ(t+1) - φ(t), φ = 1/(1+dist).
            # Static arm → Δφ = 0, no free reward.
            # Arm approaching cube → Δφ > 0, exploration guided as before.
            # Preserves design intent (morphology that enables fast approach = higher reward)
            # while eliminating the static-proximity exploitation.
            arm_ref_body = self.robot.bodies[-1].name if self.is_fixed_base else "0"
            curr_dist = np.linalg.norm(self.get_body_com("cube") - self.get_body_com(arm_ref_body))
            curr_contact_potential = 1.0 / (1.0 + curr_dist)
            if self.prev_contact_potential is None:
                self.prev_contact_potential = curr_contact_potential
            contact_weight = self.cfg.reward_specs.get('contact_weight', 1.0)
            reward_fwd_contact = contact_weight * (curr_contact_potential - self.prev_contact_potential)
            self.prev_contact_potential = curr_contact_potential
            reward_fwd = reward_fwd_cube + reward_fwd_contact
            reward_ctrl = - ctrl_cost_coeff * np.square(ctrl).mean()
            alive_bonus = self.cfg.reward_specs.get('alive_bonus', 0.0)
            reward = reward_fwd + reward_ctrl + alive_bonus
            scale = self.cfg.reward_specs.get('exec_reward_scale', 1.0)
            reward *= scale

            s = self.state_vector()
            done_condition = self.cfg.done_condition
            max_nsteps = done_condition.get('max_nsteps', 1000)
            if self.is_fixed_base:
                termination = not np.isfinite(s).all()
            else:
                height = s[2]
                zdir = quaternion_matrix(s[3:7])[:3, 2]
                ang = np.arccos(zdir[2])
                min_height = done_condition.get('min_height', 0.0)
                max_height = done_condition.get('max_height', 2.0)
                max_ang = done_condition.get('max_ang', 3600)
                termination = not (np.isfinite(s).all() and (height > min_height) and (height < max_height) and (abs(ang) < np.deg2rad(max_ang)))
            truncation = not (self.control_nsteps < max_nsteps)
            ob = self._get_obs()
            return ob, reward, termination, truncation, {'use_transform_action': False, 'stage': 'execution', 'reward_ctrl': reward_ctrl}

    def compute_reach_bonus(self):
        """Leader-only design-phase bonus (R^L): reward morphologies whose 2-link
        reach annulus [|L1-L2|, L1+L2] around the shoulder pivot geometrically
        covers the cube's (already-sampled) position.

        Uses body.bone_offset (design-space link length, exact) rather than
        get_body_com, since get_body_com returns the *subtree* COM (includes
        descendants) and so cannot isolate a single link's length. The shoulder
        pivot is read via body_xpos (frame origin), not COM, for the same reason.
        Both are pose-independent: a link's own length and its parent's frame
        origin are unaffected by the current joint angles (arm_safe_init etc).
        """
        scale = self.cfg.reward_specs.get('reach_bonus_scale', 0.0)
        if scale == 0.0 or not self.is_fixed_base or len(self.robot.bodies) < 3:
            return 0.0
        bodies = self.robot.bodies
        L1 = float(np.linalg.norm(np.asarray(bodies[1].bone_offset, dtype=float)))
        L2 = float(np.linalg.norm(np.asarray(bodies[-1].bone_offset, dtype=float)))
        shoulder_xy = self.data.body_xpos[self.model._body_name2id[bodies[1].name]][:2]
        cube_xy = self.get_body_com("cube")[:2]
        d = np.linalg.norm(cube_xy - shoulder_xy)
        reach_min, reach_max = abs(L1 - L2), L1 + L2
        excess = max(0.0, reach_min - d, d - reach_max)
        k = self.cfg.reward_specs.get('reach_bonus_k', 3.0)
        return scale * np.exp(-k * excess)

    def transit_attribute_transform(self):
        self.stage = 'attribute_transform'

    def transit_execution(self):
        self.stage = 'execution'
        self.control_nsteps = 0
        try:
            self.reset_state(True)
        except:
            print(self.cur_xml_str)
            return False
        # Snapshot φ values at episode start so step() can compute Δφ.
        arm_ref_body = self.robot.bodies[-1].name if self.is_fixed_base else "0"
        dist0 = np.linalg.norm(self.get_body_com("cube") - self.get_body_com(arm_ref_body))
        self.prev_contact_potential = 1.0 / (1.0 + dist0)
        use_target = self.cfg.reward_specs.get('use_target_reward', False)
        if use_target:
            target_x = self.cfg.reward_specs.get('target_x', 1.5)
            target_y = self.cfg.reward_specs.get('target_y', 0.0)
            cube_pos = self.get_body_com("cube")[:2]
            dist_cube0 = np.linalg.norm(cube_pos - np.array([target_x, target_y]))
            self.prev_cube_potential = 1.0 / (1.0 + dist_cube0)
        else:
            self.prev_cube_potential = None
        return True
        

    @property
    def is_fixed_base(self):
        root_joints = self.robot.bodies[0].joints
        return all(j.type != 'free' for j in root_joints)

    def if_use_transform_action(self):
        return ['skeleton_transform', 'attribute_transform', 'execution'].index(self.stage)

    def get_sim_obs(self):
        obs = []
        if 'root_offset' in self.sim_specs:
            root_pos = self.data.body_xpos[self.model._body_name2id[self.robot.bodies[0].name]]
            
        for i, body in enumerate(self.robot.bodies):
            qvel = self.data.qvel.copy()
            if self.clip_qvel:
                qvel = np.clip(qvel, -10, 10)
            if i == 0:
                arm_ref_body = self.robot.bodies[-1].name if self.is_fixed_base else "0"
                relative_dis = self.get_body_com("cube") - self.get_body_com(arm_ref_body)
                if self.is_fixed_base:
                    # fixed base: no free joint state; fill with zeros to keep 17-dim structure
                    obs_i = [np.zeros(11), relative_dis, np.zeros(3)]
                else:
                    obs_i = [self.data.qpos[2:7], qvel[:6], relative_dis, np.zeros(3)]
            else:
                qs, qe = get_single_body_qposaddr(self.model, body.name)
                if qe - qs >= 1:
                    assert qe - qs == 1
                    # jnt_dofadr accounts for free-joint qpos/qvel size mismatch (7 qpos vs 6 qvel).
                    # Choreonoid _ModelProxy lacks jnt_dofadr; fall back to jnt_qposadr which
                    # equals jnt_dofadr for fixed-base bodies (no free joint offset).
                    body_id = self.model._body_name2id[body.name]
                    jnt_adr = int(self.model.body_jntadr[body_id])
                    dof_adr = self.model.jnt_dofadr if hasattr(self.model, 'jnt_dofadr') else self.model.jnt_qposadr
                    vs = int(dof_adr[jnt_adr])
                    obs_i = [np.zeros(15), self.data.qpos[qs:qe], qvel[vs:vs+1]]
                else:
                    obs_i = [np.zeros(17)]
            if 'root_offset' in self.sim_specs:
                offset = self.data.body_xpos[self.model._body_name2id[body.name]][[0, 2]] - root_pos[[0, 2]]
                obs_i.append(offset)
            obs_i = np.concatenate(obs_i)
            obs.append(obs_i)
        obs = np.stack(obs)
        return obs

    def get_attr_fixed(self):
        obs = []
        for i, body in enumerate(self.robot.bodies):
            obs_i = []
            if 'depth' in self.attr_specs:
                obs_depth = np.zeros(self.cfg.max_body_depth)
                obs_depth[body.depth] = 1.0
                obs_i.append(obs_depth)
            if 'jrange' in self.attr_specs:
                obs_jrange = body.get_joint_range()
                obs_i.append(obs_jrange)
            if 'skel' in self.attr_specs:
                obs_add = self.allow_add_body(body)
                obs_rm = self.allow_remove_body(body)
                obs_i.append(np.array([float(obs_add), float(obs_rm)]))
            if len(obs_i) > 0:
                obs_i = np.concatenate(obs_i)
                obs.append(obs_i)
        
        if len(obs) == 0:
            return None
        obs = np.stack(obs)
        return obs

    def get_attr_design(self):
        obs = []
        for i, body in enumerate(self.robot.bodies):
            obs_i = body.get_params([], pad_zeros=True, demap_params=True)
            obs.append(obs_i)
        obs = np.stack(obs)
        return obs

    def get_body_index(self):
        index = []
        for i, body in enumerate(self.robot.bodies):
            ind = int(body.name, base=self.index_base)
            index.append(ind)
        index = np.array(index)
        return index

    def get_body_height(self):
        heights = []
        for i, body in enumerate(self.robot.bodies):
            h = body.height
            heights.append(h)
        heights = np.array(heights)
        return heights
        
    def get_body_depth(self):
        depths = []
        for i, body in enumerate(self.robot.bodies):
            d = body.depth
            depths.append(d)
        depths = np.array(depths)
        return depths

    def _get_obs(self):
        obs = []
        attr_fixed_obs = self.get_attr_fixed()
        sim_obs = self.get_sim_obs()
        design_obs = self.design_cur_params
        obs = np.concatenate(list(filter(lambda x: x is not None, [attr_fixed_obs, sim_obs, design_obs])), axis=-1)
        if self.cfg.obs_specs.get('fc_graph', False):
            edges = get_graph_fc_edges(len(self.robot.bodies))
        else:
            edges = self.robot.get_gnn_edges()
        use_transform_action = np.array([self.if_use_transform_action()])
        num_nodes = np.array([sim_obs.shape[0]])
        all_obs = [obs, edges, use_transform_action, num_nodes]
        if self.use_body_ind:
            body_index = self.get_body_index()
            all_obs.append(body_index)
        if self.use_body_depth_height:
            body_depths = self.get_body_depth()
            all_obs.append(body_depths)
            body_heights = self.get_body_height()
            all_obs.append(body_heights)
        if self.use_shortest_distance:
            distances = self.robot.get_shortest_distances()
            all_obs.append(distances)
        if self.use_position_encoding:
            lapPE = self.robot.get_laplacian_position_encoding()
            all_obs.append(lapPE)
        return all_obs

    def reset_state(self, add_noise):
        if add_noise:
            qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
            qvel = self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        else:
            qpos = self.init_qpos.copy()
            qvel = self.init_qvel.copy()

        # Cube x-position offset + per-episode noise.
        # Prevents penetration-impulse exploit: at qpos[cube_x]=0 the cube center is at
        # world x=1.0m (body pos in rrbot_arm.xml) with half-size 0.15m → left face x=0.85m.
        # Morphology optimization can grow the arm so the forearm passes through the cube
        # at episode start (v2: elbow at x=0.830, forearm diagonal through cube interior),
        # causing the physics engine to fire a separation impulse that launches the cube
        # without any active arm control.
        # cube_x_offset=0.5 → cube center at x=1.5m, left face at x=1.35m,
        # well beyond default arm reach (~0.55m). Arm must actively grow and push.
        # qpos layout (fix_skeleton=True, 2-joint arm + 2-joint cube): [j1, j11, cube_x, cube_y]
        cube_x_offset = self.env_specs.get('cube_x_offset', 0.0)
        cube_x_noise  = self.env_specs.get('cube_x_noise',  0.0)
        if cube_x_offset != 0.0 or cube_x_noise != 0.0:
            cube_x_idx = self.model.nq - 2
            base      = float(self.init_qpos[cube_x_idx]) + cube_x_offset
            extra     = self.np_random.uniform(-cube_x_noise, cube_x_noise) if add_noise else 0.0
            qpos[cube_x_idx] = base + extra

        # Safe initial arm pose: set shoulder to π/2 so arm points in +y direction.
        # Prevents penetration-impulse exploit when morphology optimizer grows arm toward
        # +x (cube direction): at qpos[0]=π/2 the arm always starts pointing away from cube,
        # so no initial overlap regardless of arm length.
        # Requires shoulder joint range widened to ±90° in rrbot_arm.xml.
        if self.env_specs.get('arm_safe_init', False):
            qpos[0] = np.pi / 2

        if self.env_specs.get('init_height', True) and not self.is_fixed_base:
            qpos[2] = 0.4

        # Cube slide joints must start at rest regardless of add_noise.
        # transit_execution() always calls reset_state(True), so ±0.1 velocity noise
        # would be applied to cube_slide / cube_slide2 even in eval mode.
        # With damping=10, τ=m/b=2.7/10=0.27s; initial velocity of 0.1 m/s takes
        # ~1s to decay — clearly visible as drift. Cube has no actuator, so velocity
        # noise provides zero exploration benefit.
        cube_x_idx = self.model.nq - 2
        qvel[cube_x_idx]     = 0.0  # cube_slide (x)
        qvel[cube_x_idx + 1] = 0.0  # cube_slide2 (y)

        self.set_state(qpos, qvel)

    def reset_robot(self):
        del self.robot
        self.robot = Robot(self.cfg.robot_cfg, xml=self.init_xml_str, is_xml_str=True)
        self.cur_xml_str = self.init_xml_str.decode('utf-8')
        self.reload_sim_model(self.cur_xml_str)
        self.design_ref_params = self.get_attr_design()
        self.design_cur_params = self.design_ref_params.copy()

    def reset_model(self):
        self.reset_robot()
        self.control_nsteps = 0
        self.stage = 'skeleton_transform'
        self.cur_t = 0
        self.reset_state(False)
        
        return self._get_obs()

    def viewer_setup(self):
        # self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = 12
        # self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.lookat[:2] = self.data.qpos[:2] 
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = 80