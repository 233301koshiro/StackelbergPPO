import numpy as np
import re
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
        except Exception:
            self._report_swallowed('apply_skel_action / reload_sim_model',
                                   'この話は骨格変形で打ち切られる')
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
        except Exception:
            self._report_swallowed('set_design_params / reload_sim_model',
                                   'この話は属性変形で打ち切られる')
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
            design_params = np.clip(design_params, -1.0, 1.0)
            succ = self.set_design_params(design_params)
            if not succ:
                return self._get_obs(), 0.0, True, False, {'use_transform_action': True, 'stage': 'attribute_transform', 'reward_ctrl': 0.0}
            reward = 0.0
            if self.cur_t == self.cfg.skel_transform_nsteps + 1:
                succ = self.transit_execution()
                if not succ:
                    return self._get_obs(), 0.0, True, False, {'use_transform_action': True, 'stage': 'attribute_transform', 'reward_ctrl': 0.0}
                # R^L: one-shot leader-only bonus for a morphology whose reach
                # annulus geometrically covers the cube (design rationale: タスク設計と
                # 報酬関数.md セクション8、実装詳細: セクション9.2/9.7/9.8).
                # Additive on top of the inherited follower return — does not
                # touch execution-phase reward, preserving Stackelberg coupling.
                reward = self.compute_reach_bonus()

            ob = self._get_obs()
            termination = truncation = False
            return ob, reward, termination, truncation, {'use_transform_action': True, 'stage': 'attribute_transform', 'reward_ctrl': 0.0}
        # execution stage
        else:
            self.control_nsteps += 1
            if getattr(self, '_init_contact_penalty_pending', False):
                # 初期接触形態: 物理を進めず 1 ステップだけペナルティを返して終了。
                # follower=1 となり EP-FILTER を通過 → Leader に負の勾配が渡る。
                self._init_contact_penalty_pending = False
                penalty = self.cfg.reward_specs.get('init_contact_penalty', 50.0)
                reward_breakdown = np.array([0.0, 0.0])
                return self._get_obs(), -penalty, True, False, {'use_transform_action': False, 'stage': 'execution', 'reward_ctrl': 0.0, 'reward_breakdown': reward_breakdown}
            assert np.all(a[:, self.control_action_dim:] == 0)
            control_a = a[:, :self.control_action_dim]
            ctrl = self.action_to_control(control_a)
            ctrl_cost_coeff = self.cfg.reward_specs.get('ctrl_cost_coeff', 1e-4)
            # Route B 対照実験用に速度ベース報酬を再有効化（use_target_reward=false 時のみ使用）。
            xposbefore = self.get_body_com("cube")[0]
            yposbefore = self.get_body_com("cube")[1]
            try:
                self.do_simulation(ctrl, self.frame_skip)
            except Exception:
                self._report_swallowed('step / do_simulation',
                                       'この話は実行中に打ち切られ、報酬 0 で返る')
                return self._get_obs(), 0, True, False, {'use_transform_action': False, 'stage': 'execution', 'reward_ctrl': 0.0}
            
            xposafter = self.get_body_com("cube")[0]
            yposafter = self.get_body_com("cube")[1]

            use_target = self.cfg.reward_specs.get('use_target_reward', False)
            target_x = self.cfg.reward_specs.get('target_x', 1.5)
            target_y = self.cfg.reward_specs.get('target_y', 0.0)
            dist_to_target = np.linalg.norm(
                np.array([xposafter, yposafter]) - np.array([target_x, target_y]))
            curr_cube_potential = 1.0 / (1.0 + dist_to_target)
            if self.prev_cube_potential is None:
                self.prev_cube_potential = curr_cube_potential
            target_pbrs = curr_cube_potential - self.prev_cube_potential
            self.prev_cube_potential = curr_cube_potential

            use_reach = self.cfg.reward_specs.get('use_reach', False)
            use_dense_target = self.cfg.reward_specs.get('use_dense_target', False)
            if use_reach:
                # Reach タスク: arm tip を 3D 目標点に近づける。シンプルな dense 報酬。
                # cube は不要（XML に残しても可）。reward_fwd_contact も無効化。
                target_z = self.cfg.reward_specs.get('target_z', 0.2)
                target_3d = np.array([target_x, target_y, target_z])
                reach_dist = np.linalg.norm(self._arm_tip_pos - target_3d)
                reward_fwd_cube = -reach_dist
                reward_fwd_contact = 0.0
                # ⭐⭐ 9-145: 障害物への食い込みを罰する。
                #   ⚠️ **既定 0.0 なので、指定しない限り既存の run は一切変わらない。**
                #   ⭐ 深さ [m] をそのまま引く（係数 1）。報酬が -距離 [m] なので単位が揃う。
                obs_scale = self.cfg.reward_specs.get('obstacle_penalty_scale', 0.0)
                obstacle_pen = 0.0
                if obs_scale > 0.0:
                    obstacle_pen = obs_scale * self._obstacle_penetration()
                reward_fwd = reward_fwd_cube - obstacle_pen
                reward_ctrl = - ctrl_cost_coeff * np.square(ctrl).mean()
                alive_bonus = self.cfg.reward_specs.get('alive_bonus', 0.0)
                reward = (reward_fwd + reward_ctrl + alive_bonus) * self.cfg.reward_specs.get('exec_reward_scale', 1.0)
                s = self.state_vector()
                termination = not np.isfinite(s).all()
                truncation = not (self.control_nsteps < self.cfg.done_condition.get('max_nsteps', 1000))
                ob = self._get_obs()
                reward_breakdown = np.array([reward_fwd_cube, -obstacle_pen])
                return ob, reward, termination, truncation, {'use_transform_action': False, 'stage': 'execution', 'reward_ctrl': reward_ctrl, 'reward_breakdown': reward_breakdown}
            elif use_target:
                # 純粋な target PBRS（目標座標のみ）
                reward_fwd_cube = target_pbrs
            elif use_dense_target:
                # Dense 報酬: vel_reward - λ * dist(cube, target)（毎ステップ勾配あり）
                # use_clipped_vel=true のとき vel は target を超えた分を加算しない
                use_clipped_vel = self.cfg.reward_specs.get('use_clipped_vel', False)
                if use_clipped_vel:
                    clipped_after  = min(xposafter,  target_x)
                    clipped_before = min(xposbefore, target_x)
                    vel_reward = (clipped_after - clipped_before) / self.dt - 0.1 * np.abs(yposafter - yposbefore) / self.dt
                else:
                    vel_reward = (xposafter - xposbefore) / self.dt - 0.1 * np.abs(yposafter - yposbefore) / self.dt
                dense_weight = self.cfg.reward_specs.get('dense_target_weight', 0.1)
                reward_fwd_cube = vel_reward - dense_weight * dist_to_target
            else:
                # 速度ベース報酬 + オプションで target PBRS を加算（組み合わせ報酬①）
                # cube_target_weight > 0 のとき: vel_reward + weight * target_pbrs
                vel_reward = (xposafter - xposbefore) / self.dt - 0.1 * np.abs(yposafter - yposbefore) / self.dt
                target_weight = self.cfg.reward_specs.get('cube_target_weight', 0.0)
                reward_fwd_cube = vel_reward + target_weight * target_pbrs

            # Potential-Based Reward Shaping: reward = φ(t+1) - φ(t), φ = 1/(1+dist).
            # Static arm → Δφ = 0, no free reward.
            # Arm approaching cube → Δφ > 0, exploration guided as before.
            # Preserves design intent (morphology that enables fast approach = higher reward)
            # while eliminating the static-proximity exploitation.
            curr_dist = np.linalg.norm(self.get_body_com("cube") - self._arm_tip_pos)
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
            # Diagnostic only: per-component reward breakdown, surfaced in log_train.txt
            # via the c_info channel (see worker_sampler.py / genesis_agent.py). Does not
            # affect the actual reward signal used for training.
            reward_breakdown = np.array([reward_fwd_cube, reward_fwd_contact])
            return ob, reward, termination, truncation, {'use_transform_action': False, 'stage': 'execution', 'reward_ctrl': reward_ctrl, 'reward_breakdown': reward_breakdown}

    def compute_reach_bonus(self):
        """Leader-only design-phase bonus (R^L): reward morphologies whose 2-link
        reach annulus [|L1-L2|, L1+L2] around the shoulder pivot geometrically
        covers the cube's (already-sampled) position, AND whose pose at the
        moment execution actually starts points toward the cube (案B: radius x
        angle, see タスク設計と報酬関数.md セクション9.5/9.7/9.8).

        Link length/shoulder pivot use body.bone_offset / body_xpos rather than
        get_body_com, since get_body_com returns the *subtree* COM (includes
        descendants) and so cannot isolate a single link's geometry.

        The angle term must reflect the REAL starting pose, not the design-space
        zero-joint-angle pose: when arm_safe_init is on, reset_state() always
        forces qpos[0] = pi/2 (shoulder rotated +90° from the design "rest"
        direction, regardless of which way the Leader grew the arm — see
        reset_state() and 進捗.md 2026-06-26 知見 "arm_safe_init と形態方向の
        不一致"). Body "0" itself never rotates (no joint), so the shoulder pivot
        (body_xpos(bodies[1])) is unaffected, but bo1+bo11 must be rotated by that
        same +90° before comparing against the cube direction, or the bonus
        rewards a "rest" direction that is never actually realized at runtime.

        2026-07-24 (Bug 16 対応): L1/L2 の長さ計算は bone_offset の全3成分の
        ノルムを使うよう修正済み（旧: [:2] 切り詰めで、縦型アームでは長さを
        常にゼロと誤算していた）。ただし angle_term 側（shoulder_xy・cube_xy・
        link_vec を使った水平面内の方向一致判定）は依然として「腕がX-Y平面内で
        動く」ことを前提にした2Dロジックのままで、縦型アーム（tripo_arm_v3等）
        での妥当性は未検証。本関数は reach_bonus_scale=0.0（既定値）のため
        現状どの run でも休眠しており実害はないが、将来 reach_bonus を
        縦型アームで有効化する際は angle_term 側の3D化を別途行うこと。
        """
        scale = self.cfg.reward_specs.get('reach_bonus_scale', 0.0)
        if scale == 0.0 or not self.is_fixed_base or len(self.robot.bodies) < 3:
            return 0.0
        bodies = self.robot.bodies
        L1 = float(np.linalg.norm(np.asarray(bodies[1].bone_offset, dtype=float)))
        L2 = float(np.linalg.norm(np.asarray(bodies[-1].bone_offset, dtype=float)))
        bo1 = np.asarray(bodies[1].bone_offset, dtype=float)[:2]
        bo11 = np.asarray(bodies[-1].bone_offset, dtype=float)[:2]
        shoulder_xy = self.data.body_xpos[self.model._body_name2id[bodies[1].name]][:2]
        cube_xy = self.get_body_com("cube")[:2]

        # radius term: does the cube fall within the 2-link reach annulus?
        d = np.linalg.norm(cube_xy - shoulder_xy)
        reach_min, reach_max = abs(L1 - L2), L1 + L2
        excess = max(0.0, reach_min - d, d - reach_max)
        k = self.cfg.reward_specs.get('reach_bonus_k', 3.0)
        radius_term = np.exp(-k * excess)

        # angle term: does the REAL starting-pose direction point toward the cube?
        # arm_safe_init forces qpos[0]=pi/2 regardless of design direction, so the
        # actually-realized link-chain vector is (bo1+bo11) rotated by that same
        # +90° about the (non-rotating) shoulder pivot.
        eps = 1e-6
        link_vec = bo1 + bo11
        if self.env_specs.get('arm_safe_init', False):
            theta = np.pi / 2
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            link_vec = rot @ link_vec
        rest_tip_xy = shoulder_xy + link_vec
        to_cube = cube_xy - shoulder_xy
        to_tip = rest_tip_xy - shoulder_xy
        cos_angle = None
        if np.linalg.norm(to_cube) < eps or np.linalg.norm(to_tip) < eps:
            angle_term = 0.0
        else:
            cos_angle = np.dot(to_cube, to_tip) / (np.linalg.norm(to_cube) * np.linalg.norm(to_tip))
            angle_power = self.cfg.reward_specs.get('reach_bonus_angle_power', 2.0)
            angle_term = max(0.0, cos_angle) ** angle_power

        return scale * radius_term * angle_term

    def transit_attribute_transform(self):
        self.stage = 'attribute_transform'

    def _get_cube_half_size(self):
        """cube_geom の半径(x)を init_xml_str から読む。env_specs.cube_half_size で
        明示上書きも可能（未指定なら XML 実測、それも失敗したら 0.5 にフォールバック）。
        2026-07-21 発覚: 従来は rrbot_arm.xml 専用の固定値 0.5 を全アーム共通で
        使っており、tripo アーム（cube half=0.15）では閾値が2.5倍過大だった。
        """
        override = self.env_specs.get('cube_half_size', None)
        if override is not None:
            return float(override)
        if not hasattr(self, '_cached_cube_half_size'):
            import re
            # init_xml_str は bytes（cur_xml_str がデコード済み str）。
            # 2026-07-21 発覚: str パターンで bytes を検索して TypeError → Choreonoid が
            # 例外を握りつぶし Qt ループだけ残る「見せかけのハング」を引き起こしていた。
            m = re.search(r'name="cube_geom"[^>]*\bsize="([^"\s]+)', self.cur_xml_str)
            self._cached_cube_half_size = float(m.group(1)) if m else 0.5
        return self._cached_cube_half_size

    def _get_max_arm_radius(self):
        """全リンクの capsule 半径の最大値。従来は rrbot 専用の固定値 0.05 だった。"""
        if not hasattr(self, '_cached_max_arm_radius'):
            radii = []
            for body in self.robot.bodies:
                for geom in body.geoms:
                    size = getattr(geom, 'size', None)
                    if size is not None:
                        radii.append(float(np.asarray(size).flat[0]))
            self._cached_max_arm_radius = max(radii) if radii else 0.05
        return self._cached_max_arm_radius

    def _check_initial_contact(self):
        """実行開始時のアーム先端位置が cube に接触しているか確認。
        接触している場合 True を返してエピソードを打ち切る
        （Leader が initial contact exploit を学習するのを防ぐ）。
        check_init_contact=false で無効化可能。arm_safe_init の有無に依存しない。

        先端位置は `_arm_tip_pos`（live simulation の body_xpos/body_xmat から
        計算される、reset_state() 後の実際の3D先端位置。root からのチェーンを
        辿り、bone_offset を各ボディの実姿勢で回転して足す）をそのまま使う。
        arm_safe_init による qpos[0]=π/2 の回転は reset_state() が既に
        シミュレーション状態へ反映済みのため、ここで改めて回転行列を
        適用する必要はない。

        2026-07-21 訂正 (Bug 13): 従来は bodies[1] + bodies[-1] の2リンクだけを
        足しており、3関節以上のアーム（tripo_arm 系）では中間リンクが無視され
        先端位置を37%（3関節）〜60%（4関節）過小評価していた
        （rrbot の2関節では偶然正しかった）。cube_half_size も rrbot 専用の
        固定値 0.5 だった点も併せて修正。

        2026-07-24 訂正 (Bug 16): Bug 13 の修正自体が bone_offset[:2]（X,Y成分
        のみ）を手動で合計する方式で、全関節Z軸（ヨー）の平面アームでは正しい
        ものの、ピッチ軸混在の縦型アーム（tripo_arm_v3）では静止姿勢の長さが
        全部Z成分に乗るため link_vec が恒常的にゼロになり、先端位置を常に
        肩関節の真上と誤認していた（Reach が ep2 で -50 に張り付き放置される
        原因）。`_arm_tip_pos`（get_sim_obs 等で既に使われている、live
        simulation ベースの N関節・軸構成非依存の先端位置）に置き換えて解決。
        """
        if (not self.env_specs.get('check_init_contact', True)
                or not self.is_fixed_base
                or len(self.robot.bodies) < 3):
            return False
        tip_xy    = self._arm_tip_pos[:2]
        cube_xy   = self.get_body_com("cube")[:2]
        cube_half = self._get_cube_half_size()
        arm_rad   = self._get_max_arm_radius()
        margin    = 0.03   # 安全マージン
        thresh    = cube_half + arm_rad + margin
        return (abs(tip_xy[0] - cube_xy[0]) < thresh and
                abs(tip_xy[1] - cube_xy[1]) < thresh)

    def _check_floor_penetration(self):
        """実行開始時にアームのいずれかの関節・先端が床（z=0）を割っているか確認。
        縦型アーム（tripo_arm_v3）用: Leader は offset の z 成分を動かせるため、
        設計フェーズで「床にめり込んだ初期姿勢」を作れてしまう。深い貫通は表面
        ベースの衝突検出では解決されず、物理的にあり得ない形態が学習に混ざる。
        cube の初期接触 exploit と同型の問題なので、同じペナルティ機構で Leader に
        負の勾配を渡す（Bug 9 の教訓 = 棄却でなくペナルティ）。
        env_specs.check_floor_penetration=true で有効化（デフォルト無効 = 既存 run 無影響）。
        """
        if (not self.env_specs.get('check_floor_penetration', False)
                or not self.is_fixed_base):
            return False
        margin = self.env_specs.get('floor_penetration_margin', 0.0)
        zs = []
        for body in self.robot.bodies:
            pos = self._body_xpos.get(body.name)
            if pos is not None and np.all(np.isfinite(pos)):
                zs.append(float(np.asarray(pos)[2]))
        tip = self._arm_tip_pos
        if np.all(np.isfinite(tip)):
            zs.append(float(tip[2]))
        return bool(zs) and min(zs) < margin

    def _report_swallowed(self, where, consequence):
        """⭐⭐⭐ **握りつぶした例外を必ず出す**（2026-10-03、Bug 53 / 系譜 9-211）。

        ⛔⛔⛔ **旧実装はここが裸の `except:` で、XML を印字するだけだった。**
          ⛔ **症状が別の症状に化ける**: `_safe_init_angle` の `TypeError` が
            「腕が動かない」「ブロックが発火しない」「学習が進まない」に見えた。
          ⛔⛔ **`hockey_bank7` は 20 epoch すべてこれで、EP-FILTER が 8336 話を落としていた。**
        ⚠️ **返り値・制御の流れは変えていない。**出力を足しただけ。
        ⭐ XML は長いので `TRANSIT_DUMP_XML=1` のときだけ出す。
        """
        import traceback
        print(f'⛔⛔ [{where}] 例外を握りつぶした。{consequence}:', flush=True)
        traceback.print_exc()
        if os.environ.get('TRANSIT_DUMP_XML'):
            print(getattr(self, 'cur_xml_str', '(cur_xml_str 無し)'), flush=True)

    def transit_execution(self):
        self.stage = 'execution'
        self.control_nsteps = 0
        try:
            self.reset_state(True)
        except Exception:
            self._report_swallowed('transit_execution / reset_state(True)',
                                   'この話は execution へ入れないまま done になる')
            return False
        # 初期接触チェック: arm tip が cube に触れている形態への対処。
        # Leader が「接触してインパルスで押す」exploit を学習するのを防ぐ。
        #
        # init_contact_penalty > 0（デフォルト 50.0）: エピソードを「1 exec ステップ +
        #   ペナルティ報酬」として成立させる。follower>0 になるため EP-FILTER を通過し、
        #   Leader がこの形態の悪さを勾配として学習できる（物理は進めないので exploit 不可）。
        #   旧挙動（棄却→ EP-FILTER 落ち）は Leader に勾配が渡らず、設計分布が棄却領域に
        #   はまると全エピソード棄却 →「All episodes are filtered」で assert 死する
        #   欠陥があった（F4・L2・TP1 の3回再現、デバッグ戦記 Bug 9）。
        # ペナルティの大きさの原則: 「正直に1エピソード動いたときの最悪リターン」より
        #   確実に悪いこと。ctrl_cost=0.2 の学習初期は正直なエピソードが ≈-16 になるため、
        #   当初のデフォルト 1.0 では「接触即終了(-1)の方が得」となり Leader が接触形態に
        #   収束した（2026-07-10 L2/TP1 再走 ep10 で実測）。50 ≈ ctrl コスト満額 + マージン。
        # init_contact_penalty <= 0: 旧挙動（棄却）。
        self._init_contact_penalty_pending = False
        _ic, _fp = self._check_initial_contact(), self._check_floor_penetration()
        if _ic or _fp:
            # ⭐⭐ **どちらの門が、何の値で発火したかを出す**（2026-10-03、系譜 9-211）。
            #   ⛔⛔ **出していなかったので「腕が動かない」と誤診した。**
            #     実際は 1 step でペナルティ終了しており、物理は一度も進んでいなかった。
            try:
                _t = np.asarray(self._arm_tip_pos, dtype=float)
                _c = np.asarray(self.get_body_com('cube'), dtype=float)
                _th = (self._get_cube_half_size() + self._get_max_arm_radius() + 0.03)
                print(f'⚠️⚠️ [init_gate] **1 step でペナルティ終了する。**'
                      f'初期接触={_ic} / 床貫通={_fp} / '
                      f'先端 ({_t[0]:.3f},{_t[1]:.3f},{_t[2]:.3f}) / '
                      f'cube ({_c[0]:.3f},{_c[1]:.3f},{_c[2]:.3f}) / '
                      f'|dx|={abs(_t[0]-_c[0]):.3f} |dy|={abs(_t[1]-_c[1]):.3f} '
                      f'（どちらも閾値 {_th:.3f} 未満なら接触判定）', flush=True)
            except Exception:
                pass
            if self.cfg.reward_specs.get('init_contact_penalty', 50.0) > 0:
                self._init_contact_penalty_pending = True
            else:
                return False
        # Snapshot φ values at episode start so step() can compute Δφ.
        dist0 = np.linalg.norm(self.get_body_com("cube") - self._arm_tip_pos)
        self.prev_contact_potential = 1.0 / (1.0 + dist0)
        # target PBRS は use_target_reward=true と組み合わせ報酬①（cube_target_weight>0）の両方で必要
        target_x = self.cfg.reward_specs.get('target_x', 1.5)
        target_y = self.cfg.reward_specs.get('target_y', 0.0)
        cube_pos = self.get_body_com("cube")[:2]
        dist_cube0 = np.linalg.norm(cube_pos - np.array([target_x, target_y]))
        self.prev_cube_potential = 1.0 / (1.0 + dist_cube0)
        return True
        

    @property
    def _obstacle_boxes(self):
        """⭐ 9-145: 静的な障害物（関節を持たない worldbody 直下の body）の直方体。

        ⚠️ **9-144 で判定器に柱を教えたとき、報酬側は柱を一度も見ていなかった。**
        **避けろと言っていないのに避けることを期待していた**（9-144）。
        XML から 1 回だけ読み、以降は使い回す。
        """
        if getattr(self, '_obs_boxes_cache', None) is not None:
            return self._obs_boxes_cache
        import xml.etree.ElementTree as ET
        boxes = []
        try:
            root = ET.fromstring(self.cur_xml_str)
            wb = root.find('worldbody')
            bodies = wb.findall('body')
            for i, b in enumerate(bodies):
                if i == 0 or b.get('name') == 'cube' or b.find('joint') is not None:
                    continue          # 腕の根元と可動体は障害物ではない
                g = b.find('geom')
                if g is None:
                    continue
                bp = np.array([float(x) for x in b.get('pos', '0 0 0').split()])
                sz = [float(x) for x in g.get('size', '0').split()]
                typ = g.get('type', 'box')
                if typ == 'box' and len(sz) >= 3:
                    half = np.array(sz[:3])
                elif typ in ('cylinder', 'capsule') and len(sz) >= 2:
                    half = np.array([sz[0], sz[0], sz[1]])
                elif typ == 'sphere' and sz:
                    half = np.array([sz[0]] * 3)
                else:
                    continue
                boxes.append((bp - half, bp + half))
        except Exception:
            boxes = []
        self._obs_boxes_cache = boxes
        return boxes

    def _obstacle_penetration(self, n_sample=9):
        """⭐ 腕のリンクが障害物に食い込んでいる最大の深さ [m]。触れていなければ 0。

        ⭐⭐ **単位を「距離 [m]」に揃えてあるのが要点**（9-145）。
        Reach の報酬は `-reach_dist` [m] なので、**深さをそのまま引けば係数は 1（無次元）で済み、
        根拠のない重みを 1 個も増やさない**。
        ⭐ 上限は柱の半幅で自動的に決まる（幾何が決めるので調整値ではない）。
        """
        boxes = self._obstacle_boxes
        if not boxes:
            return 0.0
        worst = 0.0
        for body in self.robot.bodies:
            a = self._body_xpos.get(body.name)
            if a is None or np.any(np.isnan(a)):
                continue
            a = np.asarray(a, dtype=float)
            bo = getattr(body, 'bone_offset', None)
            if bo is None:
                b = a
            else:
                mat = np.asarray(self._body_xmat.get(body.name, np.eye(3))).reshape(3, 3)
                b = a + mat @ np.asarray(bo, dtype=float)
            ts = np.linspace(0.0, 1.0, n_sample)[:, None]
            pts = a[None, :] + (b - a)[None, :] * ts       # (n_sample, 3)
            for lo, hi in boxes:
                inside = np.all((pts >= lo) & (pts <= hi), axis=1)
                if not inside.any():
                    continue
                d = np.minimum(pts - lo, hi - pts).min(axis=1)
                worst = max(worst, float(d[inside].max()))
        return worst

    @property
    def is_fixed_base(self):
        root_joints = self.robot.bodies[0].joints
        return all(j.type != 'free' for j in root_joints)

    @property
    def _arm_tip_pos(self):
        """Arm tip position for contact reward / obs. Falls back through bodies from
        the tip to handle branched structures (e.g. tripo_arm) where the deepest
        body may have NaN translation in Choreonoid's forward kinematics.
        For the last body (no children), adds bone_offset rotated by body_xmat to
        return the physical tip rather than the joint origin (elbow)."""
        if not self.is_fixed_base:
            return self.get_body_com("0")
        for body in reversed(self.robot.bodies):
            pos = self._body_xpos.get(body.name)
            if pos is not None and not np.any(np.isnan(pos)):
                pos = np.asarray(pos)
                if len(body.child) == 0 and body.bone_offset is not None:
                    mat = np.asarray(self._body_xmat.get(body.name, np.eye(3))).reshape(3, 3)
                    tip = pos + mat @ body.bone_offset
                    if np.all(np.isfinite(tip)):
                        return tip
                return pos
        return np.zeros(3)

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
                relative_dis = self.get_body_com("cube") - self._arm_tip_pos
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

    # ────────────────────────────────────────────────────────────────────
    # ⭐⭐ 順運動学（FK）— **シミュレータを引かずに XML から腕の形を復元する**
    #
    # ⛔⛔ 2026-10-02（系譜 9-196）: 旧実装は腕を「原点から伸びる長さ R の直線の棒」
    #   として扱い、`tip_y = R·sin(θ)` で初期角を決めていた。**その形状は存在しない。**
    #   ⛔ 実際は 4 関節の連鎖で、**`qpos[0]` はヨー（z 軸回り）なのに式はピッチを記述していた。**
    #   ⛔ 結果、`hockey_bank5` は先端が 1201 step 中 0 step しかリンク内に入らなかった。
    #   ⛔ さらに R を `_safe_init_cached` で初回値に固定しており、co-design が腕を
    #     1.0425 → 2.5532 m に伸ばしても再計算されなかった。
    #
    # ⭐ 是正: **XML（＝いまの形態）から連鎖を読み、実際に FK を回して先端位置を得る。**
    #   ⚠️ `reset_model()` の時点では Choreonoid の `_body_xpos` が空なので
    #     **シミュレータは引けない**（引くと KeyError → worker 死 → 9.5 時間ハング）。
    # ────────────────────────────────────────────────────────────────────

    def _parse_arm_chain(self, xml):
        """XML から腕の運動連鎖を読む。戻り値は根元から順の list。

        各要素 = dict(offset=親関節からの位置, axis=関節軸, bone=リンク先端までのベクトル)
        ⭐ すべて静的な値。シミュレータを引かない。
        """
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml if isinstance(xml, str) else xml.decode())
        # 腕の根元 body（worldbody 直下で joint を持つ子を辿れるもの）を探す
        wb = root.find('worldbody')
        if wb is None:
            raise RuntimeError('_parse_arm_chain: worldbody が無い')

        def first_hinge_child(node):
            for b in node.findall('body'):
                if b.find('joint[@type="hinge"]') is not None:
                    return b
            return None

        start, root_pos = None, np.zeros(3)
        for b in wb.findall('body'):
            if b.find('joint[@type="hinge"]') is not None:
                start, root_pos = b, np.zeros(3)
                break
            kid = first_hinge_child(b)
            if kid is not None:
                # ⭐ 根元 body（関節を持たない台座）の pos を取りこぼさない。
                #   ⛔ e2e_hockey_easy では body "0" が pos=(0,0,0.020) を持つ。
                start = kid
                root_pos = np.array([float(v) for v in b.attrib.get('pos', '0 0 0').split()])
                break
        if start is None:
            raise RuntimeError('_parse_arm_chain: hinge を持つ body が見つからない')

        chain, node = [], start
        while node is not None:
            jt = node.find('joint[@type="hinge"]')
            if jt is None:
                break
            cap = node.find('geom[@type="capsule"]')
            if cap is None or 'fromto' not in cap.attrib:
                bone = np.zeros(3)
            else:
                ft = np.array([float(v) for v in cap.attrib['fromto'].split()])
                bone = ft[3:] - ft[:3]
            chain.append(dict(
                name=node.attrib.get('name', '?'),
                offset=np.array([float(v) for v in node.attrib.get('pos', '0 0 0').split()]),
                axis=np.array([float(v) for v in jt.attrib.get('axis', '0 0 1').split()]),
                bone=bone))
            node = first_hinge_child(node)
        if not chain:
            raise RuntimeError('_parse_arm_chain: 連鎖が空')
        chain[0]['offset'] = chain[0]['offset'] + root_pos
        return chain

    @staticmethod
    def _rot(axis, ang):
        """軸 axis まわりに ang 回す回転行列（Rodrigues）。"""
        a = np.asarray(axis, dtype=float)
        n = np.linalg.norm(a)
        if n < 1e-12:
            return np.eye(3)
        a = a / n
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)

    def _fk_points_batch(self, chain, A):
        """⭐⭐ FK を **まとめて** 解く。`A` は (N, J) の関節角。

        ⛔⛔ **2026-10-03（9-209）: ピッチを探索に入れたら 1 回の探索が
        47.8 ms → 752.5 ms（15.7 倍）になり、`T_eval` が 94 → 508 s、
        ETA が 21 h → 2 日 10 h になった。**
        ⭐ **解像度を落とさずに速くするため、全組み合わせを numpy で一度に回す。**

        戻り値: `P`（N, J+1, 3）各リンクの始点、`Q`（N, J, 3）各リンクの終点。
        """
        N = A.shape[0]
        R = np.repeat(np.eye(3)[None], N, axis=0)      # (N,3,3)
        p = np.zeros((N, 3))
        starts, ends = [], []
        for j, lk in enumerate(chain):
            p = p + np.einsum('nij,j->ni', R, lk['offset'])
            # 軸まわりの回転をまとめて作る（Rodrigues）
            a = np.asarray(lk['axis'], float)
            a = a / (np.linalg.norm(a) or 1.0)
            K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            c = np.cos(A[:, j])[:, None, None]
            sn = np.sin(A[:, j])[:, None, None]
            Rj = np.eye(3)[None] + sn * K[None] + (1 - c) * (K @ K)[None]
            R = R @ Rj
            q = p + np.einsum('nij,j->ni', R, lk['bone'])
            starts.append(p.copy()); ends.append(q.copy())
        return np.stack(starts, 1), np.stack(ends, 1)    # (N,J,3), (N,J,3)

    @staticmethod
    def _seg_point_dist_batch(A3, B3, c):
        """⭐ 線分（まとめて）と 1 点の最短距離。`A3`,`B3` は (N,J,3)。戻りは (N,J)。"""
        ab = B3 - A3
        L2 = np.einsum('nji,nji->nj', ab, ab)
        t = np.where(L2 > 1e-12, np.einsum('nji,i->nj', (c - A3), np.ones(3)) * 0, 0.0)
        t = np.einsum('nji,nji->nj', (c[None, None] - A3), ab) / np.where(L2 > 1e-12, L2, 1.0)
        t = np.clip(t, 0.0, 1.0)
        proj = A3 + ab * t[..., None]
        return np.linalg.norm(proj - c[None, None], axis=2)

    def _fk_points(self, chain, angles):
        """FK。各リンクの始点と終点をワールド座標で返す (N+1, 3) ではなく区間の list。

        戻り値: [(始点, 終点), ...]  リンクごと
        """
        R = np.eye(3)
        p = np.zeros(3)
        segs = []
        for lk, th in zip(chain, angles):
            p = p + R @ lk['offset']
            R = R @ self._rot(lk['axis'], float(th))
            q = p + R @ lk['bone']
            segs.append((p.copy(), q.copy()))
            # ⚠️ p は**この body の原点**のまま次へ渡す。子の offset は親の body 原点から測る
        return segs

    @staticmethod
    def _seg_point_dist(a, b, c):
        """線分 ab と 点 c の最短距離。"""
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            return float(np.linalg.norm(c - a))
        t = float(np.clip((c - a) @ ab / L2, 0.0, 1.0))
        return float(np.linalg.norm(a + t * ab - c))

    def _safe_init_angle(self):
        r"""⭐⭐ `arm_safe_init` の初期ヨー角。**既定は従来どおり π/2（90°）。**

        ⭐ 目的は 2 つ。**①腕が初期姿勢で対象に重なり、分離インパルスが対象を無償で
        吹き飛ばすのを防ぐ。②壁のある環境で腕が壁の外に出ないようにする。**

        ⭐⭐ `arm_init_clear_y` を与えると、**FK を実際に回して条件を満たすヨー角を探す。**

        | 条件 | 判定 |
        |---|---|
        | 全リンクが壁の内側 | すべての線分上の点で `\|y\| ≤ clear_y` |
        | 対象と初期接触しない | すべてのリンクと対象の距離 ≥ `need` |

        ⭐ 満たす角のうち **先端が対象に最も近いもの**を選ぶ（動き出しやすい側）。
        ⛔ **一つも無ければ例外。**それは「この設計空間ではこの環境の初期姿勢が作れない」
        という結果である（系譜 9-193 の事前登録の読み ④）。

        ⛔⛔ **2026-10-02 の是正（系譜 9-196）**: 旧実装は腕を直線の棒と見なし、
        **ヨー角に対してピッチの式を当てていた**ので先端位置が全く合わなかった。
        ⛔ また `R` を初回値に固定していたので、co-design が腕を伸ばしても追随しなかった。
        ⭐ **本実装は毎回 XML（＝いまの形態）から連鎖を読み直す。**
        """
        clear_y = self.env_specs.get('arm_init_clear_y')
        if clear_y is None:
            return np.pi / 2                      # ⭐ 従来どおり。既存 run は無影響
        clear_y = float(clear_y)

        xml = getattr(self, 'cur_xml_str', None) or getattr(self, 'init_xml_str', '')
        if not xml:
            raise RuntimeError('arm_init_clear_y: XML が取れない')
        # ⭐ 形態が変われば作り直す（⛔ 旧実装はここを固定して壊れた）
        key = hash(xml if isinstance(xml, str) else bytes(xml))
        if getattr(self, '_safe_init_key', None) == key:
            return self._safe_init_cached

        chain = self._parse_arm_chain(xml)
        need = (self._get_cube_half_size() + self._get_max_arm_radius()
                + self.env_specs.get('arm_init_margin', 0.03))

        mm = re.search(r'<body\s+name="cube"\s+pos="([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)', 
                       xml if isinstance(xml, str) else xml.decode())
        if mm is None:
            raise RuntimeError('arm_init_clear_y: XML から cube の位置を読めない')
        cube = np.array([float(mm.group(1)), float(mm.group(2)), float(mm.group(3))])

        # ⭐ ピッチ側の角は init_qpos をそのまま使う（変数を増やさない）
        base = np.zeros(len(chain))
        iq = np.asarray(getattr(self, 'init_qpos', np.zeros(len(chain))), dtype=float)
        for k in range(1, len(chain)):
            if k < len(iq):
                base[k] = iq[k]

        # ⭐⭐ **2026-10-03（9-208）: 第2関節（ピッチ）も探索に入れる。**
        #   ⛔⛔ **ヨーだけでは先端の高さが変えられない**（9-203）。
        #     `hockey_bank6` は 1201 step すべてでパックの 39〜44 cm 上を掃いた。
        #   ⭐ seed1 は学習で下ろせたが（高さ差 0.040 m）、⛔ **seed0 は下ろせなかった**（0.269 m）。
        #   ⭐⭐ **初期姿勢で高さを合わせられるなら、そこが seed 間の差を生んでいる可能性がある。**
        #   ⚠️ **既定では無効**（`arm_init_pitch_search` を指定したときだけ）。既存 run は無影響。
        # ⭐ 床からの最小クリアランス [m]。⚠️ 台座の高さ 0.020 m を下回らせない
        FLOOR_CLEAR = float(self.env_specs.get('arm_init_floor_clear', 0.02))
        _pitch = self.env_specs.get('arm_init_pitch_search', False)
        _p_cands = (np.linspace(-np.pi / 2, np.pi / 2, 61) if _pitch else np.zeros(1))

        # ⭐⭐ **ベクトル化して一度に解く**（9-209）。⛔ 逐次版は 752.5 ms で
        #   `T_eval` が 94 → 508 s、ETA が 21 h → 2 日 10 h になった。
        #   ⭐ **同じ解像度のまま 19.9 ms（38 倍）。**逐次版と差 3e-16 で一致を確認済み。
        def _solve(use_pitch, floor):
            """⭐ 1 段分を解く。返すのは (n_ok, best)。`best` は条件を満たす解が無ければ None。"""
            yaws = np.linspace(-np.pi, np.pi, 361 if use_pitch else 1441)
            if use_pitch:
                G = np.stack(np.meshgrid(yaws, _p_cands, indexing='ij'), -1).reshape(-1, 2)
            else:
                G = np.c_[yaws, np.zeros(len(yaws))]
            A = np.repeat(base[None], len(G), axis=0)
            A[:, 0] = G[:, 0]
            if use_pitch and A.shape[1] > 1:
                A[:, 1] = G[:, 1]
            Pp, Qq = self._fk_points_batch(chain, A)
            ok = np.abs(np.concatenate([Pp[:, :, 1], Qq[:, :, 1]], 1)).max(1) <= clear_y
            if floor is not None:
                ok &= np.minimum(Pp[:, :, 2], Qq[:, :, 2]).min(1) >= floor
            ok &= self._seg_point_dist_batch(Pp, Qq, cube).min(1) >= need
            n = int(ok.sum())
            if n == 0:
                return 0, None
            dt = np.where(ok, np.linalg.norm(Qq[:, -1, :] - cube, axis=1), np.inf)
            k = int(np.argmin(dt))
            return n, (float(G[k, 0]), float(dt[k]), self._fk_points(chain, A[k]),
                       (float(G[k, 1]) if use_pitch else None))

        # ⭐⭐⭐ **段階的に緩める**（2026-10-03、系譜 9-211）。
        #   ⛔⛔⛔ **旧実装は条件を満たす角が 0 本のとき `best = None` のまま unpack していた。**
        #     ⛔ `TypeError` が `transit_execution()` の裸の `except:` に飲まれ、
        #       **エピソードは execution へ入れないまま done になる。**
        #     ⛔⛔ **`hockey_bank7` は 20 epoch すべてこれで、`train_R_eps` が完全に 0.00、
        #       EP-FILTER が 8336 エピソードを落としていた**（bank6 は −0.38）。
        #   ⭐⭐ **床のクリアランス（9-208 で足した）が厳しすぎると全滅する。**
        #     ⭐ **全滅したら緩めた段へ落とす。**最後の段は 9-207（bank6）と同一条件なので、
        #       ⭐⭐ **少なくとも bank6 が通った設計空間では必ず解がある。**
        #   ⚠️ **これは「うまくいかないので条件を緩める」ではない**（§5-2 ①）。
        #     ⭐ **緩めた段を使ったことを必ず出力する**ので、どの条件で成立したかが結果に残る。
        #   ⭐⭐⭐ **順序が大事（2026-10-03 に 1 度間違えた）。**
        #     ⛔⛔ 初稿は 2 段目で**床の制約を外した**。⛔ その結果リンクが床下へ出る姿勢が選ばれ、
        #       **初期接触ペナルティ（1 step で終了）に変わっただけだった**（step 0 → 1）。
        #     ⭐⭐ **床は最後まで残す。**先に落とすのは**ピッチ探索**の方。
        #     ⭐ 2 段目は `hockey_bank6`（9-207 で完走）と**同一条件**なので、
        #       ⭐⭐ **bank6 が通った設計空間では必ず解がある。**
        #   ⛔⛔ **2026-10-03: 段の名前を 1 度間違えた。**「ヨーのみ＋床」を
        #     「9-207 と同一条件」と書いたが、⭐ **床の制約は 9-208 で私が足したもので、
        #     `hockey_bank6`（9-207）は持っていない。**同一条件は最終段の方である。
        _stages = [('ピッチ＋床', _pitch, FLOOR_CLEAR),
                   ('ヨーのみ＋床', False, FLOOR_CLEAR),
                   ('ピッチのみ・床なし', _pitch, None),
                   ('ヨーのみ・床なし（9-207 = bank6 と同一条件）', False, None)]
        _tried, best, used = [], None, None
        _seen = set()
        for _nm, _up, _fl in _stages:
            if (_up, _fl) in _seen:
                continue                   # ⭐ ピッチ探索が無効なら 1・2 段と 3・4 段は同一
            _seen.add((_up, _fl))
            n_ok, best = _solve(_up, _fl)
            _tried.append(f'{_nm}:{n_ok}')
            if best is not None:
                used = _nm
                break

        if best is None:
            # ⭐ 9-208 以前の明示的な例外へ戻す。⛔ 黙って None を返さない
            raise RuntimeError(
                f'⛔⛔ arm_safe_init: FK で条件を満たす角が一つも無い（全段で 0）。'
                f'段ごとの件数 {" / ".join(_tried)} / リンク {len(chain)} 本 / '
                f'対象 {cube[:2]} / 必要離隔 {need:.4f} m / 壁の内側 {clear_y} m '
                f'→ ⭐ この設計空間ではこの環境の初期姿勢が作れない')

        th, d_tip, segs, tp = best
        if used != _stages[0][0]:
            # ⭐⭐ **緩めた段を使ったことを黙って通さない**（§5-2 ①「例外を握りつぶす」）
            print(f'⚠️⚠️ [arm_safe_init] **第1段「{_stages[0][0]}」が 0 件だったので '
                  f'「{used}」へ落とした。**段ごとの件数 {" / ".join(_tried)}', flush=True)
        self._safe_init_cached = float(th)
        # ⭐⭐ 採用したピッチも保持する（qpos[1] に当てる。⛔ 無いと探索した意味が無い）
        self._safe_init_pitch = (None if tp is None else float(tp))
        self._safe_init_key = key
        tip = segs[-1][1]
        reach = float(np.linalg.norm(tip))
        # ⭐⭐ **何に対して効いたかを必ず出す**（CLAUDE.md §5-2 ⑤-3-2）
        print(f'[arm_safe_init] ⭐ FK で探索: リンク {len(chain)} 本 '
              f'{[c["name"] for c in chain]} / 対象 ({cube[0]:.3f},{cube[1]:.3f}) / '
              f'必要離隔 {need:.4f} m / 壁の内側 {clear_y} m\n'
              f'[arm_safe_init] ⭐ 採用した段「{used}」/ 段ごとの件数 {" / ".join(_tried)}\n'
              f'[arm_safe_init] ⭐ 条件を満たす角 {n_ok} 本 → 採用 ヨー {np.degrees(th):.1f}° '
              f'→ 先端 ({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f}) '
              f'原点から {reach:.3f} m / ⭐ 対象まで {d_tip:.3f} m（3 次元）'
              f' / ⭐⭐ 高さの差 {abs(tip[2] - cube[2]):.3f} m'
              f' / ピッチ {"探索せず" if tp is None else f"{np.degrees(tp):+.1f}°"}', flush=True)
        return float(th)

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

        # Shot（エアホッケー、実験系譜 9-33）: パックが毎回違う y に来る。
        # **報酬は Pusher と同じ**で、変えるのは初期条件だけ（変数を1つに保つ）。
        # 方策は get_sim_obs() の relative_dis（cube - 先端）で cube を観測しているので、
        # これは**学習可能な変動**であって盲目のノイズではない。
        # 既定 0.0 なので、指定しない限り既存の run と挙動は一致する。
        cube_y_noise = self.env_specs.get('cube_y_noise', 0.0)
        if cube_y_noise != 0.0:
            cube_y_idx = self.model.nq - 1
            extra = self.np_random.uniform(-cube_y_noise, cube_y_noise) if add_noise else 0.0
            qpos[cube_y_idx] = float(self.init_qpos[cube_y_idx]) + extra

        # Safe initial arm pose: set shoulder to π/2 so arm points in +y direction.
        # Prevents penetration-impulse exploit when morphology optimizer grows arm toward
        # +x (cube direction): at qpos[0]=π/2 the arm always starts pointing away from cube,
        # so no initial overlap regardless of arm length.
        # Requires shoulder joint range widened to ±90° in rrbot_arm.xml.
        if self.env_specs.get('arm_safe_init', False):
            qpos[0] = self._safe_init_angle()
            # ⭐⭐ ピッチも探索したなら当てる（9-208）。⚠️ 既定（探索しない）では None で無影響
            _tp = getattr(self, '_safe_init_pitch', None)
            if _tp is not None and len(qpos) > 1:
                qpos[1] = _tp

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