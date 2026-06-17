from collections import OrderedDict
import os
from pathlib import Path

from gym import spaces
from gym.utils import seeding
import numpy as np
import gym

try:
    import mujoco
except Exception:
    mujoco = None  # deferred: error raised only when MujocoEnv is instantiated

DEFAULT_SIZE = 500


# ---------------------------------------------------------------------------
# Compatibility wrappers: expose mujoco-py-style API over the new mujoco pkg
# ---------------------------------------------------------------------------

class _ModelWrapper:
    """Wraps mujoco.MjModel to expose the mujoco-py attribute API."""

    def __init__(self, m):
        self._m = m
        # Build actuator_names tuple (mujoco-py style)
        self.actuator_names = tuple(
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator_{i}"
            for i in range(m.nu)
        )
        # Build body_names tuple and _body_name2id dict (mujoco-py style)
        self.body_names = tuple(
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
            for i in range(m.nbody)
        )
        self._body_name2id = {
            name: i for i, name in enumerate(self.body_names)
        }
        # Build _camera_name2id dict
        self._camera_name2id = {
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i): i
            for i in range(m.ncam)
            if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) is not None
        }

    def camera_name2id(self, name):
        return mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_CAMERA, name)

    def __getattr__(self, name):
        return getattr(self._m, name)


class _DataWrapper:
    """Wraps mujoco.MjData to expose the mujoco-py attribute API."""

    def __init__(self, m, d):
        self._m = m
        self._d = d

    # Expose all MjData attributes transparently
    def __getattr__(self, name):
        return getattr(self._d, name)

    # mujoco-py: data.body_xpos[body_id]  →  new: data.xpos[body_id]
    @property
    def body_xpos(self):
        return self._d.xpos

    # mujoco-py: data.get_body_xpos(name)
    def get_body_xpos(self, name):
        bid = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_BODY, name)
        return self._d.xpos[bid].copy()

    # mujoco-py: data.get_body_xmat(name)  →  returns (3,3)
    def get_body_xmat(self, name):
        bid = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_BODY, name)
        return self._d.xmat[bid].reshape(3, 3).copy()


class _SimWrapper:
    """Mimics mujoco_py.MjSim using the new mujoco package."""

    def __init__(self, m, d):
        self._m = m
        self._d = d
        self.data = _DataWrapper(m, d)

    def reset(self):
        mujoco.mj_resetData(self._m, self._d)

    def forward(self):
        mujoco.mj_forward(self._m, self._d)

    def step(self):
        mujoco.mj_step(self._m, self._d)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_model(xml_str=None, path=None):
    """Load MjModel from XML string or file path."""
    if xml_str is not None:
        return mujoco.MjModel.from_xml_string(xml_str)
    return mujoco.MjModel.from_xml_path(str(path))


def convert_observation_to_space(observation):
    if isinstance(observation, list):
        return None
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)
    return space


# ---------------------------------------------------------------------------
# MujocoEnv — same public API as before, now backed by new mujoco package
# ---------------------------------------------------------------------------

class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments (backed by mujoco >= 3.x)."""

    def __init__(self, fullpath, frame_skip, mujoco_xml=None):
        if mujoco is None:
            raise RuntimeError("mujoco package not installed. Run: pip install mujoco")

        if mujoco_xml is not None:
            raw_model = _load_model(xml_str=mujoco_xml)
        else:
            fp = Path(fullpath)
            if not fp.exists():
                fp = Path(__file__).parents[3] / 'assets' / 'mujoco_models' / fp.name
                if not fp.exists():
                    raise IOError(f"File {fullpath} does not exist")
            raw_model = _load_model(path=fp)

        self.frame_skip = frame_skip
        self._setup_sim(raw_model)
        self.is_inited = False

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, termination, truncation, _info = self.step(action)
        assert not (termination or truncation)

        self._set_observation_space(observation)
        self.seed()
        self.is_inited = True

    def _setup_sim(self, raw_model):
        """Create wrapper objects from a raw MjModel."""
        raw_data = mujoco.MjData(raw_model)
        self.model = _ModelWrapper(raw_model)
        self.sim = _SimWrapper(raw_model, raw_data)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}
        self.init_qpos = raw_data.qpos.ravel().copy()
        self.init_qvel = raw_data.qvel.ravel().copy()

    # ------------------------------------------------------------------

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reload_sim_model(self, xml_str):
        raw_model = _load_model(xml_str=xml_str)
        self._setup_sim(raw_model)

    # ------------------------------------------------------------------
    # Subclass hooks

    def reset_model(self):
        raise NotImplementedError

    def viewer_setup(self):
        pass

    # ------------------------------------------------------------------

    def reset(self):
        self.sim.reset()
        return self.reset_model()

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.sim._d.qpos[:] = qpos
        self.sim._d.qvel[:] = qvel
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.sim._d.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE,
               camera_id=None, camera_name=None):
        # Rendering requires a display; headless training skips this.
        pass

    def close(self):
        self.viewer = None
        self._viewers = {}

    def _get_viewer(self, mode):
        return None

    def set_custom_key_callback(self, key_func):
        pass

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([
            self.sim._d.qpos.flat,
            self.sim._d.qvel.flat
        ])

    def vec_body2world(self, body_name, vec):
        body_xmat = self.data.get_body_xmat(body_name)
        return (body_xmat @ vec[:, None]).ravel()

    def pos_body2world(self, body_name, pos):
        body_xpos = self.data.get_body_xpos(body_name)
        body_xmat = self.data.get_body_xmat(body_name)
        return (body_xmat @ pos[:, None]).ravel() + body_xpos
