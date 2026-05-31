"""
Choreonoid-based simulation environment.
Drop-in replacement for mujoco_env_gym.py.

Architecture:
  This process (Python 3.9, PPO/PyTorch) ←→ cnoid_sim_server.py (Python 3.8, Choreonoid)
  Communication via ZeroMQ REQ/REP on localhost.

Usage:
  1. Start Choreonoid server:
       xvfb-run choreonoid --python path/to/cnoid_sim_server.py
  2. Use ChoreonoidEnv in place of MujocoEnv (same API).
"""

import os
import numpy as np
import json
import zmq
from os import path
from pathlib import Path
from gym import spaces
from gym.utils import seeding

DEFAULT_SIZE = 500
DEFAULT_PORT = int(os.environ.get('CNOID_PORT', 5556))


class ChoreonoidEnv:
    """
    Same public interface as MujocoEnv (mujoco_env_gym.py).
    Simulation is delegated to Choreonoid via ZMQ.
    """

    def __init__(self, fullpath, frame_skip, mujoco_xml=None, port=DEFAULT_PORT):
        self.frame_skip = frame_skip
        self._port = port

        # ZMQ REQ socket (synchronous request-reply)
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.connect(f"tcp://localhost:{port}")

        # Load model into server
        if mujoco_xml is not None:
            xml_str = mujoco_xml
        else:
            if not path.exists(fullpath):
                fullpath = path.join(
                    Path(__file__).parent.parent.parent.parent,
                    'assets/mujoco_models', path.basename(fullpath)
                )
            with open(fullpath, 'r') as f:
                xml_str = f.read()

        self._last_xml = xml_str  # kept for reconnect() after fork
        info = self._send({'cmd': 'load_model', 'xml_str': xml_str, 'frame_skip': frame_skip})
        self._apply_model_info(info)

        self.viewer = None
        self._viewers = {}
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

    def _send(self, msg: dict) -> dict:
        self._sock.send_json(msg)
        resp = self._sock.recv_json()
        if 'error' in resp:
            raise RuntimeError(
                f"ChoreonoidEnv server error [{msg.get('cmd')}]: {resp['error']}\n"
                + resp.get('traceback', '')
            )
        return resp

    def _apply_model_info(self, info: dict):
        """Store model metadata returned by the server after load_model."""
        self.nq = info['nq']
        self.nv = info['nv']
        self._actuator_names = info['actuator_names']
        self._actuator_ctrlrange = np.array(info['ctrlrange'], dtype=np.float64)
        self._timestep = info['timestep']
        self.init_qpos = np.array(info['init_qpos'], dtype=np.float64)
        self.init_qvel = np.array(info['init_qvel'], dtype=np.float64)
        self._body_names = info['body_names']       # list of all body names in order
        self._body_jntadr = info['body_jntadr']     # same as model.body_jntadr
        self._body_jntnum = info['body_jntnum']
        self._jnt_qposadr = info['jnt_qposadr']
        # Initialize state caches to safe defaults (overwritten after first step/reset)
        self._qpos = np.zeros(self.nq)
        self._qvel = np.zeros(self.nv)
        self._ctrl = np.zeros(len(self._actuator_names))
        self._body_xpos = {n: np.zeros(3) for n in self._body_names}
        self._body_xmat = {n: np.eye(3)   for n in self._body_names}

    # ------------------------------------------------------------------
    # MujocoEnv-compatible properties

    @property
    def dt(self):
        return self._timestep * self.frame_skip

    @property
    def model(self):
        """Minimal model proxy so existing env code can read model.nq etc."""
        return _ModelProxy(self)

    @property
    def data(self):
        """Minimal data proxy so existing env code can access data.qpos etc."""
        return _DataProxy(self)

    # ------------------------------------------------------------------
    # Public API (same as MujocoEnv)

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
            self.observation_space = None  # StackelbergPPO uses list observations
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
        resp = self._send({'cmd': 'reset'})
        self._cache_state(resp)
        return self.reset_model()

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.nq,) and qvel.shape == (self.nv,)
        resp = self._send({
            'cmd': 'set_state',
            'qpos': qpos.tolist(),
            'qvel': qvel.tolist(),
        })
        self._cache_state(resp)

    def do_simulation(self, ctrl, n_frames):
        resp = self._send({
            'cmd': 'step',
            'ctrl': ctrl.tolist(),
            'n_frames': n_frames,
        })
        self._cache_state(resp)

    def reload_sim_model(self, xml_str: str):
        """
        Replace the current robot model (called when morphology changes).
        Equivalent to mujoco_py.MjSim reload.
        """
        self._last_xml = xml_str  # keep for reconnect() after fork
        info = self._send({'cmd': 'reload_model', 'xml_str': xml_str})
        self._apply_model_info(info)
        self._set_action_space()   # actuator count may have changed

    def reconnect(self, port: int):
        """
        Close the current ZMQ socket and reconnect to a different port.
        Used by forked worker processes to connect to their own Choreonoid server
        while preserving the current morphology (reloads _last_xml on the new server).
        """
        self._sock.close()
        self._ctx.term()
        self._port = port
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.connect(f"tcp://localhost:{port}")
        # Reload current morphology on the new server
        info = self._send({'cmd': 'load_model', 'xml_str': self._last_xml,
                           'frame_skip': self.frame_skip})
        self._apply_model_info(info)

    def state_vector(self):
        return np.concatenate([self._qpos, self._qvel])

    def get_body_com(self, body_name: str) -> np.ndarray:
        return self._body_xpos[body_name]

    def vec_body2world(self, body_name: str, vec: np.ndarray) -> np.ndarray:
        xmat = self._body_xmat[body_name]
        return (xmat @ vec[:, None]).ravel()

    def pos_body2world(self, body_name: str, pos: np.ndarray) -> np.ndarray:
        xpos = self._body_xpos[body_name]
        xmat = self._body_xmat[body_name]
        return (xmat @ pos[:, None]).ravel() + xpos

    def close(self):
        self._sock.close()
        self._ctx.term()

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        pass  # visualization handled by Choreonoid side

    # ------------------------------------------------------------------
    # State cache (updated after each server call)

    def _cache_state(self, resp: dict):
        self._qpos = np.array(resp['qpos'], dtype=np.float64)
        self._qvel = np.array(resp['qvel'], dtype=np.float64)
        self._body_xpos = {k: np.array(v) for k, v in resp.get('body_xpos', {}).items()}
        self._body_xmat = {k: np.array(v).reshape(3, 3)
                           for k, v in resp.get('body_xmat', {}).items()}
        self._ctrl = np.array(resp.get('ctrl', [0.0] * len(self._actuator_names)))

    # ------------------------------------------------------------------
    # Subclass hooks (same as MujocoEnv)

    def step(self, action):
        raise NotImplementedError

    def reset_model(self):
        raise NotImplementedError

    def viewer_setup(self):
        pass


# ------------------------------------------------------------------
# Lightweight proxies so existing env code like
#   self.model.nq  /  self.data.qpos  /  self.data.ctrl
# keeps working without change.

class _ModelProxy:
    def __init__(self, env: ChoreonoidEnv):
        self._env = env

    @property
    def nq(self):          return self._env.nq
    @property
    def nv(self):          return self._env.nv
    @property
    def nu(self):          return len(self._env._actuator_names)
    @property
    def actuator_names(self): return self._env._actuator_names
    @property
    def actuator_ctrlrange(self): return self._env._actuator_ctrlrange
    @property
    def body_names(self):  return self._env._body_names
    @property
    def body_jntadr(self): return self._env._body_jntadr
    @property
    def body_jntnum(self): return self._env._body_jntnum
    @property
    def jnt_qposadr(self): return self._env._jnt_qposadr

    class _Opt:
        def __init__(self, timestep): self.timestep = timestep
    @property
    def opt(self): return self._Opt(self._env._timestep)

    def _camera_name2id(self): return {}

    # body name → id
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
        # Return array indexed by body id (same as MuJoCo data.body_xpos)
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
