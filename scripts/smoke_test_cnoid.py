#!/usr/bin/env python3
"""
Choreonoid環境の動作確認スモークテスト。

使い方:
  USE_CHOREONOID=1 /choreonoid_ws/install/bin/choreonoid --no-window \
    --python scripts/smoke_test_cnoid.py cfg=pusher xml_name=rrbot_arm

判定基準:
  PASS: 3エピソード完走, NaN/Infなし, step時間 < 10s, 報酬が0超え
  FAIL: ハング(30s/step超), NaN, 報酬が常に0, 例外

choreonoid_train.py と同じ起動パターンを使用。
"""

import sys
import os
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
os.environ.setdefault('USE_CHOREONOID', '1')

# choreonoid --python は sys.argv を埋めないので /proc/self/cmdline から復元
with open('/proc/self/cmdline', 'rb') as _f:
    _cmdline = [a.decode(errors='replace') for a in _f.read().split(b'\x00') if a]
try:
    _idx = _cmdline.index('--python')
    _script = _cmdline[_idx + 1]
    sys.argv = [_script] + _cmdline[_idx + 2:]
except (ValueError, IndexError):
    pass

import numpy as np
from omegaconf import OmegaConf

# --- Config 構築 (choreonoid_train.py と同じ手順) -------------------------
_overrides = {}
for arg in sys.argv[1:]:
    if '=' in arg:
        k, v = arg.split('=', 1)
        _overrides[k.lstrip('+')] = v

_cfg_id  = _overrides.get('cfg', 'pusher')
_xml     = _overrides.get('xml_name', 'rrbot_arm')
_project = _overrides.get('project_path', _ROOT)

FLAG = OmegaConf.create({
    'cfg':          _cfg_id,
    'xml_name':     _xml,
    'project_path': _project,
    'seed':         0,
    'uni_obs_norm': False,
    'enable_wandb': False,
    'env_init_height': False,
})

from design_opt.utils.config import Config
cfg = Config(FLAG, _project, '/tmp/smoke_test_cnoid')

from design_opt.envs import env_dict

# --- 環境作成 --------------------------------------------------------------
print(f'\n[smoke] cfg={_cfg_id}  xml={_xml}', flush=True)
print('[smoke] building env...', flush=True)
t0 = time.time()
env = env_dict[cfg.env_name](cfg, agent=None)
print(f'[smoke] env built in {time.time()-t0:.1f}s', flush=True)

# --- 判定用集計 ------------------------------------------------------------
N_EPISODES   = 3
MAX_STEP_SEC = 30.0   # 1ステップがこれを超えたらハング判定
results = {
    'step_times':    [],
    'exec_rewards':  [],
    'nan_count':     0,
    'errors':        [],
}

def _check_finite(val, label):
    arr = np.atleast_1d(val)
    if not np.all(np.isfinite(arr)):
        results['nan_count'] += 1
        results['errors'].append(f'NaN/Inf in {label}')

# --- エピソードループ -------------------------------------------------------
for ep in range(N_EPISODES):
    print(f'\n[smoke] episode {ep+1}/{N_EPISODES}', flush=True)

    obs = env.reset()
    _check_finite(obs, f'ep{ep} reset obs')

    ep_exec_reward = 0.0
    ep_steps = 0
    stage_done = False

    while not stage_done:
        action = env.action_space.sample() * 0.0   # ゼロアクション

        t_step = time.time()
        obs, reward, term, trunc, info = env.step(action)
        elapsed = time.time() - t_step

        results['step_times'].append(elapsed)
        _check_finite(obs, f'ep{ep} obs')
        _check_finite(reward, f'ep{ep} reward')

        if elapsed > MAX_STEP_SEC:
            msg = f'HANG: step took {elapsed:.1f}s > {MAX_STEP_SEC}s'
            results['errors'].append(msg)
            print(f'[smoke] {msg}', flush=True)
            stage_done = True

        if info.get('stage') == 'execution':
            ep_exec_reward += float(reward)
            ep_steps += 1

        if term or trunc:
            stage_done = True

    results['exec_rewards'].append(ep_exec_reward)
    print(f'[smoke]   exec_reward={ep_exec_reward:.2f}  exec_steps={ep_steps}  '
          f'step_time_avg={np.mean(results["step_times"][-ep_steps:] or [0]):.3f}s',
          flush=True)

# --- 判定 ------------------------------------------------------------------
print('\n' + '='*50, flush=True)
print('[smoke] RESULTS', flush=True)
print('='*50, flush=True)

step_times = results['step_times']
exec_rewards = results['exec_rewards']

avg_step = np.mean(step_times) if step_times else 999
max_step = np.max(step_times)  if step_times else 999
avg_reward = np.mean(exec_rewards) if exec_rewards else 0.0

print(f'  step time: avg={avg_step:.3f}s  max={max_step:.3f}s', flush=True)
print(f'  exec_reward: {exec_rewards}  avg={avg_reward:.2f}', flush=True)
print(f'  NaN/Inf count: {results["nan_count"]}', flush=True)
if results['errors']:
    for e in results['errors']:
        print(f'  ERROR: {e}', flush=True)

checks = {
    'step time < 10s':        max_step < 10.0,
    'exec reward > 0':        avg_reward > 0.0,
    'no NaN/Inf':             results['nan_count'] == 0,
    'no errors':              len(results['errors']) == 0,
    f'{N_EPISODES} eps done': len(exec_rewards) == N_EPISODES,
}

print('', flush=True)
all_pass = True
for label, ok in checks.items():
    mark = 'PASS' if ok else 'FAIL'
    if not ok:
        all_pass = False
    print(f'  [{mark}] {label}', flush=True)

print('', flush=True)
if all_pass:
    print('[smoke] ===  OVERALL: PASS  ===', flush=True)
else:
    print('[smoke] ===  OVERALL: FAIL  ===', flush=True)
print('='*50, flush=True)

sys.exit(0 if all_pass else 1)
