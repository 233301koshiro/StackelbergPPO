#!/usr/bin/env python3
"""
数値検証スクリプト: Choreonoid バックエンドで学習済みポリシーを評価し、
cube が実際に +x 方向に押せているかを確認する。

使い方:
  EVAL_RESTORE_DIR=single_run/pusher_cnoid \
  USE_CHOREONOID=1 choreonoid --no-window --python \
      scripts/eval_cnoid_numerical.py

出力:
  - 各エポードの cube 移動距離・報酬
  - cube が +x 方向に動いているかの判定
  - 報酬の推移グラフ（PNG）
"""

import os
import sys
sys.path.append(os.getcwd())

os.environ['USE_CHOREONOID'] = '1'

import numpy as np
import yaml
from omegaconf import OmegaConf

from khrylib.utils import *
from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent
from design_opt.utils.tools import set_global_seed

project_path = os.getcwd()

# Read parameters from environment variables
# EVAL_RESTORE_DIR: required
# EVAL_EPOCH:       checkpoint to load (default: best)
# EVAL_NUM_EPISODES: number of episodes (default: 5)
class args:
    restore_dir  = os.environ.get('EVAL_RESTORE_DIR')
    epoch        = os.environ.get('EVAL_EPOCH', 'best')
    num_episodes = int(os.environ.get('EVAL_NUM_EPISODES', '5'))
    plot         = True

if not args.restore_dir:
    print("Error: EVAL_RESTORE_DIR environment variable is required.")
    print("Usage: EVAL_RESTORE_DIR=single_run/pusher_cnoid USE_CHOREONOID=1 "
          "choreonoid --no-window --python scripts/eval_cnoid_numerical.py")
    sys.exit(1)

# ── 設定読み込み ──────────────────────────────────────────────────────────────
train_config_path = os.path.join(project_path, args.restore_dir, ".hydra", "config.yaml")
FLAGS = OmegaConf.create(yaml.safe_load(open(train_config_path)))
cfg = Config(FLAGS, project_path, args.restore_dir)
cfg.restore_dir = args.restore_dir

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cpu')
set_global_seed(cfg.seed)

epoch = int(args.epoch) if isinstance(args.epoch, str) and args.epoch.isnumeric() else args.epoch

# ── エージェント初期化 ────────────────────────────────────────────────────────
print(f"[eval] Loading checkpoint: {args.restore_dir} epoch={epoch}")
agent = BodyGenAgent(cfg=cfg, dtype=dtype, device=device,
                     seed=cfg.seed, num_threads=1, training=False, checkpoint=epoch)
env = agent.env

print(f"[eval] Robot bodies: {[b.name for b in env.robot.bodies]}")
print(f"[eval] Running {args.num_episodes} episodes...\n")

# ── 評価ループ ────────────────────────────────────────────────────────────────
results = []

for ep in range(args.num_episodes):
    state = env.reset()
    done = False
    ep_reward = 0.0
    cube_x_traj = []
    cube_y_traj = []
    step = 0

    while not done:
        # 観測を正規化してポリシーに入力
        from design_opt.agents.genesis_agent import tensorfy
        state_var = tensorfy([state])
        if agent.obs_norm is not None:
            state_var = agent.normalize_observation(state_var)

        with torch.no_grad():
            action = agent.policy_net.select_action(state_var, mean_action=True)
            action = action.numpy().astype(np.float64)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if info.get('stage') == 'execution':
            ep_reward += reward
            cube_pos = env.get_body_com("cube")
            cube_x_traj.append(cube_pos[0])
            cube_y_traj.append(cube_pos[1])

        state = next_state
        step += 1

    if len(cube_x_traj) >= 2:
        cube_x_displacement = cube_x_traj[-1] - cube_x_traj[0]
        cube_y_deviation    = abs(cube_y_traj[-1] - cube_y_traj[0])
        moving_forward = cube_x_displacement > 0
    else:
        cube_x_displacement = 0.0
        cube_y_deviation    = 0.0
        moving_forward = False

    results.append({
        'episode': ep,
        'reward': ep_reward,
        'cube_x_disp': cube_x_displacement,
        'cube_y_dev': cube_y_deviation,
        'exec_steps': len(cube_x_traj),
        'moving_forward': moving_forward,
        'cube_x_traj': cube_x_traj,
    })

    status = "✓ PUSHING" if moving_forward else "✗ NOT PUSHING"
    print(f"  Ep {ep:2d}: reward={ep_reward:7.2f}  "
          f"cube_x Δ={cube_x_displacement:+.4f}m  "
          f"cube_y dev={cube_y_deviation:.4f}m  "
          f"exec_steps={len(cube_x_traj):3d}  {status}")

# ── サマリー ──────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"  Episodes evaluated : {args.num_episodes}")
print(f"  Avg reward         : {np.mean([r['reward'] for r in results]):.2f}")
print(f"  Avg cube +x disp.  : {np.mean([r['cube_x_disp'] for r in results]):.4f} m")
print(f"  Avg cube y dev.    : {np.mean([r['cube_y_dev'] for r in results]):.4f} m")
push_rate = sum(r['moving_forward'] for r in results) / len(results)
print(f"  Episodes pushing   : {sum(r['moving_forward'] for r in results)}/{args.num_episodes}  ({push_rate*100:.0f}%)")
print("=" * 60)

# ── グラフ保存 ────────────────────────────────────────────────────────────────
if args.plot and results[0]['cube_x_traj']:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # cube x 軌跡
        ax = axes[0]
        for r in results:
            ax.plot(r['cube_x_traj'], label=f"ep{r['episode']} Δ={r['cube_x_disp']:+.3f}")
        ax.axhline(y=results[0]['cube_x_traj'][0] if results[0]['cube_x_traj'] else 0,
                   color='gray', linestyle='--', alpha=0.5, label='start x')
        ax.set_xlabel('Execution step')
        ax.set_ylabel('cube x position (m)')
        ax.set_title('Cube x-position over execution steps\n(should increase → being pushed)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # エピソード報酬バー
        ax = axes[1]
        eps = [r['episode'] for r in results]
        rews = [r['reward'] for r in results]
        colors = ['green' if r['moving_forward'] else 'red' for r in results]
        ax.bar(eps, rews, color=colors, alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode reward')
        ax.set_title('Episode rewards\n(green=cube moved forward, red=did not)')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        out_path = os.path.join(args.restore_dir, 'eval_numerical.png')
        plt.savefig(out_path, dpi=120)
        print(f"\n[eval] Plot saved: {out_path}")
    except Exception as e:
        print(f"[eval] Plot failed: {e}")
