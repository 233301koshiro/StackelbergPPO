#!/usr/bin/env python3
"""
可視化スクリプト: 各ボディの 3D 座標を matplotlib でレンダリングし
mp4 動画として保存する。Choreonoid の GUI ウィンドウ不要。

使い方:
  EVAL_RESTORE_DIR=single_run/pusher_cnoid \
  USE_CHOREONOID=1 choreonoid --no-window --python \
      scripts/eval_cnoid_visual.py

描画内容:
  - 各ロボットボディ → 青い球
  - cube → オレンジの箱
  - 床 → 灰色の平面
  - 骨格のエッジ（親子リンク）→ 青い線
"""

import os
import sys
sys.path.append(os.getcwd())

os.environ['USE_CHOREONOID'] = '1'

import numpy as np
import yaml
from omegaconf import OmegaConf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection
import matplotlib.animation as animation

# Use imageio-ffmpeg bundled binary if system ffmpeg is unavailable
try:
    import imageio_ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

from khrylib.utils import *
from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent
from design_opt.utils.tools import set_global_seed

project_path = os.getcwd()

# Read parameters from environment variables
# EVAL_RESTORE_DIR:     required
# EVAL_EPOCH:           checkpoint to load (default: best)
# EVAL_OUTPUT:          output mp4 path (default: {restore_dir}/eval_visual.mp4)
# EVAL_FPS:             frames per second (default: 20)
# EVAL_MAX_EXEC_STEPS:  max execution stage steps (default: 200)
class args:
    restore_dir    = os.environ.get('EVAL_RESTORE_DIR')
    epoch          = os.environ.get('EVAL_EPOCH', 'best')
    output         = os.environ.get('EVAL_OUTPUT', None)
    fps            = int(os.environ.get('EVAL_FPS', '20'))
    max_exec_steps = int(os.environ.get('EVAL_MAX_EXEC_STEPS', '200'))

if not args.restore_dir:
    print("Error: EVAL_RESTORE_DIR environment variable is required.")
    print("Usage: EVAL_RESTORE_DIR=single_run/pusher_cnoid USE_CHOREONOID=1 "
          "choreonoid --no-window --python scripts/eval_cnoid_visual.py")
    sys.exit(1)

out_path = args.output or os.path.join(args.restore_dir, 'videos', 'eval_visual.mp4')
os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

# ── 設定・エージェント読み込み ─────────────────────────────────────────────
train_config_path = os.path.join(project_path, args.restore_dir, ".hydra", "config.yaml")
FLAGS = OmegaConf.create(yaml.safe_load(open(train_config_path)))
cfg = Config(FLAGS, project_path, args.restore_dir)
cfg.restore_dir = args.restore_dir

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cpu')
set_global_seed(cfg.seed)

epoch = int(args.epoch) if isinstance(args.epoch, str) and args.epoch.isnumeric() else args.epoch
print(f"[visual] Loading checkpoint: {args.restore_dir} epoch={epoch}")
agent = BodyGenAgent(cfg=cfg, dtype=dtype, device=device,
                     seed=cfg.seed, num_threads=1, training=False, checkpoint=epoch)
env = agent.env

# ── 1エピソード実行してフレームデータ収集 ─────────────────────────────────
print("[visual] Running episode to collect frames...")
state = env.reset()
done = False
frames = []   # list of {body_xpos, cube_pos, robot_edges, stage}
exec_steps = 0

while not done and exec_steps < args.max_exec_steps:
    from design_opt.agents.genesis_agent import tensorfy
    state_var = tensorfy([state])
    if agent.obs_norm is not None:
        state_var = agent.normalize_observation(state_var)
    with torch.no_grad():
        action = agent.policy_net.select_action(state_var, mean_action=True)
        action = action.numpy().astype(np.float64)

    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    stage = info.get('stage', '')

    # ボディ座標を収集
    body_positions = {name: env.get_body_com(name).copy()
                      for name in env._body_names}

    # ロボットの親子エッジを収集（ツリー構造）
    edges = []
    for body in env.robot.bodies:
        if body.parent is not None:
            p_name = body.parent.name
            c_name = body.name
            if p_name in body_positions and c_name in body_positions:
                edges.append((body_positions[p_name], body_positions[c_name]))

    frames.append({
        'body_positions': body_positions,
        'edges': edges,
        'stage': stage,
        'reward': reward,
    })

    if stage == 'execution':
        exec_steps += 1

    state = next_state

print(f"[visual] Collected {len(frames)} frames "
      f"({exec_steps} execution steps)")

# ── アニメーション作成 ────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

def draw_frame(i):
    ax.cla()
    frame = frames[i]
    bpos = frame['body_positions']
    stage = frame['stage']

    # 床（z=0 平面）
    xx, yy = np.meshgrid([-1.5, 2.5], [-1.5, 1.5])
    ax.plot_surface(xx, yy, np.zeros_like(xx),
                    alpha=0.15, color='gray', zorder=0)

    # ロボットボディのリンク（親子線）
    for p_pos, c_pos in frame['edges']:
        ax.plot([p_pos[0], c_pos[0]],
                [p_pos[1], c_pos[1]],
                [p_pos[2], c_pos[2]],
                'b-', linewidth=2.5, alpha=0.8)

    # ロボットボディ（球）
    robot_names = [b.name for b in env.robot.bodies]
    for name in robot_names:
        if name in bpos:
            p = bpos[name]
            ax.scatter(p[0], p[1], p[2],
                       c='royalblue', s=80, zorder=5, depthshade=True)

    # cube（オレンジ）
    if 'cube' in bpos:
        cp = bpos['cube']
        ax.scatter(cp[0], cp[1], cp[2],
                   c='darkorange', s=200, marker='s', zorder=6, depthshade=True)
        ax.text(cp[0], cp[1], cp[2] + 0.12, 'cube',
                fontsize=8, color='darkorange', ha='center')

    # +x 方向の矢印（push 方向）
    ax.quiver(0, 0, 0.05, 0.5, 0, 0,
              color='green', linewidth=1.5, arrow_length_ratio=0.3, alpha=0.6)
    ax.text(0.6, 0, 0.1, '+x (push)', fontsize=8, color='green')

    ax.set_xlim(-1.0, 2.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(0, 2.0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Frame {i+1}/{len(frames)}  stage={stage}  r={frame['reward']:.3f}",
                 fontsize=10)
    ax.view_init(elev=25, azim=-60)

print(f"[visual] Rendering {len(frames)} frames...")
ani = animation.FuncAnimation(fig, draw_frame, frames=len(frames),
                               interval=1000 // args.fps, blit=False)

writer = animation.FFMpegWriter(fps=args.fps, bitrate=1800)
ani.save(out_path, writer=writer)
plt.close()
print(f"[visual] Saved: {out_path}")
