#!/usr/bin/env python3
"""
学習済みモデルの最終形態（モルフォロジー）を静止画で可視化する。

実行フェーズ開始時点のロボット骨格を 3 アングル（正面・横・上面）で保存する。
論文図との比較用。

使い方:
  EVAL_RESTORE_DIR=single_run/pusher_cnoid \
  USE_CHOREONOID=1 choreonoid --no-window --python scripts/eval_morphology.py
"""

import os, sys
sys.path.append(os.getcwd())
os.environ['USE_CHOREONOID'] = '1'

import numpy as np
import yaml
from omegaconf import OmegaConf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from khrylib.utils import *
from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent, tensorfy
from design_opt.utils.tools import set_global_seed

project_path = os.getcwd()

class args:
    restore_dir = os.environ.get('EVAL_RESTORE_DIR', os.environ.get('VIEWER_RESTORE_DIR'))
    epoch       = os.environ.get('EVAL_EPOCH', 'best')
    output      = os.environ.get('EVAL_MORPHO_OUTPUT', None)
    episodes    = int(os.environ.get('EVAL_MORPHO_EPISODES', '5'))

if not args.restore_dir:
    print("Error: EVAL_RESTORE_DIR が必要です")
    sys.exit(1)

out_path = args.output or os.path.join(args.restore_dir, 'morphology.png')

# ── 設定・エージェント読み込み ──────────────────────────────────────────────
train_config_path = os.path.join(project_path, args.restore_dir, ".hydra", "config.yaml")
FLAGS = OmegaConf.create(yaml.safe_load(open(train_config_path)))
cfg = Config(FLAGS, project_path, args.restore_dir)
cfg.restore_dir = args.restore_dir

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cpu')
set_global_seed(cfg.seed)

epoch = int(args.epoch) if isinstance(args.epoch, str) and args.epoch.isnumeric() else args.epoch
print(f"[morpho] Loading: {args.restore_dir}  epoch={epoch}")
agent = BodyGenAgent(cfg=cfg, dtype=dtype, device=device,
                     seed=cfg.seed, num_threads=1, training=False, checkpoint=epoch)
env = agent.env

# ── 複数エピソードで形態を収集 ────────────────────────────────────────────
morphologies = []  # list of {body_positions, edges, body_names}

for ep in range(args.episodes):
    state = env.reset()
    done = False
    captured = False

    while not done and not captured:
        state_var = tensorfy([state])
        if agent.obs_norm is not None:
            state_var = agent.normalize_observation(state_var)
        with torch.no_grad():
            action = agent.policy_net.select_action(state_var, mean_action=True)
            action = action.numpy().astype(np.float64)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        stage = info.get('stage', '')

        # 実行フェーズ開始の最初のステップで形態をキャプチャ
        if stage == 'execution' and not captured:
            body_positions = {name: env.get_body_com(name).copy()
                              for name in env._body_names}
            edges = []
            for body in env.robot.bodies:
                if body.parent is not None:
                    p, c = body.parent.name, body.name
                    if p in body_positions and c in body_positions:
                        edges.append((body_positions[p], body_positions[c]))

            robot_names = [b.name for b in env.robot.bodies]
            morphologies.append({
                'body_positions': body_positions,
                'edges': edges,
                'robot_names': robot_names,
                'ep': ep,
            })
            captured = True
            print(f"[morpho] Ep {ep}: bodies={robot_names}")

        state = next_state

# ── 最も体が多いエピソードを代表として使う ──────────────────────────────────
# （骨格探索で最終的に一番進化した形態を選ぶ）
best_morph = max(morphologies, key=lambda m: len(m['robot_names']))
print(f"[morpho] 代表エピソード: ep={best_morph['ep']}  body数={len(best_morph['robot_names'])}")

bpos   = best_morph['body_positions']
edges  = best_morph['edges']
rnames = best_morph['robot_names']

# ── 3アングルで描画 ───────────────────────────────────────────────────────
VIEWS = [
    ('正面 (XZ)', 20, -60),
    ('横 (YZ)',   20,  30),
    ('上面 (XY)', 85, -60),
]

fig = plt.figure(figsize=(15, 5))
fig.suptitle(
    f"Robot Morphology — {args.restore_dir} (epoch={epoch})\n"
    f"bodies={len(rnames)}: {rnames}",
    fontsize=9
)

for col, (title, elev, azim) in enumerate(VIEWS):
    ax = fig.add_subplot(1, 3, col + 1, projection='3d')
    ax.set_title(title, fontsize=9)

    # リンク（親子線）
    for p_pos, c_pos in edges:
        ax.plot([p_pos[0], c_pos[0]],
                [p_pos[1], c_pos[1]],
                [p_pos[2], c_pos[2]],
                'b-', linewidth=2.5, alpha=0.8)

    # ロボットボディ（青丸）+ ラベル
    for name in rnames:
        if name not in bpos:
            continue
        p = bpos[name]
        ax.scatter(p[0], p[1], p[2], c='royalblue', s=80, zorder=5)
        ax.text(p[0], p[1], p[2] + 0.05, name, fontsize=6, ha='center', color='navy')

    # cube（オレンジ四角）
    if 'cube' in bpos:
        cp = bpos['cube']
        ax.scatter(cp[0], cp[1], cp[2], c='darkorange', s=150, marker='s', zorder=6)
        ax.text(cp[0], cp[1], cp[2] + 0.1, 'cube', fontsize=7, color='darkorange', ha='center')

    # 床
    xx, yy = np.meshgrid([-0.5, 2.0], [-1.0, 1.0])
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.08, color='gray')

    # +x 方向矢印
    ax.quiver(0, 0, 0.05, 0.4, 0, 0, color='green',
              linewidth=1.2, arrow_length_ratio=0.3, alpha=0.6)

    ax.set_xlim(-0.5, 2.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0, 1.5)
    ax.set_xlabel('X', fontsize=7)
    ax.set_ylabel('Y', fontsize=7)
    ax.set_zlabel('Z', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=elev, azim=azim)

plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[morpho] 保存: {out_path}")

# ── ツリー構造をテキストで出力 ────────────────────────────────────────────
def print_tree(body, prefix='', is_last=True):
    connector = '└── ' if is_last else '├── '
    p = bpos.get(body.name, np.zeros(3))
    print(f"{prefix}{connector}{body.name}  pos=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
    child_prefix = prefix + ('    ' if is_last else '│   ')
    for i, child in enumerate(body.child):
        print_tree(child, child_prefix, i == len(body.child) - 1)

print("\n[morpho] ロボット形態ツリー:")
root = env.robot.bodies[0]
print(f"└── {root.name}  pos=({bpos.get(root.name, np.zeros(3))[0]:.2f}, "
      f"{bpos.get(root.name, np.zeros(3))[1]:.2f}, "
      f"{bpos.get(root.name, np.zeros(3))[2]:.2f})")
for i, child in enumerate(root.child):
    print_tree(child, '    ', i == len(root.child) - 1)
