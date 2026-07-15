"""
eval_reach_hover.py: Reach 系 run の「収束床の内訳」を判別するプローブ。

累積 -dist 報酬の床（G 系 -66〜-73）が
  (a) 幾何的到達不能（hover 距離が大きい: ≈6cm）なのか
  (b) 移動時間 + ホバージッタの累積（hover ≈ 0〜2cm、transit が支配）なのか
を、tip-target 距離の時系列 dist(t) を直接測定して判別する。

使い方:
  EVAL_RESTORE_DIR=single_run/rrbot_arm_reach_G3 EVAL_NUM_EPISODES=3 \
    USE_CHOREONOID=1 choreonoid --no-window --python scripts/eval_reach_hover.py

背景: docs/研究応用/形態比較.md「未決の問い」（2026-07-10）
"""
import os
import sys
import numpy as np

os.environ.setdefault('USE_CHOREONOID', '1')
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_path)

import traceback
import yaml
import torch
from omegaconf import OmegaConf
from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent, tensorfy

# 例外時も os._exit で確実に終了する（Qt イベントループ残留による
# ハング → timeout SIGKILL 待ちを防ぐ。eval スクリプト共通の既知問題）
import builtins
_orig_excepthook = sys.excepthook
def _exit_on_exception(etype, e, tb):
    traceback.print_exception(etype, e, tb)
    os._exit(1)
sys.excepthook = _exit_on_exception

restore_dir = os.environ.get('EVAL_RESTORE_DIR')
num_episodes = int(os.environ.get('EVAL_NUM_EPISODES', '3'))
epoch = os.environ.get('EVAL_EPOCH', 'best')

if not restore_dir:
    print("Error: set EVAL_RESTORE_DIR")
    os._exit(1)

train_config_path = os.path.join(project_path, restore_dir, ".hydra", "config.yaml")
FLAGS = OmegaConf.create(yaml.safe_load(open(train_config_path)))
# 学習時の restore_dir（例: 既に存在しない旧 run）が残っていると Config が
# FileNotFoundError になるため、この run 自身で上書きする（compare_morphology.py と同じ手法）
OmegaConf.update(FLAGS, 'restore_dir', restore_dir)
cfg = Config(FLAGS, project_path, restore_dir)
cfg.restore_dir = restore_dir
# 完走 run 自身の checkpoint を再評価する際は転用フィルタを無効化する。
# control_prior/morph_prior が true のままだと load_checkpoint が Leader/Follower の
# 一方と obs_norm を読み込まず、ランダム初期化ネットで形態・行動が再現される
# （2026-07-15 発覚。デバッグ戦記 Bug 10）。
cfg.control_prior = False
cfg.morph_prior = False

if not cfg.reward_specs.get('use_reach', False):
    print(f"Error: {restore_dir} は Reach run ではありません (use_reach=false)")
    os._exit(1)

print(f"[hover-probe] Loading: {restore_dir} epoch={epoch}")
torch.set_default_dtype(torch.float64)
device = torch.device('cpu')
agent = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=device,
                     seed=cfg.seed, num_threads=1, training=False, checkpoint=epoch)
env = agent.env

target = np.array([
    cfg.reward_specs.get('target_x', 1.5),
    cfg.reward_specs.get('target_y', 0.0),
    cfg.reward_specs.get('target_z', 0.15),
])
print(f"[hover-probe] target={target}")

all_hover, all_transit, all_min, all_mean = [], [], [], []
for ep in range(num_episodes):
    state = env.reset()
    done = False
    dists = []
    while not done:
        state_var = tensorfy([state])
        if agent.obs_norm is not None:
            state_var = agent.normalize_observation(state_var)
        with torch.no_grad():
            action = agent.policy_net.select_action(state_var, mean_action=(os.environ.get('EVAL_STOCHASTIC') != '1')).numpy().astype(np.float64)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if info.get('stage') == 'execution':
            dists.append(float(np.linalg.norm(env._arm_tip_pos - target)))

    d = np.array(dists)
    if len(d) < 50:
        print(f"  Ep {ep}: exec steps={len(d)} (短すぎ、スキップ)")
        continue
    hover = d[-200:].mean() if len(d) >= 400 else d[len(d)//2:].mean()
    hover_std = d[-200:].std() if len(d) >= 400 else d[len(d)//2:].std()
    # transit = hover+1cm を初めて下回るまでのステップ数
    below = np.nonzero(d < hover + 0.01)[0]
    transit = int(below[0]) if len(below) else len(d)
    marks = {i: d[i] for i in [0, 50, 100, 200, 500, len(d)-1] if i < len(d)}
    print(f"  Ep {ep}: steps={len(d)}  mean={d.mean():.4f}  min={d.min():.4f}  "
          f"hover(last)={hover:.4f}±{hover_std:.4f}  transit≈{transit}steps")
    print(f"         dist(t): " + "  ".join(f"t={i}:{v:.3f}" for i, v in sorted(marks.items())))
    all_hover.append(hover); all_transit.append(transit); all_min.append(d.min()); all_mean.append(d.mean())

if all_hover:
    print("\n[hover-probe] === 判定 ===")
    print(f"  mean dist (≈ -exec_R)     : {np.mean(all_mean):.4f}")
    print(f"  hover dist（終盤平均）     : {np.mean(all_hover):.4f}")
    print(f"  min dist                  : {np.mean(all_min):.4f}")
    print(f"  transit steps             : {np.mean(all_transit):.0f}")
    hv = np.mean(all_hover)
    if hv > 0.04:
        print("  → (a) 幾何/姿勢的に届いていない床（hover が大きい）")
    else:
        print("  → (b) transit+ジッタの累積が床の主因（hover は小さい）")

os._exit(0)
