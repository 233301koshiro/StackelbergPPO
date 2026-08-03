#!/usr/bin/env python3
"""
probe_reach_convergence.py: Reach 実行フェーズで「target に収束するまでに何ステップ
かかったか」を実測する。

用途（実験系譜.md 第8段「次の仮説1」= 収束速度仮説の検証）:
  tripo_arm_v2c_reach の best_R が rrbot_L1・TA より低いのは「収束に失敗したから」
  ではなく（3run とも最終的にサブミリ精度で収束することは確認済み）、「収束までの
  過渡期間が長く、その間のコストが積算されるから」という仮説を検証する。
  開始距離が遠いほど・gear が低いほど収束が遅い、という予測を数値で確かめる。

probe_reach_trajectory.py との違い: あちらは決められた時刻の推移を表示するだけで
「収束ステップ数」を出さない。本スクリプトは閾値到達ステップを直接算出する。

使い方:
  EVAL_RESTORE_DIR=single_run/tripo_arm_v2c_reach EVAL_CHECKPOINT=best \
  USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
      --no-window --python scripts/probe_reach_convergence.py

環境変数:
  EVAL_RESTORE_DIR  対象 run（必須）
  EVAL_CHECKPOINT   既定 'best'
"""
import os
import sys

sys.path.append(os.getcwd())
os.environ['USE_CHOREONOID'] = '1'

import numpy as np
import yaml
import torch
from omegaconf import OmegaConf

from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent, tensorfy
from design_opt.utils.tools import set_global_seed

# 収束とみなす距離のしきい値[m]。実験系譜.md の仮説記述に合わせて 0.05 / 0.01 を測り、
# サブミリ収束の確認用に 0.001 も加える。
THRESHOLDS = [0.05, 0.01, 0.001]

project_path = os.getcwd()
restore_dir = os.environ['EVAL_RESTORE_DIR']
checkpoint = os.environ.get('EVAL_CHECKPOINT', 'best')

FLAGS = OmegaConf.create(yaml.safe_load(open(f'{restore_dir}/.hydra/config.yaml')))
flags_dict = OmegaConf.to_container(FLAGS, resolve=True)
flags_dict.pop('restore_dir', None)
FLAGS = OmegaConf.create(flags_dict)
cfg = Config(FLAGS, project_path, restore_dir)
cfg.restore_dir = restore_dir
# Bug 10: 再評価時に転用フィルタが残ると重みが読み込まれない
cfg.control_prior = False
cfg.morph_prior = False
torch.set_default_dtype(torch.float64)
set_global_seed(cfg.seed)

target = np.array([
    cfg.env_specs.get('target_x', 0.8),
    cfg.env_specs.get('target_y', 0.0),
    cfg.env_specs.get('target_z', 0.15),
])

ckpt_arg = int(checkpoint) if checkpoint != 'best' else 'best'
agent = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                     seed=cfg.seed, num_threads=1, training=False, checkpoint=ckpt_arg)
env = agent.env


def get_arm_tip(env):
    """腕の末端リンクの tip 位置。env._body_names[-1] は cube を指すことがあるため
    env.robot.bodies（設計対象の腕のみ）から取る（第8段の計測バグ対策）。"""
    b_last = env.robot.bodies[-1]
    bo_last = getattr(b_last, 'bone_offset', None)
    if bo_last is None:
        bo_last = np.array([0.25, 0.0, 0.0])
    pos_last = np.array(env._body_xpos.get(b_last.name, np.zeros(3)))
    R_last = np.array(env._body_xmat.get(b_last.name, np.eye(3)))
    return pos_last + R_last @ np.asarray(bo_last, dtype=float)


state = env.reset()
dists = []
for _ in range(cfg.skel_transform_nsteps + 2 + 1100):
    in_exec = env.stage == 'execution'
    sv = tensorfy([state])
    if agent.obs_norm is not None:
        sv = agent.normalize_observation(sv)
    with torch.no_grad():
        action = agent.policy_net.select_action(sv, mean_action=True).numpy().astype(np.float64)
    state, reward, done, _, info = env.step(action)
    if in_exec:
        dists.append(float(np.linalg.norm(get_arm_tip(env) - target)))
    if done:
        break

# 形態パラメータ（リンク長と gear）— 仮説は「開始距離が遠く gear が低いほど遅い」
parts = []
total_reach = 0.0
for b in env.robot.bodies[1:]:
    off = np.asarray(b.bone_offset, dtype=float)
    length = float(np.linalg.norm(off))
    total_reach += length
    gear = None
    for joint in b.joints:
        if joint.actuator:
            gear = float(joint.actuator.gear)
    parts.append(f'{b.name}:len={length:.3f},gear={gear:.1f}' if gear is not None
                 else f'{b.name}:len={length:.3f}')

print(f'[conv] run={restore_dir} ckpt={checkpoint}')
print(f'[conv] target={target}  総リーチ={total_reach:.3f}m')
print(f'[conv] 形態: {" ".join(parts)}')

if not dists:
    print('[conv] 実行フェーズのステップが0件。収束判定不能')
    os._exit(1)

print(f'[conv] 実行ステップ数={len(dists)}  開始dist={dists[0]:.4f}  '
      f'最小dist={min(dists):.5f}  最終dist={dists[-1]:.5f}')

# 報酬は概ね距離のマイナスの積算になるため、「過渡期の損」と「定常保持の損」を
# 分けて出す。仮説（過渡が長いほど総報酬が下がる）の検証に直接使う。
n = len(dists)
tail_start = min(200, n - 1)          # ep 序盤200ステップを過渡、以降を定常とみなす
transient = dists[:tail_start]
steady = dists[tail_start:]
print(f'[conv] 距離の積算: 全体={sum(dists):.2f}  '
      f'過渡(0-{tail_start})={sum(transient):.2f}  定常({tail_start}-)={sum(steady):.2f}')
if steady:
    print(f'[conv] 定常区間の平均dist={np.mean(steady):.5f}  最大dist={max(steady):.5f}')

for th in THRESHOLDS:
    # 初めて閾値を下回ったステップ（以降ずっと下回るとは限らないので初回到達を報告）
    first = next((i for i, d in enumerate(dists) if d < th), None)
    if first is None:
        print(f'[conv] dist<{th}: 未到達')
    else:
        # 到達後に再び上回った回数（保持の安定性）
        excursions = sum(1 for i in range(first, len(dists) - 1)
                         if dists[i] < th <= dists[i + 1])
        print(f'[conv] dist<{th}: {first} ステップで到達'
              f'（以降 {excursions} 回だけ再逸脱）')

sys.stdout.flush()
os._exit(0)
