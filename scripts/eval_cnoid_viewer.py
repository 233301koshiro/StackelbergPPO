"""
Choreonoid の GUI ビューアでポリシーを再生するスクリプト。

使い方:
  vglrun choreonoid --python scripts/eval_cnoid_viewer.py -- \
      --restore_dir single_run/pusher_cnoid

  または（VGL なし・X11 ディスプレイあり）:
  choreonoid --python scripts/eval_cnoid_viewer.py -- \
      --restore_dir single_run/pusher_cnoid

オプション:
  --restore_dir   学習ディレクトリ（必須）
  --epoch         使用するチェックポイント（デフォルト: best）
  --fps           再生フレームレート（デフォルト: 25）
  --episodes      繰り返しエピソード数（デフォルト: 3、0 で無限ループ）
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['USE_CHOREONOID'] = '1'

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
import cnoid.IRSLUtil as IU

from khrylib.utils import *
from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent, tensorfy
from design_opt.utils.tools import set_global_seed

# ---- 引数パース -------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--restore_dir', type=str, required=True)
parser.add_argument('--epoch',    default='best')
parser.add_argument('--fps',      type=int, default=25)
parser.add_argument('--episodes', type=int, default=3,
                    help='再生エピソード数（0 で無限ループ）')
args = parser.parse_args()

step_interval = 1.0 / args.fps   # 各シミュレーションステップの待ち時間

# ---- 設定読み込み -------------------------------------------------------
project_path = os.getcwd()
train_cfg_path = os.path.join(project_path, args.restore_dir, '.hydra', 'config.yaml')
FLAGS = OmegaConf.create(yaml.safe_load(open(train_cfg_path)))
cfg = Config(FLAGS, project_path, args.restore_dir)
cfg.restore_dir = args.restore_dir

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cpu')
set_global_seed(cfg.seed)

epoch = int(args.epoch) if isinstance(args.epoch, str) and args.epoch.isnumeric() else args.epoch

# ---- エージェント読み込み -----------------------------------------------
print(f'[viewer] チェックポイント読み込み: {args.restore_dir}  epoch={epoch}')
agent = BodyGenAgent(cfg=cfg, dtype=dtype, device=device,
                     seed=cfg.seed, num_threads=1, training=False, checkpoint=epoch)
env   = agent.env
policy = agent.policy_net
policy.eval()
if agent.obs_norm is not None:
    agent.obs_norm.eval()
    agent.obs_norm.to(device)

print(f'[viewer] ロボット形態: {[b.name for b in env.robot.bodies]}')
print(f'[viewer] {args.fps} fps で再生開始  （Ctrl+C で停止）')
print()

# ---- 再生ループ ----------------------------------------------------------
ep = 0
try:
    while args.episodes == 0 or ep < args.episodes:
        state = env.reset()
        step  = 0
        total_reward = 0.0

        while True:
            # ネットワーク推論
            state_var = tensorfy([state])
            if agent.obs_norm is not None:
                state_var = agent.normalize_observation(state_var)
            with torch.no_grad():
                action = policy.select_action(state_var, mean_action=True).numpy().astype(np.float64)

            # シミュレーション 1 ステップ
            next_state, reward, term, trunc, info = env.step(action)

            # GUI 更新：processEvent() でシーンビューを描画
            IU.processEvent()

            if info.get('stage') == 'execution':
                total_reward += reward
                # 実行フェーズのみスリープ（形態変換フェーズは速送り）
                time.sleep(step_interval)

            done = term or trunc
            step += 1

            if done:
                break
            state = next_state

        ep += 1
        print(f'[viewer] Ep {ep}: reward={total_reward:.1f}  steps={step}  '
              f'bodies={[b.name for b in env.robot.bodies]}')

except KeyboardInterrupt:
    print('\n[viewer] 停止しました')

env.close()
