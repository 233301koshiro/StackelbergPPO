#!/usr/bin/env python3
"""
学習済みモデルの最終形態（モルフォロジー）を URDF と MuJoCo XML で保存する。

設計フェーズ（骨格変換・属性変換）をポリシーに従って実行し、
実行フェーズ開始時点の形態ファイルを保存する。

使い方:
  EVAL_RESTORE_DIR=single_run/pusher_cnoid \
  USE_CHOREONOID=1 choreonoid --no-window --python scripts/save_morphology_urdf.py

出力:
  {restore_dir}/morphology/morphology_best.urdf   ← Choreonoid / RViz で直接開ける
  {restore_dir}/morphology/morphology_best.xml    ← MuJoCo XML（元フォーマット）
"""

import os, sys
sys.path.append(os.getcwd())
os.environ['USE_CHOREONOID'] = '1'

import numpy as np, yaml
from omegaconf import OmegaConf

from khrylib.utils import *
from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent, tensorfy
from design_opt.utils.tools import set_global_seed
from khrylib.rl.envs.common.mujoco_env_choreonoid import mujoco_xml_to_urdf

project_path = os.getcwd()

class args:
    restore_dir = os.environ.get('EVAL_RESTORE_DIR', os.environ.get('VIEWER_RESTORE_DIR'))
    epoch       = os.environ.get('EVAL_EPOCH', 'best')
    episodes    = int(os.environ.get('EVAL_MORPHO_EPISODES', '10'))

if not args.restore_dir:
    print("Error: EVAL_RESTORE_DIR が必要です")
    sys.exit(1)

# ── 設定・エージェント読み込み ──────────────────────────────────────────────
FLAGS = OmegaConf.create(yaml.safe_load(
    open(os.path.join(project_path, args.restore_dir, ".hydra", "config.yaml"))))
cfg   = Config(FLAGS, project_path, args.restore_dir)
cfg.restore_dir = args.restore_dir

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cpu')
set_global_seed(cfg.seed)

epoch = int(args.epoch) if isinstance(args.epoch, str) and args.epoch.isnumeric() else args.epoch
print(f"[save_urdf] Loading: {args.restore_dir}  epoch={epoch}")
agent = BodyGenAgent(cfg=cfg, dtype=dtype, device=device,
                     seed=cfg.seed, num_threads=1, training=False, checkpoint=epoch)
env = agent.env

# ── 複数エピソードで形態を収集（最も体が多いものを代表とする）────────────────
candidates = []

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

        if info.get('stage') == 'execution' and not captured:
            xml_bytes = env.robot.export_xml_string()
            body_names = [b.name for b in env.robot.bodies]
            candidates.append({'xml': xml_bytes, 'bodies': body_names, 'ep': ep})
            print(f"[save_urdf] Ep {ep:2d}: bodies({len(body_names)})={body_names}")
            captured = True

        state = next_state

if not candidates:
    print("Error: 形態を1つも取得できませんでした")
    sys.exit(1)

# 最もボディ数が多い形態を代表として選ぶ
best = max(candidates, key=lambda c: len(c['bodies']))
print(f"\n[save_urdf] 代表: ep={best['ep']}  body数={len(best['bodies'])}")
print(f"[save_urdf] ツリー: {best['bodies']}")

# ── URDF と XML を保存 ──────────────────────────────────────────────────────
morph_dir = os.path.join(args.restore_dir, 'morphology')
os.makedirs(morph_dir, exist_ok=True)
prefix = os.path.join(morph_dir, f'morphology_{args.epoch}')

# MuJoCo XML
xml_path = prefix + '.xml'
with open(xml_path, 'wb') as f:
    f.write(best['xml'])
print(f"[save_urdf] MuJoCo XML 保存: {xml_path}")

# URDF
urdf_str, _, _, _, _ = mujoco_xml_to_urdf(best['xml'].decode())
urdf_path = prefix + '.urdf'
with open(urdf_path, 'w') as f:
    f.write(urdf_str)
print(f"[save_urdf] URDF 保存:       {urdf_path}")
