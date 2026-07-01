#!/usr/bin/env python3
"""
visualize_morph_changes.py: 学習中の形態（bone_offset）変化を可視化する。

複数チェックポイントの bone_length・rest_tip 位置・arm_safe_init 補正後の実際の先端方向を
1プロセス内で連続評価してプロットする（Choreonoid 起動は1回のみ）。

使い方:
  EVAL_RESTORE_DIR=single_run/rrbot_arm_cnoid_curriculum_v1 \
  USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
  /choreonoid_ws/install/bin/choreonoid --no-window \
      --python scripts/visualize_morph_changes.py

環境変数:
  EVAL_RESTORE_DIR   : 必須。分析対象の run ディレクトリ
  EVAL_SAMPLE_EVERY  : 何 epoch おきにサンプリングするか（デフォルト20）
  EVAL_OUTPUT        : 出力 PNG パス（デフォルト: {restore_dir}/videos/morph_changes.png）
"""
import os, sys, math
sys.path.append(os.getcwd())
os.environ['USE_CHOREONOID'] = '1'

import numpy as np
import yaml
import pickle
import torch
from pathlib import Path
from omegaconf import OmegaConf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent, tensorfy
from design_opt.utils.tools import set_global_seed

project_path = os.getcwd()
restore_dir = os.environ['EVAL_RESTORE_DIR']
sample_every = int(os.environ.get('EVAL_SAMPLE_EVERY', '20'))
output_path = os.environ.get('EVAL_OUTPUT',
    os.path.join(restore_dir, 'videos', 'morph_changes.png'))

# ── 設定読み込み ──────────────────────────────────────────────────────────
FLAGS = OmegaConf.create(yaml.safe_load(open(f'{restore_dir}/.hydra/config.yaml')))
cfg = Config(FLAGS, project_path, restore_dir)
cfg.restore_dir = restore_dir
torch.set_default_dtype(torch.float64)
set_global_seed(cfg.seed)

# ── チェックポイント一覧 ─────────────────────────────────────────────────
models_dir = Path(restore_dir) / 'models'
ckpts = sorted(
    [int(p.stem.split('_')[1]) for p in models_dir.glob('epoch_*.p')],
    key=int
)
# サンプリング
sampled = [e for e in ckpts if e % sample_every == 0]
if ckpts and ckpts[-1] not in sampled:
    sampled.append(ckpts[-1])
# best も追加
best_path = models_dir / 'best.p'
has_best = best_path.exists()
print(f'[morph_vis] チェックポイント {len(ckpts)} 件 → {len(sampled)} 件をサンプリング')

# ── 形態メトリクス収集 ───────────────────────────────────────────────────
def get_morph_metrics(agent, env):
    """形態フェーズを回してボディの bone_offset を返す。"""
    total_adim = env.control_action_dim + env.attr_design_dim + 1
    state = env.reset()
    for _ in range(cfg.skel_transform_nsteps + 2):
        sv = tensorfy([state])
        if agent.obs_norm is not None:
            sv = agent.normalize_observation(sv)
        with torch.no_grad():
            action = agent.policy_net.select_action(sv, mean_action=True).numpy().astype(np.float64)
        state, _, _, _, info = env.step(action)
        if info.get('stage') == 'execution':
            break
    bodies = env.robot.bodies
    metrics = {}
    for b in bodies:
        if b.bone_offset is not None:
            bo = np.asarray(b.bone_offset, dtype=float)
            metrics[b.name] = {
                'bone_length': float(np.linalg.norm(bo)),
                'bone_offset': bo[:2].copy()
            }
    # rest_tip (arm_safe_init 補正前)
    shoulder_xy = env.data.body_xpos[env.model._body_name2id[bodies[1].name]][:2]
    bo1  = np.asarray(bodies[1].bone_offset, dtype=float)[:2]
    bo11 = np.asarray(bodies[-1].bone_offset, dtype=float)[:2]
    rest_tip = shoulder_xy + bo1 + bo11
    # arm_safe_init 補正後の実際の先端方向
    link_vec = bo1 + bo11
    if cfg.env_specs.get('arm_safe_init', False):
        th = math.pi / 2
        rot = np.array([[math.cos(th), -math.sin(th)],
                         [math.sin(th),  math.cos(th)]])
        link_vec = rot @ link_vec
    actual_tip = shoulder_xy + link_vec
    metrics['_rest_tip']   = rest_tip
    metrics['_actual_tip'] = actual_tip
    return metrics

records = []
agent = None

def load_agent(epoch_arg):
    global agent
    if agent is None:
        agent = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                             seed=cfg.seed, num_threads=1, training=False,
                             checkpoint=epoch_arg)
    else:
        agent.load_checkpoint(epoch_arg)
    return agent

for ep in sampled:
    try:
        ag = load_agent(ep)
        m = get_morph_metrics(ag, ag.env)
        m['epoch'] = ep
        records.append(m)
        bl0 = m.get('0', {}).get('bone_length', 0)
        bl1 = m.get('1', {}).get('bone_length', 0)
        bl11= m.get('11',{}).get('bone_length', 0)
        tip = m['_actual_tip']
        print(f'  epoch={ep:5d}: L0={bl0:.3f} L1={bl1:.3f} L11={bl11:.3f} '
              f'actual_tip=({tip[0]:.3f},{tip[1]:.3f})')
    except Exception as e:
        print(f'  epoch={ep}: SKIP ({e})')

if has_best:
    try:
        ag = load_agent('best')
        m = get_morph_metrics(ag, ag.env)
        m['epoch'] = 'best'
        records.append(m)
        print(f'  best: L0={m.get("0",{}).get("bone_length",0):.3f} '
              f'L1={m.get("1",{}).get("bone_length",0):.3f} '
              f'L11={m.get("11",{}).get("bone_length",0):.3f}')
    except Exception as e:
        print(f'  best: SKIP ({e})')

if not records:
    print('[morph_vis] データなし。終了。')
    os._exit(1)

# ── プロット ─────────────────────────────────────────────────────────────
eps_num = [r['epoch'] for r in records if r['epoch'] != 'best']
best_rec = next((r for r in records if r['epoch'] == 'best'), None)

fig = plt.figure(figsize=(14, 10))
fig.suptitle(f'Morphology Changes: {Path(restore_dir).name}', fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# --- 左上: リンク長の推移 ---
ax1 = fig.add_subplot(gs[0, 0])
for bname, color, label in [('0','#888888','L0 (mount)'), ('1','#2196F3','L1 (shoulder→elbow)'), ('11','#FF5722','L2 (elbow→tip)')]:
    vals = [r.get(bname, {}).get('bone_length', 0) for r in records if r['epoch'] != 'best']
    ax1.plot(eps_num, vals, '-o', color=color, label=label, ms=4)
if best_rec:
    for bname, color in [('0','#888888'), ('1','#2196F3'), ('11','#FF5722')]:
        ax1.axhline(best_rec.get(bname, {}).get('bone_length', 0), color=color, ls='--', alpha=0.5)
ax1.set_xlabel('epoch'); ax1.set_ylabel('bone_length (m)')
ax1.set_title('Link Lengths over Training')
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# --- 右上: total reach の推移 ---
ax2 = fig.add_subplot(gs[0, 1])
reach = [sum(r.get(b,{}).get('bone_length',0) for b in ['0','1','11'])
         for r in records if r['epoch'] != 'best']
ax2.plot(eps_num, reach, '-o', color='#4CAF50', ms=4)
ax2.axhline(1.5, color='orange', ls='--', alpha=0.7, label='cube left face x≈1.35m')
ax2.axhline(1.0, color='red',    ls='--', alpha=0.5, label='cube_x_offset≈1.0')
ax2.set_xlabel('epoch'); ax2.set_ylabel('total reach (m)')
ax2.set_title('Total Arm Reach')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# --- 左下: actual_tip x/y の推移（arm_safe_init 補正後） ---
ax3 = fig.add_subplot(gs[1, 0])
tip_x = [r['_actual_tip'][0] for r in records if r['epoch'] != 'best']
tip_y = [r['_actual_tip'][1] for r in records if r['epoch'] != 'best']
ax3.plot(eps_num, tip_x, '-o', color='#2196F3', label='tip x (cube方向)', ms=4)
ax3.plot(eps_num, tip_y, '-o', color='#FF5722', label='tip y', ms=4)
ax3.axhline(0, color='gray', ls=':', alpha=0.5)
ax3.set_xlabel('epoch'); ax3.set_ylabel('position (m)')
ax3.set_title('Actual Tip Direction (arm_safe_init 補正後)')
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

# --- 右下: X-Y 平面上のアーム軌跡（最初・途中・最後） ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_aspect('equal')
cmap = plt.cm.viridis
for i, r in enumerate([rec for rec in records if rec['epoch'] != 'best']):
    color = cmap(i / max(len(eps_num)-1, 1))
    tip = r['_actual_tip']
    ax4.plot([0, tip[0]], [0, tip[1]], '-', color=color, alpha=0.6, lw=1.5)
    if i in (0, len(eps_num)//2, len(eps_num)-1):
        ax4.annotate(f"ep{r['epoch']}", tip, fontsize=7, color=color)
ax4.scatter(1.5, 0, marker='*', s=200, color='orange', zorder=5, label='cube center (offset=0.5)')
ax4.scatter(1.1, 0, marker='*', s=200, color='gold', zorder=5, label='cube center (offset=0.1)')
ax4.axhline(0, color='gray', ls=':', alpha=0.3)
ax4.axvline(0, color='gray', ls=':', alpha=0.3)
ax4.set_xlabel('X (push方向)'); ax4.set_ylabel('Y')
ax4.set_title('Tip方向の変化（X-Y 平面）'); ax4.legend(fontsize=7); ax4.grid(True, alpha=0.2)

os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
fig.savefig(output_path, dpi=120, bbox_inches='tight')
print(f'[morph_vis] 保存: {output_path}')
os._exit(0)
