#!/usr/bin/env python3
"""学習済み run を再生し、**log の best を再現できるか**を確かめる（CLAUDE.md §5-2 ⑤-3）。

**なぜ要るか**: 2026-09-10 に「再生と log が 60 倍合わない」と誤診した（9-66）。
原因は **1 エピソードで判定したこと**で、読み込み直後の 1〜2 話が当てにならない。
**3 話目から log の値に落ち着く。** その運用を毎回手で守るのは無理なので道具にした。

**何を比べるか**: 報酬の式を再構成せず、**env が返す reward をそのまま積む**。
これは logger の `exec_episode_reward` と同じ量なので、log の best と直接比べられる。
（9-66 では式を手で再構成して二度読み違えた。再構成しないのが要点。）

使い方:
  EVAL_RESTORE_DIR=single_run/e2e_a1_reach [EVAL_EPOCH=best] [EVAL_NUM_EPISODES=5] \
  USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
      --no-window --python scripts/verify_run_replay.py
"""
import os, sys
sys.path.append(os.getcwd())
os.environ.setdefault('USE_CHOREONOID', '1')

import numpy as np, yaml, torch
from omegaconf import OmegaConf
from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent, tensorfy
from design_opt.utils.tools import set_global_seed

DISCARD = int(os.environ.get('EVAL_DISCARD', '2'))   # 頭の何話を捨てるか（9-66）

rd = os.environ['EVAL_RESTORE_DIR']
_ep = os.environ.get('EVAL_EPOCH', 'best')
epoch = int(_ep) if _ep.isdigit() else _ep           # ⚠️ int で渡さないと epoch_0010.p を探さない
nep = int(os.environ.get('EVAL_NUM_EPISODES', '5'))

FLAGS = OmegaConf.create(yaml.safe_load(open(f'{rd}/.hydra/config.yaml')))
OmegaConf.update(FLAGS, 'restore_dir', rd)
cfg = Config(FLAGS, os.getcwd(), rd)
cfg.restore_dir = rd
cfg.control_prior = False      # Bug 10: true のままだとランダム初期化ネットで再現される
cfg.morph_prior = False
torch.set_default_dtype(torch.float64)
set_global_seed(cfg.seed)
agent = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                     seed=cfg.seed, num_threads=1, training=False, checkpoint=epoch)
env = agent.env
is_reach = bool(cfg.reward_specs.get('use_reach', False))

# log の best（第①層）を読む
best_log = None
try:
    import re
    for line in open(f'{rd}/log/log_train.txt'):
        m = re.search(r'rewards (-?\d+\.\d+)', line)
        if m:
            best_log = float(m.group(1))
except Exception:
    pass

rows = []
for e in range(nep):
    state = env.reset()
    tot, n = 0.0, 0
    for _ in range(cfg.skel_transform_nsteps + 2 + 1200):
        in_exec = env.stage == 'execution'
        sv = tensorfy([state])
        if agent.obs_norm is not None:
            sv = agent.normalize_observation(sv)
        with torch.no_grad():
            a = agent.policy_net.select_action(sv, mean_action=True).numpy().astype(np.float64)
        state, r, done, trunc, _ = env.step(a)
        if in_exec:
            tot += float(r); n += 1
        if done or trunc:
            break
    rows.append((tot, n))

print(f"\n[verify] {rd}  epoch={epoch}  task={'Reach' if is_reach else 'Pusher'}  {nep} エピソード")
print(f"[verify] 各話の exec 合計報酬（= log の exec_R_eps と同じ量）:")
for i, (tot, n) in enumerate(rows):
    mark = "  ← 捨てる（読み込み直後は当てにならない。9-66）" if i < DISCARD else ""
    print(f"         ep{i+1}: {tot:10.2f}  ({n} steps){mark}")

late = [t for t, _ in rows[DISCARD:]]
print(f"\n[verify] {DISCARD+1} 話目以降の平均 : {np.mean(late):10.2f}")
print(f"[verify] 全話の単純平均       : {np.mean([t for t,_ in rows]):10.2f}  ⚠️ 頭に引きずられるので使わない")
if best_log is not None:
    diff = abs(np.mean(late) - best_log)
    rel = diff / max(abs(best_log), 1e-9)
    print(f"[verify] log の best          : {best_log:10.2f}")
    print(f"[verify] 差                   : {diff:10.2f}  （相対 {rel*100:.1f} %）")
    print(f"\n[verify] {'✅ 再現した' if rel < 0.35 else '⚠️ 再現していない — 手で確かめること'}")
else:
    print("[verify] ⚠️ log から best を読めなかった")

sys.stdout.flush()
os._exit(0)   # Choreonoid の Qt ループが残るため（Bug 26 と同型）
