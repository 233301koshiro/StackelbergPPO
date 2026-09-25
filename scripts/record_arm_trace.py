#!/usr/bin/env python3
"""学習済み方策を1エピソード実行し、**実物メッシュで動画を作るための軌跡**を記録する。

**なぜ要るか**: 発表で「描いた絵がそのまま動く」ことを見せたい。
`diagnose_morphology.py` の第3層は数値でしか結果を返さず、
`save_morphology_urdf.py` は静止した形態しか出さない。
**動きと形態の両方**を1ファイルに落とす口が無かった。

出力（`<restore_dir>/trace/arm_trace.npz`）:
  body_names   リンク名（根元から順）
  xpos         (T, L, 3)  各リンクのワールド位置
  xmat         (T, L, 3, 3) 各リンクのワールド回転
  bone_offset  (L, 3)  **最適化後**のボーン（設計図の値ではない）
  cube         (T, 3)  Pusher の対象物（無い場合は形状 (0,)）
  target       (3,)    Reach の目標

⚠️ **最適化後の形態を使うこと。** co-design はリンク長も太さも変えるので、
XML の設計値でメッシュを並べると**学習結果と違う絵**になる。

使い方:
  EVAL_RESTORE_DIR=single_run/e2e_a1v_reach EVAL_CHECKPOINT=best \\
  USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \\
      --no-window --python scripts/record_arm_trace.py
"""
import os, sys
sys.path.append(os.getcwd())
os.environ['USE_CHOREONOID'] = '1'

import numpy as np
import yaml
import torch
from omegaconf import OmegaConf

from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent, tensorfy
from design_opt.utils.tools import set_global_seed

project_path = os.getcwd()
restore_dir = os.environ['EVAL_RESTORE_DIR']
checkpoint = os.environ.get('EVAL_CHECKPOINT', 'best')
# ⚠️⚠️ 既定を 400 → 1200 にした（2026-09-20、9-137）。
#   ⛔ **旧既定 400 は 1 エピソード（実測 985〜991 step）の 40 % しか記録しない。**
#   ⭐ **打ち切られたことは出力のどこにも出ず、「全部見た」つもりで比較してしまう。**
#   ⚠️ 9-136 で実際に踏んだ: 最初の 400 step だけで「対象の移動量は +2 %」と書いた。
#   ⭐ `done` で自然に終わるので、大きくしても余計な時間はかからない。
max_steps = int(os.environ.get('TRACE_STEPS', '1200'))

FLAGS = OmegaConf.create(yaml.safe_load(open(f'{restore_dir}/.hydra/config.yaml')))
d = OmegaConf.to_container(FLAGS, resolve=True)
d.pop('restore_dir', None)
FLAGS = OmegaConf.create(d)
cfg = Config(FLAGS, project_path, restore_dir)
cfg.restore_dir = restore_dir
cfg.control_prior = False
cfg.morph_prior = False
torch.set_default_dtype(torch.float64)
set_global_seed(cfg.seed)

# ⚠️ int で渡さないと models/epoch_0010.p ではなく models/10.p を探しに行く（9-66）
ckpt_arg = int(checkpoint) if checkpoint != 'best' else 'best'
agent = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                     seed=cfg.seed, num_threads=1, training=False, checkpoint=ckpt_arg)
env = agent.env

# ⚠️ 9-96: Target-Pusher は目標を reward_specs に持つ（env_specs ではない）。
#   旧実装は env_specs しか見ず、TP の trace に既定の [0.8,0,0.15] を書いていた。
#   土日の再分析で「対象が目標から離れていく」ように見えた原因。
_rs = cfg.reward_specs if hasattr(cfg, 'reward_specs') else {}
_src = _rs if _rs.get('use_target_reward', False) else cfg.env_specs
target = np.array([_src.get('target_x', 0.8),
                   _src.get('target_y', 0.0),
                   _src.get('target_z', 0.15)])

# ⭐⭐ Bug 47 / 9-142: 「読み込み直後の 1 話目」は当てにならない（9-66）。
#   ⛔ 旧実装は 1 話しか回さず、その 1 話がまさに 1 話目だった。
#     対照 Pusher で 1 話目 15.11 m に対し 3〜6 話は 6.54〜15.97 m（幅 144 %）。
#   ⚠️ **軌跡を取り直しても直らない**（毎回 1 話目をやり直すだけ）。
#   ⭐ **1 回の読み込みのまま N 話回し、頭の SKIP 話を捨てる。**
N_EP = int(os.environ.get('TRACE_EPISODES', '5'))
SKIP = int(os.environ.get('TRACE_SKIP', '2'))

names = None
bone = None
geom_size = None
eps = []          # 採用した話ごとの (xpos, xmat, cube)

for k in range(N_EP):
    state = env.reset()
    xpos, xmat, cube = [], [], []
    for _ in range(cfg.skel_transform_nsteps + 2 + max_steps):
        in_exec = env.stage == 'execution'
        sv = tensorfy([state])
        if agent.obs_norm is not None:
            sv = agent.normalize_observation(sv)
        with torch.no_grad():
            action = agent.policy_net.select_action(sv, mean_action=True).numpy().astype(np.float64)
        state, reward, done, _, info = env.step(action)

        if in_exec:
            if names is None:
                # 設計フェーズが終わった時点の形態を確定させる（ここから先は変わらない）
                names = [b.name for b in env.robot.bodies]
                bone = np.array([np.asarray(getattr(b, 'bone_offset', [0, 0, 0]), dtype=float)
                                 for b in env.robot.bodies])
                # ⭐ リンク半径（カプセルの size[0]）。**貫通判定にはこれが要る**
                #   （`check_cube_penetration.py`。軸だけでは食い込みを過小評価する）。
                geom_size = np.array([
                    float(np.asarray(b.geoms[0].size, dtype=float).flatten()[0])
                    if getattr(b, 'geoms', None) else np.nan
                    for b in env.robot.bodies])
            xpos.append([np.asarray(env._body_xpos[n], dtype=float) for n in names])
            xmat.append([np.asarray(env._body_xmat[n], dtype=float).reshape(3, 3) for n in names])
            # Pusher の対象物。⚠️ **`_body_xpos` には cube が入っていない**（腕の body だけ）。
            # 静止した初期値を読み続けて「動かない cube」を記録する事故を起こしたので、
            # `probe_cube_trace.py` と同じ `get_body_com()` を使う。
            try:
                cube.append(np.asarray(env.get_body_com('cube'), dtype=float))
            except Exception:
                cube.append(np.zeros(3))
        if done:
            break
    if k >= SKIP:
        eps.append((np.array(xpos), np.array(xmat), np.array(cube)))
    print(f'[trace] 話 {k+1}/{N_EP}  step={len(xpos)}  '
          f'{"⭐ 採用" if k >= SKIP else "⚠️ 捨てる（頭の話は当てにならない。Bug 47）"}', flush=True)

assert eps, 'TRACE_EPISODES が TRACE_SKIP 以下です'

# ⭐ 下流（check_before_conclusion・check_obstacle_clearance）は単一話の配列を読むので、
#   **採用した話のうち中央値のもの**を従来のキーに入れる。1 話目は入れない。
def _score(e):
    c = e[2]
    return float(c[-1, 0] - c[0, 0]) if len(c) else 0.0
order = sorted(range(len(eps)), key=lambda i: _score(eps[i]))
rep = eps[order[len(order) // 2]]
xpos, xmat, cube = rep

out_dir = os.path.join(restore_dir, 'trace')
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'arm_trace.npz')
np.savez_compressed(out,
                    body_names=np.array(names),
                    xpos=xpos, xmat=xmat,
                    bone_offset=bone, geom_size=geom_size, cube=cube, target=target,
                    # ⭐ 採用した全話の要約。量を比べるときはこちらを使う（Bug 47）
                    ep_cube_dx=np.array([_score(e) for e in eps]),
                    ep_count=len(eps), ep_skipped=SKIP)

print(f'[trace] {restore_dir} ckpt={checkpoint}')
print(f'[trace] リンク: {names}')
print(f'[trace] 最適化後のボーン長: ' +
      ' / '.join(f'{np.linalg.norm(b):.4f}' for b in bone) +
      f'  合計 {sum(np.linalg.norm(b) for b in bone):.4f} m')
_dx = np.array([_score(e) for e in eps])
print(f'[trace] 採用 {len(eps)} 話（頭 {SKIP} 話を捨てた）  保存したのは中央値の話  実行ステップ {len(xpos)}')
print(f'[trace] 話ごとの cube 移動: {np.round(_dx, 3).tolist()}  '
      f'→ 幅 {(_dx.max()-_dx.min()):.3f} m')
print(f'[trace] → {out}')
# ⭐ 打ち切りを黙って通さない（9-137）
if len(xpos) >= max_steps:
    print(f'⛔ **{max_steps} step で打ち切られた。エピソードは終わっていない。**')
    print(f'⚠️ **この軌跡で「最終位置」「総移動量」を語ってはいけない。**'
          f' TRACE_STEPS を増やして取り直すこと（9-136 で実際に誤った）')

# ⚠️ **os._exit で落とす。** Choreonoid（Qt）のイベントループが残り、
#   出力を書き終えた後もプロセスが生き続ける。2026-09-10 の確認で
#   **9 月 4 日から 5 日 18 時間、孤児プロセスが 414 MB 抱えたまま残っていた**
#   （Bug 26 と同型。1 本あたり約 800 MB）。
#   CLAUDE.md §5-2 ⑤-3 で「結果を論じる前に再生する」を規律にしたので、
#   **これを直さないと再生のたびに漏れる。**
#   `eval_reach_hover.py` は既に同じ手当てをしている。
sys.stdout.flush()
os._exit(0)
