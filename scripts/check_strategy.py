#!/usr/bin/env python3
"""
戦術退化チェッカー: pusher タスクの Reward Hacking を検出する。

速度報酬・目標座標報酬どちらにも使用可能。
チェック内容は報酬関数と無関係な物理・幾何の異常。

使い方:
  EVAL_RESTORE_DIR=single_run/rrbot_arm_cnoid_v5 \
  USE_CHOREONOID=1 /choreonoid_ws/install/bin/choreonoid --no-window \
      --python scripts/check_strategy.py

または:
  scripts/run_cnoid_train.sh の代わりに環境変数を設定して実行。

チェック項目:
  [1] penetration チェック
      joints=0 の初期姿勢で前腕セグメント（elbow→tip）がキューブ内部を
      通過していないか。カプセル半径 0.04m 分だけ AABB を拡張して判定。
  [2] cube-at-target チェック
      キューブ初期位置が target_x に既に近すぎないか（目標座標報酬のみ有効）。
  [3] 早期キューブ移動チェック
      ゼロアクションで最初の 3 ステップにキューブが動いているか。
      動いていれば impulse が発生している証拠。
  [4] 腕の成長方向チェック
      bone_offset が ±45° 方向（v2 退化方向）に伸びていないか。

出力例:
  [CHECK] penetration  : PASS  (gap=0.85m)
  [CHECK] cube@target  : PASS  (dist=0.50m > threshold=0.10m)
  [CHECK] early-motion : PASS  (Δcube_x=0.000m in 3 steps)
  [CHECK] arm-direction: WARN  body_1 angle=-46.1° (suspicious ±45° diagonal)
  [VERDICT] WARN — 要注意の傾向あり、動画確認推奨
"""

import os
import sys
sys.path.append(os.getcwd())
os.environ['USE_CHOREONOID'] = '1'

import math
import numpy as np
import yaml
from omegaconf import OmegaConf

from khrylib.utils import *
from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent
from design_opt.utils.tools import set_global_seed

# ── 設定 ──────────────────────────────────────────────────────────────────
CAPSULE_RADIUS   = 0.04   # 腕ジオムのカプセル半径 [m]
CUBE_HALF_SIZE   = 0.15   # キューブのハーフサイズ [m]（rrbot_arm.xml）
CUBE_BODY_X      = 1.0    # XML の cube body pos x [m]
EARLY_MOTION_STEPS    = 3      # 初期キューブ移動チェックのステップ数
EARLY_MOTION_THRESH   = 0.005  # [m] これ以上動いたら impulse 疑惑
TARGET_DIST_THRESH    = 0.10   # [m] キューブが target_x にこれより近ければ退化疑惑
ANGLE_DIAGONAL_THRESH = 30.0   # [°] ±45° からこの範囲内なら suspicious

project_path = os.getcwd()

class args:
    restore_dir = os.environ.get('EVAL_RESTORE_DIR')
    epoch       = os.environ.get('EVAL_EPOCH', 'best')

if not args.restore_dir:
    print("Error: EVAL_RESTORE_DIR 環境変数が必要です。")
    sys.exit(1)

# ── エージェント・環境ロード ──────────────────────────────────────────────
train_config_path = os.path.join(project_path, args.restore_dir, ".hydra", "config.yaml")
FLAGS = OmegaConf.create(yaml.safe_load(open(train_config_path)))
cfg   = Config(FLAGS, project_path, args.restore_dir)
cfg.restore_dir = args.restore_dir

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cpu')
set_global_seed(cfg.seed)

epoch = int(args.epoch) if isinstance(args.epoch, str) and args.epoch.isnumeric() else args.epoch
print(f"[check] restore_dir={args.restore_dir}  epoch={epoch}")

agent = BodyGenAgent(cfg=cfg, dtype=dtype, device=device,
                     seed=cfg.seed, num_threads=1, training=False, checkpoint=epoch)
env = agent.env

# ── ユーティリティ ────────────────────────────────────────────────────────
def get_body_physics(robot):
    data = {}
    for body in robot.bodies:
        entry = {}
        if hasattr(body, 'bone_offset') and body.bone_offset is not None:
            bo = np.asarray(body.bone_offset, dtype=float).ravel()
            entry['bone_offset'] = bo.copy()
            entry['bone_length'] = float(np.linalg.norm(bo))
        data[body.name] = entry
    return data


def build_rest_arm(physics, base_world):
    """joints=0 のときの腕の 4 点を bone_offset 連鎖で計算する。"""
    base = np.array(base_world, dtype=float)
    bo0  = physics.get('0',  {}).get('bone_offset', np.zeros(3))
    bo1  = physics.get('1',  {}).get('bone_offset', np.array([0.30, 0., 0.]))
    bo11 = physics.get('11', {}).get('bone_offset', np.array([0.25, 0., 0.]))
    shoulder = base     + bo0
    elbow    = shoulder + bo1
    tip      = elbow    + bo11
    return {'base': base, 'shoulder': shoulder, 'elbow': elbow, 'tip': tip,
            'bo0': bo0, 'bo1': bo1, 'bo11': bo11}


def segment_aabb_2d(p, q, xmin, xmax, ymin, ymax):
    """線分 p→q が 2D AABB と交差するか（Liang-Barsky）。"""
    dx, dy = q[0] - p[0], q[1] - p[1]
    t_min, t_max = 0.0, 1.0
    for d, pv, lo, hi in [(dx, p[0], xmin, xmax), (dy, p[1], ymin, ymax)]:
        if abs(d) < 1e-9:
            if pv < lo or pv > hi:
                return False, None, None
        else:
            t1, t2 = (lo - pv) / d, (hi - pv) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False, None, None
    entry = p + t_min * (q - p)
    return True, t_min, entry


def bone_angle_deg(bo):
    """XY 平面での bone_offset の角度 [°]。"""
    return math.degrees(math.atan2(bo[1], bo[0]))


def is_diagonal(angle_deg, threshold=ANGLE_DIAGONAL_THRESH):
    """±45° または ±135° に近いか（v2 の退化方向）。"""
    a = angle_deg % 360
    for center in [45, 135, 225, 315]:
        if abs((a - center + 180) % 360 - 180) < threshold:
            return True
    return False


# ── エピソード実行（形態変換フェーズを通過）──────────────────────────────
print("[check] エピソード実行中（形態変換フェーズ通過）...")
state = env.reset()
done = False
step = 0
after_physics = None
cube_x_history = []   # 実行フェーズ最初の数ステップの cube x を記録

while not done and step < 300:
    from design_opt.agents.genesis_agent import tensorfy
    state_var = tensorfy([state], dtype, device)
    with torch.no_grad():
        action = agent.policy_net.select_action(state_var, mean_action=True)[0][0]
    action = action.numpy().astype(np.float64)

    # 早期キューブ移動チェック用: 実行フェーズ開始後のみゼロアクションで観察
    # → ここでは通常アクションで形態変換フェーズを通過し、
    #    実行フェーズに入った最初の数ステップの cube 位置を記録する
    next_state, reward, terminated, truncated, info = env.step(action)
    done  = terminated or truncated
    stage = info.get('stage', '')

    if stage not in ('skeleton_transform', 'attribute_transform'):
        if after_physics is None:
            after_physics = get_body_physics(env.robot)
            shoulder_world = np.array(env._world.body_items['robot'].body.rootLink.p)
            arm = build_rest_arm(after_physics, shoulder_world)
        if len(cube_x_history) < EARLY_MOTION_STEPS + 1:
            cube_pos = env.get_body_com('cube').copy() if 'cube' in env._body_names else None
            if cube_pos is not None:
                cube_x_history.append(cube_pos[0])

    state = next_state
    step += 1

if after_physics is None:
    print("[check] ERROR: 実行フェーズに到達できませんでした。")
    sys.exit(1)

shoulder_world = np.array(env._world.body_items['robot'].body.rootLink.p)
arm = build_rest_arm(after_physics, shoulder_world)

# cube の初期 x（env_specs の offset 込み）
cube_x_offset = cfg.env_specs.get('cube_x_offset', 0.0)
cube_init_x = CUBE_BODY_X + cube_x_offset   # noise なし（eval では add_noise=False）
cube_init_y = 0.0

# target_x（目標座標報酬のみ存在）
use_target = cfg.reward_specs.get('use_target_reward', False)
target_x   = cfg.reward_specs.get('target_x', 1.5) if use_target else None

print()
print("=" * 60)
print(f"  RUN     : {args.restore_dir}")
print(f"  EPOCH   : {epoch}")
print(f"  REWARD  : {'目標座標 (target_x=' + str(target_x) + ')' if use_target else '速度ベース'}")
print(f"  cube 初期: x={cube_init_x:.3f}m (offset={cube_x_offset})")
print()
print("  腕の形態（joints=0 の初期姿勢）:")
print(f"    base     : ({arm['base'][0]:.3f}, {arm['base'][1]:.3f})")
print(f"    shoulder : ({arm['shoulder'][0]:.3f}, {arm['shoulder'][1]:.3f})")
print(f"    elbow    : ({arm['elbow'][0]:.3f}, {arm['elbow'][1]:.3f})")
print(f"    tip      : ({arm['tip'][0]:.3f}, {arm['tip'][1]:.3f})")
print("=" * 60)

results = {}   # check名 → ('PASS'|'WARN'|'FAIL', message)

# ─────────────────────────────────────────────────────────────────────────
# [1] penetration チェック
# ─────────────────────────────────────────────────────────────────────────
elbow = arm['elbow'][:2]
tip   = arm['tip'][:2]

r = CAPSULE_RADIUS
xmin = cube_init_x - CUBE_HALF_SIZE - r
xmax = cube_init_x + CUBE_HALF_SIZE + r
ymin = cube_init_y - CUBE_HALF_SIZE - r
ymax = cube_init_y + CUBE_HALF_SIZE + r

hit, t_entry, entry_pt = segment_aabb_2d(elbow, tip, xmin, xmax, ymin, ymax)

if hit:
    results['penetration'] = (
        'FAIL',
        f"前腕が cube に侵入！ t={t_entry:.3f} 地点 ({entry_pt[0]:.3f}, {entry_pt[1]:.3f}) "
        f"→ エピソード開始時に impulse 発生"
    )
else:
    # tip から cube 左面までの距離
    gap = xmin - tip[0]   # tip が cube より左にいる場合は正
    if gap < 0:
        results['penetration'] = (
            'WARN',
            f"tip が cube の x 範囲内にある (tip_x={tip[0]:.3f}, cube_xmin={xmin:.3f})"
            f" — y 方向はセーフだが要確認"
        )
    else:
        results['penetration'] = ('PASS', f"gap={gap:.3f}m（tip と cube 左面の距離）")

# ─────────────────────────────────────────────────────────────────────────
# [2] cube-at-target チェック（目標座標報酬のみ）
# ─────────────────────────────────────────────────────────────────────────
if use_target:
    dist_to_target = abs(cube_init_x - target_x)
    if dist_to_target < TARGET_DIST_THRESH:
        results['cube@target'] = (
            'FAIL',
            f"cube 初期位置 x={cube_init_x:.3f} が target_x={target_x:.3f} に近すぎる "
            f"(dist={dist_to_target:.3f}m < {TARGET_DIST_THRESH}m) → 何もしなくても高報酬"
        )
    elif dist_to_target < 0.30:
        results['cube@target'] = (
            'WARN',
            f"cube 初期 x={cube_init_x:.3f} と target_x={target_x:.3f} の距離={dist_to_target:.3f}m — やや近い"
        )
    else:
        results['cube@target'] = ('PASS', f"dist={dist_to_target:.3f}m > {TARGET_DIST_THRESH}m")
else:
    results['cube@target'] = ('SKIP', "速度報酬のため非適用")

# ─────────────────────────────────────────────────────────────────────────
# [3] 早期キューブ移動チェック
# ─────────────────────────────────────────────────────────────────────────
if len(cube_x_history) >= 2:
    delta_x = cube_x_history[-1] - cube_x_history[0]
    detail = f"Δcube_x={delta_x:+.4f}m in {len(cube_x_history)-1} steps "
    detail += f"(x: {cube_x_history[0]:.3f} → {cube_x_history[-1]:.3f})"
    if abs(delta_x) > EARLY_MOTION_THRESH * 5:
        results['early-motion'] = (
            'FAIL',
            detail + " — 開始直後の大きなキューブ移動（impulse 強疑惑）"
        )
    elif abs(delta_x) > EARLY_MOTION_THRESH:
        results['early-motion'] = ('WARN', detail + " — 微小な移動あり、要確認")
    else:
        results['early-motion'] = ('PASS', detail)
else:
    results['early-motion'] = ('SKIP', "cube_x_history が取得できませんでした")

# ─────────────────────────────────────────────────────────────────────────
# [4] 腕の成長方向チェック
# ─────────────────────────────────────────────────────────────────────────
angle_msgs = []
suspicious = False
for bn, bo_key in [('1', 'bo1'), ('11', 'bo11')]:
    bo = arm[bo_key]
    length = float(np.linalg.norm(bo))
    if length < 0.05:
        angle_msgs.append(f"    body_{bn}: 長さ={length:.3f}m（短すぎて方向不明）")
        continue
    angle = bone_angle_deg(bo)
    diag  = is_diagonal(angle, ANGLE_DIAGONAL_THRESH)
    flag  = " ← suspicious (±45° diagonal)" if diag else ""
    angle_msgs.append(f"    body_{bn}: 長さ={length:.3f}m  角度={angle:.1f}°{flag}")
    if diag:
        suspicious = True

direction_detail = "\n" + "\n".join(angle_msgs)
if suspicious:
    results['arm-direction'] = (
        'WARN',
        direction_detail + "\n    v2 の退化方向（±45°）と類似。腕が伸びた後に要再確認"
    )
else:
    results['arm-direction'] = ('PASS', direction_detail)

# ─────────────────────────────────────────────────────────────────────────
# 結果表示
# ─────────────────────────────────────────────────────────────────────────
ICONS = {'PASS': '✅', 'WARN': '⚠️ ', 'FAIL': '❌', 'SKIP': '⏭️ '}

print()
for name, (status, msg) in results.items():
    icon = ICONS.get(status, '?')
    print(f"[CHECK] {name:<14}: {icon} {status}  {msg}")

# 総合判定
statuses = [s for s, _ in results.values()]
if 'FAIL' in statuses:
    verdict = 'FAIL'
    verdict_msg = '退化戦略の疑いが強い。動画で確認し、run を停止・修正を検討してください。'
elif 'WARN' in statuses:
    verdict = 'WARN'
    verdict_msg = '要注意の傾向あり。epoch 30〜50 で再チェック推奨。'
else:
    verdict = 'PASS'
    verdict_msg = '現時点では退化戦略の兆候なし。'

print()
print(f"[VERDICT] {ICONS[verdict]} {verdict} — {verdict_msg}")
print()
