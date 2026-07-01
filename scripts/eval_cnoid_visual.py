#!/usr/bin/env python3
"""
可視化スクリプト: 形態変化動画 + 実行動画 の2本を出力する。

使い方:
  EVAL_RESTORE_DIR=single_run/rrbot_arm_cnoid_v2 \
  USE_CHOREONOID=1 /choreonoid_ws/install/bin/choreonoid --no-window --python \
      scripts/eval_cnoid_visual.py

出力:
  {restore_dir}/videos/eval_morphology.mp4  ... 変化前→変化後の形態比較
  {restore_dir}/videos/eval_execution.mp4   ... 実行フェーズのみ

環境変数:
  EVAL_RESTORE_DIR    : 必須
  EVAL_EPOCH          : ロードするチェックポイント (default: best)
  EVAL_OUTPUT_MORPH   : 形態動画パス (default: {restore_dir}/videos/eval_morphology.mp4)
  EVAL_OUTPUT_EXEC    : 実行動画パス (default: {restore_dir}/videos/eval_execution.mp4)
  EVAL_FPS            : fps (default: 20)
  EVAL_MAX_EXEC_STEPS : 実行フェーズの最大ステップ数 (default: 200)
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
import matplotlib.patches as mpatches
import matplotlib.animation as animation

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

class args:
    restore_dir    = os.environ.get('EVAL_RESTORE_DIR')
    epoch          = os.environ.get('EVAL_EPOCH', 'best')
    output_morph   = os.environ.get('EVAL_OUTPUT_MORPH', None)
    output_exec    = os.environ.get('EVAL_OUTPUT_EXEC', None)
    fps            = int(os.environ.get('EVAL_FPS', '20'))
    max_exec_steps = int(os.environ.get('EVAL_MAX_EXEC_STEPS', '200'))

if not args.restore_dir:
    print("Error: EVAL_RESTORE_DIR is required.")
    sys.exit(1)

video_dir = os.path.join(args.restore_dir, 'videos')
os.makedirs(video_dir, exist_ok=True)
out_morph = args.output_morph or os.path.join(video_dir, 'eval_morphology.mp4')
out_exec  = args.output_exec  or os.path.join(video_dir, 'eval_execution.mp4')

# ── 設定・エージェント読み込み ────────────────────────────────────────────
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


# ── ボディ物理パラメータ読み取り ─────────────────────────────────────────
def get_body_physics(robot):
    """Robot.bodies から bone_offset / ext_start を直接読む（非正規化済み）。"""
    data = {}
    for body in robot.bodies:
        entry = {}
        if hasattr(body, 'bone_offset') and body.bone_offset is not None:
            bo = np.asarray(body.bone_offset, dtype=float).ravel()
            entry['bone_offset'] = bo.copy()
            entry['bone_length'] = float(np.linalg.norm(bo))
        for geom in body.geoms:
            if hasattr(geom, 'ext_start') and geom.ext_start is not None:
                es = np.asarray(geom.ext_start, dtype=float).ravel()
                entry['ext_start'] = float(es[0])
                break
        data[body.name] = entry
    return data


def build_rest_arm(physics, base_world):
    """
    全関節 0° のときの 4 点を bone_offset の連鎖から計算する。
    rrbot_arm ではジョイント軸が Z（上方向）なので bone_offset [x,y,0] が
    X-Y 水平面内の方向を表す。

    返り値の 4 点:
      base          = root_link.p (固定ベース)
      shoulder_jnt  = base  + body_0.bone_offset (link "1" のジョイント位置)
      elbow_jnt     = shoulder_jnt + body_1.bone_offset  (link "11" のジョイント位置)
      tip           = elbow_jnt   + body_11.bone_offset
    """
    base = np.array(base_world, dtype=float)
    bo0  = physics.get('0',  {}).get('bone_offset', np.zeros(3))
    bo1  = physics.get('1',  {}).get('bone_offset', np.array([0.30, 0.0, 0.0]))
    bo11 = physics.get('11', {}).get('bone_offset', np.array([0.25, 0.0, 0.0]))
    shoulder_jnt = base         + bo0
    elbow_jnt    = shoulder_jnt + bo1
    tip          = elbow_jnt    + bo11
    return {
        'base': base,
        'shoulder': shoulder_jnt,   # ← 既存コードとの互換性のため 'shoulder' キーを残す
        'elbow': elbow_jnt,
        'tip': tip,
        'l0': float(np.linalg.norm(bo0)),
        'l1': float(np.linalg.norm(bo1)),
        'l11': float(np.linalg.norm(bo11)),
    }


# ── Choreonoid からリンク位置取得 ─────────────────────────────────────────
def collect_arm_skeleton(env):
    """link.p + link.R から shoulder / elbow / tip を返す（実行中用）。"""
    cnoid_body = env._world.body_items['robot'].body
    root_link  = cnoid_body.rootLink
    link_11    = cnoid_body.link('11')
    if link_11 is None:
        return None

    shoulder = np.array(root_link.p)
    elbow    = np.array(link_11.p)

    # bone_offset から最新のリンク長を取得（set_design_params 後に更新済み）
    physics = get_body_physics(env.robot)
    bo11 = physics.get('11', {}).get('bone_offset', np.array([0.25, 0.0, 0.0]))
    R11  = np.array(link_11.R)
    tip  = elbow + R11 @ bo11

    return {'shoulder': shoulder, 'elbow': elbow, 'tip': tip}


# ── エピソード実行 ───────────────────────────────────────────────────────
print("[visual] Running episode to collect frames...")

# reset 直後（変換前）の形態を取得
state = env.reset()
before_physics  = get_body_physics(env.robot)
shoulder_world  = np.array(env._world.body_items['robot'].body.rootLink.p)
before_arm_rest = build_rest_arm(before_physics, shoulder_world)

morph_frames = []
exec_frames  = []
after_physics  = None
after_arm_rest = None
exec_steps = 0
done = False
prev_stage = 'skeleton_transform'

while not done and exec_steps < args.max_exec_steps:
    from design_opt.agents.genesis_agent import tensorfy
    state_var = tensorfy([state])
    if agent.obs_norm is not None:
        state_var = agent.normalize_observation(state_var)
    with torch.no_grad():
        action = agent.policy_net.select_action(state_var, mean_action=True)
        action = action.numpy().astype(np.float64)

    next_state, reward, terminated, truncated, info = env.step(action)
    done  = terminated or truncated
    stage = info.get('stage', '')

    arm = collect_arm_skeleton(env)

    cube_pos = None
    if 'cube' in env._body_names:
        cube_pos = env.get_body_com('cube').copy()

    if stage in ('skeleton_transform', 'attribute_transform'):
        morph_frames.append({'arm': arm, 'stage': stage})
    else:
        # 実行フェーズに入った瞬間に「変換後」形態を記録
        if after_physics is None:
            after_physics  = get_body_physics(env.robot)
            after_arm_rest = build_rest_arm(after_physics, shoulder_world)
            print(f"[visual] Morphology captured at execution start.")
            for bn in ['0', '1', '11']:
                bo = after_physics.get(bn, {}).get('bone_offset', None)
                if bo is not None:
                    print(f"[visual] body_{bn} bone_offset: x={bo[0]:.4f}, y={bo[1]:.4f}")
            for k in ('base','shoulder','elbow','tip'):
                p = after_arm_rest[k]
                print(f"[visual] rest arm {k}: x={p[0]:.4f}, y={p[1]:.4f}")
        exec_frames.append({
            'arm': arm, 'cube_pos': cube_pos,
            'stage': stage, 'reward': reward,
        })
        exec_steps += 1

    prev_stage = stage
    state = next_state

print(f"[visual] morph_frames={len(morph_frames)}, exec_frames={len(exec_frames)}")

if after_physics is None:
    after_physics  = before_physics
    after_arm_rest = before_arm_rest


# ── 形態変化サマリー計算 ──────────────────────────────────────────────────
def delta_str(val, unit='m', threshold=1e-3):
    if abs(val) < threshold:
        return '±0'
    sign = '+' if val > 0 else ''
    return f'{sign}{val:.3f}{unit}'

body_names = ['0', '1', '11']
param_rows = []
for bn in body_names:
    bp = before_physics.get(bn, {})
    ap = after_physics.get(bn, {})
    bl = bp.get('bone_length', None)
    al = ap.get('bone_length', None)
    be = bp.get('ext_start', None)
    ae = ap.get('ext_start', None)
    bbo = bp.get('bone_offset', None)
    abo = ap.get('bone_offset', None)
    param_rows.append({
        'name': f'body_{bn}',
        'bl_before': bl, 'bl_after': al,
        'be_before': be, 'be_after': ae,
        'bbo': bbo, 'abo': abo,
    })

print("[visual] Morphology change summary:")
for r in param_rows:
    bn = r['name']
    if r['bl_before'] is not None and r['bl_after'] is not None:
        d = r['bl_after'] - r['bl_before']
        print(f"  {bn}: bone_length {r['bl_before']:.3f} → {r['bl_after']:.3f}  Δ={delta_str(d)}")
    if r['be_before'] is not None and r['be_after'] is not None:
        d = r['be_after'] - r['be_before']
        print(f"  {bn}: ext_start   {r['be_before']:.3f} → {r['be_after']:.3f}  Δ={delta_str(d)}")


# ── 形態変化動画レンダリング ─────────────────────────────────────────────
HOLD = args.fps * 3   # 変化前/後を3秒ずつ表示
TRANS = args.fps // 2 # 0.5秒でグラデーション遷移

THRESHOLD = 1e-3  # 変化ありとみなすΔ (m)

def link_color(before_len, after_len, is_after_frame):
    """before/after の変化量に応じて色を返す。"""
    if before_len is None or after_len is None:
        return 'steelblue'
    delta = after_len - before_len
    if not is_after_frame:
        return '#888888'   # before: グレー
    if delta > THRESHOLD:
        return '#FF6B2B'   # 伸びた: オレンジ
    if delta < -THRESHOLD:
        return '#00BFFF'   # 縮んだ: シアン
    return '#4CAF50'       # 変化なし: 緑


def draw_arm_2d(ax, arm_rest, is_after, bl0_b, bl0_a, bl1_b, bl1_a, bl11_b, bl11_a):
    """X-Y 俯瞰（ジョイント軸が Z なので回転面が X-Y）で 4 点の腕を描く。"""
    b  = arm_rest['base']       # 固定ベース
    s  = arm_rest['shoulder']   # link "1" ジョイント（shoulder joint）
    e  = arm_rest['elbow']      # link "11" ジョイント（elbow joint）
    t  = arm_rest['tip']        # 先端

    col0  = link_color(bl0_b,  bl0_a,  is_after)   # ベース→shoulder
    col1  = link_color(bl1_b,  bl1_a,  is_after)   # shoulder→elbow
    col11 = link_color(bl11_b, bl11_a, is_after)   # elbow→tip

    # セグメント描画（X-Y平面: X=前後, Y=左右）
    ax.plot([b[0], s[0]], [b[1], s[1]], '-', color=col0,  lw=3,
            solid_capstyle='round', ls='--', zorder=3, label='base offset')
    ax.plot([s[0], e[0]], [s[1], e[1]], '-', color=col1,  lw=5,
            solid_capstyle='round', zorder=3, label='link 1')
    ax.plot([e[0], t[0]], [e[1], t[1]], '-', color=col11, lw=4,
            solid_capstyle='round', zorder=3, label='link 11')

    # ジョイント点
    ax.plot(b[0], b[1], 's', color='#444444', ms=10, zorder=5)   # base: 四角
    ax.plot(s[0], s[1], 'o', color='#222222', ms=9,  zorder=5)   # shoulder joint
    ax.plot(e[0], e[1], 'o', color=col1,      ms=8,  zorder=5)   # elbow
    ax.plot(t[0], t[1], '^', color=col11,     ms=7,  zorder=5)   # tip: 三角

    # 座標ラベル（視認性のためオフセット調整）
    pad = max(abs(MORPH_XLIM[1] - MORPH_XLIM[0]),
              abs(MORPH_YLIM[1] - MORPH_YLIM[0])) * 0.07
    ax.annotate(f'base\n({b[0]:.2f},{b[1]:.2f})', xy=(b[0], b[1]),
                xytext=(b[0] - pad, b[1] - pad), fontsize=7, color='#444444',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.7))
    ax.annotate(f'shoulder\n({s[0]:.2f},{s[1]:.2f})', xy=(s[0], s[1]),
                xytext=(s[0] - pad * 0.8, s[1] + pad), fontsize=7, color='#222222',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.7))
    ax.annotate(f'elbow\n({e[0]:.2f},{e[1]:.2f})', xy=(e[0], e[1]),
                xytext=(e[0] + pad * 0.3, e[1] + pad), fontsize=7, color='#333333',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.7))
    ax.annotate(f'tip\n({t[0]:.2f},{t[1]:.2f})', xy=(t[0], t[1]),
                xytext=(t[0] + pad * 0.3, t[1] - pad), fontsize=7, color='#333333',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.7))


def make_morph_figure(arm_rest, is_after, bl1_b, bl1_a, bl11_b, bl11_a, alpha=1.0):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # --- 左パネル: 俯瞰 (X-Y) ---
    ax = axes[0]
    ax.set_facecolor('#F5F5F0')
    draw_arm_2d(ax, arm_rest, is_after,
                bl0_b, bl0_a, bl1_b, bl1_a, bl11_b, bl11_a)
    ax.set_xlim(*MORPH_XLIM)
    ax.set_ylim(*MORPH_YLIM)
    ax.set_aspect('equal')
    ax.set_xlabel('X (push direction) [m]', fontsize=9)
    ax.set_ylabel('Y (lateral) [m]', fontsize=9)
    ax.grid(True, alpha=0.3)
    # 目標キューブ位置（参考）
    ax.axvline(x=0.60, color='orange', lw=1, ls='--', alpha=0.6, label='cube x₀=0.60')
    ax.axvline(x=1.50, color='red',    lw=1, ls=':', alpha=0.5, label='target x=1.50')
    ax.legend(fontsize=7.5, loc='upper left')
    view_label = 'After (Learned Morphology)' if is_after else 'Before (Default Morphology)'
    ax.set_title(f'Top View (X-Y)  [{view_label}]  [epoch={epoch}]', fontsize=11, fontweight='bold')

    # --- 右パネル: パラメータ表 ---
    ax2 = axes[1]
    ax2.set_facecolor('#FAFAF8')
    ax2.axis('off')

    l1_b  = bl1_b  if bl1_b  is not None else 0.0
    l1_a  = bl1_a  if bl1_a  is not None else 0.0
    l11_b = bl11_b if bl11_b is not None else 0.0
    l11_a = bl11_a if bl11_a is not None else 0.0

    dl1   = l1_a  - l1_b
    dl11  = l11_a - l11_b
    reach_b = l1_b + l11_b
    reach_a = l1_a + l11_a
    d_reach = reach_a - reach_b

    if is_after:
        title_color = '#CC3300'
        lines = [
            ('Parameter', 'Before', 'After', 'Delta'),
            ('─' * 12, '─' * 8, '─' * 8, '─' * 8),
            ('Link 1 length',  f'{l1_b:.3f} m',  f'{l1_a:.3f} m',  delta_str(dl1)),
            ('Link 11 length', f'{l11_b:.3f} m', f'{l11_a:.3f} m', delta_str(dl11)),
            ('Total reach',    f'{reach_b:.3f} m', f'{reach_a:.3f} m', delta_str(d_reach)),
        ]
        col_colors = ['#DDDDDD', '#FFFFFF', '#FFEDD8', '#FFE0E0']
        extra = [
            '',
            ('凡例: ', '■ 伸びた', '■ 縮んだ', '■ 変化なし / default'),
        ]
    else:
        title_color = '#334455'
        lines = [
            ('Parameter', 'Value', '', ''),
            ('─' * 12, '─' * 8, '', ''),
            ('Link 1 length',  f'{l1_b:.3f} m', '', ''),
            ('Link 11 length', f'{l11_b:.3f} m', '', ''),
            ('Total reach',    f'{reach_b:.3f} m', '', ''),
        ]
        col_colors = ['#DDDDDD', '#F0F0F0', '#FFFFFF', '#FFFFFF']
        extra = []

    y0 = 0.92
    dy = 0.10
    ax2.text(0.5, y0 + 0.05, view_label, transform=ax2.transAxes,
             fontsize=14, fontweight='bold', ha='center', va='top', color=title_color)

    cols_x = [0.02, 0.35, 0.58, 0.78]
    for i, row in enumerate(lines):
        for j, cell in enumerate(row):
            color = '#333333'
            if i == 0:
                color = '#111111'
                fw = 'bold'
            else:
                fw = 'normal'
            if is_after and i >= 2 and j == 3:  # Δ列
                try:
                    val = float(cell.replace('+', '').replace('m', '').strip())
                    if val > THRESHOLD:
                        color = '#CC4400'
                    elif val < -THRESHOLD:
                        color = '#006699'
                except:
                    pass
            ax2.text(cols_x[j], y0 - i * dy - 0.06, cell, transform=ax2.transAxes,
                     fontsize=10, ha='left', va='top', color=color, fontweight=fw)

    if is_after:
        legend_y = y0 - len(lines) * dy - 0.08
        ax2.add_patch(mpatches.Rectangle((0.04, legend_y - 0.03), 0.12, 0.05,
                                          color='#FF6B2B', transform=ax2.transAxes))
        ax2.text(0.18, legend_y, 'Grew (Delta > 0)', transform=ax2.transAxes,
                 fontsize=9, va='center', color='#CC4400')
        ax2.add_patch(mpatches.Rectangle((0.04, legend_y - 0.10), 0.12, 0.05,
                                          color='#00BFFF', transform=ax2.transAxes))
        ax2.text(0.18, legend_y - 0.07, 'Shrank (Delta < 0)', transform=ax2.transAxes,
                 fontsize=9, va='center', color='#006699')
        ax2.add_patch(mpatches.Rectangle((0.04, legend_y - 0.17), 0.12, 0.05,
                                          color='#4CAF50', transform=ax2.transAxes))
        ax2.text(0.18, legend_y - 0.14, 'No change', transform=ax2.transAxes,
                 fontsize=9, va='center', color='#2E7D32')

    fig.patch.set_alpha(alpha)
    fig.tight_layout(pad=1.5)
    return fig


# モーフィング動画フレームリスト構築
bl0_b  = before_physics.get('0',  {}).get('bone_length', None)
bl0_a  = after_physics.get('0',   {}).get('bone_length', None)
bl1_b  = before_physics.get('1',  {}).get('bone_length', None)
bl1_a  = after_physics.get('1',   {}).get('bone_length', None)
bl11_b = before_physics.get('11', {}).get('bone_length', None)
bl11_a = after_physics.get('11',  {}).get('bone_length', None)

# 軸範囲: before/after/transition の全頂点を包むように動的決定
def _arm_pts(arm):
    return [arm['base'], arm['shoulder'], arm['elbow'], arm['tip']]

all_pts = _arm_pts(before_arm_rest) + _arm_pts(after_arm_rest)
xs = [p[0] for p in all_pts]
ys = [p[1] for p in all_pts]
PAD = 0.15
MORPH_XLIM = (min(xs) - PAD, max(xs) + PAD)
MORPH_YLIM = (min(ys) - PAD, max(ys) + PAD)
# y 軸は対称にして見やすくする
y_half = max(abs(MORPH_YLIM[0]), abs(MORPH_YLIM[1]))
MORPH_YLIM = (-y_half, y_half)
print(f"[visual] Morph axis: x={MORPH_XLIM}, y={MORPH_YLIM}")

print(f"[visual] Rendering morphology video → {out_morph}")

writer_morph = animation.FFMpegWriter(fps=args.fps, bitrate=2000)
fig_m, axes_m = plt.subplots(1, 2, figsize=(13, 6))

def draw_morph_frame(fi):
    for ax in axes_m:
        ax.cla()
    total = HOLD + TRANS + HOLD

    if fi < HOLD:
        # 変化前
        arm = before_arm_rest
        is_after = False
        label = 'Before (Default Morphology)'
        lcolor = '#334455'
    elif fi < HOLD + TRANS:
        # トランジション: before → after を補間
        t = (fi - HOLD) / TRANS
        arm = {}
        for k in ('base', 'shoulder', 'elbow', 'tip'):
            arm[k] = (1 - t) * before_arm_rest[k] + t * after_arm_rest[k]
        arm['l0']  = (1 - t) * before_arm_rest['l0']  + t * after_arm_rest['l0']
        arm['l1']  = (1 - t) * before_arm_rest['l1']  + t * after_arm_rest['l1']
        arm['l11'] = (1 - t) * before_arm_rest['l11'] + t * after_arm_rest['l11']
        is_after = t > 0.5
        label = 'Morphing...'
        lcolor = '#666633'
    else:
        # 変化後
        arm = after_arm_rest
        is_after = True
        label = 'After (Learned Morphology)'
        lcolor = '#CC3300'

    # 左: 俯瞰
    ax = axes_m[0]
    ax.set_facecolor('#F5F5F0')
    draw_arm_2d(ax, arm, is_after,
                bl0_b, bl0_a, bl1_b, bl1_a, bl11_b, bl11_a)
    ax.set_xlim(*MORPH_XLIM)
    ax.set_ylim(*MORPH_YLIM)
    ax.set_aspect('equal')
    ax.set_xlabel('X (push direction) [m]', fontsize=9)
    ax.set_ylabel('Y (lateral) [m]', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0.60, color='orange', lw=1, ls='--', alpha=0.6, label='cube x₀=0.60')
    ax.axvline(x=1.50, color='red',    lw=1, ls=':',  alpha=0.5, label='target x=1.50')
    ax.legend(fontsize=7.5, loc='upper left')
    ax.set_title(f'Top View (X-Y)  [{label}]  frame {fi+1}/{total}',
                 fontsize=11, fontweight='bold', color=lcolor)

    # 右: パラメータ表
    ax2 = axes_m[1]
    ax2.set_facecolor('#FAFAF8')
    ax2.axis('off')

    l1_b   = bl1_b  if bl1_b  is not None else 0.0
    l1_a   = bl1_a  if bl1_a  is not None else l1_b
    l11_b_ = bl11_b if bl11_b is not None else 0.0
    l11_a_ = bl11_a if bl11_a is not None else l11_b_
    dl1    = l1_a  - l1_b
    dl11   = l11_a_ - l11_b_
    reach_b  = l1_b + l11_b_
    reach_a_ = l1_a + l11_a_
    d_reach  = reach_a_ - reach_b

    ax2.text(0.5, 0.97, label, transform=ax2.transAxes,
             fontsize=14, fontweight='bold', ha='center', va='top', color=lcolor)

    rows = [
        ('Parameter',    'Before',           'After',            'Delta'),
        ('Link 1 length',f'{l1_b:.3f} m',   f'{l1_a:.3f} m',   delta_str(dl1)),
        ('Link 11 len',  f'{l11_b_:.3f} m', f'{l11_a_:.3f} m', delta_str(dl11)),
        ('Total reach',  f'{reach_b:.3f} m',f'{reach_a_:.3f} m',delta_str(d_reach)),
    ]
    cx = [0.02, 0.35, 0.58, 0.78]
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            fw = 'bold' if i == 0 else 'normal'
            c  = '#111111' if i == 0 else '#333333'
            if is_after and i >= 1 and j == 3:
                try:
                    v = float(cell.replace('+','').replace('m','').strip())
                    if   v >  THRESHOLD: c = '#CC4400'
                    elif v < -THRESHOLD: c = '#006699'
                except: pass
            ax2.text(cx[j], 0.88 - i * 0.10, cell, transform=ax2.transAxes,
                     fontsize=10, ha='left', va='top', color=c, fontweight=fw)

    if is_after:
        ly = 0.44
        for col, txt, tc in [
            ('#FF6B2B', 'Grew (Delta > 0)',   '#CC4400'),
            ('#00BFFF', 'Shrank (Delta < 0)', '#006699'),
            ('#4CAF50', 'No change',           '#2E7D32'),
        ]:
            ax2.add_patch(mpatches.Rectangle((0.04, ly), 0.10, 0.05,
                                              color=col, transform=ax2.transAxes))
            ax2.text(0.17, ly + 0.025, txt, transform=ax2.transAxes,
                     fontsize=9, va='center', color=tc)
            ly -= 0.09

    fig_m.tight_layout(pad=1.5)


total_morph_frames = HOLD + TRANS + HOLD
ani_m = animation.FuncAnimation(fig_m, draw_morph_frame,
                                  frames=total_morph_frames,
                                  interval=1000 // args.fps, blit=False)
ani_m.save(out_morph, writer=animation.FFMpegWriter(fps=args.fps, bitrate=2000))
plt.close(fig_m)
print(f"[visual] Saved morphology: {out_morph}")


# ── 実行フェーズ動画レンダリング ─────────────────────────────────────────
print(f"[visual] Rendering execution video ({len(exec_frames)} frames) → {out_exec}")

X_MIN, X_MAX = -0.4, 2.0
Y_MIN, Y_MAX = -0.8, 0.8
Z_MIN, Z_MAX =  0.0, 1.2

fig_e = plt.figure(figsize=(10, 7))
ax_e  = fig_e.add_subplot(111, projection='3d')

def draw_exec_frame(i):
    ax_e.cla()
    frame = exec_frames[i]
    arm   = frame['arm']
    cp    = frame['cube_pos']
    stage = frame['stage']

    # 床
    xx, yy = np.meshgrid([X_MIN, X_MAX], [Y_MIN, Y_MAX])
    ax_e.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.12, color='gray', zorder=0)

    # アーム
    if arm is not None:
        s, e, t = arm['shoulder'], arm['elbow'], arm['tip']
        ax_e.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]],
                  'b-', lw=3.5, alpha=0.9, zorder=4)
        ax_e.plot([e[0], t[0]], [e[1], t[1]], [e[2], t[2]],
                  color='cornflowerblue', lw=2.5, alpha=0.9, zorder=4)
        ax_e.scatter(*s, c='navy',      s=120, zorder=5, depthshade=False)
        ax_e.scatter(*e, c='royalblue', s=100, zorder=5, depthshade=False)
        ax_e.scatter(*t, c='steelblue', s=80,  zorder=5, depthshade=False)

    # キューブ
    if cp is not None:
        in_bounds = X_MIN <= cp[0] <= X_MAX
        if in_bounds:
            ax_e.scatter(cp[0], cp[1], cp[2],
                         c='darkorange', s=300, marker='s', zorder=6, depthshade=False)
            ax_e.text(cp[0], cp[1], cp[2] + 0.1, f'cube\nx={cp[0]:.2f}',
                      fontsize=7, color='darkorange', ha='center')
        else:
            ax_e.text2D(0.98, 0.92,
                        f'cube x={cp[0]:.2f}m (out of view →)',
                        transform=ax_e.transAxes, fontsize=9,
                        color='darkorange', ha='right', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))

    # +x 矢印
    ax_e.quiver(0, 0, 0.05, 0.4, 0, 0,
                color='green', lw=1.5, arrow_length_ratio=0.3, alpha=0.7)
    ax_e.text(0.5, 0, 0.08, '+x (push)', fontsize=8, color='green')

    ax_e.set_xlim(X_MIN, X_MAX)
    ax_e.set_ylim(Y_MIN, Y_MAX)
    ax_e.set_zlim(Z_MIN, Z_MAX)
    ax_e.set_xlabel('X'); ax_e.set_ylabel('Y'); ax_e.set_zlabel('Z')
    cube_info = f'  cube x={cp[0]:.2f}' if cp is not None else ''
    ax_e.set_title(
        f"[epoch={epoch}]  Step {i+1}/{len(exec_frames)}  r={frame['reward']:.3f}{cube_info}",
        fontsize=10)
    ax_e.view_init(elev=30, azim=-55)


ani_e = animation.FuncAnimation(fig_e, draw_exec_frame,
                                  frames=len(exec_frames),
                                  interval=1000 // args.fps, blit=False)
ani_e.save(out_exec, writer=animation.FFMpegWriter(fps=args.fps, bitrate=1800))
plt.close(fig_e)
print(f"[visual] Saved execution: {out_exec}")
print("[visual] Done.")

# choreonoid は --python スクリプト終了後も Qt イベントループが残り続け、プロセスが
# 終了しないことがある（eval_cross_env.py と同根の問題）。timeout コマンドの SIGTERM も
# 効かずハングし続けるケースを確認済みのため、ここで明示的に強制終了する。
os._exit(0)
