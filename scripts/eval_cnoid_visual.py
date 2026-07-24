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
    # EVAL_HIDE_CUBE: '1'=常に消す / '0'=常に描く / 'auto'(デフォルト)=Reach 系
    # （reward_specs.use_reach=true の run）なら消して代わりに目標点を描く。
    # cube はタスク間の交絡排除のため XML には残しているが、Reach では報酬に
    # 一切関与しないため、動画上は紛らわしいだけ（2026-07-17 ユーザー要望）。
    hide_cube      = os.environ.get('EVAL_HIDE_CUBE', 'auto')

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
# 完走 run 自身の checkpoint を再評価する際は転用フィルタを無効化する。
# control_prior/morph_prior が true のままだと load_checkpoint が Leader/Follower の
# 一方と obs_norm を読み込まず、ランダム初期化ネットで形態・行動が再現される
# （2026-07-15 発覚。デバッグ戦記 Bug 10。本スクリプトは修正5本から漏れていた）。
cfg.control_prior = False
cfg.morph_prior = False

# EVAL_HIDE_CUBE の解決: 'auto' は Reach 系（use_reach=true）のとき cube を隠す
_is_reach = bool(cfg.reward_specs.get('use_reach', False))
if args.hide_cube == 'auto':
    HIDE_CUBE = _is_reach
else:
    HIDE_CUBE = args.hide_cube == '1'
# Reach のときは cube の代わりに目標点を描く
REACH_TARGET = None
if _is_reach:
    REACH_TARGET = (float(cfg.reward_specs.get('target_x', 1.5)),
                    float(cfg.reward_specs.get('target_y', 0.0)),
                    float(cfg.reward_specs.get('target_z', 0.2)))
print(f"[visual] hide_cube={HIDE_CUBE} (mode={args.hide_cube}, use_reach={_is_reach})"
      + (f" reach_target={REACH_TARGET}" if REACH_TARGET else ""))

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
    """Robot.bodies から bone_offset / ext_start を直接読む（非正規化済み）。
    no_root_offset=true のとき root(depth=0)の bone_offset は Leader の生出力に
    過ぎず rebuild() で無効化される（台座固定）ため、報告上もゼロとして扱う
    （2026-07-21 発覚: この関数が死んだパラメータを「成長した」ように表示していた）。"""
    no_root_offset = cfg.robot_cfg.get('no_root_offset', False)
    data = {}
    for body in robot.bodies:
        entry = {}
        root_frozen = no_root_offset and body.parent is None
        if root_frozen:
            entry['bone_offset'] = np.zeros(3)
            entry['bone_length'] = 0.0
        elif hasattr(body, 'bone_offset') and body.bone_offset is not None:
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


def build_rest_arm(physics, base_world, body_names):
    """
    全関節 0° のときの N+1 点を bone_offset の連鎖から計算する（任意関節数対応）。
    rrbot_arm ではジョイント軸が Z（上方向）なので bone_offset [x,y,0] が
    X-Y 水平面内の方向を表す。

    body_names は root→tip の順（env.robot.bodies の順）。
    points[0] = base（固定ベース）、points[i+1] = points[i] + body_names[i].bone_offset。
    body_names[0]（root, '0'）は no_root_offset で常にゼロなので points[0]==points[1]。
    最終点 points[-1] が先端（tip）。
    """
    base = np.array(base_world, dtype=float)
    points = [base.copy()]
    lengths = []
    cur = base.copy()
    for bn in body_names:
        bo = np.asarray(physics.get(bn, {}).get('bone_offset', np.zeros(3)), dtype=float)
        cur = cur + bo
        points.append(cur.copy())
        lengths.append(float(np.linalg.norm(bo)))
    return {
        'points': points,      # N+1 点（base を含む）
        'lengths': lengths,    # N セグメント長（points[i]→points[i+1] の長さ）
        'base': points[0],
        'tip': points[-1],
    }


# ── シミュレーション live 状態からリンク位置取得 ──────────────────────────
def collect_arm_skeleton(env):
    """env._body_xpos / _body_xmat（シミュレーション live データ）から
    shoulder / elbow / tip を返す。

    以前は Choreonoid の body_items['robot'].body.link('11').p を使っていたが、
    これは参照ボディ（静的）のため実行中に更新されず、動画でアームが静止して
    見えるバグがあった。body_xpos は AISTSimulator から毎ステップ取得される
    live データであり、MuJoCo body_xpos と一致することを確認済み。
    """
    bodies  = env.robot.bodies
    physics = get_body_physics(env.robot)

    # 全ボディの world 位置を連ねた polyline（任意関節数のチェーンに対応）。
    # 末尾ボディはジョイント位置なので、その bone_offset を回転して先端を足す。
    points = [np.array(env._body_xpos.get(b.name, np.zeros(3))) for b in bodies]
    b_last = bodies[-1].name
    bo_last = physics.get(b_last, {}).get('bone_offset', np.array([0.25, 0.0, 0.0]))
    R_last  = np.array(env._body_xmat.get(b_last, np.eye(3)))
    tip = points[-1] + R_last @ bo_last
    points.append(tip)

    # 旧2関節コードとの互換キー（shoulder=根本, elbow=末尾ジョイント, tip=先端）
    return {'shoulder': points[0], 'elbow': points[-2], 'tip': tip, 'points': points}


# ── エピソード実行 ───────────────────────────────────────────────────────
print("[visual] Running episode to collect frames...")

# reset 直後（変換前）の形態を取得
state = env.reset()
# チェーンの body 名（root→tip の順）。関節数に依らずここから動的に取得する
# （旧コードは ['0','1','11'] 決め打ちで3関節目以降を無視していた）。
CHAIN_BODY_NAMES = [b.name for b in env.robot.bodies]
print(f"[visual] chain body names: {CHAIN_BODY_NAMES}")
before_physics  = get_body_physics(env.robot)
shoulder_world  = np.array(env._world.body_items['robot'].body.rootLink.p)
before_arm_rest = build_rest_arm(before_physics, shoulder_world, CHAIN_BODY_NAMES)

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
        # attribute_transform の最終ステップ内で transit_execution() が呼ばれ、
        # env.stage がすでに 'execution' に変わっている。
        # この時点でのアーム・cube 状態が「実行開始直前（step 0）」に相当するため、
        # do_simulation() 前の静止状態を step 0 フレームとして記録する。
        if getattr(env, 'stage', '') == 'execution' and after_physics is None:
            after_physics  = get_body_physics(env.robot)
            after_arm_rest = build_rest_arm(after_physics, shoulder_world, CHAIN_BODY_NAMES)
            print(f"[visual] Morphology captured at execution start.")
            for bn in CHAIN_BODY_NAMES:
                bo = after_physics.get(bn, {}).get('bone_offset', None)
                if bo is not None:
                    print(f"[visual] body_{bn} bone_offset: x={bo[0]:.4f}, y={bo[1]:.4f}")
            for pi, p in enumerate(after_arm_rest['points']):
                label = 'base' if pi == 0 else ('tip' if pi == len(after_arm_rest['points']) - 1 else f'joint{pi}')
                print(f"[visual] rest arm {label}: x={p[0]:.4f}, y={p[1]:.4f}")
            cube_pos_s0 = env.get_body_com('cube').copy() if 'cube' in env._body_names else None
            nv = env.model.nv
            qpos_s0 = env.data.qpos.copy()
            vx_s0   = float(env.data.qvel[nv - 2])
            print(f"[visual] Step 0 (before do_simulation): qpos0={qpos_s0[0]:.4f} qpos1={qpos_s0[1]:.4f} cube_x={cube_pos_s0[0] if cube_pos_s0 is not None else 0:.3f} cube_vx={vx_s0:.4f}")
            exec_frames.append({
                'arm': collect_arm_skeleton(env),
                'cube_pos': cube_pos_s0,
                'stage': 'execution',
                'reward': 0.0,
                'qpos': qpos_s0.copy(),
                'cube_vx': vx_s0,
                'step0': True,
            })
    else:
        # 実行フェーズに入った瞬間に「変換後」形態を記録（フォールバック）
        if after_physics is None:
            after_physics  = get_body_physics(env.robot)
            after_arm_rest = build_rest_arm(after_physics, shoulder_world, CHAIN_BODY_NAMES)
            print(f"[visual] Morphology captured at execution start.")
            for bn in CHAIN_BODY_NAMES:
                bo = after_physics.get(bn, {}).get('bone_offset', None)
                if bo is not None:
                    print(f"[visual] body_{bn} bone_offset: x={bo[0]:.4f}, y={bo[1]:.4f}")
            for pi, p in enumerate(after_arm_rest['points']):
                label = 'base' if pi == 0 else ('tip' if pi == len(after_arm_rest['points']) - 1 else f'joint{pi}')
                print(f"[visual] rest arm {label}: x={p[0]:.4f}, y={p[1]:.4f}")
        nv = env.model.nv
        exec_frames.append({
            'arm': arm, 'cube_pos': cube_pos,
            'stage': stage, 'reward': reward,
            'qpos': env.data.qpos.copy(),
            'cube_vx': float(env.data.qvel[nv - 2]),
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

param_rows = []
for bn in CHAIN_BODY_NAMES:
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


def draw_arm_2d(ax, arm_rest, is_after, lengths_before, lengths_after, body_names):
    """X-Y 俯瞰（ジョイント軸が Z なので回転面が X-Y）で N+1 点の腕を描く（任意関節数対応）。
    セグメント0（root, body_names[0]）は no_root_offset で常にゼロ長。
    最終セグメントが先端（tip）へのリンク。"""
    points = arm_rest['points']
    n_seg = len(points) - 1
    seg_colors = [
        link_color(lengths_before[i] if i < len(lengths_before) else None,
                   lengths_after[i]  if i < len(lengths_after)  else None,
                   is_after)
        for i in range(n_seg)
    ]

    # セグメント描画（X-Y平面: X=前後, Y=左右）
    for i in range(n_seg):
        p0, p1 = points[i], points[i + 1]
        if i == 0:
            lw, ls, label = 3, '--', 'base offset'
        elif i == n_seg - 1:
            lw, ls, label = 4, '-', f'link {body_names[i]}'
        else:
            lw, ls, label = 5, '-', f'link {body_names[i]}'
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], ls, color=seg_colors[i], lw=lw,
                solid_capstyle='round', zorder=3, label=label)

    # ジョイント点
    pad = max(abs(MORPH_XLIM[1] - MORPH_XLIM[0]),
              abs(MORPH_YLIM[1] - MORPH_YLIM[0])) * 0.07
    n_pts = len(points)
    for i, p in enumerate(points):
        if i == 0:
            marker, color, name = 's', '#444444', 'base'
        elif i == n_pts - 1:
            marker, color, name = '^', seg_colors[-1], 'tip'
        elif i == 1:
            marker, color, name = 'o', '#222222', 'shoulder'
        else:
            marker, color, name = 'o', seg_colors[i - 1], f'j{body_names[i - 1]}'
        ms = 10 if i == 0 else (7 if i == n_pts - 1 else (9 if i == 1 else 8))
        ax.plot(p[0], p[1], marker, color=color, ms=ms, zorder=5)
        dx = pad * (0.3 if i % 2 == 0 else -0.8)
        dy = pad if i % 2 == 0 else -pad
        ax.annotate(f'{name}\n({p[0]:.2f},{p[1]:.2f})', xy=(p[0], p[1]),
                    xytext=(p[0] + dx, p[1] + dy), fontsize=7, color=color,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.7))


# モーフィング動画フレームリスト構築（任意関節数対応: セグメント長は build_rest_arm の
# 'lengths' をそのまま使う。各セグメントは CHAIN_BODY_NAMES と同じ順に対応する）
lengths_before = before_arm_rest['lengths']
lengths_after  = after_arm_rest['lengths']

# 軸範囲: before/after/transition の全頂点を包むように動的決定
def _arm_pts(arm):
    return arm['points']

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
        seg_lengths = lengths_before
        is_after = False
        label = 'Before (Default Morphology)'
        lcolor = '#334455'
    elif fi < HOLD + TRANS:
        # トランジション: before → after を補間（任意関節数対応）
        t = (fi - HOLD) / TRANS
        arm = {
            'points': [(1 - t) * bp + t * ap
                       for bp, ap in zip(before_arm_rest['points'], after_arm_rest['points'])],
        }
        seg_lengths = [(1 - t) * lb + t * la
                       for lb, la in zip(lengths_before, lengths_after)]
        is_after = t > 0.5
        label = 'Morphing...'
        lcolor = '#666633'
    else:
        # 変化後
        arm = after_arm_rest
        seg_lengths = lengths_after
        is_after = True
        label = 'After (Learned Morphology)'
        lcolor = '#CC3300'

    # 左: 俯瞰
    ax = axes_m[0]
    ax.set_facecolor('#F5F5F0')
    draw_arm_2d(ax, arm, is_after, lengths_before, lengths_after, CHAIN_BODY_NAMES)
    ax.set_xlim(*MORPH_XLIM)
    ax.set_ylim(*MORPH_YLIM)
    ax.set_aspect('equal')
    ax.set_xlabel('X (push direction) [m]', fontsize=9)
    ax.set_ylabel('Y (lateral) [m]', fontsize=9)
    ax.grid(True, alpha=0.3)
    if not HIDE_CUBE:
        ax.axvline(x=0.60, color='orange', lw=1, ls='--', alpha=0.6, label='cube x₀=0.60')
    if REACH_TARGET is not None:
        ax.axvline(x=REACH_TARGET[0], color='red', lw=1, ls=':', alpha=0.5,
                   label=f'reach target x={REACH_TARGET[0]:.2f}')
    else:
        ax.axvline(x=1.50, color='red', lw=1, ls=':', alpha=0.5, label='target x=1.50')
    ax.legend(fontsize=7.5, loc='upper left')
    ax.set_title(f'Top View (X-Y)  [{label}]  frame {fi+1}/{total}',
                 fontsize=11, fontweight='bold', color=lcolor)

    # 右: パラメータ表（root セグメント[0]は常にゼロなので除外し、実リンクのみ表示）
    ax2 = axes_m[1]
    ax2.set_facecolor('#FAFAF8')
    ax2.axis('off')

    real_idx = list(range(1, len(CHAIN_BODY_NAMES)))
    reach_b  = sum(lengths_before[i] for i in real_idx)
    reach_a_ = sum(lengths_after[i]  for i in real_idx)
    d_reach  = reach_a_ - reach_b

    ax2.text(0.5, 0.97, label, transform=ax2.transAxes,
             fontsize=14, fontweight='bold', ha='center', va='top', color=lcolor)

    rows = [('Parameter', 'Before', 'After', 'Delta')]
    for i in real_idx:
        lb, la = lengths_before[i], lengths_after[i]
        rows.append((f'Link {CHAIN_BODY_NAMES[i]} len', f'{lb:.3f} m', f'{la:.3f} m', delta_str(la - lb)))
    rows.append(('Total reach', f'{reach_b:.3f} m', f'{reach_a_:.3f} m', delta_str(d_reach)))

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

    # アーム（'points' があれば全リンクの polyline、なければ旧2関節描画）
    if arm is not None:
        pts = arm.get('points')
        if pts is not None and len(pts) >= 2:
            P = np.asarray(pts)
            ax_e.plot(P[:, 0], P[:, 1], P[:, 2], 'b-', lw=3.0, alpha=0.9, zorder=4)
            ax_e.scatter(P[0, 0], P[0, 1], P[0, 2], c='navy', s=120, zorder=5, depthshade=False)
            if len(P) > 2:
                ax_e.scatter(P[1:-1, 0], P[1:-1, 1], P[1:-1, 2],
                             c='royalblue', s=90, zorder=5, depthshade=False)
            ax_e.scatter(P[-1, 0], P[-1, 1], P[-1, 2], c='steelblue', s=80, zorder=5, depthshade=False)
        else:
            s, e, t = arm['shoulder'], arm['elbow'], arm['tip']
            ax_e.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]],
                      'b-', lw=3.5, alpha=0.9, zorder=4)
            ax_e.plot([e[0], t[0]], [e[1], t[1]], [e[2], t[2]],
                      color='cornflowerblue', lw=2.5, alpha=0.9, zorder=4)
            ax_e.scatter(*s, c='navy',      s=120, zorder=5, depthshade=False)
            ax_e.scatter(*e, c='royalblue', s=100, zorder=5, depthshade=False)
            ax_e.scatter(*t, c='steelblue', s=80,  zorder=5, depthshade=False)

    # キューブ（HIDE_CUBE=True のとき非表示。Reach では報酬に関与しないため）
    if cp is not None and not HIDE_CUBE:
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

    # Reach の目標点（cube の代わりに到達目標を描く）
    if REACH_TARGET is not None:
        tx, ty, tz = REACH_TARGET
        ax_e.scatter(tx, ty, tz, c='red', s=200, marker='*', zorder=6, depthshade=False)
        ax_e.text(tx, ty, tz + 0.1, f'target\n({tx:.2f}, {ty:.2f}, {tz:.2f})',
                  fontsize=7, color='red', ha='center')

    # +x 矢印
    ax_e.quiver(0, 0, 0.05, 0.4, 0, 0,
                color='green', lw=1.5, arrow_length_ratio=0.3, alpha=0.7)
    ax_e.text(0.5, 0, 0.08, '+x (push)', fontsize=8, color='green')

    ax_e.set_xlim(X_MIN, X_MAX)
    ax_e.set_ylim(Y_MIN, Y_MAX)
    ax_e.set_zlim(Z_MIN, Z_MAX)
    ax_e.set_xlabel('X'); ax_e.set_ylabel('Y'); ax_e.set_zlabel('Z')
    if REACH_TARGET is not None and arm is not None:
        _d = np.linalg.norm(np.asarray(arm['tip']) - np.asarray(REACH_TARGET))
        cube_info = f'  dist_to_target={_d:.3f}'
    elif cp is not None and not HIDE_CUBE:
        cube_info = f'  cube x={cp[0]:.2f}'
    else:
        cube_info = ''
    qpos = frame.get('qpos', None)
    cube_vx = frame.get('cube_vx', None)
    step_label = '0 (before sim)' if frame.get('step0') else str(i)
    qpos_str = f'  q0={qpos[0]:.3f} q1={qpos[1]:.3f}' if qpos is not None else ''
    vx_str   = f'  vx={cube_vx:.3f}' if cube_vx is not None else ''
    ax_e.set_title(
        f"[epoch={epoch}]  Step {step_label}/{len(exec_frames)-1}{cube_info}{qpos_str}{vx_str}",
        fontsize=8)
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
