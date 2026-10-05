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

# GUI モードでは Python stdout が Choreonoid の Message View に行き端末に出ない。
# ファイルにもリダイレクトしてデバッグを容易にする。
_log_path = os.environ.get('VIEWER_LOG', '/tmp/cnoid_viewer.log')
_log_file = open(_log_path, 'w', buffering=1)
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams:
            try: s.write(data); s.flush()
            except Exception: pass
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except Exception: pass
sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['USE_CHOREONOID'] = '1'

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
import cnoid.IRSLUtil as IU
try:
    from cnoid.Base import MessageView as _MV
    def _msg(text): _MV.instance.putln(text)
except Exception:
    def _msg(text): pass

from khrylib.utils import *
from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent, tensorfy
from design_opt.utils.tools import set_global_seed

# ---- 引数: 環境変数から取得（choreonoid は sys.argv を渡さないため）-------
# 例: VIEWER_RESTORE_DIR=single_run/pusher_cnoid VIEWER_FPS=25 choreonoid --python ...
restore_dir = os.environ.get('VIEWER_RESTORE_DIR', '')
if not restore_dir:
    print('[viewer] ERROR: VIEWER_RESTORE_DIR を設定してください')
    import os as _os; _os._exit(1)

class args:
    restore_dir = os.environ.get('VIEWER_RESTORE_DIR', '')
    epoch       = os.environ.get('VIEWER_EPOCH', 'best')
    fps         = int(os.environ.get('VIEWER_FPS', '25'))
    episodes    = int(os.environ.get('VIEWER_EPISODES', '3'))
    # ⭐ デモ用（2026-10-05）。既定 0 なので従来どおり
    start_delay  = float(os.environ.get('VIEWER_START_DELAY', '0'))    # 起動してから再生を始めるまで [s]（録画のウィンドウを選ぶ時間）
    design_pause = float(os.environ.get('VIEWER_DESIGN_PAUSE', '0'))   # 形態が変わるたびに止める [s]（形が育つ様子を見せる）

step_interval = 1.0 / args.fps

# ---- 設定読み込み -------------------------------------------------------
project_path = os.getcwd()
train_cfg_path = os.path.join(project_path, args.restore_dir, '.hydra', 'config.yaml')
FLAGS = OmegaConf.create(yaml.safe_load(open(train_cfg_path)))
cfg = Config(FLAGS, project_path, args.restore_dir)
cfg.restore_dir = args.restore_dir
# 転用起動 run の再評価時に転用フィルタが残ると重みが読み込まれない（Bug 10）。
cfg.control_prior = False
cfg.morph_prior = False

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

# ⭐ 最初のカメラ（2026-10-05、デモ用）: VIEWER_CAMERA=視点x,y,z,注視点x,y,z。
#   ⚠️ `viewAll` は床全体に合わせて腕が小さく映るので使わない。指定が無ければ Choreonoid の既定のまま
_cam = os.environ.get('VIEWER_CAMERA')
if _cam:
    try:
        from cnoid.Base import SceneView
        v = [float(x) for x in _cam.split(',')]
        eye, ctr = np.array(v[:3]), np.array(v[3:6])
        SceneView.instance.sceneWidget.setCameraPosition(eye, (ctr - eye) / np.linalg.norm(ctr - eye),
                                                         np.array([0.0, 0.0, 1.0]))
        print(f'[viewer] カメラ: 視点 {eye.tolist()} → 注視点 {ctr.tolist()}')
    except Exception as e:
        print(f'[viewer] ⚠️ カメラを設定できない（既定のまま）: {e!r}')

def _say(text):
    """ターミナルと Choreonoid のメッセージ欄の両方に出す。"""
    print(text, flush=True)
    _msg(text)


def _wait(sec):
    """画面を止めずに待つ（Qt のイベントを回し続ける）。"""
    t_end = time.time() + sec
    while time.time() < t_end:
        IU.processEvent()
        time.sleep(0.03)


# ⭐ 何を見ているかを言葉で出すための準備（2026-10-05、デモ用）
_rs = cfg.reward_specs or {}
_is_reach = bool(_rs.get('use_reach', False))
_tsrc = _rs if (_rs.get('use_target_reward', False) or _is_reach) else cfg.env_specs
_target = np.array([_tsrc.get('target_x', 0.8), _tsrc.get('target_y', 0.0), _tsrc.get('target_z', 0.15)])
_has_target = _is_reach or bool(_rs.get('use_target_reward', False))
_task = 'Reach（先端を目標へ）' if _is_reach else ('Target-Pusher（箱を目標へ）' if _rs.get('use_target_reward') else 'Pusher（箱を押す）')


def _lengths():
    """いまのリンク長 [m]（取り付け用の球は除く）。"""
    return [float(np.linalg.norm(b.bone_offset)) for b in env.robot.bodies[1:]
            if getattr(b, 'bone_offset', None) is not None]


def _fmt(ls):
    return ' / '.join(f'{x:.2f}' for x in ls)


if args.start_delay > 0:
    _say(f'[デモ] {args.restore_dir}  タスク: {_task}')
    _say(f'[デモ] {args.start_delay:.0f} 秒後に再生を始めます（録画するウィンドウを選んでください）')
    for _k in range(int(args.start_delay), 0, -1):
        if _k <= 5 or _k % 5 == 0:
            _say(f'[デモ] あと {_k} 秒')
        _wait(1.0)

# ---- 再生ループ ----------------------------------------------------------
ep = 0
try:
    while args.episodes == 0 or ep < args.episodes:
        state = env.reset()
        step  = 0
        design_steps = 0
        sketch_len = _lengths()
        _say('=' * 50)
        _say(f'[デモ] 第 {ep + 1} 話  タスク: {_task}')
        _say(f'[デモ] 描いた形（初期）のリンク長 [m]: {_fmt(sketch_len)}')
        _say('[デモ] 形態変化を始めます（Leader が形を決めます）')
        exec_steps = 0
        total_reward = 0.0
        cube_start_x = None
        cube_end_x   = None

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

            if info.get('stage') == 'skeleton_transform':
                # 骨格（リンクの本数）を決める段。本研究ではスケッチのまま固定（fix_skeleton）なので形は変わらない
                _say('[デモ] 骨格（リンクの本数）はスケッチのまま固定します')
                if args.design_pause > 0:
                    _wait(args.design_pause)
            elif info.get('stage') != 'execution':
                design_steps += 1
                _say(f'[デモ] 形態変化 {design_steps} 回目  リンク長 [m]: {_fmt(_lengths())}')
                if args.design_pause > 0:
                    _wait(args.design_pause)
            elif exec_steps == 0:
                now = _lengths()
                _say('[デモ] 形態変化が完了しました。実行に移ります（Follower が関節を動かします）')
                _say('[デモ] リンク長 [m]  描いた形 → 学習後: '
                     + '  '.join(f'{a:.2f}→{b:.2f}' for a, b in zip(sketch_len, now)))
                if args.design_pause > 0:
                    _wait(args.design_pause * 2)

            if info.get('stage') == 'execution':
                total_reward += reward
                exec_steps += 1
                cube_pos = env.get_body_com('cube')
                if cube_start_x is None:
                    cube_start_x = cube_pos[0]
                cube_end_x = cube_pos[0]
                # 実行フェーズのみスリープ（形態変換フェーズは速送り）
                time.sleep(step_interval)
                if exec_steps % 250 == 0:
                    if _is_reach:
                        _d = float(np.linalg.norm(np.asarray(env._arm_tip_pos) - _target)) * 1000
                        _say(f'[デモ] 実行中 {exec_steps} step  先端と目標の距離 {_d:.0f} mm')
                    else:
                        _say(f'[デモ] 実行中 {exec_steps} step  箱の移動 {cube_end_x - cube_start_x:.2f} m'
                             + (f'  目標まで {float(np.linalg.norm(np.asarray(cube_pos)[:2] - _target[:2])):.2f} m'
                                if _has_target else ''))

            done = term or trunc
            step += 1

            if done:
                break
            state = next_state

        ep += 1
        pushed = (cube_end_x - cube_start_x) if (cube_start_x is not None and cube_end_x is not None) else 0.0
        term_reason = 'truncation(max steps)' if trunc else 'termination(fallen/NaN)'
        line1 = (f'[viewer] Ep {ep}: reward={total_reward:.1f}  実行ステップ={exec_steps}  '
                 f'cube移動={pushed:.3f}m  終了={term_reason}')
        line2 = f'         bodies({len(env.robot.bodies)}体)={[b.name for b in env.robot.bodies]}'
        print(line1)
        print(line2)
        print('-' * 70)
        _msg('-' * 50)
        _msg(line1)
        _msg(line2)
        _say(f'[デモ] 第 {ep} 話の終わり')
        if args.design_pause > 0:
            _wait(args.design_pause * 2)

except KeyboardInterrupt:
    print('\n[viewer] 停止しました')

env.close()

# choreonoid は --python スクリプト終了後も Qt イベントループが残り続け、プロセスが
# 終了しないことがある（eval_cross_env.py / eval_cnoid_visual.py と同根の問題）。
os._exit(0)
