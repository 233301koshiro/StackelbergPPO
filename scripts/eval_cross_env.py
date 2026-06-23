#!/usr/bin/env python3
"""
クロス環境評価スクリプト
各 run をその学習時と同じ物理エンジン（MuJoCo / Choreonoid）で評価し比較する。

使い方:
  cd /userdir/StackelbergPPO
  python scripts/eval_cross_env.py \
    --runs single_run/pusher_resume single_run/pusher_cnoid_v3 \
    --labels MuJoCo Choreonoid \
    --epochs best best \
    --n_episodes 20 \
    --output single_run/comparison/eval_results.json

環境の自動判定:
  hydra config の cfg 名に "cnoid" が含まれていれば USE_CHOREONOID=1 で評価する。
  --envs mujoco choreonoid で明示的に指定することも可能。

実装上の注意:
  USE_CHOREONOID は pusher.py の import 時に読まれるため、
  run ごとにサブプロセスを立てて環境変数を分離する。
  サブプロセスは --_worker モードで 1 run を評価して JSON を標準出力に返す。
  Choreonoid の場合は cnoid モジュールを直接 import できないため、
  choreonoid --no-window --python 経由でサブプロセスを起動する。
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import numpy as np

# ─── Choreonoid --python 経由のワーカー起動 ──────────────────────────────────
# choreonoid --no-window --python <script> で起動された場合、
# Choreonoid が独自に argv を解析して --_worker 等のフラグをエラーにするため、
# 引数は環境変数 _EVAL_* 経由で渡す。
# この処理はモジュールレベルで実行され、メインの argparse より先に動作する。
if os.environ.get('_EVAL_RESTORE_DIR'):
    # 環境変数からパラメータを読んでワーカーとして直接実行
    def _run_from_env():
        _restore = os.environ['_EVAL_RESTORE_DIR']
        _epoch   = os.environ.get('_EVAL_EPOCH', 'best')
        _n_eps   = int(os.environ.get('_EVAL_N_EPISODES', '10'))
        _thresh  = float(os.environ.get('_EVAL_THRESHOLD', '0.5'))
        _out     = os.environ['_EVAL_TMPOUT']
        # run_worker はこのファイル内で後に定義されるため、ここでは定義のみ
        # 実際の呼び出しはモジュール末尾で行う
    _CHOREONOID_ENV_MODE = True
else:
    _CHOREONOID_ENV_MODE = False


# ─── 環境判定 ──────────────────────────────────────────────────────────────────

def detect_engine(restore_dir: str) -> str:
    """学習に使った物理エンジンを判定する。

    判定優先順位:
    1. .hydra/config.yaml の cfg 名に 'cnoid' が含まれる
    2. stdout.log / train.log の冒頭に Choreonoid 固有メッセージが含まれる
    3. 上記いずれも該当しない場合は mujoco
    """
    import yaml

    config_path = os.path.join(restore_dir, '.hydra', 'config.yaml')
    if os.path.exists(config_path):
        cfg = yaml.safe_load(open(config_path))
        cfg_name = str(cfg.get('cfg', ''))
        if 'cnoid' in cfg_name.lower():
            return 'choreonoid'

    for log_name in ('stdout.log', 'train.log'):
        log_path = os.path.join(restore_dir, log_name)
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', errors='ignore') as f:
                    header = f.read(1024)
                if 'cnoid' in header.lower() or 'choreonoid' in header.lower():
                    return 'choreonoid'
            except OSError:
                pass

    return 'mujoco'


# ─── ワーカーモード（サブプロセスから呼ばれる） ────────────────────────────────

def run_worker(restore_dir: str, epoch, n_episodes: int,
               success_threshold: float, output_file: str):
    """
    1つの run を評価して JSON ファイルに保存する。
    このモードではプロセス起動前に USE_CHOREONOID が正しくセットされている前提。
    """
    import torch
    import yaml
    from omegaconf import OmegaConf
    sys.path.insert(0, os.getcwd())

    from design_opt.agents.genesis_agent import tensorfy, BodyGenAgent
    from design_opt.utils.config import Config
    from design_opt.utils.tools import set_global_seed

    config_path = os.path.join(restore_dir, '.hydra', 'config.yaml')
    FLAGS = OmegaConf.create(yaml.safe_load(open(config_path)))
    project_path = os.getcwd()
    cfg = Config(FLAGS, project_path, restore_dir)
    cfg.restore_dir = restore_dir

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cpu')
    set_global_seed(cfg.seed)

    ep_arg = int(epoch) if isinstance(epoch, str) and epoch.isnumeric() else epoch
    engine = 'Choreonoid' if os.environ.get('USE_CHOREONOID') == '1' else 'MuJoCo'
    print(f"[worker] {restore_dir}  epoch={ep_arg}  engine={engine}", flush=True)

    agent = BodyGenAgent(cfg=cfg, dtype=dtype, device=device,
                         seed=cfg.seed, num_threads=1, training=False,
                         checkpoint=ep_arg)
    agent.policy_net.eval()
    if agent.obs_norm is not None:
        agent.obs_norm.eval()
        agent.obs_norm.to(device)

    env = agent.env
    per_episode = []

    for ep_idx in range(n_episodes):
        state = env.reset()
        cube_x_start = None
        cube_x_end   = None
        exec_reward  = 0.0
        exec_steps   = 0

        while True:
            state_var = tensorfy([state])
            if cfg.uni_obs_norm and agent.obs_norm is not None:
                state_var = agent.normalize_observation(state_var)

            with torch.no_grad():
                action = agent.policy_net.select_action(state_var, mean_action=True)
                action = action.numpy().astype(np.float64)

            next_state, env_reward, terminated, truncated, info = env.step(action)

            if info.get('stage') == 'execution':
                if cube_x_start is None:
                    cube_x_start = float(env.get_body_com("cube")[0])
                cube_x_end = float(env.get_body_com("cube")[0])
                exec_reward += float(env_reward)
                exec_steps  += 1

            if terminated or truncated:
                break
            state = next_state

        cube_disp = (cube_x_end - cube_x_start) \
                    if cube_x_start is not None and cube_x_end is not None else 0.0

        per_episode.append({
            'cube_disp_m': float(cube_disp),
            'exec_reward':  float(exec_reward),
            'exec_steps':   exec_steps,
            'success':      cube_disp >= success_threshold,
        })
        mark = "✓" if cube_disp >= success_threshold else "✗"
        print(f"  ep {ep_idx:2d}: exec_R={exec_reward:7.1f}  "
              f"cube_x Δ={cube_disp:+.3f}m  steps={exec_steps}  {mark}", flush=True)

    disps   = [e['cube_disp_m'] for e in per_episode]
    rewards = [e['exec_reward']  for e in per_episode]
    # NaN (数値発散エピソード) を除いて統計を計算
    rewards_finite = [r for r in rewards if np.isfinite(r)]

    result = {
        'restore_dir':         restore_dir,
        'epoch':               str(epoch),
        'engine':              engine,
        'n_episodes':          n_episodes,
        'success_threshold_m': success_threshold,
        'mean_exec_reward':    float(np.mean(rewards_finite)) if rewards_finite else float('nan'),
        'std_exec_reward':     float(np.std(rewards_finite))  if rewards_finite else float('nan'),
        'mean_cube_disp_m':    float(np.mean(disps)),
        'max_cube_disp_m':     float(np.max(disps)),
        'success_rate':        float(np.mean([e['success'] for e in per_episode])),
        'per_episode':         per_episode,
    }
    print(f"  → mean_exec_R={result['mean_exec_reward']:.1f}  "
          f"mean_disp={result['mean_cube_disp_m']:.3f}m  "
          f"success_rate={result['success_rate']*100:.0f}%", flush=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)


# ─── オーケストレーター（メインプロセス） ─────────────────────────────────────

def eval_run_subprocess(restore_dir: str, label: str, epoch: str,
                        n_episodes: int, threshold: float,
                        engine: str, tmp_dir: str) -> dict:
    """
    サブプロセスを立てて 1 run を評価し、結果 dict を返す。
    engine: 'mujoco' | 'choreonoid'
    """
    tmp_out = os.path.join(tmp_dir, label.replace('/', '_') + '_result.json')

    env = os.environ.copy()
    env['USE_CHOREONOID'] = '1' if engine == 'choreonoid' else '0'

    if engine == 'choreonoid':
        # Choreonoid の cnoid モジュールは直接 python3 から import できないため
        # choreonoid --no-window --python 経由で起動する。
        # Choreonoid は argv の -- 付きフラグを自分の引数として解析してしまうため、
        # パラメータは環境変数 _EVAL_* 経由で渡す。
        env['_EVAL_RESTORE_DIR']  = restore_dir
        env['_EVAL_EPOCH']        = str(epoch)
        env['_EVAL_N_EPISODES']   = str(n_episodes)
        env['_EVAL_THRESHOLD']    = str(threshold)
        env['_EVAL_TMPOUT']       = tmp_out
        env['_EVAL_LABEL']        = label
        choreonoid_bin = '/choreonoid_ws/install/bin/choreonoid'
        cmd = [choreonoid_bin, '--no-window', '--python', __file__]
    else:
        worker_args = [
            '--_worker',
            '--runs',       restore_dir,
            '--labels',     label,
            '--epochs',     epoch,
            '--n_episodes', str(n_episodes),
            '--threshold',  str(threshold),
            '--_tmpout',    tmp_out,
        ]
        cmd = [sys.executable, __file__] + worker_args

    print(f"\n[eval] Spawning subprocess: engine={engine}  run={restore_dir}")
    ret = subprocess.run(cmd, env=env, cwd=os.getcwd())
    if engine == 'choreonoid':
        # choreonoid では sys.exit が TypeError になるため、exit code より JSON の存在で判定
        if not os.path.exists(tmp_out):
            raise RuntimeError(f"Worker failed (code {ret.returncode}, no output) for {restore_dir}")
    elif ret.returncode != 0:
        raise RuntimeError(f"Worker failed (code {ret.returncode}) for {restore_dir}")

    with open(tmp_out, encoding='utf-8') as f:
        result = json.load(f)
    result['label'] = label
    return result


# ─── メイン ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs',       nargs='+', required=True)
    parser.add_argument('--labels',     nargs='+', default=None)
    parser.add_argument('--epochs',     nargs='+', default=None)
    parser.add_argument('--envs',       nargs='+', default=None,
                        help='各 run の評価エンジン: mujoco|choreonoid|auto (default: auto)')
    parser.add_argument('--n_episodes', type=int,   default=20)
    parser.add_argument('--threshold',  type=float, default=0.5,
                        help='成功判定の cube +x 移動距離 [m] (default: 0.5)')
    parser.add_argument('--output',     type=str,
                        default='single_run/comparison/eval_results.json')
    # 内部用（サブプロセスワーカーモード）
    parser.add_argument('--_worker',  action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--_tmpout',  type=str,            help=argparse.SUPPRESS)
    args = parser.parse_args()

    # ── ワーカーモード ─────────────────────────────────────────────────────────
    if args._worker:
        run_worker(
            restore_dir=args.runs[0],
            epoch=args.epochs[0] if args.epochs else 'best',
            n_episodes=args.n_episodes,
            success_threshold=args.threshold,
            output_file=args._tmpout,
        )
        return

    # ── オーケストレーターモード ───────────────────────────────────────────────
    labels = args.labels or args.runs
    epochs = args.epochs or ['best'] * len(args.runs)
    envs   = args.envs   or ['auto'] * len(args.runs)

    if not (len(labels) == len(epochs) == len(envs) == len(args.runs)):
        parser.error('--runs / --labels / --epochs / --envs の要素数を揃えてください')

    # auto → config から判定
    resolved_engines = []
    for run, env_spec in zip(args.runs, envs):
        if env_spec == 'auto':
            resolved_engines.append(detect_engine(run))
        else:
            resolved_engines.append(env_spec.lower())

    all_results = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for run, label, epoch, engine in zip(args.runs, labels, epochs, resolved_engines):
            result = eval_run_subprocess(run, label, epoch,
                                         args.n_episodes, args.threshold,
                                         engine, tmp_dir)
            all_results.append(result)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[eval] Results saved → {args.output}")

    print("\n" + "=" * 65)
    print(f"{'Label':<20} {'Engine':<12} {'mean_exec_R':>12} {'mean_disp':>10} {'success%':>9}")
    print("-" * 65)
    for r in all_results:
        print(f"{r['label']:<20} {r['engine']:<12} {r['mean_exec_reward']:>12.1f} "
              f"{r['mean_cube_disp_m']:>10.3f}m {r['success_rate']*100:>8.0f}%")
    print("=" * 65)


# choreonoid --no-window --python 経由で起動された場合のエントリポイント
# __name__ が '__main__' にならない場合もあるためトップレベルでチェック
if _CHOREONOID_ENV_MODE:
    _restore = os.environ['_EVAL_RESTORE_DIR']
    _epoch   = os.environ.get('_EVAL_EPOCH', 'best')
    _n_eps   = int(os.environ.get('_EVAL_N_EPISODES', '10'))
    _thresh  = float(os.environ.get('_EVAL_THRESHOLD', '0.5'))
    _out     = os.environ['_EVAL_TMPOUT']
    run_worker(_restore, _epoch, _n_eps, _thresh, _out)
    os._exit(0)  # choreonoid では sys.exit が TypeError になるため強制終了

if __name__ == '__main__':
    main()
