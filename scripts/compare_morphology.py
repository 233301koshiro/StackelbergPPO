"""
タスク別最適形態を比較するスクリプト。

使い方 (choreonoid は --xxx を自分の引数として解釈するため環境変数で渡す):
  COMPARE_RUNS="single_run/rrbot_arm_bigcube_B:single_run/rrbot_arm_reach_F" \
  COMPARE_LABELS="Pusher-B:Reach-F" \
  USE_CHOREONOID=1 /choreonoid_ws/install/bin/choreonoid --no-window \
    --python scripts/compare_morphology.py
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('USE_CHOREONOID', '1')

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from design_opt.utils.config import Config
from design_opt.utils.tools import set_global_seed
from design_opt.agents.genesis_agent import BodyGenAgent

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_morphology(run_dir: str, label: str) -> dict:
    """run_dir の best checkpoint から形態パラメータを抽出して dict で返す。"""
    raw = yaml.safe_load(open(f'{run_dir}/.hydra/config.yaml'))
    FLAGS = OmegaConf.create(raw)
    rel_run = os.path.relpath(run_dir, PROJECT)
    OmegaConf.update(FLAGS, 'restore_dir', rel_run)

    cfg = Config(FLAGS, PROJECT, run_dir)
    torch.set_default_dtype(torch.float64)
    set_global_seed(cfg.seed)

    cfg.num_threads = 1
    agent = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                         seed=cfg.seed, num_threads=1, training=False, checkpoint='best')
    env = agent.env

    _, log_eval = agent.sample(500, mean_action=True)

    ckpt = pickle.load(open(f'{run_dir}/models/best.p', 'rb'))

    bodies = []
    for body in env.robot.bodies:
        bd = {'name': body.name, 'depth': body.depth}
        if body.bone_offset is not None:
            bx = float(np.asarray(body.bone_offset[0]).flat[0])
            by = float(np.asarray(body.bone_offset[1]).flat[0])
            bd['bone_offset'] = (bx, by)
            bd['bone_len'] = float(np.linalg.norm(body.bone_offset))
        bd['joints'] = []
        for joint in body.joints:
            j = {'name': joint.name, 'range': getattr(joint, 'range', None)}
            if joint.actuator:
                j['gear'] = float(joint.actuator.gear)
            bd['joints'].append(j)
        bd['geoms'] = []
        for geom in body.geoms:
            g = {'radius': float(np.asarray(geom.size).flat[0])}
            if hasattr(geom, 'ext_start') and geom.ext_start is not None:
                g['ext_start'] = float(np.asarray(geom.ext_start).flat[0])
            bd['geoms'].append(g)
        bodies.append(bd)

    return {
        'label': label,
        'run_dir': run_dir,
        'epoch': ckpt['epoch'],
        'best_rewards': float(ckpt['best_rewards']),
        'exec_R_eps': log_eval.avg_exec_episode_reward,
        'num_bodies': len(bodies),
        'bodies': bodies,
        'design_param_names': list(env.design_param_names),
        'design_cur_params': env.design_cur_params.tolist(),
    }


def print_morphology(m: dict):
    print(f"\n{'='*60}")
    print(f"  {m['label']}  (ep={m['epoch']}, best_R={m['best_rewards']:.2f}, exec_R={m['exec_R_eps']:.3f})")
    print(f"  run: {m['run_dir']}")
    print(f"  bodies: {m['num_bodies']}")
    print(f"{'='*60}")
    total_len = 0.0
    for body in m['bodies']:
        print(f"\n  [{body['name']}]  depth={body['depth']}")
        if 'bone_offset' in body:
            bx, by = body['bone_offset']
            blen = body['bone_len']
            total_len += blen
            print(f"    bone_offset: x={bx:+.3f}, y={by:+.3f}  len={blen:.3f} m")
        for joint in body['joints']:
            r = joint.get('range')
            gear = joint.get('gear', 'N/A')
            rng_str = f"[{r[0]:.3f}, {r[1]:.3f}]" if r is not None else 'N/A'
            print(f"    joint '{joint['name']}': range={rng_str} rad  gear={gear:.1f}" if isinstance(gear, float) else f"    joint '{joint['name']}': range={rng_str}")
        for geom in body['geoms']:
            print(f"    capsule radius={geom['radius']:.4f} m", end='')
            if 'ext_start' in geom:
                print(f"  ext_start={geom['ext_start']:.4f} m", end='')
            print()
    print(f"\n  total arm length: {total_len:.3f} m")


def print_comparison_table(morphologies: list):
    print(f"\n{'='*60}")
    print("  COMPARISON TABLE")
    print(f"{'='*60}")
    labels = [m['label'] for m in morphologies]
    header = f"{'Metric':<25}" + "".join(f"{l:>15}" for l in labels)
    print(header)
    print("-" * len(header))

    def row(name, vals):
        print(f"{name:<25}" + "".join(f"{v:>15}" for v in vals))

    row("num_bodies", [str(m['num_bodies']) for m in morphologies])
    row("best_rewards", [f"{m['best_rewards']:.2f}" for m in morphologies])

    max_bodies = max(m['num_bodies'] for m in morphologies)
    for i in range(max_bodies):
        row(f"body[{i}] len (m)",
            [f"{m['bodies'][i]['bone_len']:.3f}" if i < m['num_bodies'] and 'bone_len' in m['bodies'][i] else "N/A"
             for m in morphologies])
        row(f"body[{i}] gear",
            [f"{m['bodies'][i]['joints'][0]['gear']:.0f}" if i < m['num_bodies'] and m['bodies'][i]['joints'] else "N/A"
             for m in morphologies])

    total_lens = []
    for m in morphologies:
        tl = sum(b.get('bone_len', 0) for b in m['bodies'])
        total_lens.append(f"{tl:.3f}")
    row("total arm len (m)", total_lens)


def main():
    # choreonoid が --args を自分のものとして解釈するため環境変数で受け取る
    # 使い方:
    #   COMPARE_RUNS="single_run/A:single_run/B" COMPARE_LABELS="A:B" \
    #     /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/compare_morphology.py
    runs_env = os.environ.get('COMPARE_RUNS', '')
    labels_env = os.environ.get('COMPARE_LABELS', '')

    if not runs_env:
        print("Error: set COMPARE_RUNS env var (colon-separated run dirs)")
        print("  e.g. COMPARE_RUNS='single_run/A:single_run/B' COMPARE_LABELS='A:B' choreonoid --no-window --python scripts/compare_morphology.py")
        os._exit(1)

    run_dirs = [r.strip() for r in runs_env.split(':')]
    if labels_env:
        labels = [l.strip() for l in labels_env.split(':')]
    else:
        labels = [os.path.basename(r) for r in run_dirs]

    morphologies = []
    for run_dir, label in zip(run_dirs, labels):
        run_path = os.path.join(PROJECT, run_dir) if not os.path.isabs(run_dir) else run_dir
        print(f"\n[Loading] {label} from {run_path} ...")
        m = extract_morphology(run_path, label)
        morphologies.append(m)
        print_morphology(m)

    if len(morphologies) > 1:
        print_comparison_table(morphologies)

    os._exit(0)


if __name__ == '__main__':
    main()
