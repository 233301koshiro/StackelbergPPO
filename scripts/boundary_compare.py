"""
⑤ 境界張り付き比較（matched-epoch）。方針レビュー ⑤ の格上げ検証用。

compare_morphology.py の抽出ロジックを checkpoint 指定対応にし、
design_cur_params（±1.0 正規化の設計パラメータ）の境界張り付き個数を集計する。
compare_morphology.py と同様 os._exit(0) で確実に終了する（choreonoid をハングさせない）。

使い方（choreonoid は --xxx を自分の引数に取るため環境変数で渡す）:
  BC_RUNS="single_run/tripo_arm_v2c_pusher:single_run/tripo_arm_v2c_pusher_ns2:single_run/tripo_arm_v2c_pusher_ns1" \
  BC_LABELS="nsteps5:nsteps2:nsteps1" BC_CKPT="epoch_0200" \
  USE_CHOREONOID=1 /choreonoid_ws/install/bin/choreonoid --no-window \
    --python scripts/boundary_compare.py
指定 ckpt が無い run は自動で best にフォールバックする。
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
CKPT = os.environ.get('BC_CKPT', 'best')
THRESH = float(os.environ.get('BC_THRESH', '0.95'))


def extract(run_dir, label):
    ck = CKPT if os.path.exists(f'{run_dir}/models/{CKPT}.p') else 'best'
    raw = yaml.safe_load(open(f'{run_dir}/.hydra/config.yaml'))
    FLAGS = OmegaConf.create(raw)
    OmegaConf.update(FLAGS, 'restore_dir', os.path.relpath(run_dir, PROJECT))
    cfg = Config(FLAGS, PROJECT, run_dir)
    # compare_morphology.py と同じ Bug 10 対策（転用フィルタ無効化）
    cfg.control_prior = False
    cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)
    set_global_seed(cfg.seed)
    cfg.num_threads = 1
    agent = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                         seed=cfg.seed, num_threads=1, training=False, checkpoint=ck)
    agent.sample(500, mean_action=True)
    meta = pickle.load(open(f'{run_dir}/models/{ck}.p', 'rb'))
    env = agent.env
    # ⚠️ design_cur_params の要素はスカラとは限らない（bone_offset のように多成分の
    # パラメータが 1 要素として入る）。素朴に float() すると
    # 「TypeError: only 0-dimensional arrays can be converted to Python scalars」で落ちる
    # （2026-08-07 に実際に失敗し、境界比較が実行できていなかった）。
    # 成分ごとに展開し、名前にも添字を付けて対応を保つ。
    names, vals = [], []
    for nm, v in zip(env.design_param_names, env.design_cur_params):
        a = np.asarray(v, dtype=float).ravel()
        if a.size == 1:
            names.append(str(nm)); vals.append(float(a[0]))
        else:
            for k, x in enumerate(a):
                names.append(f'{nm}[{k}]'); vals.append(float(x))

    br = np.asarray(meta['best_rewards'], dtype=float).ravel()
    return {
        'label': label, 'ckpt_used': ck, 'epoch': meta['epoch'],
        'best': float(br[0]) if br.size else float('nan'),
        'names': names,
        'vals': vals,
    }


def main():
    runs = [r.strip() for r in os.environ.get('BC_RUNS', '').split(':') if r.strip()]
    if not runs:
        print("Error: set BC_RUNS env var (colon-separated run dirs)")
        os._exit(1)
    labels_env = os.environ.get('BC_LABELS', '')
    labels = [l.strip() for l in labels_env.split(':')] if labels_env else [os.path.basename(r) for r in runs]

    results = []
    for r, lab in zip(runs, labels):
        rp = os.path.join(PROJECT, r) if not os.path.isabs(r) else r
        print(f"[load] {lab} <- {rp} (ckpt={CKPT})")
        results.append(extract(rp, lab))

    print(f"\n{'='*70}")
    print(f"  BOUNDARY-STICKING SUMMARY  (|design_cur_param| >= {THRESH})")
    print(f"{'='*70}")
    for m in results:
        stuck = [n for n, v in zip(m['names'], m['vals']) if abs(v) >= THRESH]
        print(f"\n  {m['label']}  (ckpt={m['ckpt_used']}, ep={m['epoch']}, best={m['best']:.2f})"
              f"  ->  {len(stuck)}/{len(m['vals'])} が境界(±1.0)に張り付き")
        for n, v in zip(m['names'], m['vals']):
            mark = '   <== BOUND' if abs(v) >= THRESH else ''
            print(f"      {n:<24} = {v:+.4f}{mark}")

    # 1行サマリ（ログ末尾で grep しやすく）
    print(f"\n{'='*70}\n  ONE-LINE:")
    for m in results:
        nstuck = sum(1 for v in m['vals'] if abs(v) >= THRESH)
        print(f"    {m['label']}: boundary={nstuck}/{len(m['vals'])} best={m['best']:.2f} ep={m['epoch']}")
    os._exit(0)


if __name__ == '__main__':
    main()
