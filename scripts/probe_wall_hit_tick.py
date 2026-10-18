#!/usr/bin/env python3
"""⭐ 壁の反射が env step の「何番目の tick」で起きたかと、跳ね返った後の速度の比を並べる（Bug 70、系譜 9-263 ⑥）。

仮説: 反射した速度（−e 倍）は同じ env step の残り tick でだけ当て直される。最後の tick で起きると
次の env step で当て直されず、接触解決の押し出しが残って比が e ≈ 0.75 ではなく約 1.7 になる。
→ 予測: 起きた tick が 4（最後）のときだけ比が 1 を超え、1〜3 では e に近い。

パックを側壁へ初速 V0 で打ち、初期位置 y0 を少しずつずらして発火する tick を変える。
物理は BASE_RUN の launch_env.json（学習時の環境変数）をそのまま使う。

    BASE_RUN=single_run/hockey_bank9_phys_g10 USE_CHOREONOID=1 OMP_NUM_THREADS=1 \\
    timeout -k 30 900 /choreonoid_ws/install/bin/choreonoid --no-window \\
        --python scripts/probe_wall_hit_tick.py > /tmp/probe_wall_hit_tick.log 2>&1

⚠️ Bug 43: パイプへ繋がずファイルへ。末尾は os._exit(0)（Choreonoid は自分では終わらない、系譜 9-257）。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

BASE = os.environ.get('BASE_RUN', 'single_run/hockey_bank9_phys_g10')
for _k, _v in json.load(open(os.path.join(BASE, 'launch_env.json'))).items():
    os.environ.setdefault(_k, _v)
os.environ['USE_CHOREONOID'] = '1'
V0 = float(os.environ.get('V0', '1.5'))
N_TRIAL = int(os.environ.get('N_TRIAL', '16'))


def main():
    import torch
    import yaml
    from omegaconf import OmegaConf
    from design_opt.utils.config import Config
    from design_opt.agents.genesis_agent import BodyGenAgent

    d = OmegaConf.to_container(
        OmegaConf.create(yaml.safe_load(open(f'{BASE}/.hydra/config.yaml'))), resolve=True)
    d.pop('restore_dir', None)
    cfg = Config(OmegaConf.create(d), os.getcwd(), '/tmp/_probe_hit_tick')
    cfg.control_prior = cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)
    print(f'[tick] BASE={BASE} xml={d["xml_name"]} V0={V0} '
          f'e={os.environ.get("HOCKEY_WALL_RESTITUTION")}', flush=True)

    ag = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                      seed=0, num_threads=1, training=False, checkpoint=0)
    env = ag.env
    env.reset()
    W = env.control_action_dim + np.shape(env.design_cur_params)[1] + 1

    def zero_action():
        return np.zeros((len(env.robot.bodies), W))

    info = {}
    for _ in range(64):
        *_, info = env.step(zero_action())
        if info.get('stage') == 'execution':
            break
    env.step(zero_action())      # ⚠️ 入った次の step まで進めてから設定する（9-103）

    # ⭐ tick の番号と発火を記録する（インスタンスの属性で包むだけ。本体は変えない）
    st = {'tick': 0, 'hits': []}
    w = env._world                               # ChoreonoidSimWorld（tick と反射はこちら）
    orig_tick, orig_reflect = w._tick, w._reflect_puck_on_walls

    def tick(*a, **k):
        st['tick'] += 1
        return orig_tick(*a, **k)

    def reflect(pre):
        n = orig_reflect(pre)
        if n:
            st['hits'].append(st['tick'])
        return n

    w._tick, w._reflect_puck_on_walls = tick, reflect
    print(f'[tick] ⭐ 包んだ: _world._tick・_world._reflect_puck_on_walls（壁の箱 {len(w._wall_boxes)} 個）', flush=True)

    q0 = np.array(env.data.qpos, dtype=float)
    nj = len(q0) - 2
    rows = []
    for i in range(N_TRIAL):
        y0 = 0.30 + 0.06 * i / N_TRIAL          # 壁（中心の上限 0.45）までの距離をずらす
        q = q0.copy()
        q[:nj] = 0.0
        q[nj], q[nj + 1] = 0.0, y0
        v = np.zeros_like(np.array(env.data.qvel, dtype=float))
        v[nj + 1] = V0
        w._puck_dq_target = None
        env.set_state(q, v)
        st['tick'], st['hits'] = 0, []
        vys = []
        for _ in range(12):
            env.step(zero_action())
            vys.append(float(np.array(env.data.qvel, dtype=float)[nj + 1]))
        if not st['hits']:
            print(f'[tick] y0={y0:.4f} 壁に届かない vy={vys}', flush=True)
            continue
        t = st['hits'][0]
        n_ft = t and (t - 1) // 4                # 発火した env step（0 始まり）
        in_step = (t - 1) % 4 + 1                # その env step の何番目の tick か
        before = vys[n_ft - 1] if n_ft >= 1 else V0
        after2 = vys[min(n_ft + 2, len(vys) - 1)]
        rows.append((in_step, abs(after2 / before)))
        print(f'[tick] y0={y0:.4f} 発火 tick {t}（step {n_ft} の {in_step} 番目）発火 {len(st["hits"])} 回  '
              f'vy 前 {before:+.3f} → 発火した step の末 {vys[n_ft]:+.3f} → 次 {vys[n_ft + 1]:+.3f} → 2 つ後 {after2:+.3f}  '
              f'比 {abs(after2 / before):.2f}', flush=True)

    print('\n[tick] ⭐ 何番目の tick で発火したか → 比（2 step 後の |vy| / 前の |vy|）', flush=True)
    for k in (1, 2, 3, 4):
        r = [x for s, x in rows if s == k]
        print(f'[tick]   {k} 番目: {len(r)} 回  比 {", ".join(f"{x:.2f}" for x in r) or "—"}', flush=True)


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
os._exit(0)
