#!/usr/bin/env python3
"""⭐⭐ **腕を台本で壁へ突っ込ませて、ブロックが効くかを測る**（系譜 9-205）。

⛔⛔⛔ **なぜ台本が要るか（9-204）**: 方策の再生に頼ったプローブは、
**5 つの run すべてで腕が壁の箱に一度も入らず、ブロックを一度も試せなかった。**

| restore | 腕の到達 |
|---|---|
| `hockey_bank5` | リンクに入らない（9-196） |
| `hockey_wall2` | 壁 0 個（xml が 9-98 以前） |
| `hockey_bank` | x 最大 0.032（壁は x∈[0.15,1.95]） |
| `hockey_bank2` | x 最大 0.000 |
| `hockey_bank6` | \\|y\\| 最大 0.348（側壁の内面 0.50 に届かない） |

⭐ **本プローブは方策を使わない。**⭐ **FK で「壁のすぐ内側」に先端が来る関節角を探し、
そこへ `set_state` で置いて、壁へ向かう関節速度を与える。**

⚠️ **9-103 の罠を踏まないこと**:
  ⛔ `env.data.qpos[...] = x` は**届かない**。`env.set_state(qpos, qvel)` が唯一の入口
  ⛔ `transit_execution()` は execution へ入る瞬間に `reset_state(True)` で上書きする
     → **入った「次の step」まで進めてから設定する**

⭐ **既知の失敗を再現できるか**を先に見る: `ARM_SWEEP_OFF=1` で旧実装（掃過 2 点）に戻す。
⛔ **旧でもすり抜けなければ、台本が壁を突いていない。**

    EVAL_RESTORE_DIR=single_run/hockey_bank6 USE_CHOREONOID=1 OMP_NUM_THREADS=1 \\
    HOCKEY_ARM_BLOCK=1 [ARM_SWEEP_OFF=1] \\
    timeout -k 30 1800 /choreonoid_ws/install/bin/choreonoid --no-window \\
    --python scripts/probe_arm_block_scripted.py > out.txt 2>&1

⛔⛔⛔ **`HOCKEY_ARM_BLOCK=1` だけでは壁の箱が作られない**（2026-10-02 発見）:

```python
self._wall_e = float(os.environ.get('HOCKEY_WALL_RESTITUTION', '0') or 0)
if self._wall_e <= 0:
    return          # ⛔ 壁の箱を作らずに帰る
```

⭐ **したがって `HOCKEY_WALL_RESTITUTION` を必ず一緒に指定する。**
⚠️ **腕ブロックだけを有効にした run があれば、ブロックは一切働いていない。**

⚠️ Bug 43: **パイプへ繋がず必ずファイルへ。**Choreonoid は起動だけで約 105 秒。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

RESTORE = os.environ.get('EVAL_RESTORE_DIR', 'single_run/hockey_bank6')
STEPS = int(os.environ.get('PROBE_STEPS', '120'))
WYAW = float(os.environ.get('PROBE_YAW_VEL', '6.0'))   # ヨーの角速度 [rad/s]


def main():
    import torch
    import yaml
    from omegaconf import OmegaConf
    from design_opt.utils.config import Config
    from design_opt.agents.genesis_agent import BodyGenAgent

    d = OmegaConf.to_container(
        OmegaConf.create(yaml.safe_load(open(f'{RESTORE}/.hydra/config.yaml'))), resolve=True)
    d.pop('restore_dir', None)
    # ⚠️ 第3引数は restore_dir。⛔ 架空のパスを渡すと join() で NoneType になる
    #   （2026-10-02 に踏んだ）。⭐ `probe_arm_wall_block.py`（動いている実例）に合わせる。
    cfg = Config(OmegaConf.create(d), os.getcwd(), RESTORE)
    cfg.restore_dir = RESTORE
    cfg.control_prior = cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)

    ag = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                      seed=0, num_threads=1, training=False, checkpoint='best')
    env = ag.env
    env.reset()
    W = env.control_action_dim + np.shape(env.design_cur_params)[1] + 1

    def zero_action():
        return np.zeros((len(env.robot.bodies), W))

    # ── execution へ入り、その次の step まで進める（9-103）──
    info = {}
    for _ in range(64):
        *_, info = env.step(zero_action())
        if info.get('stage') == 'execution':
            break
    env.step(zero_action())

    # ── 壁の箱と腕の連鎖を掴む ─────────────────────────────
    inner = None
    stack, seen = [env], []
    for _ in range(6):
        nxt = []
        for o in stack:
            if o is None or id(o) in seen:
                continue
            seen.append(id(o))
            if hasattr(o, '_arm_caps'):
                inner = o
                break
            nxt += [getattr(o, a, None) for a in ('env', '_world', 'unwrapped', 'wrapped')]
        if inner is not None:
            break
        stack = nxt
    if inner is None:
        raise RuntimeError('⛔⛔ `_arm_caps` を持つ層が見つからない')

    boxes = list(getattr(inner, '_wall_boxes', []) or [])
    caps = dict(getattr(inner, '_arm_caps', {}) or {})
    eb = getattr(inner, '_arm_body', None)
    sweep = getattr(inner, '_ARM_SWEEP', None)
    print(f'[scr] ⭐ 壁 {len(boxes)} 個: {[b[0] for b in boxes]}', flush=True)
    print(f'[scr] ⭐ 腕のカプセル本体: {len(caps)} / {eb.numLinks if eb else 0} リンク', flush=True)
    print(f'[scr] ⭐ 掃過の点数 _ARM_SWEEP = {sweep}', flush=True)
    if not boxes or not caps or sweep is None:
        raise RuntimeError('⛔⛔ 壁・カプセル・掃過のどれかが取れない。測っても意味が無い')

    # ── FK で「壁のすぐ内側」に先端が来る関節角を探す ───────────
    xml = getattr(env, 'cur_xml_str', None) or getattr(env, 'init_xml_str', '')
    chain = env._parse_arm_chain(xml) if hasattr(env, '_parse_arm_chain') else None
    if chain is None:
        from design_opt.envs.pusher import PusherEnv
        chain = PusherEnv._parse_arm_chain(env, xml)

    # 目標: 先端が x∈[0.6,1.2] かつ |y|∈[0.30,0.45]（壁の内面 0.50 のすぐ手前）
    best = None
    for th0 in np.linspace(-np.pi, np.pi, 181):
        for th1 in np.linspace(-1.5, 1.5, 61):
            segs = env._fk_points(chain, [th0, th1, 0.0, 0.0])
            t = segs[-1][1]
            if not (0.6 <= t[0] <= 1.2 and 0.30 <= abs(t[1]) <= 0.45):
                continue
            score = abs(abs(t[1]) - 0.42)            # 壁にできるだけ近い
            if best is None or score < best[0]:
                best = (score, th0, th1, t)
    if best is None:
        raise RuntimeError('⛔⛔ 壁のすぐ内側に先端を置ける関節角が見つからない。'
                           '⚠️ この形態では台本を組めない')
    _s, th0, th1, tip = best
    print(f'[scr] ⭐⭐ 台本の初期姿勢: ヨー {np.degrees(th0):+.1f}° / 第2関節 {np.degrees(th1):+.1f}° '
          f'→ 先端 ({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f})', flush=True)

    q = np.array(env.data.qpos, dtype=float)
    v = np.zeros_like(np.array(env.data.qvel, dtype=float))
    q[0], q[1], q[2], q[3] = th0, th1, 0.0, 0.0
    # ⭐ 壁へ向かう向きにヨーを回す（先端の y が増える向き）
    v[0] = WYAW if tip[1] >= 0 else -WYAW
    env.set_state(q, v)
    print(f'[scr] ⭐ ヨー角速度 {v[0]:+.2f} rad/s を与えた', flush=True)

    # ── 走らせて測る ───────────────────────────────────
    inside, outside_max, n = 0, 0.0, 0
    reach_y, reach_x = 0.0, -9.9
    fires0 = getattr(inner, '_arm_blocks', 0)
    for _ in range(STEPS):
        # ⛔⛔ **関節ダンピングと速度クランプが 1 step で速度を殺す。**
        #   ⭐ **毎 step 与え直さないと壁まで届かない**（2026-10-02 に実測: |y| 0.420 → 0.471 止まり）。
        qn = np.array(env.data.qpos, dtype=float)
        vn = np.zeros_like(np.array(env.data.qvel, dtype=float))
        vn[0] = WYAW if tip[1] >= 0 else -WYAW
        env.set_state(qn, vn)
        *_, info = env.step(zero_action())
        if info.get('stage') != 'execution':
            break
        n += 1
        eb = getattr(inner, '_arm_body', None)          # ⭐ 毎 step 取り直す（9-202 ①）
        bx = getattr(env, '_body_xpos', None)           # ⭐ worker の応答（9-204 ④）
        bm = getattr(env, '_body_xmat', None)
        if not bx:
            raise RuntimeError('⛔⛔ `_body_xpos` が取れない')
        for k in range(eb.numLinks):
            nm = eb.link(k).name
            if nm not in bx:
                continue
            o = np.asarray(bx[nm], dtype=float)
            R = np.asarray(bm[nm], dtype=float).reshape(3, 3)
            c = caps.get(nm)
            pts = [o[:2]] if c is None else [(o + R @ c[0])[:2], (o + R @ c[1])[:2]]
            for p in pts:
                for _nm2, lo, hi in boxes:
                    if np.all(p >= lo[:2]) and np.all(p <= hi[:2]):
                        inside += 1
                if 0.15 <= p[0] <= 1.95:
                    outside_max = max(outside_max, abs(p[1]) - 0.90)
                reach_y = max(reach_y, abs(p[1]))
                reach_x = max(reach_x, p[0])
    fires = getattr(inner, '_arm_blocks', 0) - fires0

    print(f'\n[scr] ⭐ step数 {n}')
    print(f'[scr] ⭐⭐ 腕の到達: |y| 最大 {reach_y:.3f} m / x 最大 {reach_x:.3f} m')
    print(f'[scr]    側壁の内面 |y|=0.50 ・ 外面 |y|=0.90 ・ 箱の x 範囲 [0.15, 1.95]')
    print(f'[scr] ⭐ ブロックの発火回数: {fires}')
    print(f'[scr] ⭐⭐ 腕が壁の中の標本: {inside}')
    print(f'[scr] ⭐⭐⭐ 側壁の外面を越えた最大（x∈[0.15,1.95] の点のみ）: {outside_max*1000:+.1f} mm')
    # ⭐⭐ **判定の順序が大事**（2026-10-02）。
    #   ⛔ 「|y| が内面 0.50 に届かない」を先に見ると、**ブロックが止めた場合も
    #     「台本が届いていない」と誤読する。**⭐ **発火回数を先に見る。**
    adv = reach_y - abs(tip[1])      # 初期姿勢からどれだけ壁へ進めたか
    print(f'[scr] ⭐⭐ 初期姿勢からの前進: {adv*1000:+.1f} mm（初期 |y|={abs(tip[1]):.3f}）')
    if fires > 0 and outside_max <= 0.001 and inside == 0:
        print('[scr] ⭐⭐⭐ **ブロックが効いている。**'
              f'{fires} 回発火し、腕は壁の中へ一度も入らず外へも抜けていない')
    elif fires == 0 and reach_y < 0.50:
        print('[scr] ⛔⛔ **台本が壁まで届いておらず、ブロックも発火していない。**'
              'PROBE_YAW_VEL を上げるか、初期姿勢の条件を見直すこと')
    elif outside_max > 0.001:
        print('[scr] ⛔⛔ **すり抜けている。**掃過判定が効いていない')
    elif inside > 0:
        print('[scr] ⚠️ **壁の中には入ったが外へは抜けていない。**'
              '⭐ ブロックは効いているが食い込みは許している')
    else:
        print('[scr] ⭐⭐ **壁の手前で止まっている。**ブロックが効いている')
    return 0


main()      # ⚠️ Bug 43: Choreonoid は sys.exit(0) を受け付けない
