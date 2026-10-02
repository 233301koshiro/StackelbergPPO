"""⭐⭐ **腕の壁貫通ブロックが本当に効いているかを測る**（2026-09-30・系譜 9-189）。

⛔⛔ **9-185 では約 14 GPU 時間を崩れた前提で使った。**
有効化ログも発火回数も出ていたが、⛔ **「何リンクを見ているか」を出していなかった**ので
末端リンクが抜けていたことに気づけなかった（§5-2 ⑤-3-2）。

⭐ **このプローブは「何に対して効いたか」を必ず出す**:
  ① 捕捉したリンクの**名前**と本数
  ② 壁の**名前**と範囲
  ③ **ブロックの発火回数**
  ④ ⭐⭐ **腕が壁の中に入った標本数**と、⭐⭐⭐ **壁の外面を越えた最大距離**（＝すり抜けの証拠）

⭐ **既知の失敗を再現できるか**を先に確かめる（`--reproduce`）。
`ARM_SWEEP_OFF=1` で掃過判定を切ると**旧実装（2 点だけ）**に戻るので、
⛔ **すり抜けが再現できなければプローブが壁を見ていない。**

⚠️ **出力はファイルへリダイレクトすること**（Bug 43。パイプだと timeout 時に 1 行も残らない）。
⚠️ **Choreonoid は起動だけで約 105 秒**かかるので timeout は 1200 秒以上。

起動:
  XML=e2e_hockey_easy USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
    HOCKEY_WALL_RESTITUTION=0.75 HOCKEY_ARM_BLOCK=1 \
    timeout -k 30 1800 /choreonoid_ws/install/bin/choreonoid --no-window \
    --python scripts/probe_arm_wall_block.py > out.txt 2>&1
"""
import os, sys
sys.path.append(os.getcwd())
import numpy as np, yaml, torch
from omegaconf import OmegaConf
from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent, tensorfy
from design_opt.utils.tools import set_global_seed

RESTORE = os.environ.get('EVAL_RESTORE_DIR', 'single_run/hockey_bank2')
STEPS = int(os.environ.get('PROBE_STEPS', '600'))


def main():
    FLAGS = OmegaConf.create(yaml.safe_load(open(f'{RESTORE}/.hydra/config.yaml')))
    d = OmegaConf.to_container(FLAGS, resolve=True); d.pop('restore_dir', None)
    cfg = Config(OmegaConf.create(d), os.getcwd(), RESTORE)
    cfg.restore_dir = RESTORE
    cfg.control_prior = cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)
    set_global_seed(cfg.seed)
    agent = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                         seed=cfg.seed, num_threads=1, training=False, checkpoint='best')
    env = agent.env
    # ⭐⭐ `_arm_caps` を持つ層を**探して**掴む。
    #   ⛔ `env.env` を決め打ちにして **0 リンク・0 壁**を見た（2026-09-30）。
    #   ⭐ 実体は `ChoreonoidEnv._world`（= `ChoreonoidSimWorld`）にある。
    #   ⚠️ **`ChoreonoidEnv` と `ChoreonoidSimWorld` の取り違えは 9-184 でも踏んだ。**
    inner, seen = None, []
    stack = [env]
    for _ in range(6):
        nxt = []
        for o in stack:
            if o is None or id(o) in seen:
                continue
            seen.append(id(o))
            if hasattr(o, '_arm_caps'):
                inner = o
                break
            nxt += [getattr(o, '_world', None), getattr(o, 'env', None),
                    getattr(o, 'unwrapped', None)]
        if inner is not None:
            break
        stack = nxt
    if inner is None:
        inner = env

    # ── ① 何を掴んでいるか（§5-2 ⑤-3-2）──────────────────────────────
    caps = getattr(inner, '_arm_caps', {})
    eb = getattr(inner, '_arm_body', None)
    nlink = eb.numLinks if eb is not None else 0
    print(f'[probe] ⭐ 捕捉した腕のカプセル本体: {len(caps)} / {nlink} リンク')
    print(f'[probe]    名前: {sorted(caps.keys())}')
    print(f'[probe]    全リンク: {[eb.link(k).name for k in range(nlink)] if eb else []}')
    boxes = getattr(inner, '_wall_boxes', [])
    print(f'[probe] ⭐ 壁 {len(boxes)} 個:')
    for nm, lo, hi in boxes:
        print(f'[probe]    {nm:14s} x[{lo[0]:+.3f},{hi[0]:+.3f}] y[{lo[1]:+.3f},{hi[1]:+.3f}]')
    print(f'[probe] ⭐ 掃過の点数 _ARM_SWEEP = {getattr(inner, "_ARM_SWEEP", "なし")}')
    # ⭐⭐⭐ **測れていないなら判定を拒否する**（`check_cube_penetration.py` と同じ流儀）。
    #   ⛔⛔ **これが無いと「0 リンク・0 壁」で「すり抜けなし」という偽の合格を出す**
    #   （2026-09-30 に実際に出した）。⭐ **9-185 の 14 GPU 時間はこの型で消えた。**
    if nlink == 0 or len(boxes) == 0 or getattr(inner, '_ARM_SWEEP', None) is None:
        print('[probe] ⛔⛔ **判定を拒否する。**腕または壁を掴めていない '
              f'(リンク {nlink} / 壁 {len(boxes)} / 掃過 {getattr(inner, "_ARM_SWEEP", None)})。'
              f'  env の層: {type(env).__name__} → {type(inner).__name__}')
        print('[probe] 完了')
        return
    if len(caps) < nlink - 1:
        print('[probe] ⛔ **捕捉できていないリンクがある。**9-184 の再発')

    # ── 1 エピソード走らせて腕の位置を全部見る ──────────────────────
    state = env.reset()
    inside, outside_max, n = 0, 0.0, 0
    reach_y, reach_x = 0.0, -9.9   # ⭐ 9-96: 動く物体の極値を必ず出す
    for _ in range(STEPS):
        sv = tensorfy([state])
        if agent.obs_norm is not None:
            sv = agent.normalize_observation(sv)
        with torch.no_grad():
            a = agent.policy_net.select_action(sv, mean_action=True).numpy().astype(np.float64)
        state, _r, done, _, _ = env.step(a)
        # ⛔⛔⛔ **2026-10-02: body は設計フェーズごとに作り直される。**
        #   ⭐ ループ前に掴んだ `eb` は**古い実体**で、位置が更新されず全部ゼロになる。
        #   ⚠️ **`腕のカプセル本体: 4 / 5 リンクを捕捉` が何度も出る**のがその証拠だった。
        #   ⭐ **毎 step 取り直す。**
        eb = getattr(inner, '_arm_body', None)
        if env.stage != 'execution' or eb is None:
            if done: break
            continue
        n += 1
        nlink = eb.numLinks        # ⭐ 作り直しでリンク数が変わることがある
        # ⛔⛔⛔ **2026-10-02: これが無いと `lk.p` が更新されず全部ゼロになる。**
        #   ⭐ `_block_arm_on_walls` は位置を読む前に必ずこれを呼んでいる。
        #   ⛔⛔ **プローブが呼んでいなかったので、貫通を一度も検出できていなかった**
        #     （「腕が壁の中の標本 0」「すり抜けなし」は偽の合格だった）。
        #   ⭐ **発火回数 2070 なのに腕の位置が全部 0.000 という矛盾で気づけた**（9-96）。
        eb.calcForwardKinematics()
        for k in range(nlink):
            lk = eb.link(k)
            o = np.asarray(lk.p, dtype=float); R = np.asarray(lk.R, dtype=float).reshape(3, 3)
            c = caps.get(lk.name)
            pts = [o[:2]] if c is None else [(o + R @ c[0])[:2], (o + R @ c[1])[:2]]
            for q in pts:
                for _nm, lo, hi in boxes:
                    if np.all(q >= lo[:2]) and np.all(q <= hi[:2]):
                        inside += 1
                # ⭐⭐ すり抜けの証拠: **側壁の外面を越えた距離**
                # ⛔⛔ **2026-10-02 訂正: x を見ていなかった。**
                #   ⭐ 壁は x∈[0.15, 1.95] にしかない。⛔ **x=0.03 で |y|=1.38 は
                #   「すり抜け」ではなく「壁の手前で横に振れている」だけ**だった。
                #   ⚠️ **これで「すり抜けている」と誤読しかけた。**
                if 0.15 <= q[0] <= 1.95:
                    outside_max = max(outside_max, abs(q[1]) - 0.90)
                # ⭐⭐ 9-96: **動く物体の極値を必ず出す。**
                #   ⛔⛔ これが無いと「腕が壁に近づきもしていない」ことに気づけず、
                #     **「すり抜けなし」を合格と読んでしまう**（2026-10-02 に実際に踏んだ）。
                reach_y = max(reach_y, abs(q[1]))
                reach_x = max(reach_x, q[0])
        if done: break

    fires = getattr(inner, '_arm_blocks', 0)
    print(f'[probe] ⭐ step数 {n}')
    # ⭐⭐ **腕がそもそも壁に近づいたか**を先に出す（9-96）
    print(f'[probe] ⭐⭐ 腕の到達: |y| 最大 {reach_y:.3f} m / x 最大 {reach_x:.3f} m')
    print(f'[probe]    側壁の内面 |y|=0.50 ・ 外面 |y|=0.90 ・ リンクの x 範囲 [0.15, 1.95]')
    if reach_y < 0.45:
        print('[probe] ⛔⛔ **腕が側壁に近づいてすらいない。**'
              'このプローブはブロックを一度も試していない。'
              '⭐ **腕が壁へ到達する run を `EVAL_RESTORE_DIR` に指定すること。**')
    print(f'[probe] ⭐ ブロックの発火回数: {fires}')
    print(f'[probe] ⭐⭐ 腕が壁の中の標本: {inside}')
    print(f'[probe] ⭐⭐⭐ 側壁の外面を越えた最大（⭐ **x∈[0.15,1.95] の点のみ**）: {outside_max*1000:+.1f} mm')
    if outside_max > 0.001:
        print('[probe] ⛔⛔ **すり抜けている。**掃過判定が効いていない')
    else:
        print('[probe] ⭐⭐ **すり抜けなし。**')
    print('[probe] 完了')


main()   # ⚠️ sys.exit を使わない（Bug 43。Choreonoid が受け付けない）
