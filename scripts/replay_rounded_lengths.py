#!/usr/bin/env python3
"""軸6 L-4b: β の best の上腕・前腕の長さを、**部品で作れる最も近い長さへ丸めて**再生し、50 話の exec 合計報酬を出す（読み ㉝、系譜 9-329）。

**なぜ要るか**: β は 57.6〜185.2 mm の連続な長さを学習するが、実物は組み方の系統（9-325、Hinge_Short_Plug＋Frame_Long_Socket、
サーボは connect_point ↔ right-side、噛み合い 8 個以上）で作れる長さしか組めない（隣どうしの隙間 ≤ 1.7 mm）。
丸めても性能が保たれるなら、学習した形はそのまま組める。

**何をするか**: 学習はしない。方策は best のまま、設計の段（attribute_transform）で env に渡る長さの設計変数だけを、
丸めた長さになる値へ置き換える（`set_design_params` を包む）。質量・重心・慣性は β と同じ模型（`part_mass_file`）が長さから作り直す。
方策が見る設計の観測も丸めた値になる（実物を組んだときと同じ）。報酬は `verify_run_replay.py` と同じく env が返す値を
実行の段だけ積む（= log の exec_R_eps と同じ量）。

    cd /userdir/StackelbergPPO_ra      # ⚠️ ブランチの worktree から回す（cfg `pusher_ra15` と長さの模型はブランチにだけある）
    EVAL_RESTORE_DIR=single_run/ra15_pusher_beta [EVAL_EPOCH=best] [EVAL_NUM_EPISODES=50] [ROUND=1] \
    USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
        --no-window --python scripts/replay_rounded_lengths.py > /tmp/l4b.log 2>&1
    # → single_run/<run>/trace/l4b_round1.json（話ごとの報酬・丸める前後の長さ）と l4b_round1.xml（丸めた形の XML）

`ROUND=0` は丸めずに同じ 50 話を回す（対照。丸めの効果と、50 話に増やした効果を分ける）。
組み方の表は `FAMILY_CSV`（既定は main の `data/axis6/irsl_link_all_v2.csv`、git の外。9-325 の `enum_criteria_links.py` の `ALL_OUT`）。
⚠️ 質量は β と同じ 1 次式の模型で、丸めた長さの組み方の実際の質量とは違う（9-325 の残差 2.3 g）。その差は出力に出す。
"""
import os, sys
sys.path.append(os.getcwd())
os.environ.setdefault('USE_CHOREONOID', '1')

# 学習時の物理の環境変数（verify_run_replay.py と同じ。明示的に渡した値は上書きしない）
import json
_le = os.path.join(os.environ['EVAL_RESTORE_DIR'], 'launch_env.json')
LAUNCH_ENV = json.load(open(_le)) if os.path.exists(_le) else None
if LAUNCH_ENV is None:
    print('[l4b] ⚠️ launch_env.json が無い。物理の環境変数は手で渡したものだけ', flush=True)
else:
    for _k, _v in LAUNCH_ENV.items():
        if _k not in os.environ:
            os.environ[_k] = _v
        elif os.environ[_k] != _v:
            print(f'[l4b] ⚠️ {_k}: 学習時 {_v} ／ 再生 {os.environ[_k]}（手で渡した値を使う）', flush=True)
    print(f'[l4b] ⭐ 学習時の環境で再生: {LAUNCH_ENV}', flush=True)

import csv
import numpy as np, yaml, torch
from omegaconf import OmegaConf
from design_opt.utils.config import Config, replay_robot_param_scale
from design_opt.agents.genesis_agent import BodyGenAgent, tensorfy
from design_opt.utils.tools import set_global_seed
from khrylib.robot.xml_robot import normalize_range, denormalize_range

ROUND = os.environ.get('ROUND', '1') == '1'
BODIES = ('11', '111')            # 上腕・前腕（cfg `pusher_ra15` の bone_len の `only`）
FAMILY_CSV = os.environ.get('FAMILY_CSV', '/userdir/StackelbergPPO/data/axis6/irsl_link_all_v2.csv')
FAMILY = dict(Hinge='Hinge_Short_Plug', Frame='Frame_Long_Socket', **{'Frame→サーボ': 'connect_point', 'サーボの点': 'right-side'})
MESH_MIN = 8                      # 9-325 で系統を比べた条件（噛み合い 8 個以上）

rd = os.environ['EVAL_RESTORE_DIR']
_ep = os.environ.get('EVAL_EPOCH', 'best')
epoch = int(_ep) if _ep.isdigit() else _ep           # ⚠️ int で渡さないと epoch_0010.p を探さない
nep = int(os.environ.get('EVAL_NUM_EPISODES', '50'))

# --- 作れる長さの表（長さごとに、噛み合いが多い → 部品が少ない、の順で 1 つ。9-323 と同じ選び方） ---
table = {}
for r in csv.DictReader(open(FAMILY_CSV)):
    if any(r[k] != v for k, v in FAMILY.items()) or int(r['噛み合い']) < MESH_MIN:
        continue
    L = float(r['軸間距離_mm'])
    key = (-int(r['噛み合い']), int(r['部品の数']))
    if L not in table or key < table[L][0]:
        table[L] = (key, r)
LENS = np.array(sorted(table))
_gap = np.diff(LENS).max()
print(f'[l4b] 作れる長さ: {len(LENS)} 種類・{LENS[0]:.1f}〜{LENS[-1]:.1f} mm・隙間の最大 {_gap:.2f} mm'
      f'（9-325: 57.6〜185.2 mm・≤ 1.7 mm）', flush=True)
assert len(LENS) > 100 and _gap <= 1.7, '組み方の表が 9-325 と合わない（FAMILY_CSV を確かめる）'

FLAGS = OmegaConf.create(yaml.safe_load(open(f'{rd}/.hydra/config.yaml')))
OmegaConf.update(FLAGS, 'restore_dir', rd)
cfg = Config(FLAGS, os.getcwd(), rd)
cfg.restore_dir = rd
replay_robot_param_scale(cfg, rd)
cfg.control_prior = False      # Bug 10
cfg.morph_prior = False
torch.set_default_dtype(torch.float64)
set_global_seed(cfg.seed)
agent = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                     seed=cfg.seed, num_threads=1, training=False, checkpoint=epoch)
env = agent.env

idx = [i for i, b in enumerate(env.robot.bodies) if b.name in BODIES]
for i in idx:
    ps = env.robot.bodies[i].param_specs
    # 長さが設計変数の先頭にある形（offset が無い）だけを扱う。違う cfg なら止める
    assert 'bone_len' in ps and 'offset' not in ps, f'体 {env.robot.bodies[i].name}: bone_len が先頭の設計変数でない'
print(f'[l4b] ⭐ 丸める体: {[env.robot.bodies[i].name for i in idx]}（{len(idx)} / {len(BODIES)} 個を捕捉）', flush=True)
assert len(idx) == len(BODIES)

mass_spec = json.load(open(cfg.robot_cfg['part_mass_file'])) if cfg.robot_cfg.get('part_mass_file') else None
log_lens = []      # 話ごとの (丸める前, 丸めた後) mm
_orig_set = env.set_design_params


def _len_of(p, spec):
    """設計の行動 p（[-1, 1]）が作る長さ [m]（xml_robot の set_params と同じ写像）"""
    return denormalize_range(env.robot.map_params(np.array([p]))[0], spec['lb'], spec['ub'])


def _p_of(L, spec):
    """長さ L [m] を作る設計の行動（_len_of の逆）"""
    return env.robot.demap_params(np.array([normalize_range(L, spec['lb'], spec['ub'])]))[0]


def rounded_set(in_design_params):
    params = np.array(in_design_params, dtype=float).copy()
    # ⚠️ 設計の段は `fix_skeleton` で 5 step あり、長さは前の値に足していく（abs_design でない）。
    #   途中で丸めると方策の観測が変わって別の長さへ進む（スモークで 100.7 → 101.5 mm）ので、実行へ移る最後の step だけ丸める
    last = env.cur_t == cfg.skel_transform_nsteps + 1
    row = []
    for i in idx:
        spec = env.robot.bodies[i].param_specs['bone_len']
        L = _len_of(params[i][0], spec) * 1000
        Lr = float(LENS[np.argmin(np.abs(LENS - L))]) if ROUND and last else L
        params[i][0] = _p_of(Lr / 1000, spec) if Lr != L else params[i][0]
        row.append((L, Lr))
    ok = _orig_set(params)
    if not last:
        return ok
    # ⭐ 置き換えが効いたかを、env の中の骨の長さで確かめる（⑤-3-2: 期待値と実測を並べる）
    got = [np.linalg.norm(env.robot.bodies[i].bone_offset) * 1000 for i in idx]
    log_lens.append([(L, Lr, g) for (L, Lr), g in zip(row, got)])
    return ok


env.set_design_params = rounded_set

# log の best（第①層）
best_log = None
import re
for line in open(f'{rd}/log/log_train.txt'):
    m = re.search(r'rewards (-?\d+\.\d+)', line)
    if m:
        best_log = float(m.group(1))

rows, xml = [], None
for e in range(nep):
    state = env.reset()
    tot, n = 0.0, 0
    for _ in range(cfg.skel_transform_nsteps + 2 + 1200):
        in_exec = env.stage == 'execution'
        if in_exec and xml is None:
            xml = env.cur_xml_str
        sv = tensorfy([state])
        if agent.obs_norm is not None:
            sv = agent.normalize_observation(sv)
        with torch.no_grad():
            a = agent.policy_net.select_action(sv, mean_action=True).numpy().astype(np.float64)
        state, r, done, trunc, _ = env.step(a)
        if in_exec:
            tot += float(r); n += 1
        if done or trunc:
            break
    rows.append((tot, n))
    print(f'[l4b] 話 {e+1}/{nep}: {tot:9.2f}  ({n} steps)  長さ '
          + ' / '.join(f'{L:.1f}→{Lr:.1f}（env {g:.1f}）' for L, Lr, g in log_lens[-1]), flush=True)

R = np.array([t for t, _ in rows])
err = max(abs(Lr - g) for ls in log_lens for _, Lr, g in ls)
print(f'\n[l4b] {rd}  epoch={epoch}  {"丸めた" if ROUND else "丸めない（対照）"}  {nep} 話')
print(f'[l4b] exec 合計報酬: 中央値 {np.median(R):.2f}・平均 {R.mean():.2f}・範囲 [{R.min():.2f}, {R.max():.2f}]')
if best_log is not None:
    print(f'[l4b] log の best: {best_log:.2f}')
print(f'[l4b] 置き換えた長さと env の骨の長さの差の最大: {err:.3f} mm' + ('' if err < 0.05 else '  ⛔ 置き換えが効いていない'))
comp = {}
for k, i in enumerate(idx):
    L, Lr, _ = log_lens[-1][k]
    r = table[Lr][1] if ROUND else None
    line = f'[l4b] 体 {env.robot.bodies[i].name}: 学習 {L:.1f} mm → {Lr:.1f} mm（差 {Lr - L:+.1f}）'
    if r is not None:
        line += f'  組み方: 段 {r["段"]}・噛み合い {r["噛み合い"]}・部品 {r["部品の数"]} 個'
        if mass_spec:
            b = mass_spec['bodies'][env.robot.bodies[i].name]
            model = (b['struct']['mass'] + mass_spec['slope_kg_per_m'] * (Lr / 1000 - b['L0'])) * 1000
            line += f'  構造の質量: 模型 {model:.1f} g ／ この組み方 {float(r["質量_構造_g"]):.1f} g'
        comp[env.robot.bodies[i].name] = r
    print(line)

d = os.path.join(rd, 'trace')
os.makedirs(d, exist_ok=True)
tag = f'l4b_round{int(ROUND)}'
json.dump(dict(run=rd, epoch=str(epoch), round=ROUND, episodes=nep, best_log=best_log,
               ep_reward=R.tolist(), ep_steps=[n for _, n in rows], median=float(np.median(R)),
               lengths_mm=log_lens, max_replace_err_mm=err, composition=comp,
               launch_env=LAUNCH_ENV, family=FAMILY, mesh_min=MESH_MIN),
          open(os.path.join(d, tag + '.json'), 'w'), ensure_ascii=False, indent=1)
if xml:
    open(os.path.join(d, tag + '.xml'), 'w').write(xml)
print(f'[l4b] → {d}/{tag}.json・{tag}.xml')
sys.stdout.flush()
os._exit(0)   # Choreonoid の Qt ループが残るため（Bug 26 と同型）
