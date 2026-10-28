#!/usr/bin/env python3
"""**共設計の形が「いつ」「どちら向きに」決まったかを、学習途中の形から測る**（実験系譜 9-314、仮説 H1）。

H1: 共設計の形は、その時点の方策で得になる向きへ動く。到達は最初の step から距離を毎 step 引くので形が早く決まり、
押し・ホッケーは対象に触れるまで形に向きが付かないので遅く決まる、と読む（事前登録は系譜 9-314）。

測るもの（設計変数ごとに、範囲の中の位置 p ＝（値 − 下限）÷（上限 − 下限）。0 が下限、1 が上限）:
  - 骨の長さ（cfg に body_params.offset がある縦型だけ）: 骨を初期の骨の向きへ射影した長さ ÷ 初期の長さを、rel_frac の範囲で
  - 太さ（カプセルの半径）・ギア比: cfg の geom_params.size・actuator_params.gear の範囲で
  - 一度も動かない量（根元の台座の球など）は設計変数ではないとして外す（全時点で初期と 1e-5 以内）
時点: 初期（XML）・ep10・ep50・ep100・ep150（trace_mid/ep*.npz）・best（trace/arm_trace.npz）。どれも話 0 の形（形は話ごとに同じ）。
「決まった時期」: それより後のすべての時点で、どの変数も best の位置から 0.1 以内に収まる最初の時点。
「報酬の半分」: exec_R_eps が ep0 から最大までの改善の半分に初めて達した epoch。

    python3 scripts/design_trajectory.py            # 9-314 の本線 40 本と、読み ㉔㉕㉖ の判定
    python3 scripts/design_trajectory.py --all      # trace_mid/ のある全 run（取り下げ・置き換えた run を含む。確かめ用）
    python3 scripts/design_trajectory.py <run> ...  # 指定した run だけ（組は名前から推定）

⚠️ 途中の形が無い run は「軌跡が足りない」と出して飛ばす。作るには record_arm_trace.py を EVAL_CHECKPOINT=<epoch>・
   TRACE_OUT=single_run/<run>/trace_mid/ep00NN.npz で回す（watch_midrun.sh と同じ）。
⚠️ 機序（なぜその向きか）は測らない。形を固定した対照が無いので、言えるのは時期と向きがタスクと対応するかまで。
"""
import re
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
MIDS = [10, 50, 100, 150]
TOL_SETTLE = 0.1
EARLY = 50

# 系譜 9-314 の事前登録の本線（修論が今使う組）。組の名前は読み ㉔㉕ の単位
MAIN = {
    '到達': ['tripo_pj_long_phys', 'tripo_pj_long_phys_s2_ws', 'tripo_pj_mid_phys', 'tripo_pj_mid_phys_s2_ws',
             'tripo_pj_short_phys_cg', 'tripo_pjd_prox_phys', 'tripo_pjd_prox_phys_s2_ws', 'tripo_pjd_dist_phys_cg',
             'tripo_pjd_dist_phys_cg_s2_ws', 'tripo_pjr199_phys', 'tripo_pjr199_phys_s2_ws', 'e2e_a1_reach_phys',
             'e2e_a1_reach_phys_s2_ws', 'e2e_a1_fix3_reach_phys', 'e2e_a1_fix3_reach_phys_s2_ws',
             'e2e_a1v_reach_phys', 'e2e_a1v_reach_phys_s2_ws',
             'e2e_a1v_obspen_reach_phys_sub5', 'e2e_a1v_obspen_reach_phys_sub5_s2',
             'e2e_a1v_obspen_reach_phys_long', 'e2e_a1v_obspen_reach_phys_long_s2'],
    '押し': ['tripo_pjp_long_phys', 'tripo_pjp_long_phys_s2_ws', 'tripo_pjp_mid_phys', 'tripo_pjp_mid_phys_s2_ws',
             'tripo_pjp_short_phys_cg', 'tripo_pjdp_prox_phys', 'tripo_pjdp_prox_phys_s2_ws', 'tripo_pjdp_dist_phys_cg',
             'tripo_pjdp_dist_phys_s2_ws', 'tripo_pjr211p_phys', 'tripo_pjr211p_phys_s2_ws', 'e2e_a1_pusher_phys',
             'e2e_a1_pusher_phys_s2', 'e2e_a1_fix3_pusher_phys', 'e2e_a1_fix3_pusher_phys_s2',
             'e2e_a1v_pusher_phys', 'e2e_a1v_pusher_phys_s2_ws'],
    'ホッケー': ['hockey_bank9_phys_g10_blk3d_fs_ax', 'hockey_bank9_phys_g10_blk3d_fs_ax_s2'],
}


def guess_group(run, overrides):
    if 'hockey' in run:
        return 'ホッケー'
    return '到達' if 'use_reach=true' in overrides else '押し'


def xml_initial(xml_path, names):
    """初期の骨のベクトル・半径・ギア比を body 名ごとに返す。"""
    root = ET.parse(xml_path).getroot()
    gear_of = {m.get('joint'): float(m.get('gear')) for m in root.iter('motor') if m.get('gear')}
    bone, rad, gear = {}, {}, {}
    for n in names:
        b = root.find(f".//body[@name='{n}']")
        if b is None:
            continue
        ch = b.find('body')
        g = next((x for x in b.findall('geom') if x.get('type') == 'capsule'), None)
        if ch is not None:
            bone[n] = np.array([float(v) for v in ch.get('pos').split()])
        elif g is not None and g.get('fromto'):
            v = [float(x) for x in g.get('fromto').split()]
            bone[n] = np.subtract(v[3:], v[:3])
        if g is not None:
            rad[n] = float(g.get('size').split()[0])
        j = b.find('joint')
        if j is not None and j.get('name') in gear_of:
            gear[n] = gear_of[j.get('name')]
    return bone, rad, gear


def reward_half(log_path):
    ep, R = [], []
    for line in open(log_path, encoding='utf-8', errors='replace'):
        m = re.match(r'^\[[^\]]*\]\s+(\d+)\t.*exec_R_eps\s+(-?[\d.]+)', line)
        if m:
            ep.append(int(m.group(1)))
            R.append(float(m.group(2)))
    if len(R) < 2:
        return None, None
    R = np.array(R)
    thr = R[0] + 0.5 * (R.max() - R[0])
    i = int(np.argmax(R >= thr))
    return ep[i], ep


def best_epoch(log_path):
    last_ep, best = None, None
    for line in open(log_path, encoding='utf-8', errors='replace'):
        m = re.match(r'^\[[^\]]*\]\s+(\d+)\t', line)
        if m:
            last_ep = int(m.group(1))
        if 'save best checkpoint' in line and last_ep is not None:
            best = last_ep
    return best


def analyze(run):
    D = ROOT / 'single_run' / run
    ov = (D / '.hydra' / 'overrides.yaml').read_text(encoding='utf-8')
    cfg_name = re.search(r'cfg=([A-Za-z_0-9]+)', ov).group(1)
    xml_name = re.search(r'^xml_name:\s*(\S+)', (D / '.hydra' / 'config.yaml').read_text(encoding='utf-8'), re.M).group(1)
    rob = yaml.safe_load(open(ROOT / 'design_opt' / 'cfg' / f'{cfg_name}.yml', encoding='utf-8'))['robot']
    off = (rob.get('body_params') or {}).get('offset')
    s_lb, s_ub = rob['geom_params']['size']['lb'], rob['geom_params']['size']['ub']
    g_lb, g_ub = rob['actuator_params']['gear']['lb'], rob['actuator_params']['gear']['ub']

    paths = [D / 'trace_mid' / f'ep{e:04d}.npz' for e in MIDS] + [D / 'trace' / 'arm_trace.npz']
    miss = [p.relative_to(D) for p in paths if not p.exists()]
    if miss:
        return {'run': run, 'missing': [str(m) for m in miss]}
    snaps = []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        names = [str(s) for s in z['body_names']]
        snaps.append((names, z['ep_bone'].astype(float)[0], z['ep_geom'].astype(float)[0], z['ep_gear'].astype(float)[0]))
    names = snaps[-1][0]
    bone0, rad0, gear0 = xml_initial(ROOT / 'assets' / 'mujoco_envs' / f'{xml_name}.xml', names)

    # 変数ごとの時系列 [初期, ep10, 50, 100, 150, best]
    series = {}
    for k, n in enumerate(names):
        if off is not None and n not in ('0', '1') and n in bone0 and np.linalg.norm(bone0[n]) > 1e-6:
            L0 = np.linalg.norm(bone0[n]); u = bone0[n] / L0
            lo, hi = off['rel_frac'][0], off['rel_frac'][1]
            vals = [1.0] + [float(np.dot(s[1][k], u)) / L0 for s in snaps]
            series[f'長さ {n}'] = [(v - lo) / (hi - lo) for v in vals]
        if n in rad0:
            vals = [rad0[n]] + [s[2][k] for s in snaps]
            series[f'太さ {n}'] = [(v - s_lb) / (s_ub - s_lb) for v in vals]
        if n in gear0:
            vals = [gear0[n]] + [s[3][k] for s in snaps]
            series[f'ギア {n}'] = [(v - g_lb) / (g_ub - g_lb) for v in vals]
    # 台座の球（body 0）は設計変数でない。ほかに一度も動かない量は外すが、下限・上限に張り付いたまま動かない量は残す
    #   （平面の肩のギア比は初期値 150 が上限で、最後まで 150。外すと「掴んだ変数」から黙って消える）
    def keep(k, v):
        if k.endswith(' 0') or not np.isfinite(v).all():
            return False
        moved = np.max(np.abs(np.subtract(v, v[0]))) > 1e-5
        return moved or min(abs(v[0]), abs(v[0] - 1)) < 1e-3
    series = {k: v for k, v in series.items() if keep(k, v)}

    log = D / 'log' / 'log_train.txt'
    bep = best_epoch(log)
    times = [0] + MIDS + [bep]
    P = np.array(list(series.values()))           # 変数 × 時点
    dev = np.abs(P - P[:, -1:]).max(axis=0)       # 時点ごとの best からの最大のずれ
    settle_i = next(i for i in range(len(times)) if (dev[i:] <= TOL_SETTLE).all())
    last_var = list(series)[int(np.argmax(np.abs(P[:, settle_i - 1] - P[:, -1])))] if settle_i > 0 else '-'
    half, _ = reward_half(log)
    dlen = [v[-1] - v[0] for k, v in series.items() if k.startswith('長さ')]
    return {'run': run, 'cfg': cfg_name, 'series': series, 'times': times, 'n': {t: sum(k.startswith(t) for k in series) for t in ('長さ', '太さ', 'ギア')},
            'settle': times[settle_i], 'last_var': last_var, 'half': half, 'best_ep': bep,
            'dlen': float(np.mean(dlen)) if dlen else None,
            'dmean': {t: float(np.mean([v[-1] - v[0] for k, v in series.items() if k.startswith(t)])) if any(k.startswith(t) for k in series) else None
                      for t in ('長さ', '太さ', 'ギア')}}


def fmt(x, f='{:+.2f}'):
    return '  -  ' if x is None else f.format(x)


def report(groups):
    res = {}
    for g, runs in groups.items():
        print(f'\n=== {g}（{len(runs)} 本）')
        print(f'  {"run":40s} 掴んだ変数（長・太・ギ）  決まった ep  最後まで動いた変数   報酬の半分 ep  Δ長さ   Δ太さ   Δギア')
        res[g] = []
        for r in runs:
            a = analyze(r)
            if 'missing' in a:
                print(f'  {r:40s} ⚠️ 軌跡が足りない: {", ".join(a["missing"])}')
                continue
            res[g].append(a)
            n = a['n']
            print(f'  {r:40s} {n["長さ"]}・{n["太さ"]}・{n["ギア"]:<14} {a["settle"]:>5}       {a["last_var"]:14s} {a["half"]!s:>8}     '
                  f'{fmt(a["dmean"]["長さ"])}  {fmt(a["dmean"]["太さ"])}  {fmt(a["dmean"]["ギア"])}')
    return res


def verdict(res):
    print('\n=== 読みの判定（系譜 9-314 の事前登録）')
    for g, rs in res.items():
        if not rs:
            continue
        early = sum(a['settle'] <= EARLY for a in rs)
        want_early = g == '到達'
        hit = early if want_early else len(rs) - early
        need = len(rs) if len(rs) <= 2 else int(np.ceil(0.75 * len(rs)))
        print(f'  ㉔ {g}: ep{EARLY} までに決まった {early} / {len(rs)}（予想: {"早い" if want_early else "遅い"}）→ '
              f'{"✅ 当たり" if hit >= need else "⛔ 外れ"}（必要 {need} 本）')
    for g, rs in res.items():
        L = [a for a in rs if a['dlen'] is not None]
        if not L:
            continue
        down = sum(a['dlen'] < 0 for a in L)
        want_down = g == '到達'
        hit = down if want_down else len(L) - down
        need = len(L) if len(L) <= 2 else int(np.ceil(0.75 * len(L)))
        print(f'  ㉕ {g}（縦型 {len(L)} 本）: 骨の長さが縮んだ {down} / {len(L)}（予想: {"縮む" if want_down else "伸びる"}）→ '
              f'{"✅ 当たり" if hit >= need else "⛔ 外れ"}')
    allr = [a for rs in res.values() for a in rs if a['half'] is not None]
    after = sum(a['settle'] >= a['half'] for a in allr)
    print(f'  ㉖ 形が決まった時点が報酬の半分以後: {after} / {len(allr)}')
    for g, rs in res.items():
        if rs:
            print(f'     {g}: 決まった ep の中央値 {np.median([a["settle"] for a in rs]):.0f}・報酬の半分の中央値 '
                  f'{np.median([a["half"] for a in rs if a["half"] is not None]):.0f}')


def main():
    args = sys.argv[1:]
    if args and args[0] == '--all':
        groups = {}
        for d in sorted((ROOT / 'single_run').glob('*/trace_mid/ep0010.npz')):
            r = d.parent.parent.name
            ov = (d.parent.parent / '.hydra' / 'overrides.yaml')
            if not ov.exists() or 'robot_param_scale=0.0001' in ov.read_text(encoding='utf-8'):
                continue
            groups.setdefault(guess_group(r, ov.read_text(encoding='utf-8')), []).append(r)
    elif args:
        groups = {}
        for r in args:
            ov = (ROOT / 'single_run' / r / '.hydra' / 'overrides.yaml').read_text(encoding='utf-8')
            groups.setdefault(guess_group(r, ov), []).append(r)
    else:
        groups = MAIN
    verdict(report(groups))


if __name__ == '__main__':
    main()
