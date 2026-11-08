#!/usr/bin/env python3
"""**四脚の軌跡を図にし、話ごとに歩けたかを数で出す**（系譜 9-342 の追記、2026-11-06 夜。四脚の遊び、修論に入れない）。

なぜ: 四脚を学習で歩かせる準備（ユーザー「いいよ」）の ③。腕の図（`plot_run.py`）は箱・目標・柱を描き、胴が動く体を見ない。
軌跡は腕と同じ `record_arm_trace.py` が取る（全部の体の位置と姿勢を記録するので、胴 `0` も入る）。この道具はそれを読むだけ。

出すもの（全話。1 話で判定しない、CLAUDE.md ⑤-3）:
  - 話ごと: step 数・終わり方（胴の高さの下限 / 傾きの上限 / 打ち切り）・胴の x の進み・y のずれ・胴の高さの最低・傾きの最大・足先の高さの最低
    終わり方の判定は env（`ant.py`）と同じ量で: 高さ ＝ 胴の原点の z、傾き ＝ 胴の z 軸と鉛直の角。閾値は run の cfg の `done_condition`
  - 最後の行 `[quad] 判定:` に ⛔（nan・足先のめり込み・体の原点が床の下・跳ねた・ギア比が話で違う）と ⚠️（すぐ倒れる・横へ・後ろへ）。
    見張り（`queue_quad_walk.sh`）がこの行を拾う
  - 図（<軌跡>.png、既定は `<run>/trace/quad_view.png`）: ① 上から見た胴の道筋と足先の着いた点 ② 胴の高さ ③ 傾き ④ 足先の高さ（中央値の話。歩き方の型を見る）
  - 足先 ＝ 脛（名前が `11` で始まる 3 桁の体）のカプセルの終点からカプセルの半径だけ下（床に着く点の近似）。全部の体の z の最低も出す（床より下へ出ていないか）
    ⚠️ 実行の段階に入るときの姿勢の乱れ（ant.py の ±0.1、傾き 3〜13°）で、足が床に最大 22 mm 埋まって始まり 5 step ほどで抜ける（Bug 98 の残り）。
    なので足先の最低は最初の 10 step とその後を分けて出し、⛔ はその後だけで判定する

    python3 scripts/plot_quad_trace.py <run のフォルダ>            # <run>/trace/arm_trace.npz を読む
    python3 scripts/plot_quad_trace.py <run のフォルダ> <npz>       # 別の軌跡（TRACE_OUT で書いたもの）
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def done_condition(run):
    hc = yaml.safe_load(open(os.path.join(run, '.hydra', 'config.yaml')))
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'design_opt', 'cfg', f"{hc['cfg']}.yml")))
    dc = cfg.get('done_condition', {}) or {}
    return dict(min_height=dc.get('min_height', 0.0), max_ang=dc.get('max_ang', 3600), max_nsteps=dc.get('max_nsteps', 1000)), hc


def main():
    run = sys.argv[1].rstrip('/')
    npz = sys.argv[2] if len(sys.argv) > 2 else os.path.join(run, 'trace', 'arm_trace.npz')
    out = os.path.join(run, 'trace', 'quad_view.png') if len(sys.argv) <= 2 else npz.replace('.npz', '.png')
    dc, hc = done_condition(run)
    d = np.load(npz, allow_pickle=True)
    names = [str(n) for n in d['body_names']]
    i0 = names.index('0')
    shins = [i for i, n in enumerate(names) if len(n) == 3 and n.startswith('11')]
    print(f'[quad] {run}  cfg={hc["cfg"]}  xml={hc.get("xml_name")}  話 {int(d["ep_count"])}  '
          f'終わりの条件: 高さ ≤ {dc["min_height"]} m・傾き ≥ {dc["max_ang"]}°・{dc["max_nsteps"]} step で打ち切り')
    print(f'[quad] 足先にする体: {[names[i] for i in shins]}（脛のカプセルの終点 − 半径）')
    eps = []
    for k, (xp, xm, cap, gs) in enumerate(zip(d['ep_xpos'], d['ep_xmat'], d['ep_cap'], d['ep_geom'])):
        xp, xm = np.asarray(xp, float), np.asarray(xm, float)
        tor = xp[:, i0]
        tilt = np.degrees(np.arccos(np.clip(xm[:, i0, 2, 2], -1, 1)))
        cap = np.asarray(cap, float)
        feet = np.stack([xp[:, i] + np.einsum('tij,j->ti', xm[:, i], cap[i, 3:]) for i in shins], 1)
        feet[..., 2] -= np.asarray(gs, float)[shins]
        T = len(tor)
        if tor[-1, 2] <= dc['min_height']:
            how = '高さの下限'
        elif tilt[-1] >= dc['max_ang']:
            how = '傾きの上限'
        elif T >= dc['max_nsteps']:
            how = '打ち切り（最後まで）'
        else:
            how = '⚠️ 不明（再生の step の上限か）'
        e = dict(T=T, how=how, dx=tor[-1, 0] - tor[0, 0], dx250=tor[min(T, 250) - 1, 0] - tor[0, 0], zmax=tor[:, 2].max(), finite=bool(np.isfinite(xp).all()), dy=tor[-1, 1] - tor[0, 1], hmin=tor[:, 2].min(), tmax=tilt.max(),
                 fmin=feet[10:, ..., 2].min() if T > 10 else np.nan, f0=feet[:10, ..., 2].min(), zmin=xp[..., 2].min(), tor=tor, tilt=tilt, feet=feet)
        eps.append(e)
        print(f'[quad] 話 {k+1}: {T} step・{how}・x の進み {e["dx"]:+.3f} m（250 step で {e["dx250"]:+.3f}）・y {e["dy"]:+.3f} m・胴の高さの最低 {e["hmin"]:.3f} m・'
              f'傾きの最大 {e["tmax"]:.1f}°・足先の最低 {e["fmin"]*1000:+.1f} mm（最初の 10 step は {e["f0"]*1000:+.1f} mm）・体の原点の最低 {e["zmin"]*1000:+.1f} mm')
    dx = np.array([e['dx'] for e in eps])
    print(f'[quad] x の進み: 中央値 {np.median(dx):+.3f} m・幅 {dx.min():+.3f}〜{dx.max():+.3f} m（{len(eps)} 話）')
    # ⭐ 見張り（queue_quad_walk.sh）が拾う判定。⛔ は「止めて調べる」、⚠️ は「見ておく」
    bad, warn = [], []
    if not all(e['finite'] for e in eps):
        bad.append('位置に nan / inf')
    if min(e['fmin'] for e in eps) < -0.005:
        bad.append(f'最初の 10 step の後に足先が床より 5 mm 以上下（最低 {min(e["fmin"] for e in eps)*1000:+.1f} mm、めり込み）')
    if min(e['zmin'] for e in eps) < 0:
        bad.append(f'体の原点が床より下（{min(e["zmin"] for e in eps)*1000:+.1f} mm）')
    h0 = float(np.median([e['tor'][0, 2] for e in eps]))
    if max(e['zmax'] for e in eps) > 2 * h0:
        bad.append(f'胴が最初の高さの 2 倍より上へ出た（{max(e["zmax"] for e in eps):.3f} m、跳ねた・飛んだ）')
    gear = [np.asarray(g, float) for g in d['ep_gear'] if g is not None]
    if gear and not all(np.allclose(g[np.isfinite(g)], gear[0][np.isfinite(gear[0])]) for g in gear):
        bad.append('話によってギア比が違う（固定のはず）')
    if gear:
        print(f'[quad] ギア比（1 話目）: {np.round(gear[0][np.isfinite(gear[0])], 3).tolist()}')
    if sum(e['how'] in ('高さの下限', '傾きの上限') for e in eps) == len(eps) and np.median([e['T'] for e in eps]) < 50:
        warn.append(f'全話が 50 step 未満で倒れて終わる（中央値 {np.median([e["T"] for e in eps]):.0f} step）')
    if np.median([abs(e['dy']) for e in eps]) > max(abs(np.median(dx)), 0.05):
        warn.append('y のずれが x の進みより大きい（横へ進んでいる）')
    if np.median(dx) < -0.05:
        warn.append(f'後ろへ進む（x の進み 中央値 {np.median(dx):+.3f} m。報酬は +x）')
    print(f'[quad] 250 step（10 s）の x の進み: 中央値 {np.median([e["dx250"] for e in eps]):+.3f} m')
    print('[quad] 判定: ' + ('⛔ ' + ' / '.join(bad) if bad else '✅ 異常なし') + (('  ⚠️ ' + ' / '.join(warn)) if warn else ''))
    rep = eps[int(np.argsort(dx)[len(dx) // 2])]

    _jp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
    fm.fontManager.addfont(_jp)      # 登録しないと名前だけでは見つからない（豆腐になる）
    plt.rcParams['font.family'] = fm.FontProperties(fname=_jp).get_name()
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for e in eps:
        a = 1.0 if e is rep else 0.35
        ax[0, 0].plot(e['tor'][:, 0], e['tor'][:, 1], alpha=a)
        ax[0, 1].plot(e['tor'][:, 2], alpha=a)
        ax[1, 0].plot(e['tilt'], alpha=a)
    for j in range(len(shins)):
        f = rep['feet'][:, j]
        on = f[:, 2] < 0.005
        ax[0, 0].scatter(f[on, 0], f[on, 1], s=3, alpha=0.4)
        ax[1, 1].plot(f[:, 2] * 1000, label=f'脛 {names[shins[j]]}')
    ax[0, 0].set(title='上から見た胴の道筋（点は中央値の話の足先が床から 5 mm 以内）', xlabel='x [m]', ylabel='y [m]')
    ax[0, 0].axis('equal')
    ax[0, 1].axhline(dc['min_height'], color='r', ls='--')
    ax[0, 1].set(title='胴の高さ（赤: 下限）', xlabel='step', ylabel='z [m]')
    ax[1, 0].axhline(dc['max_ang'], color='r', ls='--')
    ax[1, 0].set(title='傾き（赤: 上限）', xlabel='step', ylabel='[°]')
    ax[1, 1].set(title='足先の高さ（中央値の話）', xlabel='step', ylabel='z [mm]')
    ax[1, 1].legend(fontsize=8)
    fig.suptitle(f'{os.path.basename(run)}  x の進み 中央値 {np.median(dx):+.3f} m（{len(eps)} 話）・'
                 f'足先の最低 {min(e["fmin"] for e in eps)*1000:+.1f} mm')
    for a_ in ax.flat:
        a_.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=90)
    print(f'[quad] → {out}')


if __name__ == '__main__':
    main()
