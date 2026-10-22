#!/usr/bin/env python3
"""**物理的にありえない値**を学習ログと再生の軌跡から機械で拾う（不変条件の見張り、2026-10-12）。

[異常を能動的に見つける方法.md](../docs/リポジトリ説明/異常を能動的に見つける方法.md) の方法 2。
結果がどうであれ必ず成り立つはずの量だけを見る（順位や良し悪しは見ない）。

    python3 scripts/check_invariants.py <run> [<run> ...]
    python3 scripts/check_invariants.py --trace <npz> <run>     # 途中の epoch の軌跡（watch_midrun.sh）

| 項目 | 破ったときの過去の例 |
|---|---|
| 報酬の桁（学習ログの 1 step 平均が、その run の中央値の 20 倍を超える epoch） | Bug 60（ep0 の −367） |
| 制御入力（再生の |ctrl| が ±1 を超え、しかも切っていない） | Bug 60 |
| 触れる前に動かない（対象が速く動き出した step で、腕が離れている）・置いた瞬間に重ならない | Bug 55・Bug 6 |
| 床（可動リンクの下面が床より 10 mm 以上下の step が 1 % を超える。⭐ 10-22: 割合によらず 50 mm 以上下なら ⛔） | 9-154・9-288 |
| 壁（ホッケー: パック・腕の先端が側壁の内面を越える） | Bug 39・9-93 |
| 有限（軌跡の位置に NaN・inf が無い。NaN は比較が偽になり、床・壁の検査を黙って素通りする） | Bug 68（hockey_bank9_phys） |
| 置いた瞬間に静的な箱（板・壁）に重ならない | Bug 69（初期姿勢の探索が板を見ず、重なった話が発散した） |

`check_before_conclusion.py` の「不変条件」が `run(r)` を呼ぶ。返り値は (ok, 文) の列で、ok は True／None（⚠️）／False（⛔）。
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from judge_hockey_goal import gap  # noqa: E402  カプセル列と箱の最小距離
from check_cube_penetration import cube_half  # noqa: E402

FLOOR_TOL = 0.010   # 床より下の許容 [m]
FLOOR_FRAC = 0.01   # 床下にいてよい step の割合
# ⭐ 2026-10-22（系譜 9-283 ③・9-288）: 割合だけでは深く短い沈み込み（打撃の一瞬・開始の姿勢）を素通りした。
#   0.5 % の step だけ 182 mm 沈めた軌跡が ✅ だった。深さは離れた組のめり込みの ⛔ と同じ 50 mm
FLOOR_DEEP = 0.050
START_V = 0.5       # 「動き出した」とみなす対象の速さ [m/s]
DT = 0.04           # 記録 1 step


def _log_reward(r):
    p = ROOT / 'single_run' / r / 'log' / 'log_train.txt'
    if not p.exists():
        return None, '学習ログが無い'
    tr = [(int(l.split('\t')[0].split()[-1]), float(m.group(1)))
          for l in p.read_text(errors='ignore').splitlines()
          if 'T_sample' in l and (m := re.search(r'\ttrain_R (\S+)', l))]
    if len(tr) < 3:
        return None, 'epoch が少ない'
    a = np.array([v for _, v in tr])
    lim = max(20 * (np.median(np.abs(a)) + 1e-3), 5.0)
    bad = [(e, v) for e, v in tr if abs(v) > lim]
    if not bad:
        return True, f'学習の 1 step 平均報酬はすべて ±{lim:.1f} 以内（{len(tr)} epoch）'
    # ep0 だけで初期形態の門の罰が乗るのは正常にありうる → ⚠️（Bug 60 は ep0 だったので消さない）
    return None, f'⚠️ 1 step 平均報酬が中央値の 20 倍超: {[(e, round(v, 1)) for e, v in bad[:3]]}（Bug 60 型か門の罰か、再現して確かめる）'


def _ctrl(r, d):
    if 'ep_u_abs' not in d.files:
        return True, '（軌跡に |ctrl| が無い＝10-12 より前の再生）'
    mx = max(float(np.asarray(u, float).max()) for u in d['ep_u_abs'])
    le = json.loads(str(d['launch_env'])) if str(d['launch_env']) else {}
    le.update(json.loads(str(d['replay_env'])) if 'replay_env' in d.files and str(d['replay_env']) else {})
    if le.get('CNOID_CTRL_CLIP'):
        return True, f'|ctrl| 最大 {mx:.2f}（±1 で切っている＝トルクはギア比まで）'
    if mx > 1 + 1e-6:
        return False, f'⛔ |ctrl| 最大 {mx:.2f} で ±1 を超え、切っていない＝トルクがギア比の {mx:.2f} 倍（Bug 60）'
    return True, f'|ctrl| 最大 {mx:.2f}'


def _caps(d, e):
    xp = np.asarray(d['ep_xpos'][e], float)
    xm = np.asarray(d['ep_xmat'][e], float)
    bo = np.asarray(d['ep_bone'][e], float)
    rad = np.asarray(d['ep_geom'][e], float).ravel()[:xp.shape[1]]
    return xp, xm, bo.copy(), rad


def _touch(d, xml):
    _, half, _ = cube_half(xml)
    if half is None:
        return True, '対象が無い'
    out = []
    for e in range(len(d['ep_cubes'])):
        c = np.asarray(d['ep_cubes'][e], float)
        if np.abs(c[:, 0]).max() > 50:          # 到達で退避した箱（9-234）
            continue
        xp, xm, bo, rad = _caps(d, e)
        bo[0] = 0                               # 根の骨は運動学に効かない（9-154）
        v = np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1) / DT
        idx = np.nonzero(v > START_V)[0]
        if not len(idx):
            continue
        k = int(idx[0])                          # c[k] → c[k+1] で速く動いた
        # 標本の刻み（Bug 48）: その間に腕の点が動いた距離までは「触れていない」と言えない
        ends = xp + np.einsum('tlij,lj->tli', xm, bo)
        res = float(np.linalg.norm(ends[k + 1] - ends[k], axis=1).max())
        g = min(gap(xp[j], xm[j], bo, rad, c[j], half) for j in (k, k + 1))
        if g > res + 0.02:
            out.append((e + 1, k, round(g * 1000), round(res * 1000)))
    # ⭐ 置いた瞬間に重なっていないか（Bug 55: 重なりの押し出しは「触れてから動く」を満たしてしまう）
    over = []
    for e in range(len(d['ep_init'])):
        if d['ep_init'][e] is None or d['ep_init'][e] is False:
            continue
        ix, im, ic = (np.asarray(a, float) for a in d['ep_init'][e])
        if abs(ic[0]) > 50:
            continue
        _, _, bo, rad = _caps(d, e)
        bo[0] = 0
        g0 = gap(ix, im, bo, rad, ic, half)
        if g0 < -0.005:
            over.append((e + 1, round(g0 * 1000)))
    if over:
        return False, f'⛔ 置いた瞬間に腕と対象が重なっている話: (話, 隙間 mm) {over[:3]}（Bug 55 型。動いても打撃ではない）'
    if out:
        return False, f'⛔ 腕が離れているのに対象が動き出した話: (話, step, 隙間 mm, 刻み mm) {out[:3]}（Bug 55 型）'
    return True, '対象は腕に触れてから動き出す'


def _finite(d):
    bad = []
    for e in range(len(d['ep_xpos'])):
        a = np.concatenate([np.asarray(d['ep_xpos'][e], float).reshape(len(d['ep_xpos'][e]), -1),
                            np.asarray(d['ep_cubes'][e], float).reshape(len(d['ep_cubes'][e]), -1)], axis=1)
        nf = ~np.isfinite(a).all(axis=1)
        if nf.any():
            bad.append((e + 1, int(np.argmax(nf)), len(a)))
    if bad:
        return False, f'⛔ 位置が NaN・inf になった話: (話, step, 話の長さ) {bad[:5]}（Bug 68。物理が発散して話が途中で切れる）'
    return True, f'全 {len(d["ep_xpos"])} 話で位置は有限'


def _spawn_static(d, xml):
    t = ET.parse(ROOT / 'assets' / 'mujoco_envs' / f'{xml}.xml')
    boxes = [(np.array(b.get('pos').split(), float), np.array(b.find('geom').get('size').split(), float), b.get('name'))
             for b in t.findall('worldbody/body') if b.find('joint') is None and b.find('geom') is not None
             and b.find('geom').get('type') == 'box']
    if not boxes:
        return True, '静的な箱なし'
    over = []
    for e in range(len(d['ep_init'])):
        if d['ep_init'][e] is None or d['ep_init'][e] is False:
            continue
        ix, im, _ = (np.asarray(a, float) for a in d['ep_init'][e])
        _, _, bo, rad = _caps(d, e)
        bo[0] = 0
        g, nm = min((gap(ix, im, bo, rad, c, h), n) for c, h, n in boxes)
        if g < -0.005:
            over.append((e + 1, nm, round(g * 1000)))
    if over:
        return False, f'⛔ 置いた瞬間に腕が静的な箱に重なる話 {len(over)}/{len(d["ep_init"])}: (話, 箱, 隙間 mm) {over[:3]}（Bug 69）'
    return True, f'置いた瞬間に静的な箱（{len(boxes)} 個）と重ならない'


def _floor(d):
    names = [str(n) for n in d['body_names']]
    worst, frac = 0.0, 0.0
    for e in range(len(d['ep_xpos'])):
        xp, xm, bo, rad = _caps(d, e)
        ends = xp + np.einsum('tlij,lj->tli', xm, bo)
        zb = np.minimum(xp[..., 2], ends[..., 2]) - rad
        keep = [i for i in range(1, len(names))
                if not (i == 1 and abs(bo[1][2]) > 0.9 * np.linalg.norm(bo[1]) + 1e-9)]   # 根と縦の柱は除く
        z = zb[:, keep].min(axis=1)
        worst = min(worst, float(z.min()))
        frac = max(frac, float((z < -FLOOR_TOL).mean()))
    if frac > FLOOR_FRAC:
        return False, f'⛔ 可動リンクが床より {-worst*1000:.0f} mm 下、{frac*100:.1f} % の step（9-154 型）'
    if worst < -FLOOR_DEEP:
        return False, f'⛔ 可動リンクが床より {-worst*1000:.0f} mm 下（{FLOOR_DEEP*1000:.0f} mm 超。床下は {frac*100:.1f} % の step だけ、9-288 型）'
    return True, f'床: 最も下 {worst*1000:.0f} mm・床下 {frac*100:.1f} % の step（一瞬の接触まで）'


def _walls(d, xml):
    t = ET.parse(ROOT / 'assets' / 'mujoco_envs' / f'{xml}.xml')
    side = [b for b in t.findall('worldbody/body') if b.get('name') == 'wall_side_p']
    if not side:
        return True, '壁なし'
    inner = float(side[0].get('pos').split()[1]) - float(side[0].find('geom').get('size').split()[1])
    _, half, _ = cube_half(xml)
    py = max(float(np.abs(np.asarray(c, float)[:, 1]).max()) for c in d['ep_cubes'])
    ty = 0.0
    for e in range(len(d['ep_xpos'])):
        xp, xm, bo, _ = _caps(d, e)
        ty = max(ty, float(np.abs((xp + np.einsum('tlij,lj->tli', xm, bo))[:, 1:, 1]).max()))
    bad = []
    if py > inner - half[1] + 0.02:
        bad.append(f'パック |y| {py:.2f}')
    if ty > inner + 0.02:
        bad.append(f'腕 |y| {ty:.2f}')
    if bad:
        return False, f'⛔ 側壁の内面 {inner:.2f} を越えた: {", ".join(bad)}（Bug 39・9-93 型）'
    return True, f'壁の内側（パック |y| ≤ {py:.2f}・腕 ≤ {ty:.2f}、内面 {inner:.2f}）'


def run(r, trace=None):
    """trace: 軌跡の場所（既定は run の trace/arm_trace.npz）。途中の epoch の見張り（watch_midrun.sh）が渡す。"""
    out = [('報酬の桁',) + _log_reward(r)]
    p = Path(trace) if trace else ROOT / 'single_run' / r / 'trace' / 'arm_trace.npz'
    if not p.exists():
        return out + [('軌跡', None, '⚠️ 軌跡が無い')]
    d = np.load(p, allow_pickle=True)
    if 'ep_xpos' not in d.files:
        return out + [('軌跡', None, '⚠️ 古い形式の軌跡（話ごとの記録が無い）')]
    xml = yaml.safe_load(open(ROOT / 'single_run' / r / '.hydra' / 'config.yaml'))['xml_name']
    for name, fn in (('有限', lambda: _finite(d)), ('制御入力', lambda: _ctrl(r, d)), ('触れる前', lambda: _touch(d, xml)),
                     ('置いた瞬間の箱', lambda: _spawn_static(d, xml)),
                     ('床', lambda: _floor(d)), ('壁', lambda: _walls(d, xml))):
        try:
            out.append((name,) + fn())
        except Exception as ex:   # 検査の失敗は黙らない
            out.append((name, None, f'⚠️ 検査できない: {type(ex).__name__}: {ex}'))
    return out


def main():
    # `--trace <npz>` は直後の run 1 本にだけ効く（途中の epoch の軌跡を見るとき）
    args, trace = sys.argv[1:], None
    if args[:1] == ['--trace']:
        trace, args = args[1], args[2:]
    for r in args:
        print(f'=== {r}')
        for name, ok, msg in run(r, trace):
            print(f"  {'✅' if ok else ('⚠️' if ok is None else '⛔')} {name:<6} {msg}")


if __name__ == '__main__':
    main()
