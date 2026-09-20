#!/usr/bin/env python3
"""**結論を書く前に必ず通す検査**（実験系譜 9-114）。

⚠️⚠️ **なぜ要るか**: CLAUDE.md に書いた規律のうち、
**散文だけのものは今週 10 件破られ、スクリプトがあるものは 0 件だった。**
⭐ **「md に書く」では守られない。実行できる検査に変換する。**

本スクリプトは、**ある run について結論を書いてよいか**を機械で判定する。

    python3 scripts/check_before_conclusion.py e2e_a1_fix3_auto13
    python3 scripts/check_before_conclusion.py --all        # 完走 run 全部

検査するもの（すべて実際に破った規律）:
  1. **完走しているか**（training done!）
  2. ⭐ **再生して軌跡があるか**（§5-2 ⑤-3。9-96・9-113 で破った）
  3. ⭐ **タスクを達成しているか**（未到達どうしを比較しない。9-113 で破った）
  4. **版が混入していないか**（9-105 で判明。audit_run_validity と同じ判定）
  5. **事前登録があるか**（キューか台帳に「読み」が書かれているか。9-101 で破った）
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def run_dir(r):
    return ROOT / 'single_run' / r


def ck_done(r):
    log = run_dir(r) / 'log' / 'log_train.txt'
    if not log.exists():
        return False, '学習ログが無い'
    return ('training done!' in log.read_text(encoding='utf-8', errors='ignore'),
            '完走' if 'training done!' in log.read_text(encoding='utf-8', errors='ignore') else '未完走')


def ck_trace(r):
    p = run_dir(r) / 'trace' / 'arm_trace.npz'
    if not p.exists():
        return False, '⛔ 再生していない（§5-2 ⑤-3）。record_arm_trace.py を回すこと'
    return True, f'軌跡あり'


def has_static_body(r):
    """⭐ XML に固定 body（柱・壁）があるか。あると再生ごとに結果が揺れる（9-138）。"""
    import xml.etree.ElementTree as ET
    import yaml
    try:
        cfg = yaml.safe_load(open(run_dir(r) / '.hydra' / 'config.yaml'))
        t = ET.parse(ROOT / 'assets' / 'mujoco_envs' / f"{cfg.get('xml_name')}.xml")
    except Exception:
        return []
    out = []
    for b in t.findall('worldbody/body'):
        n = b.get('name', '')
        if n in ('0',) or b.find('joint') is not None:
            continue          # 腕の根元と、関節を持つ可動体（cube 等）は除く
        if b.find('geom') is not None:
            out.append(n)
    return out


def ck_achieved(r):
    """⭐ タスクを達成しているか。未到達どうしの比較を防ぐ（9-113）。"""
    p = run_dir(r) / 'trace' / 'arm_trace.npz'
    if not p.exists():
        return None, '（再生していないので判定不能）'
    d = np.load(p)
    # ⛔⛔ 9-138: 固定 body と接触するタスクは、同じ checkpoint でも再生ごとに揺れる
    #   （ホッケー 64〜308 mm・障害物 254 %）。1 エピソードで達成を判定してはいけない。
    static = has_static_body(r)
    # ⛔⛔ Bug 47 / 9-142: 軌跡は「読み込み直後の 1 話目」であり、その 1 話は当てにならない
    #   （対照 Pusher で 1 話目 15.11 m 対 3〜6 話 6.54〜15.97 m）。
    #   ⚠️ 反復しても検出できない（毎回 1 話目をやり直すだけ）。
    #   ⭐ 二値の到達判定は 1 話目でも変わらないことが多いが、**量の比較には使えない**。
    warn = ('  ⚠️ **この値は「読み込み直後の 1 話目」である（Bug 47）。'
            '量を比べるなら `probe_episode_bias.py` で 3 話目以降を使うこと** ')
    warn += ('' if not static else
            f"  ⛔ **固定 body {static} がある。再生ごとに結果が揺れる（9-138）。"
            f"1 エピソードで判定しないこと** ")
    # ⛔ 9-137: 打ち切られた軌跡で最終位置・到達を語らない。
    #   ⭐ ただし「末尾でまだ動いている」ときだけ警告する。
    #     対象が止まっていれば打ち切りは無害で、毎回出る警告は読まれなくなる（CLAUDE.md §5-2 ①）。
    ov = (run_dir(r) / '.hydra' / 'overrides.yaml')
    txt = ov.read_text(encoding='utf-8') if ov.exists() else ''
    is_reach = 'use_reach=true' in txt
    xp, xm, bo = d['xpos'], d['xmat'], d['bone_offset']
    tip = xp[:, -1, :] + np.einsum('tij,j->ti', xm[:, -1], bo[-1])
    # ⛔ 9-137: 打ち切られた軌跡で最終位置・到達を語らない。
    #   ⭐ ただし **判定に使う量そのもの**が末尾でまだ動いているときだけ警告する。
    #     Pusher は対象が 400 step で止まるので、腕の動きで警告を出すと毎回出て読まれなくなる
    #     （CLAUDE.md §5-2 ①「毎回同じ件数が出る検査は読まれなくなる」）。
    watched = tip if is_reach else d.get('cube', tip)
    n = len(watched)
    tail = watched[int(n * 0.9):]
    moving = float(np.abs(np.diff(tail, axis=0)).max()) if len(tail) > 1 else 0.0
    if n >= 1200 and moving > 1e-3:
        warn += (f'  ⚠️ **{n} step で打ち切られ、判定対象が末尾でもまだ動いている'
                 f'（{moving*1000:.1f} mm/step）。TRACE_STEPS を増やすこと（9-137）** ')
    if is_reach:
        tgt = d['target']
        dist = np.linalg.norm(tip - tgt, axis=1)
        if dist.min() < 0.01:
            return True, f'到達（最小 {dist.min()*1000:.0f} mm）' + warn
        return False, (f'⛔ **未到達**（最小 {dist.min()*1000:.0f} mm・最終 {dist[-1]*1000:.0f} mm）。'
                        f'⚠️ **未到達どうしの差を「良化」と呼ばないこと**（9-113）' + warn)
    c = d['cube']
    moved = abs(c[-1, 0] - c[0, 0])
    if moved < 1e-3:
        return False, f'⛔ **対象が動いていない**（{moved*1000:.1f} mm）' + warn
    return True, f'対象が {moved:.3f} m 動いた' + warn


def ck_version(r):
    """同じ xml_name の他 run と版が違わないか（9-105）。"""
    out = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'audit_run_validity.py')],
                         capture_output=True, text=True, cwd=ROOT).stdout
    seg = out.split('同じ xml_name なのに中身が違う組')[-1]
    for block in seg.split('⛔ ')[1:]:
        if re.search(rf'^\s+{re.escape(r)}\s', block, re.M):
            xml = block.split(':')[0]
            return False, f'⛔ **{xml} に版の混入あり**。同じ版の run とだけ比較すること（9-105）'
    return True, '版の混入なし'


def ck_prereg(r):
    """事前登録（読み）がキューか台帳にあるか。"""
    for q in (ROOT / 'scripts').glob('queue_*.sh'):
        t = q.read_text(encoding='utf-8')
        if r in t and ('読み' in t or '停止条件' in t):
            return True, f'{q.name} に読みあり'
    led = (ROOT / 'docs/研究応用/台帳/実験系譜.md').read_text(encoding='utf-8')
    for m in re.finditer(r'^### .*$', led, re.M):
        seg = led[m.start():m.start() + 6000]
        if r in seg and ('事前' in seg or '読み' in seg):
            return True, '台帳に読みあり'
    return False, '⚠️ 事前登録が見つからない（結果を見てから読み方を決めていないか）'


CHECKS = [('完走', ck_done), ('再生', ck_trace), ('⭐ 達成', ck_achieved),
          ('版', ck_version), ('事前登録', ck_prereg)]


def audit(r):
    print(f'\n=== {r}')
    ng = 0
    for name, fn in CHECKS:
        ok, msg = fn(r)
        mark = '✅' if ok else ('⚠️' if ok is None else '⛔')
        if ok is False:
            ng += 1
        print(f'  {mark} {name:<8} {msg}')
    print(f'  → {"⛔ 結論を書く前に上を直すこと" if ng else "✅ 結論を書いてよい"}')
    return ng


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*')
    ap.add_argument('--all', action='store_true', help='完走した run 全部')
    a = ap.parse_args()
    runs = a.runs
    if a.all:
        runs = sorted(d.name for d in (ROOT / 'single_run').iterdir()
                      if d.is_dir() and (d / 'log' / 'log_train.txt').exists()
                      and 'training done!' in (d / 'log' / 'log_train.txt').read_text(
                          encoding='utf-8', errors='ignore'))
    if not runs:
        ap.error('run 名か --all を指定してください')
    tot = sum(audit(r) for r in runs)
    print(f'\n合計 {tot} 件の要対応')
    return 0


if __name__ == '__main__':
    main()
