#!/usr/bin/env python3
"""完走した run が「正当な実験」として扱えるかを検査する（実験系譜 9-105）。

**なぜ要るか**: XML と cfg は**名前で参照される**が、**中身は書き換わる**。
同じ `xml_name` の run でも、起動時刻が違えば**別の物理で走っている**ことがある。
⚠️ **ホッケーでは実際に、同じ `e2e_hockey_wall` で 3 種類の物理が混ざっていた**（9-105）。

各 run について、**起動時刻の時点で有効だった XML / cfg の版**を git 履歴から特定し、
**同じ名前なのに中身が違う組**を検出する。

    python3 scripts/audit_run_validity.py                    # 全 run
    python3 scripts/audit_run_validity.py --prefix hockey    # 名前で絞る
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def started_at(run):
    log = ROOT / 'single_run' / run / 'log' / 'log_train.txt'
    if not log.exists():
        return None
    head = log.open(encoding='utf-8', errors='ignore').readline()
    m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', head)
    return m.group(1) if m else None


def overrides(run):
    p = ROOT / 'single_run' / run / '.hydra' / 'overrides.yaml'
    if not p.exists():
        return {}
    out = {}
    for ln in p.read_text(encoding='utf-8').split('\n'):
        m = re.match(r'-\s*\+?([\w.]+)=(.*)', ln.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def blob_at(path, when):
    """`when` 時点でのファイルの内容ハッシュ。無ければ None。"""
    r = subprocess.run(['git', 'log', '-1', '--format=%H', f'--before={when}', '--', path],
                       cwd=ROOT, capture_output=True, text=True)
    commit = r.stdout.strip()
    if not commit:
        return None
    r2 = subprocess.run(['git', 'rev-parse', f'{commit}:{path}'],
                        cwd=ROOT, capture_output=True, text=True)
    return r2.stdout.strip()[:12] or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='')
    a = ap.parse_args()

    runs = sorted(d.name for d in (ROOT / 'single_run').iterdir()
                  if d.is_dir() and (d / 'log' / 'log_train.txt').exists()
                  and d.name.startswith(a.prefix))
    rows = []
    for run in runs:
        t = started_at(run)
        ov = overrides(run)
        xml = ov.get('xml_name')
        cfg = ov.get('cfg')
        if not (t and xml):
            continue
        rows.append((run, t, xml, cfg,
                     blob_at(f'assets/mujoco_envs/{xml}.xml', t),
                     blob_at(f'design_opt/cfg/{cfg}.yml', t) if cfg else None))

    print(f"{'run':<26}{'起動':<18}{'xml_name':<22}{'XMLの版':<14}{'cfgの版'}")
    for run, t, xml, cfg, xb, cb in rows:
        print(f"{run:<26}{t[5:]:<18}{xml:<22}{str(xb)[:10]:<14}{str(cb)[:10]}")

    # ⚠️ 同じ xml_name で版が違う組を検出
    print("\n=== ⚠️ 同じ xml_name なのに中身が違う組")
    by_xml = {}
    for run, t, xml, cfg, xb, cb in rows:
        by_xml.setdefault(xml, []).append((run, xb))
    bad = 0
    for xml, lst in by_xml.items():
        versions = {v for _, v in lst if v}
        if len(versions) > 1:
            bad += 1
            print(f"  ⛔ {xml}: {len(versions)} 種類の中身")
            for run, v in lst:
                print(f"       {run:<26} {str(v)[:10]}")
    if not bad:
        print("  ✅ 無し")
    print(f"\n⚠️ **同じ名前で中身が違う組は、そのままでは比較できない。**")
    print("   比較するなら (a) 同じ版どうしを選ぶ か (b) 新しい版で回し直す。")
    return 0


if __name__ == '__main__':
    main()
