#!/usr/bin/env python3
"""**由来の監査**: 書いていない値（既定値）が、どこに・どの run に掛かっているかを並べる（2026-10-13）。学習はしない。

[異常を能動的に見つける方法.md](../docs/リポジトリ説明/異常を能動的に見つける方法.md) の方法 5。
⛔ 既定値は「書いていないので見えない」。過去にこの型で 3 回間違えた:
  - Bug 59: 押しの一部で `ctrl_cost_coeff` を書き忘れ、コードの既定 1e-4（原論文の 2000 分の 1）が効いていた
  - Bug 62: XML の default の `armature="1"` が、腕だけでなく箱・パックのスライド関節にも掛かっていた
  - 9-241: 平面アームの台座の球（半径 0.1 m）は生成スクリプトの既定値で、スケッチから測っていなかった

    python3 scripts/audit_provenance.py                      # queue_phys.sh の run すべて
    python3 scripts/audit_provenance.py --queue scripts/queue_phys.sh

出すもの:
  ① コードの既定値: `reward_specs` / `env_specs` / `done_condition` の `.get(キー, 既定)` のうち、
     キューの run の cfg にも起動行にも書いていないもの（＝既定値が効いている）。run の数つき
  ② XML の default クラス: 各 XML の `<default>` の joint / geom の属性が、どの関節・geom に掛かっているか。
     ⭐ 対象（cube）の関節・geom に掛かっているものを ⚠️ で出す
  ③ 生成スクリプトの既定値: `topology_to_xml.py` の `DEFAULT_*`（スケッチから測っていない値）
判断は人がする（既定値が正しいこともある）。⭐ 判断した結果は `シミュレーションの値の変更表.md` の 5 節に書く。
"""
import argparse
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CODE = [ROOT / 'design_opt' / 'envs' / 'pusher.py', ROOT / 'scripts' / 'worker_sampler.py',
        *sorted((ROOT / 'khrylib' / 'rl' / 'envs' / 'common').glob('*.py'))]
DYN = {'armature', 'damping', 'density', 'mass', 'frictionloss', 'stiffness', 'friction', 'solref', 'solimp'}
PAT = re.compile(r"(reward_specs|env_specs|done_condition)\.get\(\s*'([a-z_0-9]+)'\s*,\s*([^)]+?)\)")


def code_defaults():
    out = {}
    for f in CODE:
        for m in PAT.finditer(f.read_text(encoding='utf-8')):
            out.setdefault((m.group(1), m.group(2)), m.group(3).strip())
    return out


def queue_runs(queue):
    """キューを空打ちして (run 名, 引数の列) を返す（launch を echo に差し替える）。"""
    src = Path(queue).read_text(encoding='utf-8')
    src = re.sub(r'^launch\(\)\{.*?\n(?=\S)', 'launch(){ echo "LAUNCH $*"; return; }\n', src, flags=re.S | re.M)
    src = re.sub(r'^Q=.*$', 'Q=/dev/null', src, flags=re.M)
    with tempfile.NamedTemporaryFile('w', suffix='.sh', delete=False) as f:
        f.write(src)
    out = subprocess.run(['bash', f.name], capture_output=True, text=True, cwd=ROOT).stdout
    # ⭐ launch の中で全 run に足している引数（例: +env_specs.init_self_contact=true）も拾う
    body = re.search(r'^launch\(\)\{(.*?)\n(?=\S)', Path(queue).read_text(encoding='utf-8'), re.S | re.M)
    fixed = re.findall(r'\+?(?:reward_specs|env_specs|done_condition)\.[a-z_0-9]+=\S+', body.group(1)) if body else []
    runs = []
    for line in out.splitlines():
        if line.startswith('LAUNCH '):
            parts = line.split()[1:]
            runs.append((parts[0], parts[1:] + fixed))
    return runs


def specs_of(args):
    """cfg ファイルと起動行の上書きから、書いてあるキーの集合と xml_name を返す。"""
    cfg = next((a.split('=', 1)[1] for a in args if a.startswith('cfg=')), None)
    xml = next((a.split('=', 1)[1] for a in args if a.startswith('xml_name=')), None)
    keys = set()
    if cfg:
        c = yaml.safe_load(open(ROOT / 'design_opt' / 'cfg' / f'{cfg}.yml')) or {}
        for sec in ('reward_specs', 'env_specs', 'done_condition'):
            keys |= {(sec, k) for k in (c.get(sec) or {})}
    for a in args:
        m = re.match(r'\+?(reward_specs|env_specs|done_condition)\.([a-z_0-9]+)=', a)
        if m:
            keys.add((m.group(1), m.group(2)))
    return cfg, xml, keys


def xml_defaults(xml):
    t = ET.parse(ROOT / 'assets' / 'mujoco_envs' / f'{xml}.xml').getroot()
    d = t.find('default')
    rows = []
    if d is None:
        return rows
    for tag in ('joint', 'geom'):
        de = d.find(tag)
        if de is None:
            continue
        for attr, val in de.attrib.items():
            inherit = [e.get('name', '?') for e in t.iter(tag) if e is not de and attr not in e.attrib
                       and e.get('name') != 'floor']
            # ⚠️ 対象の動き方を変える属性だけを旗にする（limited・margin・condim は範囲と接触の判定の設定）
            obj = [n for n in inherit if n.startswith('cube')] if attr in DYN else []
            rows.append((tag, attr, val, len(inherit), obj))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', default=str(ROOT / 'scripts' / 'queue_phys.sh'))
    a = ap.parse_args()
    runs = queue_runs(a.queue)
    print(f'=== 由来の監査（{Path(a.queue).name}、{len(runs)} run）')
    cd = code_defaults()
    used = defaultdict(list)
    xmls = set()
    for name, args in runs:
        cfg, xml, keys = specs_of(args)
        xmls.add(xml)
        for k in cd:
            if k not in keys:
                used[k].append(name)
    print('\n① コードの既定値が効いているキー（cfg にも起動行にも書いていない）')
    for (sec, key), names in sorted(used.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        print(f'  {sec}.{key:<28} 既定 {cd[(sec, key)]:<10} {len(names):>3}/{len(runs)} run')
    print('\n② XML の default クラスの属性（⚠️ 対象＝cube に掛かるもの）')
    for xml in sorted(x for x in xmls if x):
        for tag, attr, val, n, obj in xml_defaults(xml):
            if obj:
                print(f'  ⚠️ {xml:<34} <{tag}> {attr}="{val}" が対象にも掛かる: {obj}')
    seen = set()
    for xml in sorted(x for x in xmls if x):
        for tag, attr, val, n, obj in xml_defaults(xml):
            if (tag, attr, val) not in seen:
                seen.add((tag, attr, val))
                print(f'     {tag}.{attr} = {val}（{xml} ほか）')
    print('\n③ 生成スクリプトの既定値（スケッチから測っていない）')
    src = (ROOT / 'scripts' / 'topology_to_xml.py').read_text(encoding='utf-8')
    for m in re.finditer(r'^(DEFAULT_[A-Z_]+)\s*=\s*\{(.*?)^\}', src, re.S | re.M):
        body = ' '.join(l.strip() for l in m.group(2).splitlines() if l.strip() and not l.strip().startswith('#'))
        print(f'  {m.group(1)}: {body[:200]}')


if __name__ == '__main__':
    main()
