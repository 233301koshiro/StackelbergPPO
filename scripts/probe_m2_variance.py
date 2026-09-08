#!/usr/bin/env python3
"""同じ M1 画像から作った複数の GLB を通し、**M2（Tripo3D）だけの分散**を測る。

**なぜ要るか**: 9-22 で「手描きを揃えても XML が 18 % ずれる」と分かったが、
**M1（画像整形）と M2（三次元化）のどちらが揺れているかは未解決**だった（9-22③）。
同じ M1 画像を n 回 M2 に通せば、**残る分散は M2 だけのもの**になる。

**あわせて 9-36 の「閾値付近で判定がまたがる」も測る。** hockey は水平到達 0.848 m で
閾値 0.8 m の 6 % 上しかないため、M2 が ±8 % 揺れるなら **❌ が混ざるはず**である。
n 本のうち ✅ / ❌ が何回かを数える。

前提: GLB は `<dir>/<name>.glb`, `<name>_r2.glb`, ... の形で置く。
      毎回**同じ設定**で生成すること（設定を変えると M2 の分散ではなく設定の影響を測る）。

⚠️ **作業ディレクトリは `data/test/` の外に作る。** `data/test/` は E2E の一次証拠で
`rm -rf data/test*` が deny されており（CLAUDE.md §6）、**中に scratch を作ると片付けられない**。
2026-09-08 に実際に踏んだ。`--work` で場所を指定でき、既定は `/tmp/m2_variance`。

    python3 scripts/probe_m2_variance.py --name hockey
"""
import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent


def reach_of(xml_name):
    r = subprocess.run([sys.executable, 'scripts/diagnose_morphology.py',
                        '--xml', xml_name, '--task', 'reach',
                        '--target', '0.8', '0.0', '0.15'],
                       cwd=ROOT, capture_output=True, text=True)
    m = re.search(r'水平到達限界は約 ([0-9.]+)', r.stdout)
    if m:
        return float(m.group(1)), ('✅' if '可動範囲の内側' in r.stdout else '❌')
    m = re.search(r'総リーチ ([0-9.]+)', r.stdout)     # 平面のときはこちら
    return (float(m.group(1)) if m else float('nan')), ('✅' if '可動範囲の内側' in r.stdout else '❌')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, help='data/test/<name>')
    ap.add_argument('--target', type=float, default=0.8)
    ap.add_argument('--work', default='/tmp/m2_variance',
                    help='作業ディレクトリ。⚠️ data/test/ の中に作らないこと（deny で消せない）')
    args = ap.parse_args()

    base = ROOT / 'data' / 'test' / args.name
    glbs = sorted(base.glob('3D/*.glb'), key=lambda p: (len(p.stem), p.stem))
    if len(glbs) < 2:
        raise SystemExit(f'{base}/3D に GLB が {len(glbs)} 個しかない。n≥2 が要る')

    rows = []
    for i, g in enumerate(glbs, 1):
        tag = f'{args.name}_v{i}'
        work = pathlib.Path(args.work) / tag
        if work.exists():
            shutil.rmtree(work)
        (work / '3D').mkdir(parents=True)
        shutil.copy(g, work / '3D' / f'{tag}.glb')

        # 姿勢チェック（落ちたものも結果として数える）
        pose = subprocess.run([sys.executable, 'scripts/check_glb_pose.py', str(g), '--joints', '3'],
                              cwd=ROOT, capture_output=True, text=True)
        if pose.returncode != 0:
            rows.append(dict(glb=g.name, ok=False, note='姿勢チェックで落ちた'))
            print(f'  {g.name:20} ❌ 姿勢チェックで落ちた')
            continue

        env = dict(VERTICAL='1', LINK_NAMES='base upper_arm forearm hand', FIXED_BASE='0')
        r = subprocess.run(['bash', 'scripts/run_tripo_pipeline.sh', str(g), str(work), tag],
                           cwd=ROOT, capture_output=True, text=True,
                           env={**__import__('os').environ, **env})
        if r.returncode != 0:
            rows.append(dict(glb=g.name, ok=False, note='パイプラインで落ちた'))
            print(f'  {g.name:20} ❌ パイプラインで落ちた')
            continue

        reach, verdict = reach_of(tag)
        jp = work / 'meshes' / 'joints.json'
        total = float('nan')
        if jp.exists():
            fo = json.loads(jp.read_text())['frame_origins']
            ks = list(fo)
            total = sum(float(np.linalg.norm(np.array(fo[ks[i+1]]) - np.array(fo[ks[i]])))
                        for i in range(len(ks) - 1))
        rows.append(dict(glb=g.name, ok=True, reach=reach, verdict=verdict, joint_sum=total))
        print(f'  {g.name:20} 水平到達 {reach:.4f} m  {verdict}  関節間の合計 {total:.4f}')

    ok = [r for r in rows if r['ok']]
    print(f'\n=== まとめ（n={len(rows)}、うち通ったもの {len(ok)}） ===')
    if len(ok) >= 2:
        v = np.array([r['reach'] for r in ok])
        print(f'  水平到達  平均 {v.mean():.4f}  幅 [{v.min():.4f}, {v.max():.4f}]  '
              f'±{(v.max()-v.min())/2/v.mean()*100:.1f} %')
        good = sum(1 for r in ok if r['verdict'] == '✅')
        print(f'  判定      ✅ {good} / ❌ {len(ok)-good}'
              + ('   ⚠️ **同じ画像から判定が割れた**' if 0 < good < len(ok) else ''))
        js = np.array([r['joint_sum'] for r in ok])
        print(f'  関節間合計 幅 ±{(js.max()-js.min())/2/js.mean()*100:.1f} %  '
              f'（9-23 の実測は 8 %）')
    print('\n  ⚠️ ここで出るのは **M2 だけの分散**。9-22 の 18 % から引くと M1 の寄与が出る。')


if __name__ == '__main__':
    main()
