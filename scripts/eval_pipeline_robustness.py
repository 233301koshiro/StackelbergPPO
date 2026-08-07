#!/usr/bin/env python3
"""M3（関節検出・リンク分割）の頑健性を測る — 第4章「軸1」の評価スクリプト。

**何を測るか**: 生成 AI（Tripo3D）はプロンプトで指定したマーカー色をそのまま出力しない。
公称マゼンタ #FF00FF を指定しても、実際のメッシュ頂点色は暗色化してずれる。
このずれに対して決定論的な色検出がどこまで耐えるか、そして自動キャリブレーションが
どれだけそれを吸収しているかを、**検出成否**と**関節位置の誤差**の2つで定量化する。

**なぜ成功率ではなく感度解析か**: 規約に準拠した GLB は現在2本しかなく（M1/M2 が
Web インタフェース経由のため機械的に増やせない）、成功率としては統計にならない。
一方、色の許容幅は既存メッシュ上で連続的に掃引でき、こちらは統計量として意味を持つ。

**基準値の取り方**: `--auto-color` が返す関節 Z 位置を基準とする。第2回実走の
手実測（メッシュXMLパイプライン.md §8-2、Z = -0.298 / 0.001 / 0.298）と一致することを
確認済みなので、基準として妥当である。

使い方:
    python3 scripts/eval_pipeline_robustness.py
    python3 scripts/eval_pipeline_robustness.py --out docs/研究応用/軸1_頑健性結果.md
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLB2LINKS = os.path.join(ROOT, 'scripts', 'glb_to_links.py')

# 規約準拠 = マゼンタ関節マーカーを付けて生成したメッシュ。
# joystick は規約制定**前**の第1世代でマーカーが無く、成功率の分母には入れない
# （§8 では --joints で関節位置を手指定していた）。検出が明示的に失敗することの
# 確認材料としてのみ用いる。
MESHES = [
    ('v2c 系（積み棒）', 'data/tripo_arm_colorful2/colorful stacking rod 3d model.glb', True),
    ('グリッパ付き',     'data/tripo_arm_colorful_griper/colorful robotic arm 3d model.glb', True),
    ('第1世代（規約前）', 'data/tripo_arm_colorful/mechanical_joystick_3d_model.glb', False),
]
NOMINAL = (255, 0, 255)          # プロンプトで指定している公称マゼンタ
TOLS = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120]


def run(glb, outdir, tol=None, auto=False):
    """glb_to_links を実行し (成否, 関節Z一覧, 較正色) を返す。"""
    cmd = [sys.executable, GLB2LINKS, '--glb', glb, '--out-dir', outdir]
    if auto:
        cmd.append('--auto-color')
    else:
        cmd += ['--joint-color', *map(str, NOMINAL), '--joint-tol', str(tol)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    txt = p.stdout + p.stderr
    m = re.search(r"検出 Z 位置: \[([^\]]+)\]", txt)
    zs = [float(x) for x in re.findall(r'(-?[\d.]+)m', m.group(1))] if m else None
    c = re.search(r'目標色 \((\d+), (\d+), (\d+)\)', txt)
    color = tuple(int(g) for g in c.groups()) if c else None
    return (zs is not None), zs, color


def max_err_mm(zs, ref):
    if zs is None or ref is None or len(zs) != len(ref):
        return None
    return max(abs(a - b) for a, b in zip(sorted(zs), sorted(ref))) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', help='結果を書き出す Markdown パス（省略時は標準出力のみ）')
    args = ap.parse_args()

    lines = []
    def emit(s=''):
        print(s)
        lines.append(s)

    tmp = tempfile.mkdtemp(prefix='robust_')
    emit('# 軸1: 関節検出の頑健性（色ドリフトに対する感度解析）')
    emit()

    n_ok = n_total = 0
    for label, rel, compliant in MESHES:
        glb = os.path.join(ROOT, rel)
        if not os.path.exists(glb):
            emit(f'## {label}\n\n⚠️ GLB が見つからない: `{rel}`\n')
            continue
        emit(f'## {label}{"" if compliant else "（マーカー規約**以前**の生成物）"}')
        emit()
        emit(f'- ファイル: `{rel}`')

        ok_a, ref, color = run(glb, os.path.join(tmp, 'auto'), auto=True)
        if compliant:
            n_total += 1
            n_ok += int(ok_a)
        if ok_a:
            d = max(abs(a - b) for a, b in zip(color, NOMINAL))
            emit(f'- 自動キャリブレーションが復元した実際のマーカー色: **RGB{color}**'
                 f'（公称 RGB{NOMINAL} から最大 **{d}**/チャネルのずれ）')
            emit(f'- 基準とする関節 Z 位置: {", ".join(f"{z:+.4f} m" for z in ref)}')
        else:
            emit('- ❌ **自動キャリブレーションでもマーカーを検出できない**'
                 '（マゼンタ色相域のクラスタが存在しない）。'
                 'エラーとして明示的に停止し、誤ったモデルを生成しない')
            emit()
            continue
        emit()
        emit('| 公称色 + 固定 tolerance | 検出 | 関節位置の最大誤差 |')
        emit('|---|---|---|')
        for t in TOLS:
            ok, zs, _ = run(glb, os.path.join(tmp, f't{t}'), tol=t)
            if not ok:
                emit(f'| tol = {t} | ❌ 失敗 | — |')
            else:
                e = max_err_mm(zs, ref)
                mark = '✅' if e is not None and e < 2.0 else '⚠️'
                emit(f'| tol = {t} | {mark} 検出 | {e:.1f} mm |')
        emit()

    emit('---')
    emit()
    emit(f'**規約準拠メッシュでの自動キャリブレーションの成否: {n_ok}/{n_total}**')
    emit()
    emit('> N が小さく成功率としては統計にならないため、本節の主証拠は上表の'
         '**色ドリフト許容幅**に置く。M1・M2 が Web インタフェース経由であり、'
         'メッシュを機械的に量産できないことによる制約である（限界として第6章に記載）。')

    if args.out:
        p = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
        print(f'\n[write] {p}')


if __name__ == '__main__':
    main()
