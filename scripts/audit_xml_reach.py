#!/usr/bin/env python3
"""アーム XML の「公称リーチ」と「実効リーチ」が一致するかを検算する（Bug 23 の再発防止）。

**なぜ要るか**: MuJoCo の運動学は **body の pos の連鎖**で決まる。geom の `fromto` は
見た目を決めるだけで、関節の取り付け位置には効かない。したがって
`fromto` だけを伸ばした XML は**見た目は伸びているのに実効リーチが伸びない**。
実際に `make_scaled_arm.py` がこの欠陥を持ち、1.99 倍に伸ばしたつもりの形態の
実効リーチが公称の 63% しかないまま 200 エポックの学習を 3 本回してしまった（Bug 23）。

**検算の式**（直列チェーンのみ）:
    実効リーチ = Σ(腕 body の pos の長さ) + 末端リンクの geom 長
    公称リーチ = Σ(geom の fromto 長)
両者がずれていれば、その XML は運動学と見た目が食い違っている。

⚠️ **分岐トポロジー（ant・hopper・walker 等）には適用できない。**
   脚が複数あると body pos の総和が経路長にならないため、誤検出になる。
   本スクリプトは XML を構文解析し、**兄弟 body を持つものは対象外**として除外する。

使い方:
    python3 scripts/audit_xml_reach.py                    # assets/mujoco_envs/*.xml を全部
    python3 scripts/audit_xml_reach.py <file.xml> ...     # 個別指定
    終了コード 1 = 不一致あり（CI やフックで使える）
"""
import glob
import math
import os
import sys
import xml.etree.ElementTree as ET

TOL = 1e-3

# 原論文（Stackelberg PPO / BodyGen）由来の環境。**この検算の対象外**。
# これらは geom が関節をまたいで書かれており（標準的な MuJoCo のモデリング）、
# 「geom 長の総和 = 経路長」という本スクリプトの前提が成り立たない。
# 実際 hopper/gap/swimmer は 50〜53% と出るが欠陥ではない。
# 対象はこのリポジトリのツールが生成したアーム（rrbot_* / tripo_*）に限る。
UPSTREAM_ENVS = {
    'ant.xml', 'climber.xml', 'gap.xml', 'hopper.xml', 'pusher.xml',
    'stair.xml', 'stair-hard.xml', 'swimmer.xml', 'walker.xml',
}


def chain_and_geoms(elem, acc_pos, acc_geom):
    """worldbody 以下を辿り、直列チェーンなら (body pos 長の列, geom 長の列) を返す。
    途中で兄弟 body が現れたら None（＝分岐トポロジー）。"""
    bodies = [c for c in elem if c.tag == 'body']
    if len(bodies) > 1:
        return None
    for g in elem:
        if g.tag == 'geom' and g.get('fromto'):
            v = [float(x) for x in g.get('fromto').split()]
            if len(v) == 6:
                acc_geom.append(math.dist(v[:3], v[3:]))
    if not bodies:
        return acc_pos, acc_geom
    b = bodies[0]
    pos = b.get('pos')
    if pos:
        acc_pos.append(math.dist([0, 0, 0], [float(x) for x in pos.split()]))
    return chain_and_geoms(b, acc_pos, acc_geom)


def audit(path):
    """(状態, 公称, 実効) を返す。状態は 'ok' / 'mismatch' / 'skip'"""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        return 'skip', f'XML を解析できない: {e}', None
    wb = root.find('worldbody')
    if wb is None:
        return 'skip', 'worldbody が無い', None
    # worldbody 直下には腕の根元と対象物（cube 等）が並ぶ。
    # このリポジトリの生成物は腕の body 名を数字にする規約なので、それで腕を選ぶ
    # （`make_scaled_arm.py` の BODYPOS も同じ規約に依存している）。
    tops = [c for c in wb if c.tag == 'body']
    arms = [c for c in tops if (c.get('name') or '').isdigit()]
    if len(arms) != 1:
        return 'skip', f'腕の根元（数字名の body）を一意に特定できない（直下の body {len(tops)} 個）', None
    r = chain_and_geoms(arms[0], [], [])
    if r is None:
        return 'skip', '分岐トポロジー（兄弟 body あり）', None
    poses, geoms = r
    if not geoms or not poses:
        return 'skip', 'geom または body pos が足りない', None
    # 根元 body の pos は設置高さなので経路長に含めない（poses[0] を除く）
    effective = sum(poses[1:]) + geoms[-1]
    nominal = sum(geoms)
    return ('ok' if abs(nominal - effective) < TOL else 'mismatch'), nominal, effective


def main():
    files = sys.argv[1:] or sorted(glob.glob('assets/mujoco_envs/*.xml'))
    bad, ok, skipped = [], [], []
    for p in files:
        if os.path.basename(p) in UPSTREAM_ENVS:
            skipped.append((os.path.basename(p), '原論文由来の環境（前提が成り立たない）'))
            continue
        st, a, b = audit(p)
        if st == 'skip':
            skipped.append((os.path.basename(p), a))
        elif st == 'ok':
            ok.append(os.path.basename(p))
        else:
            bad.append((os.path.basename(p), a, b))

    print(f"検査 {len(files)} 件 → 一致 {len(ok)} / 不一致 {len(bad)} / 対象外 {len(skipped)}\n")
    if bad:
        print("⛔ 公称と実効が食い違う（Bug 23 型。この XML で学習してはいけない）")
        for n, a, b in bad:
            print(f"    {n:38s} 公称 {a:.4f} m → 実効 {b:.4f} m  ({b / a * 100:.0f}%)")
        print()
    if skipped:
        print("－ 対象外（直列チェーンでない等。この検算は適用できない）")
        for n, why in skipped:
            print(f"    {n:38s} {why}")
        print()
    if not bad:
        print("✅ 直列チェーンのアーム XML はすべて公称=実効。")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
