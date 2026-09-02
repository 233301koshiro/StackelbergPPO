#!/usr/bin/env python3
"""指定した関節を固定（可動域ゼロ・アクチュエータ削除）した XML を作る。

**用途（第3層の関節使用率の助言を閉ループで検証する）**:
第3層は「この関節はほとんど動いていません。固定すれば質量と制御の自由度を
減らせる可能性があります」と提案する（3.12.2 節、実験系譜 9-20）。
その提案が**本当に有効か**を確かめるには、提案どおり固定した形態を作って再学習し、
**性能が落ちないこと**を見る必要がある。本スクリプトはその形態を生成する。

落ちなければ助言は有効。落ちれば「使用率の低さは冗長性を含意しない」という
反証になる。**どちらでも結論が出る。** 第1層の助言に対する 9-11 と同じ構造である。

**固定の仕方**: `<joint>` を削除するのではなく **range を 0 にして actuator を外す**。
削除するとリンクの親子関係の記述が変わり、リンク長・質量が同じであることを
保証しづらい。range=0 なら**形態は完全に同一で、動く自由度だけが減る**。

使い方:
    python3 scripts/make_fixed_joint_arm.py --base e2e_a1 --fix 3 --name e2e_a1_fix3
"""
import argparse
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENVS = os.path.join(ROOT, 'assets', 'mujoco_envs')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='元の XML 名（拡張子なし）')
    ap.add_argument('--fix', type=int, nargs='+', required=True,
                    help='固定する関節の番号（根元から 1 始まり）')
    ap.add_argument('--name', required=True, help='出力 XML 名（拡張子なし）')
    args = ap.parse_args()

    src = os.path.join(ENVS, args.base + '.xml')
    dst = os.path.join(ENVS, args.name + '.xml')
    if not os.path.exists(src):
        raise SystemExit(f'元 XML が無い: {src}')
    if os.path.exists(dst):
        raise SystemExit(f'出力先が既にある（上書きしない）: {dst}')

    text = open(src, encoding='utf-8').read()

    # 腕の関節名は 1_joint, 11_joint, 111_joint ... と根元から桁が増える
    names = re.findall(r'<joint name="(1+)_joint"', text)
    if not names:
        raise SystemExit('腕の関節（1_joint 形式）が見つからない')
    names.sort(key=len)
    print(f'  関節: {", ".join(n + "_joint" for n in names)}（根元から）')

    for k in args.fix:
        if not 1 <= k <= len(names):
            raise SystemExit(f'--fix {k} は範囲外（関節は {len(names)} 個）')
        jn = names[k - 1] + '_joint'
        # ① 可動域を 0 にする
        before = text
        text = re.sub(rf'(<joint name="{jn}"[^>]*range=")[^"]*(")',
                      r'\g<1>0.0 0.0\g<2>', text)
        if text == before:
            raise SystemExit(f'{jn} の range を書き換えられなかった')
        # ② アクチュエータを外す（制御次元が減る）
        text = re.sub(rf'\s*<motor[^>]*joint="{jn}"[^>]*/>', '', text)
        print(f'  関節{k}（{jn}）: range=0.0 0.0、motor を削除')

    text = re.sub(r'(<mujoco model=")[^"]*(")',
                  rf'\g<1>{args.name}_(fix{"+".join(map(str, args.fix))}_of_{args.base})\g<2>',
                  text, count=1)
    open(dst, 'w', encoding='utf-8').write(text)
    print(f'[write] assets/mujoco_envs/{args.name}.xml')
    print('  ※ リンク長・太さ・質量は元と完全に同一。動く自由度だけが減っている')
    print('  ※ 検証: 同条件で再学習し、性能が落ちなければ助言は有効')


if __name__ == '__main__':
    main()
