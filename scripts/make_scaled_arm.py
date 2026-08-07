#!/usr/bin/env python3
"""既存アーム XML のリンク長だけを一定倍率でスケールした XML を作る。

**用途（第4章「軸3」= 診断フィードバックの閉ループ検証）**:
診断の第1層は不適合な形態に対し「腕全体を N 倍以上に伸ばしてください」という
反実仮想的な指示を返す。その指示が**実際に有効か**を確かめるには、指示された倍率
ちょうどの形態を作って再学習し、達成可能な状態へ遷移するかを見る必要がある。
本スクリプトはその形態を生成する。

**リンク長のみを変える**（capsule 半径・ギア比・目標位置は据え置く）。これは
既存のスケール変種 `tripo_arm_v2c_pj_{short,mid,long}` と同じ流儀であり、
診断が言う「腕全体を N 倍に伸ばす」という操作の忠実な実装でもある。
半径まで変えると比較に交絡が入る。

使い方:
    python3 scripts/make_scaled_arm.py --base tripo_arm_v2c_pj_short \
        --scale 1.98 --name tripo_arm_v2c_pj_rec_reach
"""
import argparse
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENVS = os.path.join(ROOT, 'assets', 'mujoco_envs')

FROMTO = re.compile(
    r'(fromto=")([-\d.eE]+) +([-\d.eE]+) +([-\d.eE]+) +([-\d.eE]+) +([-\d.eE]+) +([-\d.eE]+)(")')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='元の XML 名（拡張子なし）')
    ap.add_argument('--scale', type=float, required=True, help='リンク長の倍率')
    ap.add_argument('--name', required=True, help='出力 XML 名（拡張子なし）')
    args = ap.parse_args()

    src = os.path.join(ENVS, args.base + '.xml')
    dst = os.path.join(ENVS, args.name + '.xml')
    if not os.path.exists(src):
        raise SystemExit(f'元 XML が無い: {src}')
    if os.path.exists(dst):
        raise SystemExit(f'出力先が既にある（上書きしない）: {dst}')

    text = open(src, encoding='utf-8').read()
    before, after = [], []

    def sub(m):
        vals = [float(m.group(i)) for i in range(2, 8)]
        before.append(sum(v * v for v in vals[3:]) ** 0.5)
        vals = [v * args.scale for v in vals]
        after.append(sum(v * v for v in vals[3:]) ** 0.5)
        body = ' '.join(f'{v:.6f}' for v in vals[:3]) + '  ' + \
               ' '.join(f'{v:.6f}' for v in vals[3:])
        return m.group(1) + body + m.group(8)

    text = FROMTO.sub(sub, text)
    text = re.sub(r'(<mujoco model=")[^"]*(")',
                  rf'\g<1>3-joint_serial_arm_({args.name}, {args.scale}x_of_{args.base})\g<2>',
                  text, count=1)
    open(dst, 'w', encoding='utf-8').write(text)

    print(f'[write] assets/mujoco_envs/{args.name}.xml')
    print(f'  倍率 {args.scale}x  リンク {len(before)} 本')
    print(f'  リンク長: {", ".join(f"{b:.4f}" for b in before)}  '
          f'(合計 {sum(before):.4f} m)')
    print(f'       →   {", ".join(f"{a:.4f}" for a in after)}  '
          f'(合計 {sum(after):.4f} m)')
    print('  ※ capsule 半径・ギア比・目標位置は変更していない')


if __name__ == '__main__':
    main()
