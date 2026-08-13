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

    # ⚠️ Bug 23（2026-08-10）: geom の fromto だけを伸ばすと、関節の取り付け位置
    # （子 body の pos）が元のまま残り、リンクが重なった形態になる。MuJoCo の運動学は
    # body pos の連鎖で決まるため、見た目だけ伸びて**実効リーチは伸びない**。
    # 腕の body（name が数字）の pos も同じ倍率でスケールする。
    # 根元 body "0"（ベース設置高さ）と cube は対象外。
    BODYPOS = re.compile(r'(<body name=")(\d+)("\s+pos=")([-\d.eE ]+)(")')

    def sub_pos(m):
        if m.group(2) == '0':
            return m.group(0)
        vals = [float(v) * args.scale for v in m.group(4).split()]
        return m.group(1) + m.group(2) + m.group(3) + \
            ' '.join(f'{v:.6f}' for v in vals) + m.group(5)

    text = BODYPOS.sub(sub_pos, text)

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

    # Bug 23 再発防止: 公称（geom 長の合計）ではなく、body pos 連鎖から出る
    # **実効リーチ**を出力ファイルから読み直して検算する。両者がずれていたら失敗させる。
    out = open(dst, encoding='utf-8').read()
    offs = [sum(float(v) ** 2 for v in m.group(4).split()) ** 0.5
            for m in BODYPOS.finditer(out)]
    lens = []
    for m in FROMTO.finditer(out):
        v = [float(x) for x in m.group(0).split('"')[1].split()]
        lens.append(sum((v[i + 3] - v[i]) ** 2 for i in range(3)) ** 0.5)
    effective = sum(offs[2:]) + lens[-1]
    nominal = sum(lens)
    print(f'  実効リーチ（body pos 連鎖 + 末端 geom）: {effective:.4f} m')
    if abs(effective - nominal) > 1e-4:
        raise SystemExit(
            f'❌ 公称 {nominal:.4f} m と実効 {effective:.4f} m が食い違う。'
            'Bug 23 と同じ欠陥。生成物を使ってはいけない')
    print('  ✅ 公称と実効が一致（Bug 23 の検算を通過）')


if __name__ == '__main__':
    main()
