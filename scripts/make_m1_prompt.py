#!/usr/bin/env python3
"""M1（画像整形）用の Gemini プロンプトを、貼ってすぐ使える形で組み立てて出力する。

**なぜスクリプトにするか**: プロンプトは共通部（30 行以上）と 1 枚ごとの比のロック
（1 行）から成る。共通部を md に置いて 5 枚ぶんコピーすると、共通部を直したときに
5 箇所を直す羽目になり必ずどれかが腐る。実際に 2026-09-02、描画の基準を
「台座比」から「腕÷絵の最大寸法」へ変えたとき、**実寸表だけ直してプロンプト側の
比のロックを放置**し、B1 に設計値 1.0/1.0/0.7 を指示するところだった
（実際に描かれていたのは 0.55/0.5/0.5。そのまま渡していれば Gemini が腕を
約 2 倍に伸ばし、短さが目的の入力が壊れていた）。

**共通部の正本はこのファイル。** md は「各行がなぜ要るか」の説明を持つ。

使い方:
    python3 scripts/make_m1_prompt.py --sketch B1
    python3 scripts/make_m1_prompt.py --ratios 0.55 0.5 0.5 --emphasis short

比は **実際に描いた紙を測った値**を入れること。設計値ではない。
M1 の役割は描いたものを保存することなので、設計値を書くと生成側がそちらへ寄せる。
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# 1 枚ごとの強調。狙いが「短い」「長い」の枚は、生成側が親切心で補正するのを止める必要がある。
EMPHASIS = {
    'normal': 'この比を変えないこと',
    # 短さは「腕 ÷ 絵全体の最大寸法」で決まり、その分母の大半が台座である。
    # 腕を伸ばさなくても台座を縮められれば同じことが起きるので、両方を止める。
    'short': ('短く見えても伸ばさないこと。台座を小さくしないこと。'
              '短さがこの入力の目的である'),
    'long': ('長すぎて不格好でも縮めないこと。台座を大きくしないこと。'
             '長さがこの入力の目的である'),
}

TEMPLATE = """添付した手描きスケッチのロボットアームを、下記の仕様に従って
機械設計図スタイルの画像に描き直してください。

【最優先: スケッチの構造を保つこと】
- リンクの本数、各リンクの長さの比、関節の位置は、スケッチに忠実に従うこと
- スケッチにない関節を足したり、あるものを省いたりしないこと
- 全体のプロポーション（台座の高さに対する各リンクの長さの比）を変えないこと
- 台座の高さを 1 としたとき、リンクの長さは根元から {ratios} である。{emphasis}
- スケッチが粗くても、読み取れる構造をそのまま反映すること。
  「きれいに直す」ためにリンク長の比を変えるのは禁止
- スケッチの紙にある罫線は無視すること。罫線を図の一部として描かないこと

【形状・姿勢 — 最重要】
- 固定台座から垂直に伸びるシリアルチェーンアーム（根元が下・先端が上）
- 各リンクは円柱カプセル。先端リンクのみ、やや太い円柱でよい
- 関節が同じ高さに並ばないよう、各リンクに角度をつけて描くこと

【関節マーカー — 最重要・省略厳禁】
すべての接続部に、純マゼンタ (#FF00FF) の球を1個ずつ描くこと。具体的には以下の{n}箇所:
{joint_list}
- スケッチで隣り合うリンクがほぼ一直線に並んでいる場合も、その境目に必ずマゼンタ球を描くこと。
  2本のリンクを1本の長いリンクとして描くのは禁止
- {n}箇所すべて同じ様式（リンクより太いマゼンタの球）で統一すること。
  先端側の関節だけを金具・ヒンジ・ブラケット等の機械要素で描くのは禁止
- 描き終えたら、マゼンタの球がちょうど{n}個あることを確認すること

【色分け — グラデーション禁止・各色を単色で塗ること】
- ベース台座: 濃いグレー (#333333)
- リンク1（上腕）: 鮮明な赤 (#FF2222)
- リンク2（前腕）: 鮮明な青 (#2222FF)
- リンク3（先端）: 鮮明な緑 (#22CC22)
- 関節球: 純マゼンタ (#FF00FF)
  ※ マゼンタはアーム本体色（赤・青・緑・グレー）には絶対に使わないこと（自動検出用のため）

【スタイル】
- 白背景
- CADソフトウェアのレンダリング風（プラスチック質感・影あり・幾何学的）
- 斜め45度視点（3/4アングル）、ロボット全体が収まる構図"""

JOINTS_3 = """1. 台座と上腕（赤）の間
2. 上腕（赤）と前腕（青）の間
3. 前腕（青）と先端（緑）の間 ← ここも省略しないこと"""


def build(ratios, emphasis):
    if len(ratios) != 3:
        raise SystemExit('リンクは 3 本を前提にしている（--ratios を 3 つ指定）')
    return TEMPLATE.format(
        ratios=' / '.join(f'{r:g}' for r in ratios),
        emphasis=EMPHASIS[emphasis],
        n=len(ratios),
        joint_list=JOINTS_3,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sketch', help='data/test/<名前>/sketch/measured.json から読む')
    ap.add_argument('--ratios', nargs=3, type=float,
                    help='台座高を 1 としたときのリンク長（実測値）')
    ap.add_argument('--emphasis', choices=sorted(EMPHASIS), default='normal',
                    help='1 枚ごとの強調。short/long は生成側の「親切な補正」を止める')
    args = ap.parse_args()

    ratios, emphasis = args.ratios, args.emphasis
    if args.sketch:
        f = ROOT / 'data' / 'test' / args.sketch / 'sketch' / 'measured.json'
        if not f.exists():
            raise SystemExit(
                f'{f} が無い。先に紙を測って次の形式で作ること:\n'
                '  {"ratios_base1": [0.55, 0.5, 0.5], "emphasis": "short",\n'
                '   "arm_over_bbox": 0.705, "note": "三稿。実測 base 343 px / links 191,178,177 px"}')
        d = json.loads(f.read_text(encoding='utf-8'))
        ratios = d['ratios_base1']
        emphasis = d.get('emphasis', 'normal')
        print(f'# {args.sketch}: {f.relative_to(ROOT)} より', file=sys.stderr)
        if 'arm_over_bbox' in d:
            r = d['arm_over_bbox']
            print(f'#   腕÷絵の最大寸法 = {r}  → 予測される絶対リーチ 約 {r:.2f} m'
                  f'（Reach の目標 0.8 m に対し{"棄却される見込み" if r < 0.8 else "⚠️ 棄却されない見込み"}）',
                  file=sys.stderr)
        if 'note' in d:
            print(f'#   {d["note"]}', file=sys.stderr)
        print('', file=sys.stderr)

    if not ratios:
        raise SystemExit('--sketch か --ratios のどちらかを指定すること')
    print(build(ratios, emphasis))


if __name__ == '__main__':
    main()
