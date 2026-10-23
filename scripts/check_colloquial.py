#!/usr/bin/env python3
"""修論の正式版（docs/研究応用/修論ドラフト/）の本文から話し言葉を拾う。

2026-10-23 新設（ユーザー「なので は話し言葉、〜であるためにする。評価するしかない も、評価する方針を取る でいい。
あなたも話し言葉を注意している感じだったけど抜けがあるので気をつけて」）。
文体の基準は docs/研究応用/執筆/わかりにくい文の判定.md の「0. 文体の基準」。散文の基準は破られるので検査にした（CLAUDE.md §5-1-2）。

見るのは PDF に入る本文だけ（「## 執筆メモ」以降・行頭が > の引用・コードブロック〔付録A のプロンプト全文〕を除く）。
口語版（執筆/修論_口語版/）は話し言葉で書く版なので見ない。

  python3 scripts/check_colloquial.py            # 出た行を並べる。0 件なら終了コード 0
  python3 scripts/check_colloquial.py --reproduce # 既知の話し言葉を拾えるかを確かめる
"""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFT = ROOT / 'docs' / '研究応用' / '修論ドラフト'
FILES = sorted(DRAFT.glob('第*.md')) + sorted(DRAFT.glob('付録*.md')) + [DRAFT / '要旨.md']

# (話し言葉, 書き言葉の言い換え)。語の途中に当たらないよう、前後を絞ったものがある
RULES = [
    (r'なので', '〜であるため・〜のため'),
    (r'しかない', '〜する方針を取る・〜する必要がある'),
    (r'けれど|けど', '〜が'),
    (r'(?:^|。)\s*(?:でも|だから|なのに|ということは)、', 'しかし・したがって・それにもかかわらず'),
    (r'ちゃんと', '正しく・確実に'),
    (r'すごく|とても|かなり', '（程度は数値で言う）'),
    (r'ちょっと|いろいろ|たくさん|どんどん', '（量は数値で言う）'),
    (r'やってみ', '実行する・試行する'),
    (r'じゃない|じゃなく', 'ではない・ではなく'),
    (r'みたいな|みたいに|っていう|なんて', '〜のような・〜という'),
    (r'ですが|です。|ます。|でした。|ました。', 'である調'),
]

def body(path: Path) -> str:
    t = re.split(r'^## 執筆メモ', path.read_text(encoding='utf-8'), flags=re.M)[0]
    t = re.sub(r'^```.*?^```', '', t, flags=re.M | re.S)
    return '\n'.join('' if l.startswith('>') else l for l in t.split('\n'))

def scan(text: str):
    for n, line in enumerate(text.split('\n'), 1):
        for pat, alt in RULES:
            for m in re.finditer(pat, line):
                yield n, m.group(0).strip('。、 '), alt, line[max(0, m.start() - 25):m.end() + 15]

def main():
    if '--reproduce' in sys.argv:
        sample = '得られるのは大まかな情報だけなので、事後に評価するしかない。でも、ちゃんと動く。'
        found = {w for _, w, _, _ in scan(sample)}
        want = {'なので', 'しかない', 'でも', 'ちゃんと'}
        print('拾えた:', sorted(found), '／ 期待:', sorted(want))
        sys.exit(0 if want <= found else 1)
    total = 0
    for f in FILES:
        for n, w, alt, ctx in scan(body(f)):
            total += 1
            print(f'{f.name}:{n}: 「{w}」→ {alt}\n    …{ctx}…')
    print(f'\n話し言葉: {total} 件' if total else '✅ 話し言葉は見つからなかった。')
    sys.exit(1 if total else 0)

if __name__ == '__main__':
    main()
