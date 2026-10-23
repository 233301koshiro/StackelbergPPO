#!/usr/bin/env python3
"""修論の正式版（docs/研究応用/修論ドラフト/）の本文から、長すぎる文を拾う。

2026-10-23 新設（ユーザー「あとはシンプルに長過ぎる一行とかを探してみよう」）。
文体の目標は先輩の修論（docs/研究応用/執筆/わかりにくい文の判定.md の「0. 文体の基準」）。
先輩の序論は 1 文の平均 82 字、上位 1 割が 125 字を超える。既定の閾値 120 字はこれに合わせた。
⚠️ 長さだけでは決めない。先輩は 100 字超の文を 3 割書いていて読みやすい。直すのは、長いうえに
修飾が入れ子になっている文（長い連体修飾のあとに主名詞が来る、「A だけでなく〔長い B〕に依存する」など）。
拾った文を人が読み、一本道なら残す。

見るのは PDF に入る本文だけ（check_colloquial.py と同じ: 「## 執筆メモ」以降・行頭が > の引用・コードブロックを除く）。
表の行・見出し・英文（要旨の Abstract）も除く。箇条書きは 1 項目を 1 段落として扱う（。で終わらない項目も 1 文と数える）。

  python3 scripts/check_long_sentences.py              # 正式版の全章
  python3 scripts/check_long_sentences.py 第1章        # ファイル名に「第1章」を含むものだけ
  python3 scripts/check_long_sentences.py --max 100    # 閾値を変える
  python3 scripts/check_long_sentences.py --reproduce  # 長い文を拾い、短い文を拾わないかを確かめる
"""
import re, sys, statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_colloquial import FILES, body  # 対象ファイルと本文の切り出しをそろえる

DEFAULT_MAX = 120


def plain(line: str) -> str:
    line = re.sub(r'^\s*(?:\d+\.|[-*])\s+', '', line)        # 箇条書きの記号
    line = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', line)       # リンクは表示文字だけ
    return line.replace('**', '').strip()


def sentences(text: str):
    """(行番号, 文) を返す。"""
    for n, line in enumerate(text.split('\n'), 1):
        if not line.strip() or line.lstrip().startswith(('#', '|', '<', '$$')):
            continue
        if sum(c.isascii() for c in line) > 0.7 * len(line):   # 英文要旨は見ない
            continue
        for s in re.split(r'(?<=。)', plain(line)):
            if len(s.strip()) > 5:
                yield n, s.strip()


def main():
    args = sys.argv[1:]
    limit = DEFAULT_MAX
    if '--max' in args:
        i = args.index('--max'); limit = int(args[i + 1]); del args[i:i + 2]
    if '--reproduce' in args:
        long_ = 'あ' * 130 + '。'
        short = 'い' * 90 + '。'
        hits = [s for _, s in sentences(long_ + short) if len(s) > limit]
        print(f'拾えた: {len(hits)} 件（期待 1 件、130 字の文だけ）')
        sys.exit(0 if hits == [long_] else 1)
    files = [f for f in FILES if not args or any(a in f.name for a in args)]
    total = 0
    for f in files:
        ss = list(sentences(body(f)))
        if not ss:
            continue
        lens = [len(s) for _, s in ss]
        hits = [(n, s) for n, s in ss if len(s) > limit]
        total += len(hits)
        print(f'■ {f.name}: {len(ss)} 文、平均 {statistics.mean(lens):.0f} 字、{limit} 字超 {len(hits)} 文')
        for n, s in sorted(hits, key=lambda x: -len(x[1])):
            print(f'  {f.name}:{n} （{len(s)} 字）{s}')
    print(f'\n{limit} 字を超える文: {total} 件（先輩の序論は 84 文中 10 文）。長さだけでは直さない。入れ子になっているものを直す。')


if __name__ == '__main__':
    main()
