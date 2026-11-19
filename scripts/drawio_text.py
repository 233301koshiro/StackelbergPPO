#!/usr/bin/env python3
"""drawio の図に書かれた文字を 1 行ずつ出す（圧縮された diagram も展開する）。

**なぜ要るか**: md の検査と、PDF の文字への grep は、どれも `.drawio` を読まない。
2026-11-19 に予備審査論文の最終確認チェックリストで語の置き換え（中間→中腕 など）を洗ったとき、
論文と口語版とデッキは揃っていたが、M6の仕組み.drawio に「中間」、修論 図 3.1 の元の
システム図_論文用.drawio に「手書きスケッチ」が残っていた（ユーザー「よく考えたら drawio もするべき」）。

使い方:
    python3 scripts/drawio_text.py                       # 執筆/ の drawio すべて（md_の木 は除く）
    python3 scripts/drawio_text.py <file.drawio> ...     # 指定したものだけ
    python3 scripts/drawio_text.py | grep 中間           # 語の残りを探す

出力は「ファイル名<TAB>[ページ名] 文字」。セル内の改行は ' / ' にする。
"""
import base64
import html
import re
import sys
import urllib.parse
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / 'docs' / '研究応用' / '執筆'


def texts(path: Path):
    root = ET.parse(path).getroot()
    for d in root.iter('diagram'):
        name = d.get('name')
        inner = d.find('mxGraphModel')
        if inner is None and (d.text or '').strip():
            raw = zlib.decompress(base64.b64decode(d.text.strip()), -15)
            inner = ET.fromstring(urllib.parse.unquote(raw.decode()))
        if inner is None:
            continue
        for c in inner.iter():
            v = c.get('value') or c.get('label')
            if not v:
                continue
            v = re.sub(r'<br\s*/?>', ' / ', html.unescape(v))
            v = html.unescape(re.sub(r'<[^>]+>', '', v)).replace('\xa0', ' ')
            v = re.sub(r'\s*\n\s*', ' / ', v.strip())
            if v:
                yield f'[{name}] {v}'


def main():
    paths = [Path(p) for p in sys.argv[1:]] or sorted(
        p for p in DEFAULT_DIR.rglob('*.drawio') if p.name != 'md_の木.drawio')
    for p in paths:
        for t in texts(p):
            print(f'{p.name}\t{t}')


if __name__ == '__main__':
    main()
