#!/usr/bin/env python3
"""中間発表の配布資料（論文2枚版・6枚版）を PDF にする。

`build_thesis_pdf.py` の Markdown→LaTeX 変換をそのまま再利用し、
**プリアンブルだけ差し替える**。修論は `report` クラスで章立てが前提だが、
配布資料は数ページの読み物なので `article` にし、目次も付けない。

使い方:
    python3 scripts/build_handout_pdf.py                 # 2枚版・6枚版の両方
    python3 scripts/build_handout_pdf.py 論文2枚.md      # 個別
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_thesis_pdf as B   # 変換ロジックを再利用（重複実装を作らない）

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / 'docs' / '研究応用'
BUILD_DIR = ROOT / 'build_pdf'
TARGETS = ['論文2枚.md', '論文6枚.md']

PREAMBLE = r"""
\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}
\usepackage{xeCJK}
\setCJKmainfont{Noto Serif CJK JP}
\setCJKsansfont{Noto Sans CJK JP}
\setCJKmonofont{Noto Sans Mono CJK JP}
\setmainfont{TeX Gyre Termes}
% 配布資料なので余白を詰めて情報密度を上げる。
% ⚠️ 「2枚版」「6枚版」は**ページ数がそのまま名前になっている**ので、
%    余白・行送りを緩めると名前と実物が食い違う。変更したらページ数を必ず確認すること。
\usepackage[margin=18mm]{geometry}
\linespread{0.97}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\usepackage{xcolor}
\usepackage{amsmath}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\setlist{nosep,leftmargin=1.6em}
\setlength{\parskip}{0.3\baselineskip}
\setlength{\parindent}{0pt}
% 見出しを詰める
\usepackage{titlesec}
\titlespacing*{\section}{0pt}{0.8\baselineskip}{0.25\baselineskip}
\titlespacing*{\subsection}{0pt}{0.7\baselineskip}{0.2\baselineskip}
\begin{document}
"""


def build(md_name):
    src = SRC_DIR / md_name
    if not src.exists():
        print(f'[skip] {md_name} が無い')
        return None
    text = src.read_text(encoding='utf-8')

    # タイトルを H1 から取り、本文からは落とす
    m = re.match(r'^#\s+(.+)', text)
    title = B.md_inline(m.group(1)) if m else md_name
    text = re.sub(r'^#\s+.+\n', '', text, count=1)

    # 「**（6枚版・…）**」の副題行を拾って副題にする
    sub = ''
    m2 = re.match(r'\s*\*\*（(.+?)）\*\*\s*\n', text)
    if m2:
        sub = B.md_inline(m2.group(1))
        text = text[m2.end():]

    # 出典を示す blockquote は配布資料には出さない（内部向けメモのため）
    text = re.sub(r'^>.*$\n?', '', text, flags=re.M)

    body = B.md_to_latex_body(text, unnumbered=False)
    # article には chapter が無い。md の `##` は変換側で \section になっているので、
    # ⚠️ **連鎖置換にすると多重に落ちて 0.1 のような番号になる**（実際にそうなった）。
    # プレースホルダを挟んで1回だけ写す。
    #   md `#`   → \chapter      → （配布資料では使わない）
    #   md `##`  → \section      → \section       （そのまま）
    #   md `###` → \subsection   → \subsection    （そのまま）
    body = body.replace(r'\chapter*{', '@@SEC*{').replace(r'\chapter{', '@@SEC{')
    body = body.replace('@@SEC*{', r'\section*{').replace('@@SEC{', r'\section{')

    head = (PREAMBLE
            + r'\begin{center}{\LARGE\bfseries ' + title + r'}\\[2mm]'
            + (r'{\large ' + sub + r'}\\[1mm]' if sub else '')
            + r'\end{center}' + '\n\n')
    tex = head + body + '\n\\end{document}\n'

    BUILD_DIR.mkdir(exist_ok=True)
    stem = src.stem
    tex_path = BUILD_DIR / f'{stem}.tex'
    tex_path.write_text(tex, encoding='utf-8')

    for _ in range(2):          # 参照を解決するため2回
        r = subprocess.run(['xelatex', '-interaction=nonstopmode',
                            '-output-directory', str(BUILD_DIR), str(tex_path)],
                           capture_output=True, text=True)
    pdf = BUILD_DIR / f'{stem}.pdf'
    if not pdf.exists():
        print(f'[error] {stem}: PDF が生成されなかった')
        for line in r.stdout.splitlines():
            if line.startswith('!'):
                print(f'    {line}')
        return None
    out = ROOT / 'docs' / f'{stem}.pdf'
    out.write_bytes(pdf.read_bytes())
    pages = subprocess.run(['pdfinfo', str(out)], capture_output=True, text=True).stdout
    n = re.search(r'Pages:\s+(\d+)', pages)
    print(f'[done] {out.relative_to(ROOT)}  ({n.group(1) if n else "?"} ページ, '
          f'{out.stat().st_size // 1024} KB)')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('targets', nargs='*', default=None)
    a = ap.parse_args()
    for t in (a.targets or TARGETS):
        build(t)


if __name__ == '__main__':
    main()
