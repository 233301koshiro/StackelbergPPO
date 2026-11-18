#!/usr/bin/env python3
"""予備審査論文（docs/研究応用/執筆/予備審査論文/予備審査論文.md）を PDF にする（2026-11-17）。

体裁は先輩の予備審査論文（A4・1 段組・表紙 → 概要 → 目次 → 本文 → 謝辞 → 参考文献）にそろえた。
Markdown→LaTeX の変換は build_thesis_pdf.py を借りる。違いは次のとおり。
- 章（`# 1 序論`）を改ページせず、節と同じく地の文に続けて組む（8〜10 ページに収めるため）
- `# 概要`・`# 謝辞`・`# 参考文献` は番号を振らない。概要は表紙と目次の間、謝辞と参考文献は改ページして置く
- 参考文献は [n] の形で組む
- 冒頭の `>` の行（md で読む人向けの案内）は落とす

使い方:
    python3 scripts/build_yobi_paper_pdf.py          # → docs/pdf/予備審査論文_YYYYMMDD.pdf
"""
import argparse
import datetime
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_thesis_pdf as T  # noqa: E402

SRC = T.ROOT / 'docs' / '研究応用' / '執筆' / '予備審査論文' / '予備審査論文.md'
BUILD = T.BUILD_DIR / 'yobi'            # build_pdf/ は .gitignore 済み
SYSFLOW_SRC = T.ROOT / 'docs' / '研究応用' / '執筆' / '使えそうな画像データ' / 'sysflow_slide.png'

# 表紙。学籍番号は 11-18 にユーザーから。専攻は先輩と同じと仮定（未確認）
TITLE = '手描きスケッチからロボットモデルを生成するシステムの開発'
YEAR = '2026'
DEGREE = '修士（工学）'
MAJOR = '情報・知能工学専攻'
STUDENT_ID = '233301'
AUTHOR = '秋田 光志朗'
ADVISOR = '垣内 洋平'
UNIV = '豊橋技術科学大学'

PREAMBLE = r"""
\documentclass[11pt,a4paper]{report}
\usepackage{fontspec}
\usepackage{xeCJK}
\xeCJKsetcharclass{"2460}{"24FF}{1}
\xeCJKsetcharclass{"25A0}{"25FF}{1}
\setCJKmainfont{Noto Serif CJK JP}
\setCJKsansfont{Noto Sans CJK JP}
\setCJKmonofont{Noto Sans Mono CJK JP}
\setmainfont{TeX Gyre Termes}
\usepackage[margin=25mm]{geometry}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{xcolor}
\usepackage{fvextra}
\usepackage{titlesec}
\usepackage{etoolbox}
\usepackage[hidelinks]{hyperref}
\linespread{1.1}
\renewcommand{\contentsname}{目次}
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{2}
% 章を節と同じ見た目にし、改ページしない（先輩の予備審査論文の型）
\makeatletter
\patchcmd{\chapter}{\if@openright\cleardoublepage\else\clearpage\fi}{}{}{}
\makeatother
\titleformat{\chapter}{\Large\bfseries}{\thechapter}{1em}{}
\titlespacing*{\chapter}{0pt}{18pt}{8pt}
\titleformat{\section}{\large\bfseries}{\thesection}{1em}{}
\titlespacing*{\section}{0pt}{12pt}{4pt}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{1em}{}
\setlength{\parindent}{1em}
\begin{document}
\begin{titlepage}
\centering
\vspace*{30mm}
{\LARGE __TITLE__\par}
\vspace{40mm}
{\large __YEAR__\par}
\vspace{20mm}
{\large __DEGREE__\par}
\vspace{20mm}
{\large __MAJOR__\par}
\vfill
{\large __STUDENT_ID__\par}
\vspace{6mm}
{\large __AUTHOR__\par}
\vspace{6mm}
{\large 指導教員: __ADVISOR__\par}
\vspace{16mm}
{\large __UNIV__\par}
\vspace{20mm}
\end{titlepage}
\setcounter{page}{1}
"""

POSTAMBLE = r"""
\end{document}
"""


def split_parts(md: str):
    """md を `# ` の見出しで分け、[(見出し, 本文)] を返す。冒頭の `>` の案内と H1 の題は落とす"""
    md = re.sub(r'(?m)^>.*(?:\n|\Z)', '', md)
    parts = []
    cur = None
    for ln in md.split('\n'):
        if ln.startswith('# '):
            cur = [ln[2:].strip(), []]
            parts.append(cur)
        elif cur is not None:
            cur[1].append(ln)
    return [(h, '\n'.join(b)) for h, b in parts][1:]   # 先頭は文書の題（「予備審査論文（8〜10 ページ）」）


def refs_to_latex(body: str) -> str:
    items = re.findall(r'(?m)^(\d+)\.\s+(.+)$', body)
    out = [r'\begin{list}{}{\setlength{\leftmargin}{2.2em}\setlength{\labelwidth}{2em}'
           r'\setlength{\itemsep}{2pt}\small}']
    for n, txt in items:
        out.append(rf'\item[[{n}]] ' + T.md_inline(txt))
    out.append(r'\end{list}')
    return '\n'.join(out)


def main():
    today = datetime.date.today().strftime('%Y%m%d')
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(T.ROOT / 'docs' / 'pdf' / f'予備審査論文_{today}.pdf'))
    args = p.parse_args()

    # 図 1 はスライド用に作ったフロー図。figures/ は .gitignore 済み（生成物）なので、ビルドのたびに写す
    shutil.copy(SYSFLOW_SRC, T.FIGURES_DIR / 'yobi_sysflow.png')

    tex_body = []
    for head, body in split_parts(SRC.read_text()):
        if head == '概要':
            tex_body.append(r'\chapter*{概要}' + '\n' + T.md_to_latex_body(body, unnumbered=True)
                            + '\n\\newpage\n\\tableofcontents\n\\newpage')
        elif head == '謝辞':
            tex_body.append(r'\newpage\chapter*{謝辞}' + '\n' + T.md_to_latex_body(body, unnumbered=True))
        elif head == '参考文献':
            tex_body.append(r'\newpage\chapter*{参考文献}\addcontentsline{toc}{chapter}{参考文献}'
                            + '\n' + refs_to_latex(body))
        else:
            # `# 1 序論` の「1」は LaTeX が振るので落とす（落とさないと「1 1 序論」と印字される）
            title = re.sub(r'^\d+\s+', '', head)
            tex_body.append(T.md_to_latex_body(f'# {title}\n{body}'))

    pre = PREAMBLE
    for k, v in dict(TITLE=TITLE, YEAR=YEAR, DEGREE=DEGREE, MAJOR=MAJOR, STUDENT_ID=STUDENT_ID,
                     AUTHOR=AUTHOR, ADVISOR=ADVISOR, UNIV=UNIV).items():
        pre = pre.replace(f'__{k}__', v)

    BUILD.mkdir(parents=True, exist_ok=True)
    tex = BUILD / 'yobi.tex'
    body = '\n\n'.join(tex_body)
    # 表の固定幅の列は合計 0.92\textwidth だが、列の間の余白（\tabcolsep）が足されて右の余白へ 12〜24pt はみ出していた。
    # 修論のビルドは変えず、ここで列の幅だけ縮める
    body = re.sub(r'p\{(0\.\d+)\\textwidth\}', lambda m: 'p{%.3f\\textwidth}' % (float(m.group(1)) * 0.92), body)
    tex.write_text(pre + body + POSTAMBLE)
    for run in range(2):  # 目次のため 2 回
        r = subprocess.run(['xelatex', '-interaction=nonstopmode', '-output-directory', str(BUILD), str(tex)],
                           capture_output=True, text=True, cwd=str(BUILD))
        if r.returncode != 0 and run == 1:
            print('\n'.join(r.stdout.split('\n')[-40:]))
            sys.exit(f'[ERROR] xelatex failed（{BUILD}/yobi.log）')
    shutil.copy(BUILD / 'yobi.pdf', args.out)
    print(f'[done] PDF saved → {args.out} ({(BUILD / "yobi.pdf").stat().st_size // 1024} KB)')


if __name__ == '__main__':
    main()
