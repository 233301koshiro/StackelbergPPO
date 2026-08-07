#!/usr/bin/env python3
"""
修論ドラフトの全 Markdown を xelatex 経由で1つの PDF にまとめる。

使い方:
  python3 scripts/build_thesis_pdf.py
  python3 scripts/build_thesis_pdf.py --out docs/thesis_draft.pdf
"""
import os, re, subprocess, argparse, shutil, textwrap, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFT_DIR = ROOT / 'docs' / '研究応用' / '修論ドラフト'
FIGURES_DIR = ROOT / 'figures'
BUILD_DIR = ROOT / 'build_pdf'

# 章の読み込み順: (ファイル名, unnumbered)
# unnumbered=True → \chapter* (LaTeX の章カウンタを進めない)
# (ファイル名, unnumbered, opts)
#   opts['title'] : 章タイトルを差し替える（md の H1 を使わない）
#   opts['merge'] : True なら**章を起こさず前の章の続き**として出力する（H1 を落とす）
#
# ⚠️ 第3章の構成（2026-08-06 決定、案A）: 前提と提案手法は**1つの第3章**で、
#    前提が 3.1〜3.4、提案手法が 3.5〜3.13 を占める。
#    本スクリプトは md の節番号を捨てて LaTeX に振り直させる（strip_heading_number）ため、
#    2ファイルを別章にすると提案手法の節が 3.1 から振り直され、
#    本文中の「3.12 節」等の参照が全部ずれる。そこで merge で連結する。
CHAPTER_ORDER = [
    ('要旨.md',                 True,  {}),
    ('第1章_序論.md',           False, {}),
    ('第2章_関連研究.md',       False, {}),
    ('第3章前段_前提.md',       False, {'title': '提案手法: スケッチ起点の形態検証パイプライン'}),
    ('第3章_提案手法.md',       False, {'merge': True}),
    ('第4章_実験および評価.md', False, {}),
    ('第5章_考察.md',           False, {}),
    ('第6章_結論.md',           False, {}),
    ('付録A_プロンプト全文.md', True,  {}),
]

# ── Markdown → LaTeX 変換 ─────────────────────────────────────────────────────

def escape_latex(text: str) -> str:
    """LaTeX の特殊文字をエスケープ（数式・コマンド内は除く）"""
    replacements = [
        ('\\', r'\textbackslash{}'),
        ('&',  r'\&'),
        ('%',  r'\%'),
        ('$',  r'\$'),
        ('#',  r'\#'),
        ('_',  r'\_'),
        ('{',  r'\{'),
        ('}',  r'\}'),
        ('~',  r'\textasciitilde{}'),
        ('^',  r'\textasciicircum{}'),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)
    return text

def md_inline(text: str) -> str:
    """インライン Markdown を LaTeX に変換（太字・斜体・コード・リンク）"""
    # 数式 $...$ は保護
    parts = re.split(r'(\$[^$]+\$)', text)
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # 数式
            result.append(part)
        else:
            p = escape_latex(part)
            # **bold**
            p = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', p)
            # *italic*
            p = re.sub(r'\*(.+?)\*', r'\\textit{\1}', p)
            # `code`
            p = re.sub(r'`([^`]+)`', r'\\texttt{\1}', p)
            # [text](url) → text のみ
            p = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', p)
            # [要文献確認] → 赤字注記
            p = p.replace('[要文献確認]', r'{\small\textcolor{red}{[要文献確認]}}')
            # [要参照挿入: ...] → 赤字注記
            p = re.sub(r'\[要参照挿入: ([^\]]+)\]',
                       r'{\\small\\textcolor{red}{[要参照挿入: \1]}}', p)
            result.append(p)
    return ''.join(result)

def strip_heading_number(title: str) -> str:
    """見出し先頭の番号表記（例: '3.1 ', '第3章 '）を除去して LaTeX 自動番号に委ねる"""
    # 数字番号: "3.1 " "5.2.1 " など
    title = re.sub(r'^(?:\d+\.)+\d*\s+', '', title)
    # 日本語章番号: "第3章 " "第3章前段 " など (先頭のみ)
    title = re.sub(r'^第\d+章(?:前段)?\s+', '', title)
    return title

def md_to_latex_body(md_text: str, unnumbered: bool = False,
                     merge: bool = False, title_override: str = None) -> str:
    r"""
    unnumbered=True のとき、すべての見出しを starred コマンド（\chapter*）にし、
    LaTeX の章カウンタに影響を与えない。章タイトル（レベル1）のみ TOC に追加する。
    """
    lines = md_text.split('\n')
    out = []
    in_code = False
    i = 0

    while i < len(lines):
        line = lines[i]

        # コードブロック
        if line.strip().startswith('```'):
            if not in_code:
                in_code = True
                out.append(r'\begin{verbatim}')
            else:
                in_code = False
                out.append(r'\end{verbatim}')
            i += 1
            continue
        if in_code:
            out.append(line)
            i += 1
            continue

        # ドラフト管理メモ / blockquote → 薄いグレー
        if line.startswith('>'):
            content = line.lstrip('> ').strip()
            if content:
                out.append(r'{\small\color{gray}' + md_inline(content) + r'}\\')
            i += 1
            continue

        # 見出し
        h_match = re.match(r'^(#{1,4})\s+(.+)', line)
        if h_match:
            level = len(h_match.group(1))
            raw_title = md_inline(h_match.group(2))
            title = strip_heading_number(raw_title)
            if level == 1 and merge:
                i += 1          # 前の章の続きなので章見出しを起こさない
                continue
            if level == 1 and title_override:
                title = title_override
            if unnumbered:
                cmds = {1: 'chapter*', 2: 'section*', 3: 'subsection*', 4: 'subsubsection*'}
                out.append(f'\\{cmds.get(level, "paragraph*")}{{{title}}}')
                if level == 1:  # 章タイトルだけ目次に掲載
                    out.append(f'\\addcontentsline{{toc}}{{chapter}}{{{title}}}')
            else:
                cmds = {1: 'chapter', 2: 'section', 3: 'subsection', 4: 'subsubsection'}
                out.append(f'\\{cmds.get(level, "paragraph")}{{{title}}}')
            i += 1
            continue

        # 水平線
        if re.match(r'^---+$', line.strip()):
            out.append(r'\vspace{4pt}\hrule\vspace{4pt}')
            i += 1
            continue

        # テーブル開始判定
        if '|' in line and i + 1 < len(lines) and re.match(r'^[\|\-\s:]+$', lines[i+1]):
            table_rows = []
            while i < len(lines) and '|' in lines[i]:
                if not re.match(r'^[\|\-\s:]+$', lines[i]):
                    cells = [c.strip() for c in lines[i].strip('|').split('|')]
                    table_rows.append(cells)
                i += 1
            if table_rows:
                ncols = max(len(r) for r in table_rows)
                def _disp_width(s):
                    # 全角(CJK等)文字は半角の約2倍の表示幅として概算する
                    return sum(2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for ch in s)
                max_cell_w = max((_disp_width(c) for r in table_rows for c in r), default=0)
                if max_cell_w > 22 or ncols >= 5:
                    # 長文セル（比較表等）は折り返し可能な固定幅列にする。
                    # l| のままだと1行に収まらず紙面外へはみ出す（Overfull \hbox）。
                    col_w = f'{0.92 / ncols:.3f}\\textwidth'
                    col_spec = '|' + (f'p{{{col_w}}}|') * ncols
                else:
                    col_spec = '|' + 'l|' * ncols
                out.append(r'\begin{center}')
                out.append(r'\begin{tabular}{' + col_spec + '}')
                out.append(r'\hline')
                for row in table_rows:
                    row += [''] * (ncols - len(row))
                    cells_tex = ' & '.join(md_inline(c) for c in row)
                    out.append(cells_tex + r' \\')
                    out.append(r'\hline')
                out.append(r'\end{tabular}')
                out.append(r'\end{center}')
            continue

        # 箇条書き（ネスト対応: 先頭スペースで深さを判定）
        if re.match(r'^(\s*)[\-\*]\s+', line):
            out.append(r'\begin{itemize}')
            while i < len(lines) and re.match(r'^(\s*)[\-\*]\s+', lines[i]):
                content = re.sub(r'^\s*[\-\*]\s+', '', lines[i])
                out.append(r'  \item ' + md_inline(content))
                i += 1
            out.append(r'\end{itemize}')
            continue

        # 番号付きリスト
        if re.match(r'^\d+\.\s+', line):
            out.append(r'\begin{enumerate}')
            while i < len(lines) and re.match(r'^\d+\.\s+', lines[i]):
                out.append(r'  \item ' + md_inline(re.sub(r'^\d+\.\s+', '', lines[i])))
                i += 1
            out.append(r'\end{enumerate}')
            continue

        # 空行
        if not line.strip():
            out.append('')
            i += 1
            continue

        # 通常段落
        out.append(md_inline(line))
        i += 1

    return '\n'.join(out)

# ── 図の挿入処理 ─────────────────────────────────────────────────────────────

def insert_figures(latex_body: str) -> str:
    """PJ 学習曲線図を 4.4.3 節の後に挿入"""
    fig_path = FIGURES_DIR / 'pj_learning_curve.png'
    if not fig_path.exists():
        return latex_body

    fig_block = textwrap.dedent(r"""
    \begin{figure}[htbp]
    \centering
    \includegraphics[width=0.92\textwidth]{""" + str(fig_path.resolve()) + r"""}
    \caption{前向き判定テスト（PJ 実験）の学習曲線（200 エポック完走）。
    横軸: エポック数、縦軸: エピソード報酬 exec\_R\_eps。
    短腕（赤）は ep10 以降 $-250$ 付近に頭打ちし、幾何学的到達不可能性（目標 x=0.8\,m に対しリーチ 0.55\,m）
    を反映する。長腕（青）は ep10 で $-48$ まで急上昇し、最終 ep199 で $-8.5$ に収束する。
    順位は ep0 から一貫して長腕 $>$ 短腕であり、217 ポイント差（ep10 時点）以降逆転しない。
    参照ライン（緑破線）は Reach co-design の完走結果（L1）。}
    \label{fig:pj_learning_curve}
    \end{figure}
    """)

    marker = r'\subsection{前向き判定テスト（PJ 実験）}'
    if marker in latex_body:
        idx = latex_body.find(marker)
        ec_idx = latex_body.find(r'\end{center}', idx)
        if ec_idx != -1:
            latex_body = (latex_body[:ec_idx + len(r'\end{center}')]
                          + '\n' + fig_block
                          + latex_body[ec_idx + len(r'\end{center}'):])
    return latex_body

# ── LaTeX テンプレート ────────────────────────────────────────────────────────

PREAMBLE = r"""
\documentclass[12pt,a4paper]{report}
\usepackage{fontspec}
\usepackage{xeCJK}
\setCJKmainfont{Noto Serif CJK JP}
\setCJKsansfont{Noto Sans CJK JP}
\setCJKmonofont{Noto Sans Mono CJK JP}
\setmainfont{TeX Gyre Termes}
\usepackage[margin=25mm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{xcolor}
\usepackage{verbatim}
\usepackage{hyperref}
\usepackage{setspace}
\onehalfspacing
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}

\title{{\large 修士論文ドラフト}\\[8pt]
{\LARGE 形態-制御 co-design を用いたロボットアームの\\タスク適性判定システムの構築}}
\author{垣内研究室}
\date{__BUILD_DATE__（途中経過版）}
\begin{document}
\maketitle
\tableofcontents
\newpage
"""

POSTAMBLE = r"""
\end{document}
"""

# ── メイン ───────────────────────────────────────────────────────────────────

def main():
    import datetime
    today = datetime.date.today().strftime('%Y%m%d')
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(ROOT / 'docs' / f'修論ドラフト_{today}.pdf'))
    args = p.parse_args()

    BUILD_DIR.mkdir(exist_ok=True)
    tex_file = BUILD_DIR / 'thesis.tex'

    body_parts = []
    for fname, unnumbered, opts in CHAPTER_ORDER:
        md_path = DRAFT_DIR / fname
        if not md_path.exists():
            print(f'[skip] {fname} not found')
            continue
        print(f'[read] {fname}  (unnumbered={unnumbered})')
        md_text = md_path.read_text()
        # 執筆メモセクションは除外
        md_text = re.sub(r'## 執筆メモ（本文には含めない）.*', '', md_text, flags=re.DOTALL)
        latex_body = md_to_latex_body(md_text, unnumbered=unnumbered,
                                      merge=opts.get('merge', False),
                                      title_override=opts.get('title'))
        body_parts.append(latex_body)

    full_body = '\n\n'.join(body_parts)
    full_body = insert_figures(full_body)

    build_date_ja = f'{today[:4]}年{int(today[4:6])}月{int(today[6:8])}日'
    tex_content = (PREAMBLE + full_body + POSTAMBLE).replace('__BUILD_DATE__', build_date_ja)
    tex_file.write_text(tex_content)
    print(f'[write] {tex_file}')

    # xelatex で2回コンパイル（目次のため）
    for run in range(2):
        result = subprocess.run(
            ['xelatex', '-interaction=nonstopmode', '-output-directory', str(BUILD_DIR), str(tex_file)],
            capture_output=True, text=True, cwd=str(BUILD_DIR)
        )
        if result.returncode != 0:
            print('[xelatex stderr]')
            print(result.stderr[-3000:])
            print('[xelatex stdout (last 50 lines)]')
            print('\n'.join(result.stdout.split('\n')[-50:]))
            if run == 0:
                print('[warning] first pass failed, trying second pass anyway...')
            else:
                print('[ERROR] xelatex failed')
                return

    pdf_src = BUILD_DIR / 'thesis.pdf'
    if pdf_src.exists():
        shutil.copy(pdf_src, args.out)
        size_kb = pdf_src.stat().st_size // 1024
        print(f'[done] PDF saved → {args.out} ({size_kb} KB)')
    else:
        print('[ERROR] PDF not generated. Check build_pdf/thesis.log')

if __name__ == '__main__':
    main()
