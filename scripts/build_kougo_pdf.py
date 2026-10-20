#!/usr/bin/env python3
"""修論の口語版（docs/研究応用/執筆/修論_口語版/）を 1 冊の PDF にする（2026-10-20）。

build_thesis_pdf.py の Markdown→LaTeX 変換とプリアンブルをそのまま借り、表紙の題だけ口語版にする。
- 各ファイル冒頭の「> 」の案内（md で読む人向け。リンクや作り方の説明）は落とす。本文中の「（口語版の注）」は残す
- 00 は「## 要旨」から後だけを使う（章の一覧の表は md 用）
- 1 ファイルに `#` の見出しが 2 つ以上あるとき（第3章は前段と本体を 1 ファイルにした）、2 つ目からは太字の行にして章を割らない

使い方:
    python3 scripts/build_kougo_pdf.py            # → docs/pdf/修論_口語版_YYYYMMDD.pdf
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

SRC = T.ROOT / 'docs' / '研究応用' / '執筆' / '修論_口語版'
BUILD = T.BUILD_DIR / 'kougo'          # build_pdf/ は .gitignore 済み。修論本体のビルドと分ける
ORDER = [  # (ファイル, 章番号を振らないか)
    ('00_はじめに_要旨.md', True),
    ('01_第1章_序論.md', False),
    ('02_第2章_関連研究.md', False),
    ('03_第3章_提案手法.md', False),
    ('04_第4章_実験および評価.md', False),
    ('05_第5章_結論.md', False),
]


def prepare(fname: str) -> str:
    t = (SRC / fname).read_text()
    if fname.startswith('00_'):
        t = '# 要旨\n' + t.split('\n## 要旨\n', 1)[1]
    t = re.sub(r'\n(?:> .*\n|>\n)+', '\n', t, count=1)   # 冒頭の案内の blockquote（最初の 1 つ）だけ落とす
    t = t.replace('（口語版）', '')
    seen = False
    lines = []
    for ln in t.split('\n'):
        if ln.startswith('# '):
            if seen:
                ln = f'**{ln[2:].strip()}**'
            seen = True
        lines.append(ln)
    return '\n'.join(lines)


def main():
    today = datetime.date.today().strftime('%Y%m%d')
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(T.ROOT / 'docs' / 'pdf' / f'修論_口語版_{today}.pdf'))
    args = p.parse_args()

    BUILD.mkdir(parents=True, exist_ok=True)
    body = '\n\n'.join(T.md_to_latex_body(prepare(f), unnumbered=u) for f, u in ORDER)
    date_ja = f'{today[:4]}年{int(today[4:6])}月{int(today[6:8])}日'
    pre = T.PREAMBLE.replace('修士論文ドラフト', '修士論文ドラフト（口語版）')
    tex = BUILD / 'kougo.tex'
    tex.write_text((pre + body + T.POSTAMBLE).replace('__BUILD_DATE__', date_ja))

    for run in range(2):  # 目次のため 2 回
        r = subprocess.run(['xelatex', '-interaction=nonstopmode', '-output-directory', str(BUILD), str(tex)],
                           capture_output=True, text=True, cwd=str(BUILD))
        if r.returncode != 0 and run == 1:
            print('\n'.join(r.stdout.split('\n')[-40:]))
            sys.exit(f'[ERROR] xelatex failed（{BUILD}/kougo.log）')
    shutil.copy(BUILD / 'kougo.pdf', args.out)
    print(f'[done] PDF saved → {args.out} ({(BUILD / "kougo.pdf").stat().st_size // 1024} KB)')


if __name__ == '__main__':
    main()
