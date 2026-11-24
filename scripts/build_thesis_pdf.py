#!/usr/bin/env python3
"""
修論ドラフトの全 Markdown を xelatex 経由で1つの PDF にまとめる。

使い方:
  python3 scripts/build_thesis_pdf.py
  python3 scripts/build_thesis_pdf.py --out docs/pdf/thesis_draft.pdf
"""
import os, re, subprocess, argparse, shutil, textwrap, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFT_DIR = ROOT / 'docs' / '研究応用' / '修論ドラフト'
# 図は docs/研究応用/執筆/使えそうな画像データ/ の下に用途別に置く（2026-11-19 にルートの figures/ から移した。修論/・予備審査/・勉強/・過去/）
IMG_DIR = ROOT / 'docs' / '研究応用' / '執筆' / '使えそうな画像データ'
FIGURES_DIR = IMG_DIR / '修論'
BUILD_DIR = ROOT / 'build_pdf'


def _tex_safe_figure(path):
    """PDF の図を xelatex が読める形に書き直した写しを返す（PNG などはそのまま）。

    ⛔ 2026-11-13: draw.io（デスクトップ版）が書き出した PDF は、図の全体を URL エンコードして
    メタデータ（Subject）に埋め込む。図 3.1 に画像を貼ったら 1 MB を超え、xdvipdfmx がその PDF を
    取り込むところで Segmentation fault になった（PDF のビルドが止まる）。pdftocairo で描き直した写しは通る。
    元のファイル（ユーザーが書き出した 使えそうな画像データ/修論/*.pdf）は書き換えず、build_pdf/fig_safe/ に写しを置く。
    """
    if path.suffix.lower() != '.pdf':
        return path
    out = BUILD_DIR / 'fig_safe' / path.name
    if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(['pdftocairo', '-pdf', str(path), str(out)], capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        print(f'[warn] pdftocairo で {path.name} を書き直せなかった（元のまま使う）: {r.stderr.strip()[:200]}')
        return path
    return out

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
    ('第5章_結論.md',           False, {}),
    # 参考文献は本文の最後、付録の前に置く。章番号は振らない（unnumbered=True）
    ('参考文献.md',             True,  {}),
    ('付録A_プロンプト全文.md', True,  {}),
    ('付録D_再現情報と補足実験.md', True,  {}),
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
    # 太字が数式をまたぐと上の部分ごとの変換では対にならず、** が印字されていた（2026-10-20、3.2 節。Bug 73 追記）
    out = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', ''.join(result))
    # 数字と「%」の間の空白では改行しない（2026-11-24、予備審査論文 5.8 の「−0.6」の後で改行し、「%」だけが次の行の頭に来た）
    return re.sub(r'(\d) \\%', r'\1~\\%', out)

def heading_number(title: str):
    """見出し先頭の節番号を整数列で返す。'4.3.2 R2…' → [4, 3, 2]。無ければ None。

    ⛔ **2026-10-02 まで、この番号を捨てて LaTeX に振り直させていた。**
    ⚠️ md 側に欠番（4.3.1・4.3.3.2・4.3.4）があったため PDF の番号が md とずれ、
    **PDF 本文が「4.3.3.3 節」と書いているのに、その節が 4.3.2.2 と印字されていた。**
    ⭐ md の番号は本文・台帳・CLAUDE.md が参照する SSOT なので、**md を正とする。**
    """
    m = re.match(r'^((?:\d+\.)+\d*)\s+', title)
    if not m:
        return None
    return [int(x) for x in m.group(1).rstrip('.').split('.') if x != '']


def strip_heading_number(title: str) -> str:
    """見出し先頭の番号表記（例: '3.1 ', '第3章 '）を除去して LaTeX 自動番号に委ねる"""
    # 数字番号: "3.1 " "5.2.1 " など
    title = re.sub(r'^(?:\d+\.)+\d*\s+', '', title)
    # 日本語章番号: "第3章 " "第3章前段 " など (先頭のみ)
    title = re.sub(r'^第\d+章(?:前段)?\s+', '', title)
    return title

# md の目印の絵文字（⭐ ⚠️ ⛔ ✅ ❌ 📝 など）。PDF のフォントに字形が無く「�」になっていた
# （2026-10-23 に 43 箇所。ユーザー「反映されていないなら論文から消して」）。md には残し、PDF でだけ落とす。
# → ← ≤ ①〜⑩ ▲ は意味のある記号で字形もあるので残す。
PDF_DROP_SYMBOLS = re.compile(r'(?:[\u26A0\u26D4\u2705\u274C\u2B50]|[\U0001F000-\U0001FFFF])\uFE0F?[ \u3000]?')

def strip_pdf_symbols(md_text: str) -> str:
    return PDF_DROP_SYMBOLS.sub('', md_text).replace('\uFE0F', '')

# ── 表の列の幅（2026-11-24）────────────────────────────────────────────
# 列の幅を、字の実際の幅で行の折り返しを再現して決める。それまでは列の文字量の平方根に比例させていたので、
# 「最良エピソード報酬」の「酬」、「修正指針の 1.99 倍」の「倍」のように 1 字だけが次の行へ落ち、
# 狭い列では字間が広がっていた（ユーザー「1 文字のみの改行は好きじゃないのでセルを横に伸ばして解決できないか試して。
# その分横が余っているセルを短くして帳尻を合わせる」）。決め方: ① 1 字だけ落ちる折り返しを無くす ② その中で表の高さ
# （行数）を最小にする ③ 余った幅は、どのセルも折り返さない列へ回す。幅は em で出すので、本文の字の大きさ（BODY_PT）
# だけ呼び出し側が合わせる（予備審査論文は 11pt で組むので 10.95 にする）。
BODY_PT = 12.0              # 本文の字の大きさ（pt）。修論・口語版は 12pt
TEXTWIDTH_PT = 455.24       # A4・左右 25 mm の本文幅（160 mm）
TABCOLSEP_PT = 6.0          # LaTeX の既定。列の左右に 1 つずつ付く
ORPHAN_LINES = 3            # 1 字だけ落ちる折り返し 1 つを、表が何行伸びるのと同じに数えるか
BAD_LINES = 0      # 語の途中の割れ 1 つを行いくつと同じ重さで避けるか。0 は「同点のときだけ見る」（11-24 に 0.5・1 も試した。1 は 4 文書の表の計で割れ 117 → 63 だが行が 673 → 694 に伸びた。0.5 は割れ 84・行 678 だが、予備審査論文の表 1 の「出力」の列が 1 行増え、見積もりと LaTeX の折り返しが 1 字ずれて「い」）」だけが落ちた。割れは表の中身〔欄の名前を短くする・注記を題へ移す〕で直す）
_TERMES = {}

def _termes(bold=False):
    """TeX Gyre Termes の字幅（em）。読めなければ None（そのときは半角を 0.5 em と見なす）"""
    key = 'b' if bold else 'r'
    if key not in _TERMES:
        try:
            from fontTools.ttLib import TTFont
            path = subprocess.run(['kpsewhich', 'texgyretermes-bold.otf' if bold else 'texgyretermes-regular.otf'],
                                  capture_output=True, text=True).stdout.strip()
            f = TTFont(path)
            upm = f['head'].unitsPerEm
            cmap, hmtx = f.getBestCmap(), f['hmtx'].metrics
            _TERMES[key] = {cp: hmtx[g][0] / upm for cp, g in cmap.items()}
        except Exception:
            _TERMES[key] = None
    return _TERMES[key]

# xeCJK が和文として組む字（プリアンブルの \xeCJKsetcharclass と、xeCJK が既定で全角に組む約物）
_CJK_RANGES = [(0x2460, 0x24FF), (0x25A0, 0x25FF), (0x2015, 0x2015)]
_CJK_PUNCT = set('“”‘’—…')
_NO_LINE_START = set('、。，．）」』】〕・ー〜ぁぃぅぇぉっゃゅょァィゥェォッャュョ％)]},.:;!?')
_NO_LINE_END = set('（「『【〔([{')
_PUNCT = set('、。，．・（）「」『』【】〔〕')
_OPEN = set('（「『【〔')
_CLOSE = set('）」』】〕、。，．')

def _is_cjk(ch):
    o = ord(ch)
    if unicodedata.east_asian_width(ch) in 'WF' or ch in _CJK_PUNCT:
        return True
    return any(a <= o <= b for a, b in _CJK_RANGES)

def _cell_tokens(text):
    """セルの md を、折り返しの単位 [幅(em), 前に空白があるか] の列にする。
    和文は 1 字ずつ、欧文と数字は空白までを 1 語（ハイフネーションは見込まない＝幅を多めに見る）"""
    t = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    t = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', t)
    # 数式は命令を 1 字分の幅、括弧を 0 と見なして素の字にする
    t = re.sub(r'\$([^$]+)\$', lambda m: re.sub(r'\\[A-Za-z]+', 'x', m.group(1)).replace('{', '').replace('}', '').replace('_', '').replace('^', ''), t)
    code = re.findall(r'`([^`]+)`', t)
    t = re.sub(r'`([^`]+)`', lambda m: '\x00' * len(m.group(1)), t)   # コードは等幅（0.53 em）
    fm = _termes()
    toks, word, space = [], 0.0, False
    def flush():
        nonlocal word, space
        if word:
            toks.append([word, space]); word, space = 0.0, False
    for ch in t:
        if ch == ' ':
            flush(); space = True
        elif ch == '\x00':
            word += 0.53
        elif _is_cjk(ch):
            flush()
            # 和文どうしの間の空白は xeCJK が捨てる（「中腕 対 長腕」は「中腕対長腕」と組まれる。幅も切れ目も無い）
            if space and toks and len(toks[-1]) > 2:
                space = False
            # 約物が続くと xeCJK が半角分詰める（「）・」）。詰めを見込まないと、実際は 1 字多く行に入り、最後の 1 字だけが落ちた
            w = 0.5 if (ch in _PUNCT and toks and len(toks[-1]) > 2 and toks[-1][2] in _PUNCT) else 1.0
            toks.append([w, space, ch]); space = False
        else:
            w = fm.get(ord(ch)) if fm else None
            word += w if w is not None else (1.0 if unicodedata.east_asian_width(ch) == 'A' else 0.5)
    flush()
    return toks

def _script(ch):
    """語の途中かを見るための字の種類（漢字・カタカナ）。それ以外は None"""
    if not ch:
        return None
    o = ord(ch)
    if 0x4E00 <= o <= 0x9FFF:
        return 'kanji'
    if 0x30A0 <= o <= 0x30FF:
        return 'kana'
    return None

def _layout(toks, width, bad=None):
    """幅 width（em）で前から詰めたときの各行の幅。1 語が幅を超えるときはその語で 1 行。
    bad にリストを渡すと、語の途中（漢字どうし・カタカナどうしの間）で割った数を足す（「実／効密度」「ギ／ア比」）。
    行頭に置けない字（、。）など）は前の字と、行末に置けない字（（「 など）は次の字と、1 つの単位にしてから詰める
    （単位にせずに詰めると、「最大トルク（ギ」のように列の幅を超えた行を見積もり、実際の組版と食い違った）"""
    units = []   # [幅, 前に空白, 先頭の字, 末尾の字]
    for i, tk in enumerate(toks):
        w, space = tk[0], tk[1]
        ch = tk[2] if len(tk) > 2 else None
        prev_open = units and units[-1][3] in _NO_LINE_END
        if units and (prev_open or (ch in _NO_LINE_START and not space)):
            units[-1][0] += w + (0.25 if space else 0)
            units[-1][3] = ch
        else:
            units.append([w, space, ch, ch])
    lines, cur, prev_last = [], 0.0, None
    for w, space, first, last in units:
        # 行末の閉じ括弧・句読点と行頭の開き括弧は、xeCJK が半角に詰める
        add = w + (0.25 if space and cur > 0 else 0)
        if cur > 0 and cur + add - (0.5 if last in _CLOSE else 0) > width + 1e-6:
            lines.append(cur)
            cur = w - (0.5 if first in _OPEN else 0)
            if bad is not None and not space and _script(prev_last) and _script(prev_last) == _script(first):
                bad.append(1)
        elif cur == 0:
            cur = w - (0.5 if first in _OPEN else 0)
        else:
            cur += add
        prev_last = last
    lines.append(cur)
    return lines

def _cell_layout(cell, width):
    """セル（<br> で区切った行を含む）の (行数, 必ず避ける折り返しの数, 最後の行が 2 字の数, 語の途中で割った数)。
    必ず避けるのは、<br> で指定した 1 行が折り返すことと、最後の行が全角 1 字になること（ユーザーが嫌うのはこれ）。
    最後の行が 2 字の折り返しは、行が 1 つ増えるのと同じ重さで避ける（11-24 に必ず避けるものにしたら、修論の表 4.3 が 4 行伸びた）"""
    n = orph = short = 0
    bad = []
    segs = cell.split('<br>')
    for seg in segs:
        ls = _layout(_cell_tokens(seg), width, bad)
        n += len(ls)
        if len(ls) > 1 and (len(segs) > 1 or ls[-1] <= 1.5):
            orph += 1
        elif len(ls) > 1 and ls[-1] <= 2.6:
            short += 1
    return n, orph, short, len(bad)

def table_col_widths(rows, ncols, budget_frac=1.0):
    """各列の幅（em）を返す。表の外形（列間の余白と罫線を含む）が本文幅の budget_frac 倍に収まるようにする"""
    pad = (2 * TABCOLSEP_PT + 0.4) / BODY_PT
    budget = budget_frac * TEXTWIDTH_PT / BODY_PT - ncols * pad - 0.4 / BODY_PT
    cols = [[r[j] if j < len(r) else '' for r in rows] for j in range(ncols)]
    margin = 0.1   # 字幅の見積もりの誤差に対する余裕（em）。11-24 に LaTeX の実測と 10 セルを比べ、約物の詰め以外は 1/1000 em まで一致した
    natural = [max(max((sum(t[0] + (0.25 if t[1] else 0) for t in _cell_tokens(s))
                        for s in c.split('<br>')), default=0) for c in col) + margin for col in cols]
    # 各列の候補の幅: 0.25 em 刻みで、行数・避けたい折り返し・語の途中の割れが変わる最小の幅だけを残す
    cands = []
    for j, col in enumerate(cols):
        seen, cj = {}, []
        w = natural[j]
        # 割れない最長の単位（欧文の 1 語・禁則で組になった字）より狭くすると、その語が列からはみ出す（表 4.15 の exec_R_eps）
        longest = max((max(_layout(_cell_tokens(sg), 0.0)) for c in col for sg in c.split('<br>')), default=0) + margin
        lo = max(2.0, min(natural[j], 3.0), longest)
        while w >= lo - 1e-9:
            prof = tuple(_cell_layout(c, w - margin) for c in col)
            seen[prof] = w   # 狭くしていくので、同じ折り返し方の最小の幅で上書きされる
            w -= 0.25
        for prof, w in seen.items():
            cj.append((w, prof))
        cj.sort()
        cands.append(cj)
    nrows = len(rows)
    best = None
    order = sorted(range(ncols), key=lambda j: len(cands[j]))
    # 選び方: ① 表の行数＋最後の行が 2 字の数＋1 字落ちの数×ORPHAN_LINES＋語の途中の割れ×BAD_LINES が少ない ② 語の途中の割れが少ない ③ 幅が狭い。
    #   1 字落ちを絶対に避けると、避けきれない表（修論の表 4.3）で 1 つを減らすために他の列が細くなり、表が 7 行伸びた（11-24）
    def rec(k, used, lines, orph, nshort, nbad, pick):
        nonlocal best
        if used > budget + 1e-9:
            return
        if k == ncols:
            key = (sum(lines) + nshort + ORPHAN_LINES * orph + BAD_LINES * nbad, nbad, used)
            if best is None or key < best[0]:
                best = (key, dict(pick))
            return
        j = order[k]
        for w, prof in cands[j]:
            if used + w + sum(cands[jj][0][0] for jj in order[k+1:]) > budget + 1e-9:
                break
            nl = [max(a, p[0]) for a, p in zip(lines, prof)]
            no = orph + sum(p[1] for p in prof)
            ns = nshort + sum(p[2] for p in prof)
            nb = nbad + sum(p[3] for p in prof)
            if best is not None and sum(nl) + ORPHAN_LINES * no > best[0][0]:
                continue
            pick[j] = w
            rec(k + 1, used + w, nl, no, ns, nb, pick)
            del pick[j]
    rec(0, 0.0, [1] * nrows, 0, 0, 0, {})
    if best is None:
        return None
    widths = [best[1][j] for j in range(ncols)]
    # 余った幅は 0.25 em ずつ、足したときに表の評価（行数＋2 字の最後の行＋1 字落ち×ORPHAN_LINES、次に語の途中の割れ）が
    #   最も良くなる列へ足す。評価が同じなら折り返している列を優先する（11-24 の最初の版は 1 行に収まる列だけに配り、
    #   折り返す列が細いまま「関／節検出」「リンク分／割」と割れていた。新方針の資料の表 1）
    # 表の幅は、収まる限り本文幅の 92 %（これまでの表の幅）にそろえ、それより広く要る表だけ本文幅まで広げる
    extra = max(sum(widths), budget - (1 - 0.92) * TEXTWIDTH_PT / BODY_PT) - sum(widths)
    def score(ws):
        lines = orph = short = bad = 0
        for r in rows:
            lay = [_cell_layout(r[k] if k < len(r) else '', ws[k] - margin) for k in range(ncols)]
            lines += max(l[0] for l in lay)
            orph += sum(l[1] for l in lay); short += sum(l[2] for l in lay); bad += sum(l[3] for l in lay)
        return (lines + short + ORPHAN_LINES * orph + BAD_LINES * bad, bad)
    step = 0.25
    while extra >= step - 1e-9:
        opts = []
        for k in range(ncols):
            trial = list(widths); trial[k] += step
            multi = any(_cell_layout(c, widths[k] - margin)[0] > 1 for c in cols[k])
            opts.append((score(trial), 0 if multi else 1, k))
        _, _, k = min(opts)
        widths[k] += step
        extra -= step
    return widths


def md_to_latex_body(md_text: str, unnumbered: bool = False,
                     merge: bool = False, title_override: str = None) -> str:
    r"""
    unnumbered=True のとき、すべての見出しを starred コマンド（\chapter*）にし、
    LaTeX の章カウンタに影響を与えない。章タイトル（レベル1）のみ TOC に追加する。
    """
    md_text = strip_pdf_symbols(md_text)
    lines = md_text.split('\n')
    out = []
    in_code = False
    i = 0

    while i < len(lines):
        line = lines[i]

        # コードブロック。本文で使うのは付録A のプロンプト全文だけで、文面は再現の条件なので縮めず、
        # 文字だけ小さくする（2026-11-05、ユーザー「①で」。4 ページの付録を詰めるため）。
        # 長い行は余白で切れていたので、改行を足さずに枠の端で折り返す（折り返しには矢印が付く）
        if line.strip().startswith('```'):
            if not in_code:
                in_code = True
                out.append(r'\begin{Verbatim}[fontsize=\footnotesize,baselinestretch=1.1,breaklines,breakanywhere]')
            else:
                in_code = False
                out.append(r'\end{Verbatim}')
            i += 1
            continue
        if in_code:
            out.append(line)
            i += 1
            continue

        # 別行立ての数式 `$$...$$`（1 行）→ \[...\]。⛔ 2026-10-20 まで未対応で、中の $...$ だけが
        #   行内数式になり、外側の $ が 1 つずつ文字として印字されていた（3.1 節の J^F・J^L）
        dm = re.match(r'^\s*\$\$(.+)\$\$\s*$', line)
        if dm:
            out.append(r'\[' + dm.group(1) + r'\]')
            i += 1
            continue

        # 図: `![キャプション](docs/研究応用/執筆/使えそうな画像データ/修論/xxx.png)` を figure 環境にする。
        # 場所はリポジトリの根から書く（2026-11-19 までは figures/ で始まる行だけを図とみなしていた）。
        # 幅は `{width=0.7}` を alt 末尾に付けて指定できる（既定 0.92）。
        img = re.match(r'^!\[(.*?)\]\(([^)\s]+\.(?:png|pdf|jpe?g))\)\s*(\{width=[\d.]+\})?\s*$',
                       line.strip())
        if not img and line.strip().startswith('!['):
            # 図の行に見えるのにパースできないものを**黙って捨てない**。
            # 実際に `{width=1.0}` を閉じ括弧の外に書いた図が1点、
            # 警告も出ずに PDF から欠落した（2026-08-07）。
            print(f'[warn] 図の行を解釈できない（PDF に出ません）: {line.strip()[:70]}')
        if img:
            cap, rel = img.group(1), img.group(2)
            # `{width=}` は alt 内・閉じ括弧の外のどちらに書いてもよい
            wm = re.search(r'\{width=([\d.]+)\}\s*$', cap)
            width = float(wm.group(1)) if wm else (
                float(img.group(3)[7:-1]) if img.group(3) else 0.92)
            if wm:
                cap = cap[:wm.start()].strip()
            path = (ROOT / rel).resolve()
            if not path.exists():
                print(f'[warn] 図が見つからない: {rel}')
                i += 1
                continue
            path = _tex_safe_figure(path)
            # 本ドラフトは図表番号を**本文側で手動管理**している（表と同じ流儀）。
            # \caption を使うと LaTeX が別途採番して本文の「図 4.2」とずれるので、
            # 番号込みのキャプションを小さめの地の文として自分で置く。
            out.append(r'\begin{figure}[htbp]')
            out.append(r'\centering')
            out.append(rf'\includegraphics[width={width}\textwidth]{{{path}}}')
            if cap:
                # 1 行に収まるキャプションは中央に、折り返すものは両端そろえ（2026-11-19、ユーザー「図のキャプションは中央揃えにして」）。
                # それまでは常に左寄せで、短い「図 3: 検証に用いた水平 3 関節アーム。」が図の左下に寄っていた。
                # 折り返すものまで中央にすると各行の端がばらつくので、1 行かどうかを組んだ幅で判定する。
                c = md_inline(cap)
                out.append(r'\\[2mm]')
                out.append(r'\begin{minipage}{0.92\textwidth}{\small \sbox0{' + c + r'}'
                           r'\ifdim\wd0>\linewidth ' + c + r'\else\centering ' + c + r'\par\fi}\end{minipage}')
            out.append(r'\end{figure}')
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
                # ⭐ md の番号が在るなら、その**深さ**で LaTeX の階層を決め、
                #    カウンタを合わせて同じ番号を印字させる（md が正）。
                #    番号が無い見出し（「学習ログとは独立な再現確認」等）は # の数で決める。
                parts = heading_number(raw_title)
                cmds = {1: 'chapter', 2: 'section', 3: 'subsection', 4: 'subsubsection'}
                if parts and 2 <= len(parts) <= 4:
                    cmd = cmds[len(parts)]
                    out.append(f'\\setcounter{{{cmd}}}{{{parts[-1] - 1}}}')
                    out.append(f'\\{cmd}{{{title}}}')
                else:
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
                    # 「\|」（数式のノルムの縦棒）では列を割らない（2026-10-20 まで 4.1.2 の Reach の報酬の式が割れていた）
                    cells = [c.strip() for c in re.split(r'(?<!\\)\|', lines[i].strip().strip('|'))]
                    table_rows.append(cells)
                i += 1
            if table_rows:
                ncols = max(len(r) for r in table_rows)
                def _disp_width(s):
                    # 全角(CJK等)文字は半角の約2倍の表示幅として概算する
                    # 幅の決まらない字（A: ①・○・× など）も、U+2000 以降は日本語の書体で全角に描かれるので 2 と数える（2026-11-20、「結果①②」の「②」だけが折り返した）
                    return sum(2 if (unicodedata.east_asian_width(ch) in 'WF' or (unicodedata.east_asian_width(ch) == 'A' and ord(ch) >= 0x2000)) else 1 for ch in s)
                max_cell_w = max((_disp_width(c) for r in table_rows for c in r), default=0)
                # セルの中の改行（md の <br>）は p 列の \newline でしか効かないので、固定幅列にする（2026-11-19、表の「第1層<br>（幾何）・判定」）
                has_br = any('<br>' in c for r in table_rows for c in r)
                widths = (table_col_widths(table_rows, ncols)
                          if (max_cell_w > 22 or ncols >= 5 or has_br) else None)
                if widths:
                    # 列の幅は字の幅で折り返しを再現して決める（table_col_widths）。左寄せにして、狭い列で字間が広がるのを防ぐ
                    tabx = False
                    col_spec = '|' + ''.join(f'>{{\\raggedright\\arraybackslash}}p{{{w:.2f}em}}|' for w in widths)
                elif max_cell_w > 22 or ncols >= 5 or has_br:
                    # （以下は table_col_widths が幅を決められなかったとき〔フォントが読めない・最小の幅でも本文幅を超える〕の旧い決め方）
                    # 長文セル（比較表等）は折り返し可能な固定幅列にする。
                    # l| のままだと1行に収まらず紙面外へはみ出す（Overfull \hbox）。
                    # 列幅は列の文字量の平方根に比例させる（2026-10-21）。等分だと短い列（「判定」「報告」）が
                    #   長い列と同じ幅を取り、長い列が細く折り返されて表が縦に伸びていた（表 3.2 が 1 ページ近く）。
                    #   高さ ≒ 文字量 ÷ 幅 の和を最小にする幅は √文字量 に比例する。最小は本文幅の 8 %。
                    tot = [sum(_disp_width(r[j]) if j < len(r) else 0 for r in table_rows) for j in range(ncols)]
                    wts = [max(t, 1) ** 0.5 for t in tot]
                    fr = [w / sum(wts) for w in wts]
                    fr = [max(x, 0.08) for x in fr]
                    # <br> で行を指定した列は、その 1 行が折り返さない幅を下限にする（本文幅は全角で約 37 字＝表示幅 75。
                    #   余白の分を 3 足す）。√文字量の幅では「（収束形態）・報告」の「告」だけが次の行へ出た（2026-11-19）
                    mins = {}
                    for j in range(ncols):
                        segs = [s for r in table_rows if j < len(r) and '<br>' in r[j] for s in r[j].split('<br>')]
                        if segs:
                            mins[j] = (max(_disp_width(s) for s in segs) + 3) / 75
                    # 幅を本文幅の 92 % に配る。下限に届かない列は下限に固定し、残りを他の列で √文字量に比例して分ける
                    #   （11-19 まではすべての列を一緒に 92 % へ縮めていたので、下限も 8 % ほど削られていた。11-20）
                    total = 0.92
                    fixed = {}
                    while True:
                        free = [j for j in range(ncols) if j not in fixed]
                        rest = total - sum(fixed.values())
                        wsum = sum(wts[j] for j in free) or 1
                        share = {j: rest * wts[j] / wsum for j in free}
                        lack = [j for j in free if share[j] < max(mins.get(j, 0), 0.08 * total)]
                        if not lack or len(fixed) + len(lack) >= ncols:
                            break
                        for j in lack:
                            fixed[j] = max(mins.get(j, 0), 0.08 * total)
                    fr = [fixed.get(j, share.get(j, 0.08 * total)) for j in range(ncols)]
                    if sum(fr) > total + 1e-9:   # 下限の合計が本文幅を超えるときは、比で縮める（これまでの振る舞い）
                        fr = [x / sum(fr) * total for x in fr]
                    # 最長のセルが全角 10 字以内の短い列（見出し・段階の名前・時期）は折り返さない（2026-11-20、ユーザー
                    #   「診断と修正指針の『針』だけ改行されているのが気になる」）。字の幅の見積もりでは半角の数字や空白で多めに
                    #   取りすぎ、長い列が細くなったので、短い列は l 列にして LaTeX に中身の幅を測らせ、残りの列は tabularx の
                    #   X 列で残りの幅を √文字量の比で分ける。短い列の見積もりの合計が幅の半分を超える表と、短い列しか無い表は、
                    #   これまでの p 列のまま
                    short = [j for j in range(ncols) if j not in mins
                             and max((_disp_width(r[j]) for r in table_rows if j < len(r)), default=0) <= 20]
                    est = sum((max(_disp_width(r[j]) for r in table_rows if j < len(r)) + 3) / 75 for j in short)
                    xcols = [j for j in range(ncols) if j not in short and j not in mins]
                    tabx = bool(short) and bool(xcols) and est <= 0.5 * total
                    if tabx:
                        xs = sum(wts[j] for j in xcols)
                        spec = []
                        for j in range(ncols):
                            if j in short:
                                spec.append('l')
                            elif j in mins:
                                spec.append(f'p{{{fr[j]:.3f}\\textwidth}}')
                            else:
                                spec.append(f'>{{\\hsize={len(xcols) * wts[j] / xs:.3f}\\hsize}}X')
                        col_spec = '|' + '|'.join(spec) + '|'
                    else:
                        col_spec = '|' + ''.join(f'p{{{x:.3f}\\textwidth}}|' for x in fr)
                else:
                    tabx = False
                    col_spec = '|' + 'l|' * ncols
                # 直前の段落が表の題（**表 N.M**: …）なら、題と表を 1 つの minipage に入れて同じページに置く。
                # 分けて出すと、題だけがページの下に残り表が次のページへ送られる（2026-10-22、表 4.3）。
                j = len(out) - 1
                while j >= 0 and not out[j].strip():
                    j -= 1
                caption = None
                if j >= 0 and out[j].startswith(r'\textbf{表 '):
                    caption = out[j]
                    del out[j:]
                if caption:
                    out.append('')
                    out.append(r"\par\vspace{8pt}\noindent\begin{minipage}[t]{\textwidth}")
                    # 表の題も図のキャプションと同じく、1 行に収まるなら中央に置く（2026-11-20、ユーザー「表が中央揃いになってないよ」）。
                    # 折り返す題は行ごとの中央にすると端がばらつくので、両端揃えのまま
                    out.append(r'\sbox0{' + caption + r'}\ifdim\wd0>\linewidth ' + caption
                               + r'\par\else{\centering ' + caption + r'\par}\fi\vspace{4pt}\centering')
                else:
                    out.append(r'\begin{center}')
                out.append((r'\begin{tabularx}{0.92\textwidth}{' if tabx else r'\begin{tabular}{') + col_spec + '}')
                out.append(r'\hline')
                for row in table_rows:
                    row += [''] * (ncols - len(row))
                    cells_tex = ' & '.join(r'\newline '.join(md_inline(x) for x in c.split('<br>')) for c in row)
                    out.append(cells_tex + r' \\')
                    out.append(r'\hline')
                out.append(r'\end{tabularx}' if tabx else r'\end{tabular}')
                if caption:
                    out.append(r"\end{minipage}\par\vspace{8pt}")
                    out.append('')
                else:
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

        # 番号付きリスト。番号は md の数字をそのまま印字する（見出しと同じく md が正）。
        # 2026-11-05 まで LaTeX に 1 から振り直させており、空行や注記で切れたリストが毎回 1 に戻っていた
        # （5.3 節の課題が 1, 1, 2, … と印字され、本文の「5.3 節の 6」が別の項を指していた。参考文献は全件が 1）
        if re.match(r'^\d+\.\s+', line):
            out.append(r'\begin{enumerate}')
            while i < len(lines) and re.match(r'^\d+\.\s+', lines[i]):
                num = int(re.match(r'^(\d+)\.', lines[i]).group(1))
                out.append(r'  \setcounter{enumi}{%d}\item ' % (num - 1)
                           + md_inline(re.sub(r'^\d+\.\s+', '', lines[i])))
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

# 図の挿入は md 側の `![キャプション](docs/研究応用/執筆/使えそうな画像データ/修論/xxx.png)` で行う（2026-08-07。場所は 2026-11-19 に移した）。
# 以前は 4.3.3.1 の PJ 学習曲線だけを LaTeX 本文へ後付けする関数があったが、
# 図を増やすたびに関数を書き足す方式では追随できないため廃止した。


# ── LaTeX テンプレート ────────────────────────────────────────────────────────

PREAMBLE = r"""
\documentclass[12pt,a4paper]{report}
\usepackage{fontspec}
\usepackage{xeCJK}
% 丸数字（①〜）と図形記号（▲）は既定で欧文扱いになり、TeX Gyre Termes に字形が無く黙って消える（2026-10-21、図 3.2 の ①〜③）。和文の字として Noto で組む
\xeCJKsetcharclass{"2460}{"24FF}{1}
\xeCJKsetcharclass{"25A0}{"25FF}{1}  % ▲■○ などの図形記号も同じ（図 3.3 のキャプションの ▲ が消えた）
\setCJKmainfont{Noto Serif CJK JP}
\setCJKsansfont{Noto Sans CJK JP}
\setCJKmonofont{Noto Sans Mono CJK JP}
\setmainfont{TeX Gyre Termes}
\usepackage[margin=25mm]{geometry}
% 第4章は 4.3.2.1 のような4階層目まで使う（軸2 の要求仕様ごとの入れ子）。
% report 既定は secnumdepth=2 で subsubsection が無番号・目次にも出ないため引き上げる。
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{tabularx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{xcolor}
\usepackage{verbatim}
\usepackage{fvextra}
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
    p.add_argument('--out', default=str(ROOT / 'docs' / 'pdf' / f'修論ドラフト_{today}.pdf'))
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
        # ドラフト管理の blockquote も除外（2026-08-07）。
        # 版歴・節番号の繰り下げ履歴・ビルドスクリプトのパス・「案A」といった
        # **執筆者向けの内部メモが全章の冒頭に印字されていた**（実測11箇所）。
        # md には残す（作業に要る）が、配布物である PDF からは落とす。
        md_text = re.sub(r'^> \*\*ドラフト管理\*\*.*?(?=\n(?!>)|\Z)', '',
                         md_text, flags=re.DOTALL | re.M)
        # 本文中の「> 📝 **執筆メモ（本文に残さない）**」も同じく落とす（2026-10-19、PDF に 4 箇所出ていた）
        md_text = re.sub(r'^> 📝 \*\*執筆メモ.*?(?=\n(?!>)|\Z)', '',
                         md_text, flags=re.DOTALL | re.M)
        # 本文の md に引用（> で始まる行）の正当な使い道は無く、残っているのは執筆用の注記だけ。
        # 「ドラフト管理」「📝 執筆メモ」以外の頭（⛔⛔ 2026-10-13 … など）の注記が PDF に灰色で出ていた
        # （2026-10-23、第5章と付録D の 2 箇所）。頭の言葉で拾うと次の新しい頭がまた漏れるので、全部落とす。
        md_text = re.sub(r'(?m)^>.*(?:\n|\Z)', '', md_text)
        latex_body = md_to_latex_body(md_text, unnumbered=unnumbered,
                                      merge=opts.get('merge', False),
                                      title_override=opts.get('title'))
        body_parts.append(latex_body)

    full_body = '\n\n'.join(body_parts)

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
