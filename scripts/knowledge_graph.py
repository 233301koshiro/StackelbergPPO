#!/usr/bin/env python3
"""研究の知識ネットワーク — 知見・修論の節・想定問答・系譜の段・Bug を 1 枚のグラフにする（2026-10-23、ユーザー提案）。

**目的は研究の全体像を眺めること。** md を刈るためではない（それは doc_tree.py）。
ノードはファイルではなく、**すでに番号の付いた知識の単位**:

| 種類 | 出どころ | 例 |
|---|---|---|
| 知見 | 知見総括表 A〜D・F・G の行 | 「判定は長いほど良いの単調尺度か」 |
| 取り下げ | 知見総括表 E の行 | 「9-245 以前の学習の結果を現在の結論として言う」 |
| 修論 | 修論の章と節の見出し | 修論 4.4.3.2 |
| 想定問答 | 想定問答の Q | Q14b |
| 系譜 | 実験系譜の段 | 系譜 9-285 |
| Bug | デバッグ戦記 | Bug 76 |

- **幹（木）**: 研究の問い → 5 つの台帳 → 主題（総括表の節・系譜の索引・修論の章）→ 各ノード。
- **つながり**: あるノードの本文に別のノードの識別子（`系譜 9-285`・`Bug 76`・`修論 4.4.3.1`・`想定問答 Q14`）が
  出てきたら辺を張る。**手で保守するものは無い**（ドキュメントの構造と監査.md の「① 生成物にする」）。

    python3 scripts/knowledge_graph.py            # docs/リポジトリ説明/知識ネットワーク.html を書く
    python3 scripts/knowledge_graph.py --stats    # ノードと辺の数だけ出す
    python3 scripts/knowledge_graph.py --reproduce  # 既知のつながりを拾えるか

⚠️ 辺は「本文に識別子が出てくる」だけで、支持か反証かは区別しない。中身はノードの出どころ（ファイル:行）で読む。
"""
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / 'docs'
LEDGER = DOCS / '研究応用/台帳/実験系譜.md'
DEBUG = DOCS / 'デバッグ戦記.md'
QA = DOCS / '研究応用/執筆/想定問答.md'
SUMMARY = DOCS / '研究応用/執筆/知見総括表.md'
DRAFT = DOCS / '研究応用/修論ドラフト'
TEMPLATE = Path(__file__).resolve().parent / 'knowledge_graph_template.html'
OUT = DOCS / 'リポジトリ説明/知識ネットワーク.html'

THESIS = ['要旨.md', '第1章_序論.md', '第2章_関連研究.md', '第3章前段_前提.md', '第3章_提案手法.md',
          '第4章_実験および評価.md', '第5章_結論.md', '付録A_プロンプト全文.md', '付録D_再現情報と補足実験.md']
QUESTION = 'スケッチや形態案が、タスクに対して妥当かを素早く検証できるか'


def plain(s: str, n: int = 0) -> str:
    s = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', s)
    s = re.sub(r'[*`>#|~]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return (s[:n] + '…') if n and len(s) > n else s


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


class Graph:
    def __init__(self):
        self.nodes = collections.OrderedDict()   # id → dict
        self.bodies = {}                          # id → (本文, 文脈)

    def add(self, nid, kind, label, parent, src='', text='', flag='', body='', ctx=''):
        if nid in self.nodes:
            return
        self.nodes[nid] = dict(id=nid, kind=kind, label=label, parent=parent, src=src,
                               text=plain(text, 360), flag=flag)
        if body:
            self.bodies[nid] = (body, ctx)


def build() -> Graph:
    g = Graph()
    g.add('root', 'root', '研究の問い', None, text=QUESTION)
    for hub, label, p in [('hub:知見', '知見総括表', SUMMARY), ('hub:修論', '修論', DRAFT / '第4章_実験および評価.md'),
                          ('hub:想定問答', '想定問答', QA), ('hub:系譜', '実験系譜', LEDGER), ('hub:Bug', 'デバッグ戦記', DEBUG)]:
        g.add(hub, 'hub', label, 'root', src=rel(p) if hub != 'hub:修論' else rel(DRAFT))

    # ── 知見総括表 ──
    lines = SUMMARY.read_text(encoding='utf-8').splitlines()
    sec, header, k = None, None, 0
    for i, line in enumerate(lines, 1):
        m = re.match(r'^## ([A-G])\. (.*)', line)
        if m:
            sec, header, k = m.group(1), None, 0
            g.add(f'知見 {sec}', 'theme', f'{sec}. {plain(m.group(2))}', 'hub:知見', src=f'{rel(SUMMARY)}:{i}')
            continue
        if not sec or not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if re.match(r'^[-:\s|]+$', line):
            continue
        if header is None or (i < len(lines) and re.match(r'^\|\s*[-:]', lines[i])):
            header = cells
            continue
        k += 1
        col = {h: c for h, c in zip(header, cells)}
        if sec == 'E':
            nid, kind = f'取り下げ {k}', 'retracted'
            label = col.get('当初の主張', cells[0])
            text = f"何で崩れたか: {col.get('何で崩れたか', '')}　現在: {col.get('現在', '')}"
        elif '仮説' in col:
            nid, kind = f'知見 {sec}{k}', 'finding'
            label = col['仮説']
            text = f"知見: {col.get('知見', '')}　実験: {col.get('実験', '')}　数値: {col.get('決定的な数値', '')}　系譜の段: {col.get('系譜の段', '')}"
        else:
            nid, kind = f'知見 {sec}{k}', 'finding'
            label = cells[0]
            text = '　'.join(f'{h}: {c}' for h, c in zip(header[1:], cells[1:]))
        g.add(nid, kind, plain(label, 48), f'知見 {sec}', src=f'{rel(SUMMARY)}:{i}', text=text,
              body=line, ctx='summary')

    # ── 修論 ──
    secs = {}
    for name in THESIS:
        p = DRAFT / name
        text = p.read_text(encoding='utf-8')
        text = re.sub(r'## 執筆メモ（本文には含めない）.*', '', text, flags=re.DOTALL)
        cur, buf, start = None, [], 0
        tl = text.splitlines()

        def flush():
            if cur:
                g.bodies[cur] = ('\n'.join(buf), 'thesis')
                if not g.nodes[cur]['text']:
                    g.nodes[cur]['text'] = plain(' '.join(b for b in buf if not b.startswith('|')), 360)
        for i, line in enumerate(tl, 1):
            m = re.match(r'^#\s+(第(\d+)章|付録([A-D])|要旨)\s*(.*)', line)
            m2 = re.match(r'^#{2,5}\s+(?:付録)?([0-9A-D]+(?:\.[0-9]+)+)\s+(.*)', line)
            if m or m2:
                flush()
                buf = []
                if m:
                    chap = m.group(1) if not m.group(3) else f'付録{m.group(3)}'
                    cur = f'修論 {chap}'
                    g.add(cur, 'chapter', plain(f'{chap} {m.group(4)}', 40), 'hub:修論', src=f'{rel(p)}:{i}')
                else:
                    num = m2.group(1)
                    head = num.rsplit('.', 1)[0]
                    if '.' in head:
                        parent = f'修論 {head}'
                    else:
                        parent = f'修論 第{head}章' if head.isdigit() else f'修論 付録{head}'
                    cur = f'修論 {num}'
                    secs[num] = cur
                    if parent not in g.nodes:
                        parent = f'修論 第{num.split(".")[0]}章' if num[0].isdigit() else 'hub:修論'
                    g.add(cur, 'section', plain(f'{num} {m2.group(2)}', 44), parent, src=f'{rel(p)}:{i}')
                continue
            buf.append(line)
        flush()
    g.secs = secs

    # ── 想定問答 ──
    tl = QA.read_text(encoding='utf-8').splitlines()
    cur, buf = None, []
    for i, line in enumerate(tl, 1):
        m = re.match(r'^#{2,3}\s+(Q\d*[a-z]?)[.:：]?\s*(.*)', line)
        if m:
            if cur:
                g.bodies[cur] = ('\n'.join(buf), 'qa')
            q = m.group(1) if m.group(1) != 'Q' else f'Q@{i}'
            cur, buf = f'想定問答 {q}', []
            g.add(cur, 'qa', plain(f"{q if not q.startswith('Q@') else 'Q'} {m.group(2)}", 48), 'hub:想定問答',
                  src=f'{rel(QA)}:{i}', text=m.group(2))
            continue
        if cur:
            buf.append(line)
    if cur:
        g.bodies[cur] = ('\n'.join(buf), 'qa')

    # ── 実験系譜: 索引の主題 → 段 ──
    tl = LEDGER.read_text(encoding='utf-8').splitlines()
    theme_of, in_idx, theme = {}, False, None
    for i, line in enumerate(tl, 1):
        if line.startswith('## 📇 第9段の索引'):
            in_idx = True
            continue
        if in_idx and line.startswith('## '):
            break
        if in_idx:
            m = re.match(r'^### ([A-Z])\. (.*)', line)
            if m:
                theme = f'索引 {m.group(1)}'
                g.add(theme, 'theme', f'{m.group(1)}. {plain(m.group(2), 30)}', 'hub:系譜', src=f'{rel(LEDGER)}:{i}')
            elif theme and line.startswith('|'):
                first = line.strip('|').split('|')[0]
                for d in re.findall(r'(?<![\d.\-])(\d-\d{1,3}[a-z]?)(?![\d\-])', first):
                    theme_of.setdefault(d, theme)
    head_re = re.compile(r'^#{2,4}\s+(.*?)(?<![\d\-])(\d-\d{1,3}[a-z]?)\.\s*(.*)')
    cur, buf = None, []
    for i, line in enumerate(tl, 1):
        m = head_re.match(line)
        if m and not line.startswith('## 📇'):
            if cur:
                g.bodies[cur] = ('\n'.join(buf), 'ledger')
            d, pre, title = m.group(2), m.group(1), m.group(3)
            flag = 'retracted' if '⛔' in pre or title.lstrip('*').startswith('⛔') else (
                'corrected' if '⚠️' in pre else '')
            dan = d.split('-')[0]
            if d in theme_of:
                parent = theme_of[d]
            elif dan != '9':
                parent = f'第{dan}段'
                g.add(parent, 'theme', f'第{dan}段', 'hub:系譜')
            else:
                n = int(re.match(r'\d+', d.split('-')[1]).group(0))
                lo = n // 50 * 50
                parent = f'第9段 {lo}'
                g.add(parent, 'theme', f'第9段 {max(lo, 1)}〜{lo + 49}（索引の外）', 'hub:系譜')
            cur, buf = f'系譜 {d}', []
            g.add(cur, 'ledger', plain(f'{d} {title}', 50), parent, src=f'{rel(LEDGER)}:{i}', flag=flag)
            continue
        if cur:
            buf.append(line)
            if len(buf) <= 12 and not g.nodes[cur]['text'] and line.strip() and not line.startswith(('|', '#', '```')):
                g.nodes[cur]['text'] = plain(line, 360)
    if cur:
        g.bodies[cur] = ('\n'.join(buf), 'ledger')
    # 索引の I は後から足した段の受け皿になっていて 184 段ある（10-23 時点）。眺められるよう 50 段ずつに分ける
    kids = collections.defaultdict(list)
    for n in g.nodes.values():
        if n['kind'] == 'ledger':
            kids[n['parent']].append(n)
    for theme, ns in kids.items():
        if len(ns) <= 40:
            continue
        for n in ns:
            num = int(re.match(r'\d+', n['id'].split('-')[1]).group(0))
            lo = num // 50 * 50
            sub = f'{theme} {lo}'
            g.add(sub, 'theme', f"{g.nodes[theme]['label'][:14]}… 9-{max(lo, 1)}〜{lo + 49}", theme)
            n['parent'] = sub

    # ── デバッグ戦記 ──
    tl = DEBUG.read_text(encoding='utf-8').splitlines()
    cur, buf = None, []
    for i, line in enumerate(tl, 1):
        m = re.match(r'^#{2,4}\s+Bug\s+(\d+)[:：]?\s*(.*)', line)
        if m:
            if cur:
                g.bodies[cur] = ('\n'.join(buf), 'bug')
            n = int(m.group(1))
            lo = (n - 1) // 10 * 10 + 1
            grp = f'Bug {lo}〜{lo + 9}'
            g.add(grp, 'theme', grp, 'hub:Bug')
            cur, buf = f'Bug {n}', []
            g.add(cur, 'bug', plain(f'Bug {n} {m.group(2)}', 50), grp, src=f'{rel(DEBUG)}:{i}', text=m.group(2))
            continue
        if cur:
            buf.append(line)
    if cur:
        g.bodies[cur] = ('\n'.join(buf), 'bug')
    return g


def refs(g: Graph, body: str, ctx: str) -> set:
    """本文に出てくる識別子 → ノード id。"""
    out = set()
    for d in re.findall(r'系譜\s?(\d-\d{1,3}[a-z]?)', body):
        out.add(f'系譜 {d}')
    for s, lo, hi in re.findall(r'(?<![\d.\-])(\d)-(\d{1,3})\s?[〜~]\s?(?:\d-)?(\d{1,3})(?![\d.])', body):  # 9-241〜9-247
        if 0 < int(hi) - int(lo) <= 30:
            out.update(f'系譜 {s}-{n}' for n in range(int(lo), int(hi) + 1))
    if ctx in ('ledger', 'summary'):     # 台帳と総括表の中は接頭辞なしで段を書く
        for d in re.findall(r'(?<![\d.\-/])(\d-\d{1,3}[a-z]?)(?![\d.\-]|\s*倍)', body):
            out.add(f'系譜 {d}')
    for n in re.findall(r'Bug\s?(\d+)', body):
        out.add(f'Bug {int(n)}')
    for q in re.findall(r'想定問答\s?の?\s?(Q\d+[a-z]?)', body):
        out.add(f'想定問答 {q}')
    if ctx == 'qa':
        for q in re.findall(r'(?<![A-Za-z])(Q\d+[a-z]?)(?![\d])', body):
            out.add(f'想定問答 {q}')
    nums = re.findall(r'修論\s?(?:付録\s?)?(?:[A-D]\s)?([0-9A-D](?:\.\d+)+)', body)
    if ctx in ('thesis', 'qa'):
        nums += re.findall(r'(?<![0-9.])([0-9A-D](?:\.\d+)+)\s*(?:節|項)', body)
    if ctx == 'qa':
        nums += re.findall(r'(?<![0-9A-Za-z.\-])([1-5](?:\.\d+){2,3})(?![0-9.])', body)
    for num in nums:
        if num in g.secs:
            out.add(g.secs[num])
    return {o for o in out if o in g.nodes}


def edges(g: Graph) -> collections.Counter:
    w = collections.Counter()
    for nid, (body, ctx) in g.bodies.items():
        for t in refs(g, body, ctx):
            if t != nid and g.nodes[t]['parent'] != nid and g.nodes[nid]['parent'] != t:
                w[tuple(sorted((nid, t)))] += 1
    return w


def payload(g: Graph, w) -> dict:
    ids = list(g.nodes)
    ix = {n: i for i, n in enumerate(ids)}
    nodes = [[n['id'], n['kind'], n['label'], ix.get(n['parent'], -1), n['src'], n['text'], n['flag']]
             for n in g.nodes.values()]
    import datetime
    import subprocess
    head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return dict(nodes=nodes, edges=[[ix[a], ix[b], c] for (a, b), c in w.items()], question=QUESTION,
                generated=f"{datetime.date.today():%Y-%m-%d}（{head} の md から）")


def reproduce() -> int:
    g = build()
    w = edges(g)
    want = [('系譜 9-285', 'Bug 76'), ('想定問答 Q4', '修論 4.4.3.1'), ('取り下げ 1', '系譜 9-246')]
    ok = True
    for a, b in want:
        hit = w.get(tuple(sorted((a, b))), 0)
        ok &= hit > 0
        print(f"{'✅' if hit else '⛔'} {a} — {b}: {hit}")
    return 0 if ok else 1


def main() -> int:
    if '--reproduce' in sys.argv:
        return reproduce()
    g = build()
    w = edges(g)
    kinds = collections.Counter(n['kind'] for n in g.nodes.values())
    print('ノード', len(g.nodes), dict(kinds), '/ つながり', len(w))
    if '--stats' in sys.argv:
        lonely = [n for n in g.nodes.values() if n['kind'] in ('ledger', 'bug', 'qa', 'finding', 'retracted')
                  and not any(n['id'] in e for e in w)]
        print('つながりの無いノード', len(lonely))
        return 0
    html = TEMPLATE.read_text(encoding='utf-8').replace(
        '/*DATA*/null', json.dumps(payload(g, w), ensure_ascii=False, separators=(',', ':')))
    OUT.write_text(html, encoding='utf-8')
    print(f'→ {rel(OUT)}（{len(html) // 1024} KB）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
