#!/usr/bin/env python3
"""想定問答.md から、読むためのページ 想定問答.html を作る（2026-11-25）。

md が正（③ 派生）。ページは md の写しなので、md を直したらこれを回し直し、
Artifact に公開し直す（公開先は想定問答.md の冒頭）。

取り出すもの:
- 目次（「発表で聞かれる順の目次」）の ★★★ / ★★ の表: 順・場面・Q・素人の言い方
- 目次の ★ の行と「対象外」の行に出てくる Q 番号
- 各 Q（見出し「## Q1:」「### Q7.」など）: 見出しの問い、直下の「🗣️ **30 秒**:」の行、
  「言ってはいけない」を含む段落と「- ❌」の行、全文

使い方:
    python3 scripts/build_qa_page.py            # 想定問答.html を書く
    python3 scripts/build_qa_page.py --check    # 書かずに件数だけ出す
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'docs', '研究応用', '執筆', '想定問答.md')
OUT = os.path.join(ROOT, 'docs', '研究応用', '執筆', '想定問答.html')

HEAD_RE = re.compile(r'^(#{2,3}) (Q\d+b?)[.:：]\s*(.*)$')


def strip_links(md):
    """ページの外では相対リンクが切れるので、文字だけ残す（http は残す）。"""
    return re.sub(r'\[([^\]]+)\]\((?!https?://)[^)]*\)', r'\1', md)


def split_questions(text):
    lines = text.split('\n')
    qs, cur = [], None
    for ln in lines:
        m = HEAD_RE.match(ln)
        if m:
            if cur:
                qs.append(cur)
            cur = {'num': m.group(2), 'title': m.group(3).strip(), 'lines': []}
            continue
        if cur is not None:
            cur['lines'].append(ln)
    if cur:
        qs.append(cur)
    # Q14 は 2 つある（縦型のシード数・関節軸）。2 つ目を Q14-axes とする
    seen = {}
    for q in qs:
        n = q['num']
        seen[n] = seen.get(n, 0) + 1
        q['id'] = n if seen[n] == 1 else f'{n}-axes'
    return qs


def lay_question(title):
    m = re.search(r'「(.+)」', title)
    return m.group(1) if m else title


def short_answer(lines):
    for ln in lines[:6]:
        m = re.match(r'^> 🗣️ \*\*30 秒\*\*: ?(.*)$', ln)
        if m:
            return m.group(1).strip()
    return ''


def taboo(lines):
    out = []
    paras, buf = [], []
    for ln in lines:
        if ln.strip() == '':
            if buf:
                paras.append('\n'.join(buf))
                buf = []
        else:
            buf.append(ln)
    if buf:
        paras.append('\n'.join(buf))
    for p in paras:
        if '言ってはいけない' in p or 'やってはいけない' in p or re.search(r'^\s*- ❌', p, re.M):
            out.append(p)
    return '\n\n'.join(out)


def parse_index(text):
    """目次の表と ★ 行から、段・順・場面・素人の言い方を取る。"""
    start = text.find('## ⭐⭐ 発表で聞かれる順の目次')
    end = text.find('\n---', start)
    block = text[start:end]
    tier = None
    info, order = {}, []
    for ln in block.split('\n'):
        if ln.startswith('### ★★★'):
            tier = 3
        elif ln.startswith('### ★★'):
            tier = 2
        elif ln.startswith('### ★'):
            tier = 1
        elif ln.startswith('### 対象外'):
            tier = 0
        if tier in (3, 2) and ln.startswith('| ') and not ln.startswith('| 順') and not ln.startswith('|---'):
            cells = [c.strip() for c in ln.strip('|').split('|')]
            if len(cells) >= 4:
                qcell = cells[2]
                ids = re.findall(r'Q\d+b?', qcell)
                for qid in ids:
                    key = 'Q14-axes' if (qid == 'Q14' and '関節軸' in qcell) else qid
                    if key in info:
                        continue
                    info[key] = {'tier': tier, 'scene': cells[1], 'lay': cells[3].strip('「」'), 'no': cells[0]}
                    order.append(key)
        elif tier in (1, 0) and ln and not ln.startswith('###') and not ln.startswith('>'):
            body = re.sub(r'~~.*?~~', '', ln)
            for qid in re.findall(r'Q\d+b?', body):
                key = qid
                if tier == 1 and qid == 'Q14':
                    key = 'Q14'
                if key in info:
                    continue
                info[key] = {'tier': tier, 'scene': '', 'lay': '', 'no': ''}
                order.append(key)
    return info, order


def build(check=False):
    text = open(SRC, encoding='utf-8').read()
    qs = split_questions(text)
    info, order = parse_index(text)
    byid = {q['id']: q for q in qs}
    items = []
    rank = {k: i for i, k in enumerate(order)}
    for q in qs:
        meta = info.get(q['id'], {'tier': -1, 'scene': '', 'lay': '', 'no': ''})
        body = '\n'.join(q['lines']).strip()
        body = re.sub(r'^> 🗣️ \*\*30 秒\*\*:.*\n?', '', body, flags=re.M).strip()
        body = re.sub(r'\n---\s*$', '', body).strip()
        items.append({
            'id': q['id'], 'num': q['num'], 'tier': meta['tier'], 'scene': meta['scene'],
            'lay': meta['lay'] or lay_question(q['title']), 'title': q['title'],
            'short': short_answer(q['lines']), 'taboo': strip_links(taboo(q['lines'])),
            'full': strip_links(body),
        })
    def sort_key(it):
        if it['id'] in rank:
            return (0, rank[it['id']])
        n = int(re.sub(r'\D', '', it['num']) or 0)
        return (1, n)
    items.sort(key=sort_key)
    missing = [k for k in order if k not in byid]
    stats = {
        'questions': len(items),
        'tier3': sum(1 for i in items if i['tier'] == 3),
        'tier2': sum(1 for i in items if i['tier'] == 2),
        'tier1': sum(1 for i in items if i['tier'] == 1),
        'excluded': sum(1 for i in items if i['tier'] == 0),
        'not_in_index': [i['id'] for i in items if i['tier'] == -1],
        'short_answers': sum(1 for i in items if i['short']),
        'index_ids_without_section': missing,
    }
    print(json.dumps(stats, ensure_ascii=False))
    if check:
        return
    tpl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qa_page_template.html')
    tpl = open(tpl_path, encoding='utf-8').read()
    data = json.dumps(items, ensure_ascii=False).replace('</', '<\\/')
    html = tpl.replace('/*__DATA__*/[]', data)
    open(OUT, 'w', encoding='utf-8').write(html)
    print(f'[write] {OUT} ({len(html)} bytes)')


if __name__ == '__main__':
    build(check='--check' in sys.argv)
