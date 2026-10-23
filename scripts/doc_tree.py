#!/usr/bin/env python3
"""md の木 — どの md が、何のために読まれているかを根からの距離で並べる（2026-10-23、ユーザー提案）。

修論を「主張の木」にして根から遠い段落を刈ったのと同じことを、md でやる。
**木は手で書かない。** md にすでに書いてある参照（リンク・ファイル名・`系譜 9-49`・`修論 4.4.3.1`・
`Bug 76`・`想定問答 Q14`）から生成する（ドキュメントの構造と監査.md の「① 生成物にする」）。

根は 3 つ。md は読む目的が 3 つあるので、修論のように 1 本にはならない。
  提出: PDF に入る修論の章（build_thesis_pdf.py の CHAPTER_ORDER）
  審査: 想定問答・口語版
  運用: CLAUDE.md と、そこから毎回読む入口（新しいチャットの始め方・コマンド）

辺は「A が B を参照している」なら A → B（B は A を支える子）。根から幅優先で辿り、
距離が 1 つ小さい親のうち参照の多いものを親にする。**本当の構造はグラフで、木はその骨格。**
⚠️ 「ここを変えたらどこが古くなるか」は木では答えられない（親を 1 つしか持たないため）。
それは `doc_refs.py <識別子>` で引く。この木は棚卸しで**刈る候補を探す**ためのもの。

    python3 scripts/doc_tree.py              # 根ごとの木・どの根からも辿れない md・遠い md
    python3 scripts/doc_tree.py --drawio     # docs/リポジトリ説明/md_の木.drawio を書く（生成物）
    python3 scripts/doc_tree.py --reproduce  # 既知の参照（修論 4.4.3.1 → 第4章、系譜 → 実験系譜）を辿れるか
"""
import collections
import re
import subprocess
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / 'docs'
DRAFT = DOCS / '研究応用/修論ドラフト'
LEDGER = DOCS / '研究応用/台帳/実験系譜.md'
DEBUG = DOCS / 'デバッグ戦記.md'
QA = DOCS / '研究応用/執筆/想定問答.md'
OUT = DOCS / 'リポジトリ説明/md_の木.drawio'

# 索引。全 md へのリンクを持つので、辿るとすべてが根から 2 段に入り、刈る候補が 0 になる（10-23 の初回の実測）。
# 索引に載っていることは「読まれている」ことを意味しないので、ここからの辺は辿らない（索引自身は木に載る）
NAV = [DOCS / '目次.md']

ROOTS = {
    '提出': [DRAFT / n for n in (
        '要旨.md', '第1章_序論.md', '第2章_関連研究.md', '第3章前段_前提.md', '第3章_提案手法.md',
        '第4章_実験および評価.md', '第5章_結論.md', '参考文献.md',
        '付録A_プロンプト全文.md', '付録D_再現情報と補足実験.md')],
    '審査': [QA] + sorted((DOCS / '研究応用/執筆/修論_口語版').glob('*.md')),
    '運用': [ROOT / 'CLAUDE.md', DOCS / 'リポジトリ説明/新しいチャットの始め方.md']
            + sorted((ROOT / '.claude/commands').glob('*.md')),
}


def nodes() -> list:
    ps = [p for p in DOCS.glob('**/*.md') if '/archive/' not in str(p)]
    ps += [ROOT / 'CLAUDE.md', ROOT / 'scripts/README.md'] + sorted((ROOT / '.claude/commands').glob('*.md'))
    return [p for p in ps if p.exists()]


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def thesis_sections() -> dict:
    """修論の節番号 → それを見出しに持つ章ファイル。上位の番号も同じ章へ。"""
    out = {}
    for p in DRAFT.glob('*.md'):
        for m in re.finditer(r'^#{2,5}\s+(?:付録)?([0-9A-D]+(?:\.[0-9]+)+)\s', p.read_text(encoding='utf-8'), re.M):
            parts = m.group(1).split('.')
            for i in range(2, len(parts) + 1):
                out.setdefault('.'.join(parts[:i]), p)
    return out


def edges(ns: list) -> dict:
    """(A, B) → 参照の回数。A が B を参照している。"""
    by_name = collections.defaultdict(list)
    for p in ns:
        by_name[p.name].append(p)
        by_name[p.stem].append(p)
    # ファイル名で引けるのは一意で 4 文字以上のものだけ（「構成」「要旨」は一般語と紛れる）
    stems = {s: v[0] for s, v in by_name.items() if len(v) == 1 and len(s) >= 4 and not s.endswith('.md')}
    stem_re = re.compile('|'.join(sorted(map(re.escape, stems), key=len, reverse=True)))
    secs = thesis_sections()
    w = collections.Counter()
    nav = {p.resolve() for p in NAV}
    for a in ns:
        if a.resolve() in nav:
            continue
        text = a.read_text(encoding='utf-8')
        found = []
        for m in re.finditer(r'\]\(([^)#\s]+\.md)', text):          # リンク
            t = (a.parent / m.group(1)).resolve()
            if t.exists():
                found.append(t)
        for m in re.finditer(r'([^\s/`()\[\]|「」（）]+\.md)', text):   # リンクでないファイル名
            if len(by_name.get(m.group(1), [])) == 1:
                found.append(by_name[m.group(1)][0])
        found += [stems[m.group(0)] for m in stem_re.finditer(text)]
        found += [LEDGER] * len(re.findall(r'系譜\s?\d-\d', text))
        found += [DEBUG] * len(re.findall(r'Bug\s?\d', text))
        found += [QA] * len(re.findall(r'想定問答\s?Q\d', text))
        for m in re.finditer(r'修論\s?(?:付録\s?)?(?:[A-D]\s)?([0-9A-D]+(?:\.[0-9]+)+)', text):
            if m.group(1) in secs:
                found.append(secs[m.group(1)])
        if DRAFT in a.parents or '修論_口語版' in str(a):     # 修論の中は接頭辞なしで「4.4.2 節」と書く
            for m in re.finditer(r'(?<![0-9.])([0-9A-D]+(?:\.[0-9]+)+)\s*(?:節|項)', text):
                if m.group(1) in secs:
                    found.append(secs[m.group(1)])
        for b in found:
            b = Path(b).resolve()
            if b != a.resolve():
                w[(a.resolve(), b)] += 1
    return w


def build():
    ns = [p.resolve() for p in nodes()]
    w = edges(ns)
    out_e = collections.defaultdict(list)
    for (a, b), n in w.items():
        out_e[a].append((b, n))
    dist, parent, root_of = {}, {}, {}
    frontier = []
    for r, ps in ROOTS.items():
        for p in ps:
            p = p.resolve()
            if p.exists() and p not in dist:
                dist[p], root_of[p] = 0, r
                frontier.append(p)
    while frontier:
        cand = collections.defaultdict(list)          # 次の層: 子 → [(重み, 親)]
        for a in frontier:
            for b, n in out_e[a]:
                if b not in dist and b in set(ns):
                    cand[b].append((n, a))
        nxt = []
        for b, ps in cand.items():
            n, a = max(ps, key=lambda x: (x[0], rel(x[1])))
            dist[b], parent[b], root_of[b] = dist[a] + 1, a, root_of[a]
            nxt.append(b)
        frontier = nxt
    return ns, w, dist, parent, root_of


def last_commit(p: Path) -> str:
    r = subprocess.run(['git', 'log', '-1', '--format=%ad', '--date=format:%m-%d', '--', str(p)],
                       capture_output=True, text=True, cwd=ROOT)
    return r.stdout.strip() or '--'


def frozen(p: Path) -> bool:
    return 'doc-policy: frozen' in p.read_text(encoding='utf-8')[:600]


def report(ns, w, dist, parent, root_of):
    kids = collections.defaultdict(list)
    for c, p in parent.items():
        kids[p].append(c)
    indeg = collections.Counter(b for (a, b) in w)

    def show(p, depth):
        lines = len(p.read_text(encoding='utf-8').splitlines())
        mark = ' 🧊' if frozen(p) else ''
        print(f"{'  ' * depth}{'└ ' if depth else ''}{p.name}  （{lines} 行・最終 {last_commit(p)}・被参照 {indeg[p]}）{mark}")
        for c in sorted(kids[p], key=lambda x: -indeg[x]):
            show(c, depth + 1)

    for r, ps in ROOTS.items():
        print(f'\n━━ 根「{r}」')
        for p in ps:
            if p.exists() and dist.get(p.resolve()) == 0 and root_of[p.resolve()] == r:
                show(p.resolve(), 0)

    lost = [p for p in ns if p not in dist]
    print(f'\n━━ ⛔ どの根からも辿れない md: {len(lost)} 本（刈る候補。消す前に全文を読む。孤立＝不要ではない）')
    for p in sorted(lost, key=rel):
        print(f'  {rel(p)}  （最終 {last_commit(p)}・被参照 {indeg[p]}）{" 🧊" if frozen(p) else ""}')
    far = [p for p in ns if dist.get(p, 0) >= 3]
    print(f'\n━━ ⚠️ 根から 3 段以上遠い md: {len(far)} 本（根に効く経路が細い。経路を見て、要るかを判断する）')
    for p in sorted(far, key=lambda x: (-dist[x], rel(x))):
        chain, q = [], p
        while q in parent:
            q = parent[q]
            chain.append(q.name)
        print(f'  {dist[p]} 段  {rel(p)}  ← ' + ' ← '.join(chain))
    print(f'\n計 {len(ns)} 本 / 辿れた {len(dist)} 本。⚠️ 木は親を 1 つしか持たない。波及の確認は doc_refs.py。')


def drawio(ns, w, dist, parent, root_of):
    kids = collections.defaultdict(list)
    for c, p in parent.items():
        kids[p].append(c)
    color = [('研究応用/修論ドラフト', '#dae8fc'), ('研究応用/執筆', '#e1d5e7'), ('研究応用/台帳', '#fff2cc'),
             ('デバッグ戦記', '#fff2cc'), ('研究応用/方針', '#d5e8d4'), ('研究応用/設計', '#f5f5f5'),
             ('リポジトリ説明', '#ffe6cc'), ('', '#ffffff')]
    cells, ids = [], {}
    W, H, DX, DY = 230, 34, 260, 42

    def box(label, x, y, fill, extra=''):
        i = f'n{len(cells)}'
        cells.append(f'<mxCell id="{i}" value="{escape(label, quote=True)}" style="rounded=1;whiteSpace=wrap;html=1;'
                     f'fontSize=10;fillColor={fill};{extra}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" '
                     f'width="{W}" height="{H}" as="geometry"/></mxCell>')
        return i

    row = [0]

    def place(p, depth):
        fill = next(c for k, c in color if k in rel(p))
        ids[p] = box(f'<b>{p.stem}</b><br><font style="font-size:8px" color="#666666">'
                     f'{rel(p).replace("docs/", "")}</font>', depth * DX + 40, row[0] * DY + 60, fill,
                     'dashed=1;' if frozen(p) else '')
        my_row = row[0]
        row[0] += 1
        for c in sorted(kids[p], key=rel):
            place(c, depth + 1)
        return my_row

    for r, ps in ROOTS.items():
        cells.append(f'<mxCell id="h{r}" value="根「{r}」" style="text;fontStyle=1;fontSize=13;" vertex="1" '
                     f'parent="1"><mxGeometry x="40" y="{row[0] * DY + 60}" width="200" height="24" as="geometry"/></mxCell>')
        row[0] += 1
        for p in ps:
            p = p.resolve()
            if p.exists() and dist.get(p) == 0 and root_of[p] == r:
                place(p, 0)
        row[0] += 1
    lost = [p for p in ns if p not in dist]
    cells.append(f'<mxCell id="hlost" value="どの根からも辿れない（刈る候補）" style="text;fontStyle=1;fontSize=13;'
                 f'fontColor=#b85450;" vertex="1" parent="1"><mxGeometry x="40" y="{row[0] * DY + 60}" '
                 f'width="300" height="24" as="geometry"/></mxCell>')
    row[0] += 1
    for p in sorted(lost, key=rel):
        ids[p] = box(f'<b>{p.stem}</b><br><font style="font-size:8px">{rel(p).replace("docs/", "")}</font>',
                     40, row[0] * DY + 60, '#f8cecc')
        row[0] += 1
    for c, p in parent.items():
        cells.append(f'<mxCell id="e{ids[c]}" style="edgeStyle=orthogonalEdgeStyle;endArrow=none;strokeColor=#999999;" '
                     f'edge="1" parent="1" source="{ids[p]}" target="{ids[c]}"><mxGeometry relative="1" as="geometry"/></mxCell>')
    title = (f'md の木（生成物。手で編集しない。scripts/doc_tree.py --drawio で作り直す）'
             f' — 色: 青 修論 / 紫 執筆 / 黄 台帳 / 緑 方針 / 灰 設計 / 橙 リポジトリ説明。破線は凍結')
    cells.insert(0, f'<mxCell id="title" value="{escape(title, quote=True)}" style="text;fontSize=12;" vertex="1" '
                    f'parent="1"><mxGeometry x="40" y="10" width="1200" height="30" as="geometry"/></mxCell>')
    xml = ('<mxfile><diagram name="md の木"><mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
           + ''.join(cells) + '</root></mxGraphModel></diagram></mxfile>')
    OUT.write_text(xml, encoding='utf-8')
    print(f'→ {rel(OUT)}（{len(ids)} 本）')


def reproduce() -> int:
    """既知の参照を辿れるかを確かめる。辿れない検査の「孤立」は根拠にならない。"""
    ns, w, *_ = build()
    want = [(DRAFT / '第5章_結論.md', DRAFT / '第4章_実験および評価.md', '修論 4.x の参照'),
            (QA, LEDGER, '系譜 9-xxx の参照'),
            (QA, DEBUG, 'Bug N の参照'),
            (ROOT / 'CLAUDE.md', DOCS / '研究応用/引き継ぎ_再起動後.md', 'リンク')]
    ok = True
    for a, b, why in want:
        hit = w.get((a.resolve(), b.resolve()), 0)
        ok &= hit > 0
        print(f"{'✅' if hit else '⛔'} {a.name} → {b.name}（{why}）: {hit} 回")
    return 0 if ok else 1


def main() -> int:
    if '--reproduce' in sys.argv:
        return reproduce()
    t = build()
    if '--drawio' in sys.argv:
        drawio(*t)
    else:
        report(*t)
    return 0


if __name__ == '__main__':
    sys.exit(main())
