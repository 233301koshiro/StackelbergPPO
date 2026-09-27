#!/usr/bin/env python3
"""**コードと md がずれていないか**を機械で見る（2026-09-27 新設）。

⚠️ **既存の 4 本が見ていない領域である。**
`check_docs_consistency`（値）・`check_stale_claims`（主張）・`check_docs_inventory`（孤立・リンク）
・`check_citations`（引用）は **docs の中だけ**を見る。
⛔ **「コードを変えたのに md が追随していない」は素通りする。**

検査:
  A. スクリプトが **2 箇所とも**に載っているか（CLAUDE.md §4-2「2 箇所に書く構造」）
  B. md が参照しているスクリプト・XML・cfg が実在するか
  C. 環境変数で挙動が変わるコードが md に書かれているか
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / 'docs'
R1 = ROOT / 'scripts' / 'README.md'
R2 = DOCS / 'リポジトリ説明' / '評価スクリプト.md'

# 索引に載せなくてよいもの（使い捨て・内部）
SKIP = {'__init__.py'}

# ⚠️ **除外は「恒久的に正しい」ものだけ**（CLAUDE.md §5-2 ①: 閾値を緩めて 0 件にしない）
#   ⭐ `queue_*.sh` は run ごとの使い捨てで、**中身そのものが事前登録**である。
#     索引ではなく**実験系譜から辿れること**が要件なので、A では見ず D で見る。
QUEUE = re.compile(r'^queue_.*\.sh$')
#   ⭐ md に出てくる説明用のプレースホルダ（実在しないのが正しい）
PLACEHOLDERS = {'my_robot.xml', 'my_robot.yml'}
#   ⭐ 「消した」と明記している行の参照は正しい（過去の記録）
HISTORICAL = re.compile(r'削除|現存しない|当時|かつて|廃止|取り下げ|旧版|無くなった|消した'
                        # ⭐ **「無い」と md 自身が書いている行**は正しい記録
                        r'|存在しない|が無い|はない|❌|擬似コード|構想|scratchpad/')


def md_texts():
    return {p: p.read_text(encoding='utf-8', errors='replace') for p in DOCS.rglob('*.md')}


def main():
    issues = 0
    t1 = R1.read_text(encoding='utf-8', errors='replace')
    t2 = R2.read_text(encoding='utf-8', errors='replace')
    mds = md_texts()
    all_md = '\n'.join(mds.values()) + t1

    # --- A. 2 箇所に載っているか ---
    scripts = sorted(p.name for p in (ROOT / 'scripts').iterdir()
                     if p.suffix in ('.py', '.sh') and p.name not in SKIP)
    idx = [s for s in scripts if not QUEUE.match(s)]
    miss1 = [s for s in idx if s not in t1]
    miss2 = [s for s in idx if s not in t2]
    both = [s for s in idx if s in miss1 and s in miss2]
    print(f'【A】スクリプト {len(idx)} 本の索引（queue_*.sh {len(scripts)-len(idx)} 本は D で見る）')
    print(f'    scripts/README.md に無い     : {len(miss1)}')
    print(f'    評価スクリプト.md に無い      : {len(miss2)}')
    print(f'    ⛔ **両方に無い**             : {len(both)}')
    for s in both:
        print(f'        {s}')
    issues += len(both)

    # --- B. md が参照する実体があるか ---
    print('\n【B】md が参照しているのに実在しないもの')
    # ⚠️ **ファイル名はリポジトリ全体から探す。**`pusher.py` は design_opt/envs/、
    #   `mujoco_env_choreonoid.py` は khrylib/rl/envs/common/ にあり、
    #   scripts/ だけ見ると**実在するものを「無い」と誤検出する**（2026-09-27 に 33 件中 20 件超が誤検出）。
    here = set()
    for d, _sub, files in os.walk(ROOT):
        if '/.git' in d or '/single_run' in d:
            continue
        here.update(files)
    pats = [(r'`([\w./-]+\.(?:py|sh))`', None),
            (r'assets/mujoco_envs/([\w-]+\.xml)', None),
            (r'design_opt/cfg/([\w-]+\.yml)', None)]
    bad = {}
    for p, text in mds.items():
        # ⚠️ archive/ は過去の記録。当時存在したものを指していて正しい
        if str(p.relative_to(DOCS)).startswith('archive/'):
            continue
        lines = text.split('\n')
        for pat, _ in pats:
            for m in re.finditer(pat, text):
                name = os.path.basename(m.group(1))
                if name in here or name in PLACEHOLDERS:
                    continue
                # ⭐ **「消した」と明記してある行は正しい歴史的記述**なので飛ばす
                #   （`record_cnoid_viewer.sh` は「不採用としてファイルごと削除済み。現存しない」
                #     と書いてあった。2026-09-27）
                # ⭐ **同じ行か、そのファイル内で「存在しない」と宣言されていれば正当。**
                #   ⚠️ 宣言は冒頭の注記にまとめて書くことが多いので、**行単位では拾えない**
                #     （2026-09-27 に 7 件が恒久的な誤検出として残った）。
                ln = lines[text[:m.start()].count('\n')]
                if HISTORICAL.search(ln):
                    continue
                if any(HISTORICAL.search(text[max(0, k - 200):k + 200])
                       for k in (mm.start() for mm in re.finditer(re.escape(name), text))):
                    continue
                bad.setdefault(name, set()).add(str(p.relative_to(DOCS)))
    for k in sorted(bad):
        print(f'    ⛔ {k}  ← {", ".join(sorted(bad[k])[:3])}')
    print(f'    合計 {len(bad)} 件')
    issues += len(bad)

    # --- C. 環境変数で挙動が変わるのに md に無いもの ---
    print('\n【C】環境変数のうち **md にも自分の docstring にも**書かれていないもの')
    envs = set()
    for p in list((ROOT / 'scripts').glob('*.py')) + [
            ROOT / 'khrylib' / 'rl' / 'envs' / 'common' / 'mujoco_env_choreonoid.py']:
        if not p.exists():
            continue
        for m in re.finditer(r"os\.environ(?:\.get)?[(\[]\s*'([A-Z][A-Z0-9_]+)'", p.read_text(
                encoding='utf-8', errors='replace')):
            envs.add(m.group(1))
    # ⚠️ **worker との IPC・内部用は除外**（利用者が触らないので md に書く意味が無い）
    INTERNAL = re.compile(r'^(MUJOCO|CNOID)_(REQ|RES)_FD$|^MUJOCO_WORKER_ID$|^VIEWER_LOG$')
    # ⭐ **自分の docstring に書いてあるなら md に無くてよい。**
    #   プローブの調整つまみ（`STEP_DEG` 等）は、そのスクリプトを使う人が読む場所にある。
    #   ⛔ **どこにも書いていないものだけが問題。**
    doc_has = set()
    for p_ in list((ROOT / 'scripts').glob('*.py')) + [
            ROOT / 'khrylib' / 'rl' / 'envs' / 'common' / 'mujoco_env_choreonoid.py']:
        if not p_.exists():
            continue
        src = p_.read_text(encoding='utf-8', errors='replace')
        head = src[:src.find('def ') if 'def ' in src else 4000]   # 冒頭の説明部
        for e_ in envs:
            if e_ in head:
                doc_has.add(e_)
    undoc = sorted(e for e in envs
                   if e not in all_md and not INTERNAL.match(e) and e not in doc_has)
    for e in undoc:
        print(f'    ⛔ {e}')
    print(f'    合計 {len(undoc)} 件 / 全 {len(envs)} 個')
    issues += len(undoc)

    # --- D. queue_*.sh が実験系譜から辿れるか ---
    led = (DOCS / '研究応用' / '台帳' / '実験系譜.md').read_text(encoding='utf-8', errors='replace')
    qs = [s for s in scripts if QUEUE.match(s)]
    orphan = []
    for q in qs:
        if q in led or q in t1:
            continue
        # ⭐ **キューの価値は事前登録＝起動する run。**
        #   run が台帳にあるなら、キューはそこから辿れる。
        src = (ROOT / 'scripts' / q).read_text(encoding='utf-8', errors='replace')
        runs = set(re.findall(r'hydra\.run\.dir=single_run/(\S+)', src))
        runs |= set(re.findall(r'^launch\s+(\S+)', src, re.M))
        if runs and any(r in led for r in runs):
            continue
        # ⭐ **キュー冒頭が「実験系譜 9-NNN」と段番号で名乗っていれば、それで辿れる。**
        #   （`queue_thresh20.sh` は run 名が変数だが「実験系譜 9-120」と書いてある。2026-09-27）
        seg = re.findall(r'(?:実験系譜|系譜)\s*(9-\d+)', src[:1500])
        if seg and any(f'### ' in led and s in led for s in seg):
            continue
        orphan.append((q, sorted(runs)[:3]))
    print(f'\n【D】queue_*.sh {len(qs)} 本のうち、**キュー名でも run 名でも台帳から辿れない**もの')
    for q, rr in orphan:
        print(f'    ⛔ {q}  （起動する run: {", ".join(rr) if rr else "不明"}）')
    print(f'    合計 {len(orphan)} 件')
    issues += len(orphan)

    print(f'\n{"=" * 56}')
    if issues:
        print(f'❌ 要対応 {issues} 件。**コードを変えたら md も直す**（CLAUDE.md §4-2）')
    else:
        print('✅ コードと md は同期している。')
    return 1 if issues else 0


sys.exit(main())
