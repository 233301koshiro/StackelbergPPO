#!/usr/bin/env python3
"""docs 全体から「事実と食い違っている可能性のある記述」を洗い出す。

**なぜ要るか**: CLAUDE.md §4-2 の反映先テーブルは「どこに書くか」を定めるが、
**テーブルに載っていないファイル**（配布資料・想定問答・凍結スナップショット等）が
更新から漏れる。2026-09-02 に実際に漏れた:

  - 論文6枚/2枚・中間発表原稿が「E2E は未実施」と書いていた（9/1 に 1 例通っている）
  - 第4章は 4.5.3 を直したのに 4.6 の要約が古いまま残っていた
  - 「系→システム」の統一が配布資料へ届いていなかった

トリガー駆動の逐次反映だけでは、**同じ主張が複数ファイルに散っている**場合に漏れる。
本スクリプトは主張の側から横断して検出する。

使い方: python3 scripts/check_stale_claims.py
終了コード: 検出 0 件なら 0、1 件以上なら 1
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / 'docs'

# 「系」は複合語・run 系統の略記が大半なので、**システムの意味で使われている形だけ**を狙う。
# 「系が動く」「系の成立」「系全体」「系を構築」のように、単独の名詞として使われるもの。
SYSTEM_SENSE = re.compile(
    r'(?<![ぁ-んァ-ヶ一-龥A-Za-z0-9`_])系(?=[がをはにのも]?\s*(?:動く|成立|構築|実現|全体|'
    r'として|主軸|の内側|が要求|の中核|を作った|が頑健|としての))')

# 「余裕 10 %」は**否定されたことを述べている文**が大半。反証語が近くにあれば正常。
REFUTED = re.compile(r'(?:間違いだった|根拠を失|否定され|反証|単調に悪化|しかし|当初は|改めた|'
                     r'成立していない|再設計|誤り|撤回|取り消)')

CHECKS = [
    dict(name='E2E を「未実施」と断定している',
         pat=r'(?:E2E|end-to-end)[^。\n]{0,25}(?:実走|通し実行)[^。\n]{0,10}は?(?:未実施|未実走|未着手)',
         why='2026-09-01 に A1 で 1 例通した。「1 例のみ」「成功率を出せる件数ではない」と書く'),
    dict(name='余裕 10 % を「推奨」として肯定的に書いている',
         pat=r'余裕[^。\n]{0,12}10\s*%[^。\n]{0,16}(?:推奨|併記|加え|示す)',
         why='実験で否定済み（9-11）。到達タスクでは余裕を広げるほど単調に悪化する',
         filter='refuted'),
    # IME の誤変換で日本語のつもりがキリル・ハングルになる事故が繰り返し起きている
    # （2026-08 に 명示→明示・성績→成績、2026-09-02 に 手описき）。
    # ギリシャ文字は τ φ π Δ θ など数式記号として正当なので**対象にしない**。
    dict(name='キリル・ハングルの混入（IME の誤変換）',
         pat=r'[Ѐ-ӿ가-힯ᄀ-ᇿ]',
         why='日本語のつもりが別の文字になっている。見た目が似ていて気づきにくい'),
    dict(name='「系」をシステムの意味で使っている',
         pat=SYSTEM_SENSE,
         why='2026-08-29 に「システム」へ統一した。run 系統の略記・複合語（報酬系 等）は対象外'),
]

# 凍結スナップショットなど、意図的に古いまま残すファイル
FROZEN = {'要旨_詳細版.md'}


def main() -> int:
    total = 0
    for p in sorted(DOCS.rglob('*.md')):
        if 'archive' in p.parts:
            continue
        s = p.read_text(encoding='utf-8')
        frozen = p.name in FROZEN
        for c in CHECKS:
            hits = []
            for m in re.finditer(c['pat'], s):
                if c.get('filter') == 'refuted':
                    # 同じ段落に反証の記述があれば「否定した文」なので正常
                    # 段落境界に頼ると「撤回」が次行にある場合を取りこぼすので
                    # 前後 400 字の窓で反証語を探す
                    para = s[max(0, m.start() - 400):m.end() + 400]
                    if REFUTED.search(para):
                        continue
                hits.append(s[max(0, m.start() - 30):m.start() + 40].replace('\n', ' '))
            if not hits:
                continue
            if frozen:
                print(f"  ⏸ {p.relative_to(DOCS)}: {c['name']} {len(hits)} 件"
                      f"（凍結ファイルなので対象外）")
                continue
            total += len(hits)
            print(f"  ❌ {p.relative_to(DOCS)}: {c['name']} {len(hits)} 件")
            print(f"     → {c['why']}")
            for h in hits[:3]:
                print(f"       …{h}…")

    print()
    if total == 0:
        print("✅ 古い主張は検出されなかった。")
        return 0
    print(f"❌ 合計 {total} 件。**同じ主張が他のファイルにも無いか grep で洗うこと**"
          "（CLAUDE.md §8「訂正した主張を記憶を頼りに一部のMDだけ直して終わりにしない」）。")
    return 1


if __name__ == '__main__':
    sys.exit(main())
